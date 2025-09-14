
# -*- coding: utf-8 -*-
"""
lib/tiker_data.py — селектор тикера/экспираций и подготовка raw_records/spot в st.session_state.
Работает ТОЛЬКО с Polygon: использует provider_polygon.fetch_option_chain / fetch_stock_history,
которые возвращают "yahoo-like" структуру.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os
from datetime import datetime

import pandas as pd
import streamlit as st

# --- Provider imports (prefer relative, fallback absolute) ---
try:
    from .provider_polygon import fetch_option_chain, fetch_stock_history
except Exception:
    from provider_polygon import fetch_option_chain, fetch_stock_history  # type: ignore

# ---------- Dataclasses ----------

@dataclass
class TikerRawResult:
    ticker: str
    spot: Optional[float]
    expirations: List[str]
    selected: List[str]
    raw_by_exp: Dict[str, List[dict]]
    ohlc: Optional[pd.DataFrame] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None

# ---------- Helpers ----------

def _fmt_date(val) -> str:
    """to 'YYYY-MM-DD'"""
    if val is None:
        return ""
    if isinstance(val, str):
        # already label
        if len(val) >= 8 and "-" in val:
            return val.split("T")[0]
        try:
            # numeric string
            ts = int(val)
            return datetime.utcfromtimestamp(ts).date().isoformat()
        except Exception:
            return val
    try:
        return datetime.utcfromtimestamp(int(val)).date().isoformat()
    except Exception:
        return str(val)

def _extract_from_chain_yf(chain: dict) -> Tuple[Optional[float], List[str], Dict[str, List[dict]]]:
    """
    Ожидаем структуру:
    {"optionChain":{"result":[{"quote":{...},"options":[{"expirationDate":unix,"calls":[...],"puts":[...]}...]}]}}
    Возвращаем: spot, expirations(list[str]), raw_by_exp{exp:list[dict]}
    Каждая запись дополняется полем "type": "call"/"put".
    """
    if not isinstance(chain, dict):
        return None, [], {}
    oc = (chain.get("optionChain") or {}).get("result") or []
    if not oc:
        return None, [], {}
    root = oc[0] or {}
    q = root.get("quote") or {}
    spot = q.get("regularMarketPrice") or q.get("last") or q.get("price")
    try:
        spot = float(spot) if spot is not None else None
    except Exception:
        spot = None

    raw_by_exp: Dict[str, List[dict]] = {}
    expirations: List[str] = []

    for blk in (root.get("options") or []):
        # gather expiration label
        label = None
        # use contract string expiration if present
        for rec in (blk.get("calls") or []) + (blk.get("puts") or []):
            if isinstance(rec, dict) and rec.get("expiration"):
                label = str(rec["expiration"])
                break
        if label is None:
            label = _fmt_date(blk.get("expirationDate"))
        label = _fmt_date(label)
        if not label:
            # skip if cannot determine
            continue
        expirations.append(label)
        lst: List[dict] = raw_by_exp.setdefault(label, [])
        for rec in (blk.get("calls") or []):
            r = dict(rec); r["type"] = "call"; lst.append(r)
        for rec in (blk.get("puts") or []):
            r = dict(rec); r["type"] = "put"; lst.append(r)

    expirations = sorted(set(expirations))
    return spot, expirations, raw_by_exp

def _get_api_key() -> Optional[str]:
    for k in ("POLYGON_API_KEY", "RAPIDAPI_KEY"):
        v = os.environ.get(k)
        if v:
            return v
    return None

# ---------- Public API ----------

def get_raw_by_exp(ticker: str,
                   selected_exps: Optional[List[str]] = None,
                   need_ohlc: bool = True,
                   ohlc_interval: str = "1m",
                   ohlc_limit: int = 500) -> TikerRawResult:
    """Скачивает цепочку и (опц.) свечи, разбивает контракты по датам экспирации."""
    api_key = _get_api_key()
    # fetch option chain (yahoo-like remapped)
    chain_json, _ = fetch_option_chain(ticker=ticker, api_key=api_key, expiry_unix=None)  # type: ignore
    spot, expirations, raw_by_exp = _extract_from_chain_yf(chain_json)

    # default selection
    if not selected_exps:
        selected_exps = []
        if expirations:
            # ближайшая по дате
            try:
                parsed = [datetime.strptime(x, "%Y-%m-%d") for x in expirations]
                selected_exps = [expirations[parsed.index(sorted(parsed)[0])]]
            except Exception:
                selected_exps = [expirations[0]]

    df = None; day_hi = None; day_lo = None
    if need_ohlc:
        df, _ = fetch_stock_history(ticker=ticker, api_key=api_key, interval=ohlc_interval, limit=int(ohlc_limit), dividend=False, timeout=30)  # type: ignore
        if isinstance(df, pd.DataFrame) and not df.empty:
            day_hi = float(df["high"].max()) if "high" in df.columns else None
            day_lo = float(df["low"].min()) if "low" in df.columns else None

    return TikerRawResult(
        ticker=ticker, spot=spot, expirations=expirations,
        selected=selected_exps, raw_by_exp=raw_by_exp,
        ohlc=df, day_high=day_hi, day_low=day_lo
    )

def render_tiker_data_block(stmod) -> None:
    """UI-блок: выбор тикера/экспираций + запись данных в st.session_state"""
    st = stmod
    st.markdown("**Ticker**")
    ticker = st.text_input("Ticker", value=st.session_state.get("td_ticker", "SPY"), key="td_ticker")
    # первичная загрузка — чтобы получить список дат
    base = get_raw_by_exp(ticker, selected_exps=None, need_ohlc=True, ohlc_interval="1m", ohlc_limit=500)

    if not base.expirations:
        st.warning("Провайдер не вернул список экспираций для данного тикера.")
        return

    selected = st.selectbox("Экспирация", options=base.expirations, index=max(0, base.expirations.index(base.selected[0]) if base.selected else 0), key="td_exp_select")
    # повторный вызов под выбранную дату
    base = get_raw_by_exp(ticker, selected_exps=[selected], need_ohlc=True, ohlc_interval="1m", ohlc_limit=500)

    # ---- Запись в session_state для downstream-блоков ----
    st.session_state["raw_records"] = base.raw_by_exp.get(selected, [])
    st.session_state["spot"] = base.spot
    st.session_state["_last_exp_sig"] = selected
    st.session_state["_last_ticker"] = ticker
    st.session_state["tiker_selected_exps"] = [selected]
    st.session_state["exp_locked_by_tiker_data"] = True

    # Покажем мини-инфо
    with st.expander("Инфо (данные для таблицы и Key Levels)"):
        st.write(f"Ticker: **{ticker}**  •  Exp: **{selected}**")
        st.write(f"Contracts: {len(st.session_state['raw_records'])}  •  Spot: {base.spot}")
        if base.ohlc is not None and not base.ohlc.empty:
            st.dataframe(base.ohlc.tail(5), use_container_width=True)
