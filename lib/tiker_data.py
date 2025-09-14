
from __future__ import annotations
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# External project deps (already present in your repo):
# - provider_polygon.fetch_option_chain

@dataclass
class TikerRawResult:
    ticker: str
    spot: float
    expirations: List[str]
    selected: List[str]
    raw_by_exp: Dict[str, List[dict]]

# -------- Provider helpers --------

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_chain(ticker: str) -> dict:
    from provider_polygon import fetch_option_chain
    chain = fetch_option_chain(ticker=ticker, host=None, key=None, expiry_unix=None)
    return chain or {}

def _extract_spot_and_exps(chain: dict) -> Tuple[float, List[str]]:
    quote = chain.get("quote") or {}
    spot = quote.get("regularMarketPrice", None)
    if spot is None:
        spot = quote.get("last", None)
    if spot is None:
        spot = 0.0
    spot = float(spot)

    exps = chain.get("expirations") or chain.get("expirationDates") or []
    exps = [str(x) for x in exps]
    exps = sorted(set(exps))
    return spot, exps

def _nearest_exp(expirations: List[str], today: Optional[str] = None) -> Optional[str]:
    if not expirations:
        return None
    try:
        today_dt = datetime.strptime(today, "%Y-%m-%d") if today else datetime.utcnow()
        parsed = []
        for e in expirations:
            try:
                parsed.append((e, datetime.strptime(e[:10], "%Y-%m-%d")))
            except Exception:
                continue
        if not parsed:
            return expirations[0]
        future = [e for e, d in parsed if d >= today_dt]
        if future:
            return sorted(future, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))[0]
        return sorted([e for e, _ in parsed], key=lambda x: datetime.strptime(x, "%Y-%m-%d"))[0]
    except Exception:
        return expirations[0]

def _record_matches_exp(rec: dict, exp: str) -> bool:
    for k in ("exp","expiration","expirationDate","expiry","expDate"):
        v = rec.get(k)
        if v is None:
            continue
        if str(v).startswith(exp):
            return True
    d = rec.get("details") or {}
    for k in ("exp","expiration","expirationDate"):
        v = d.get(k)
        if v and str(v).startswith(exp):
            return True
    return False

def _split_raw_by_exp(chain: dict, selected_exps: List[str]) -> Dict[str, List[dict]]:
    items = chain.get("options") or chain.get("records") or []
    if not items:
        return {e: [] for e in selected_exps}
    out: Dict[str, List[dict]] = {e: [] for e in selected_exps}
    # Distribute each record to matching expiration(s)
    for rec in items:
        for e in selected_exps:
            if _record_matches_exp(rec, e):
                out[e].append(rec)
    # Fallback: if filter empty and provider likely returned single-exp data, copy all
    for e in selected_exps:
        if not out[e]:
            out[e] = list(items)
    return out

# -------- Public UI/block --------

def render_tiker_data_block(title: str = "Блок выбора тикера и даты экспирации (сырьё)") -> TikerRawResult:
    st.header(title)

    # 1) Ticker input (default SPY)
    ticker_default = st.session_state.get("td2_ticker", "SPY")
    ticker = st.text_input("Ticker", value=ticker_default, key="td2_ticker_input")
    st.session_state["td2_ticker"] = ticker

    # Fetch provider chain & extract expirations
    chain = _fetch_chain(ticker)
    spot, exps = _extract_spot_and_exps(chain)
    if not exps:
        st.warning("Провайдер не вернул список экспираций для данного тикера.")
        return TikerRawResult(ticker=ticker, spot=spot, expirations=[], selected=[], raw_by_exp={})

    # 2) Default to nearest expiration
    default_exp = _nearest_exp(exps)
    selected = st.multiselect("Дата(ы) экспирации", options=exps, default=[default_exp] if default_exp else [])

    # 3) Build per-exp raw arrays
    raw_by_exp = _split_raw_by_exp(chain, selected)

    # 4) STOP here: sanitize_window.py is handled downstream by consumers of this block
    return TikerRawResult(
        ticker=ticker,
        spot=spot,
        expirations=exps,
        selected=selected,
        raw_by_exp=raw_by_exp,
    )
