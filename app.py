# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, List, Tuple

import streamlit as st


# --- Intraday price loader for Key Levels (Polygon v2 aggs 1-minute) ---
def _load_session_price_df_for_key_levels(ticker: str, session_date_str: str, api_key: str, timeout: int = 30):
    import pandas as pd
    import pytz
    t = (ticker or "").strip().upper()
    if not t or not session_date_str:
        return None
    base = "https://api.polygon.io"
    url = f"{base}/v2/aggs/ticker/{t}/range/1/minute/{session_date_str}/{session_date_str}?adjusted=true&sort=asc&limit=50000"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        js = r.json()
    except Exception:
        return None
    res = js.get("results") or []
    if not res:
        return None
    tz = pytz.timezone("America/New_York")
    # Build dataframe
    times   = [pd.to_datetime(x.get("t"), unit="ms", utc=True).tz_convert(tz) for x in res]
    price   = [x.get("c") for x in res]
    volume  = [x.get("v") for x in res]
    vwap_api= [x.get("vw") for x in res]
    opens   = [x.get("o") for x in res]
    highs   = [x.get("h") for x in res]
    lows    = [x.get("l") for x in res]
    closes  = [x.get("c") for x in res]
    df = pd.DataFrame({
        "time":   times,
        "price":  pd.to_numeric(price,  errors="coerce"),
        "volume": pd.to_numeric(volume, errors="coerce"),
    })
    # Add OHLC columns (used by candlesticks)
    try:
        df["open"]  = pd.to_numeric(opens,  errors="coerce")
        df["high"]  = pd.to_numeric(highs,  errors="coerce")
        df["low"]   = pd.to_numeric(lows,   errors="coerce")
        df["close"] = pd.to_numeric(closes, errors="coerce")
    except Exception:
        pass
    # VWAP: prefer API field 'vw', else compute cumulative
    if any(v is not None for v in vwap_api):
        vw = pd.Series(pd.to_numeric(vwap_api, errors="coerce"), index=df.index)
        vol = df["volume"].fillna(0.0)
        cum_vol = vol.cumsum().replace(0, pd.NA)
        df["vwap"] = (vw * vol).cumsum() / cum_vol
    else:
        vol = df["volume"].fillna(0.0)
        pr  = df["price"].fillna(method="ffill")
        cum_vol = vol.cumsum().replace(0, pd.NA)
        df["vwap"] = (pr.mul(vol)).cumsum() / cum_vol
    return df
# --- Helpers to hide tables from main page ---
def _st_hide_df(*args, **kwargs):
    # no-op: we suppress table rendering on main page per requirements
    return None
def _st_hide_subheader(*args, **kwargs):
    # no-op: suppress section headers for tables
    return None
from lib.netgex_chart import render_netgex_bars, _compute_gamma_flip_from_table
from lib.key_levels import render_key_levels

# Project imports
from lib.sanitize_window import sanitize_and_window_pipeline
from lib.tiker_data import (
    list_future_expirations,
    download_snapshot_json,
    get_spot_price,
    PolygonError,
)
from lib.final_table import FinalTableConfig, build_final_tables_from_corr, process_from_raw  # IMPROVED: Импорт configs

st.set_page_config(page_title="GammaStrat — df_raw", layout="wide")

# --- Helpers -----------------------------------------------------------------
def _coerce_results(data: Any) -> List[Dict]:
    """
    Приводит разные варианты JSON к list[dict] записей опционов.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "options", "data"):
            v = data.get(key)
            if isinstance(v, list):
                return v
    return []


def _infer_spot_from_snapshot(raw: List[Dict]) -> float | None:
    """
    Fallback-оценка S из снимка опционов:
    - Берём страйки и |delta| близкие к 0.5 (окно 0.2..0.8, вес 1/| |delta|-0.5 |)
    """
    num = 0.0
    den = 0.0

    def get_nested(d: Dict, keys: list[str], blocks=("details","greeks","day","underlying_asset")):
        for k in keys:
            if k in d:
                return d[k]
        for b in blocks:
            sub = d.get(b, {})
            if isinstance(sub, dict):
                for k in keys:
                    if k in sub:
                        return sub[k]
        return None

    for r in raw:
        K = get_nested(r, ["strike","k","strike_price","strikePrice"])
        dlt = get_nested(r, ["delta","dlt"])
        if K is None or dlt is None:
            continue
        try:
            K = float(K); dlt = float(dlt)
        except Exception:
            continue
        if 0.2 <= abs(dlt) <= 0.8 and K > 0:
            w = 1.0 / (abs(abs(dlt) - 0.5) + 1e-6)
            num += w * K
            den += w
    if den > 0:
        return num / den
    return None


def _get_api_key() -> str | None:
    key = None
    try:
        key = st.secrets.get("POLYGON_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    if not key:
        key = os.getenv("POLYGON_API_KEY")
    return key


# --- UI: Controls -------------------------------------------------------------

api_key = _get_api_key()
if not api_key:
    st.error("POLYGON_API_KEY не задан в Streamlit Secrets или переменных окружения.")
    st.stop()



# --- Input state helpers ------------------------------------------------------
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "SPY"

def _normalize_ticker():
    t = st.session_state.get("ticker", "")
    st.session_state["ticker"] = (t or "").strip().upper()

# --- Controls moved to sidebar ----------------------------------------------
with st.sidebar:
    st.text_input("Тикер", key="ticker", on_change=_normalize_ticker)
    ticker = st.session_state.get("ticker", "")

    # NEW: Toggle для advanced_mode
    advanced_mode = st.checkbox("Advanced Mode (улучшенные расчёты)", value=False)

    # Получаем список будущих экспираций под выбранный тикер
    try:
        expirations = list_future_expirations(ticker, api_key)
    except Exception as e:
        expirations = []
        st.warning(f"Не удалось получить экспирации: {e}")

    expiration = st.selectbox("Экспирация", expirations) if expirations else None

    if expiration:
        try:
            raw_json = download_snapshot_json(ticker, expiration, api_key)
            raw_records = _coerce_results(raw_json)
            st.session_state["raw_records"] = raw_records

            # Spot price fallback
            spot = get_spot_price(ticker, api_key)[0]
            st.session_state["spot"] = spot
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")

# --- Pipeline ---
if "raw_records" in st.session_state:
    try:
        raw_records = st.session_state["raw_records"]
        S = st.session_state["spot"]

        # IMPROVED: Настраиваем config с advanced_mode
        final_cfg = FinalTableConfig(advanced_mode=advanced_mode, market_cap=654.8e9, adv=70e6)  # Пример params, можно input

        # ... (оригинальный пайплайн с process_from_raw или build_final_tables, передавая final_cfg)

        # Для чартов: render_netgex_bars и render_key_levels (без изменений)
    except Exception as e:
        st.error("Ошибка пайплайна sanitize/window.")
        st.exception(e)

# ... (остальной оригинальный код)
