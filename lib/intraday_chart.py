# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
import requests
import datetime as _dt

# ---------------- candles normalizer ----------------
def _normalize_candles_json(candles_json):
    rows = []
    for item in candles_json or []:
        ts = item.get("t") or item.get("timestamp") or item.get("ts")
        if ts is None:
            continue
        rows.append({
            "ts": ts,
            "o": item.get("o") or item.get("open"),
            "h": item.get("h") or item.get("high"),
            "l": item.get("l") or item.get("low"),
            "c": item.get("c") or item.get("close"),
            "v": item.get("v") or item.get("volume"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["ts"]).sort_values("ts")


# ---------------- polygon fetch ----------------
def _fetch_polygon_candles(ticker: str, api_key: str, interval: str = "1m", limit: int = 640):
    # Map SPX -> I:SPX
    poly_symbol = "I:SPX" if ticker.upper() in ("SPX", "^SPX") else ticker.upper()
    interval_map = {"1m": (1, "minute"), "5m": (5, "minute")}
    mult, timespan = interval_map.get(interval, (1, "minute"))

    now = _dt.datetime.utcnow()
    minutes = int(limit) * mult
    start = now - _dt.timedelta(minutes=minutes + 5)

    to_ms = int(now.timestamp() * 1000)
    from_ms = int(start.timestamp() * 1000)
    url = f"https://api.polygon.io/v2/aggs/ticker/{poly_symbol}/range/{mult}/{timespan}/{from_ms}/{to_ms}"

    r = requests.get(
        url,
        params={"adjusted": "true", "sort": "asc", "limit": int(limit), "apiKey": api_key},
        timeout=20,
    )
    r.raise_for_status()
    j = r.json()
    results = j.get("results") or []

    out = []
    for rec in results:
        out.append({
            "t": int(rec.get("t", 0)),
            "o": rec.get("o"),
            "h": rec.get("h"),
            "l": rec.get("l"),
            "c": rec.get("c"),
            "v": rec.get("v"),
        })
    return out


# ---------------- render section ----------------
def render_key_levels_section(ticker: str):
    st.subheader("Key Levels")

    # safe read controls
    interval = st.session_state.get("kl_interval") or "1m"
    limit_val = st.session_state.get("kl_limit")
    try:
        limit = int(limit_val) if limit_val is not None else 640
    except Exception:
        limit = 640

    # get polygon key
    poly_key = st.secrets.get("POLYGON_API_KEY", os.environ.get("POLYGON_API_KEY"))
    candles_json = None
    if poly_key:
        try:
            candles_json = _fetch_polygon_candles(ticker, poly_key, interval=interval, limit=limit)
        except Exception as e:
            st.warning(f"Polygon fetch error: {e}")

    if not candles_json:
        st.warning("No data for Key Levels (Polygon API).")
        return

    dfc = _normalize_candles_json(candles_json)
    if dfc.empty:
        st.warning("No candle data available after normalization.")
        return

    st.dataframe(dfc.head(20))
