# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
import requests
import datetime as _dt
from dateutil import tz
import plotly.graph_objects as go
import numpy as np

# ---------------- candles normalizer ----------------
def _normalize_candles_json(candles_json):
    rows = []
    for item in candles_json or []:
        ts = item.get("t") or item.get("timestamp") or item.get("ts")
        if ts is None:
            continue
        # polygon uses ms
        ts_ms = int(ts if int(ts) > 10_000_000_000 else int(ts)*1000)
        rows.append({
            "ts": ts_ms,
            "o": item.get("o") or item.get("open"),
            "h": item.get("h") or item.get("high"),
            "l": item.get("l") or item.get("low"),
            "c": item.get("c") or item.get("close"),
            "v": item.get("v") or item.get("volume"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # numeric & datetime
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df["dt_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    try:
        local_tz = tz.gettz()
        df["dt_local"] = df["dt_utc"].dt.tz_convert(local_tz)
    except Exception:
        df["dt_local"] = df["dt_utc"]
    # VWAP (typical-price)
    tp = (df["h"] + df["l"] + df["c"]) / 3.0
    v = df["v"].fillna(0.0)
    cum_v = v.cumsum().replace(0, np.nan)
    df["vwap"] = (tp * v).cumsum() / cum_v
    return df


# ---------------- polygon fetch ----------------
def _fetch_polygon_candles(ticker: str, api_key: str, interval: str = "1m", limit: int = 640):
    poly_symbol = "I:SPX" if ticker.upper() in ("SPX", "^SPX") else ticker.upper()
    interval_map = {"1m": (1, "minute"), "2m": (2, "minute"), "5m": (5, "minute"),
                    "15m": (15, "minute"), "30m": (30, "minute"), "1h": (60, "minute"), "1d": (1, "day")}
    mult, timespan = interval_map.get(interval, (1, "minute"))
    now = _dt.datetime.utcnow()
    to_ms = int(now.timestamp() * 1000)
    # pull 5 days back to be safe for minute bars
    from_ms = to_ms - 5*24*60*60*1000
    url = f"https://api.polygon.io/v2/aggs/ticker/{poly_symbol}/range/{mult}/{timespan}/{from_ms}/{to_ms}"
    r = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": int(limit), "apiKey": api_key}, timeout=20)
    r.raise_for_status()
    j = r.json()
    results = j.get("results") or []
    out = []
    for rec in results[-int(limit):]:
        out.append({
            "t": int(rec.get("t", 0)),
            "o": rec.get("o"), "h": rec.get("h"), "l": rec.get("l"),
            "c": rec.get("c"), "v": rec.get("v"),
        })
    return out


# ---------------- render section ----------------
def render_key_levels_section(ticker: str):
    st.subheader("Key Levels")

    # controls from sidebar
    interval = st.session_state.get("kl_interval") or "1m"
    limit_val = st.session_state.get("kl_limit")
    try:
        limit = int(limit_val) if limit_val is not None else 640
    except Exception:
        limit = 640

    # polygon key
    poly_key = st.secrets.get("POLYGON_API_KEY", os.environ.get("POLYGON_API_KEY"))
    if not poly_key:
        st.info("No POLYGON_API_KEY provided")
        return

    # toggle above chart (как было оговорено)
    show_last = st.toggle("Last session", value=False, key="kl_last_session_hdr")

    # fetch
    candles_json = None
    try:
        candles_json = _fetch_polygon_candles(ticker, poly_key, interval=interval, limit=limit)
    except Exception as e:
        st.warning(f"Polygon fetch error: {e}")

    df = _normalize_candles_json(candles_json or [])

    # session window 9:30–16:00 local
    local_tz = tz.gettz()
    now_local = _dt.datetime.now(tz=local_tz)
    ses_start = now_local.replace(hour=9, minute=30, second=0, microsecond=0)
    ses_end   = now_local.replace(hour=16, minute=0, second=0, microsecond=0)

    fig = go.Figure()

    if not df.empty:
        if show_last:
            df = df[(df["dt_local"] >= ses_start) & (df["dt_local"] <= ses_end)].copy()

        fig.add_trace(go.Candlestick(
            x=df["dt_local"], open=df["o"], high=df["h"], low=df["l"], close=df["c"],
            name="Price"
        ))
        if df["vwap"].notna().any():
            fig.add_trace(go.Scatter(x=df["dt_local"], y=df["vwap"], name="VWAP", mode="lines"))
    else:
        fig.add_annotation(x=ses_start + (ses_end - ses_start)/2, yref="paper", y=0.5,
                           text="Market closed", showarrow=False)

    # horizontal levels from first chart
    levels = st.session_state.get("first_chart_max_levels", {}) or {}
    if isinstance(levels, dict):
        if "max_pos_gex" in levels and levels["max_pos_gex"] is not None:
            fig.add_hline(y=float(levels["max_pos_gex"]), line=dict(color="#2FD06F", width=1, dash="dash"))
        if "max_neg_gex" in levels and levels["max_neg_gex"] is not None:
            fig.add_hline(y=float(levels["max_neg_gex"]), line=dict(color="#FF3B30", width=1, dash="dash"))
        if "gflip" in levels and levels["gflip"] is not None:
            fig.add_hline(y=float(levels["gflip"]), line=dict(color="#B0B8C5", width=1, dash="dot"))

    fig.update_layout(
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(tickformat="%H:%M", showgrid=True, range=[ses_start, ses_end]),
        yaxis_title="Price",
        dragmode="pan"
    )
    st.plotly_chart(fig, use_container_width=True)
