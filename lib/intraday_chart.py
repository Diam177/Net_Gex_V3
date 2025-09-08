# -*- coding: utf-8 -*-
import os
import math
import time
import json
import requests
import datetime as _dt
from dateutil import tz

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =====================================================================================
# Helpers: time & formatting
# =====================================================================================
NY_TZ = tz.gettz("America/New_York")

def _now_utc():
    return _dt.datetime.utcnow().replace(tzinfo=tz.UTC)

def _today_ny(dt_utc=None):
    """Return date (YYYY-MM-DD) for New York based on provided UTC time (or now)."""
    dt_utc = dt_utc or _now_utc()
    return dt_utc.astimezone(NY_TZ).date()

def _session_bounds_local(local_tz, day=None):
    """
    Build today's regular session [9:30, 16:00] in user's local timezone.
    """
    day = day or _dt.datetime.now(tz=local_tz).date()
    start_local = _dt.datetime(day.year, day.month, day.day, 9, 30, tzinfo=local_tz)
    end_local   = _dt.datetime(day.year, day.month, day.day, 16, 0, tzinfo=local_tz)
    return start_local, end_local

def _fmt_date_for_caption(dt_local):
    try:
        return dt_local.strftime("%b %-d, %Y")
    except Exception:
        # Windows-compatible (no %-d)
        return dt_local.strftime("%b %d, %Y")

# =====================================================================================
# Polygon fetchers
# =====================================================================================
POLYGON_BASE = "https://api.polygon.io"

def _polygon_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    return "I:SPX" if s in ("SPX", "^SPX") else s

def _fetch_polygon_candles(symbol: str, api_key: str, interval="1m", limit=640):
    """
    Fetch minute candles using polygon v2 aggs endpoint.
    Returns list of dicts with keys: t (ms), o, h, l, c, v.
    """
    sym = _polygon_symbol(symbol)
    interval_map = {
        "1m": (1, "minute"),
        "2m": (2, "minute"),
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "30m": (30, "minute"),
        "1h": (60, "minute"),
        "1d": (1, "day"),
    }
    mult, timespan = interval_map.get(interval, (1, "minute"))

    # Define a from/to around now to get the most recent limit bars
    to_ts = int(_now_utc().timestamp() * 1000)
    # Over-request enough history; polygon returns the most recent bars within range when 'sort=desc'
    # Use ~5 days back for minute bars to be safe
    from_ts = to_ts - 5 * 24 * 60 * 60 * 1000

    url = f"{POLYGON_BASE}/v2/aggs/ticker/{sym}/range/{mult}/{timespan}/{from_ts}/{to_ts}"
    params = {"limit": limit, "sort": "desc", "apiKey": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    results = js.get("results") or []
    # we want newest at bottom (asc), limit N
    results = list(reversed(results))[-limit:]
    # remap to common keys
    out = []
    for it in results:
        out.append({
            "t": int(it.get("t")),
            "o": float(it.get("o")) if it.get("o") is not None else None,
            "h": float(it.get("h")) if it.get("h") is not None else None,
            "l": float(it.get("l")) if it.get("l") is not None else None,
            "c": float(it.get("c")) if it.get("c") is not None else None,
            "v": float(it.get("v")) if it.get("v") is not None else None,
        })
    return out

# =====================================================================================
# Normalization & VWAP
# =====================================================================================
def _normalize_candles_json(candles_json, local_tz):
    rows = []
    for item in candles_json or []:
        ts = item.get("t") or item.get("timestamp") or item.get("ts")
        if ts is None:
            continue
        # polygon gives ms
        if ts > 10_000_000_000:
            ts_ms = int(ts)
        else:
            ts_ms = int(ts * 1000)
        dt_utc = _dt.datetime.utcfromtimestamp(ts_ms/1000.0).replace(tzinfo=tz.UTC)
        dt_local = dt_utc.astimezone(local_tz)
        rows.append({
            "ts": ts_ms,
            "dt_local": dt_local,
            "o": item.get("o") or item.get("open"),
            "h": item.get("h") or item.get("high"),
            "l": item.get("l") or item.get("low"),
            "c": item.get("c") or item.get("close"),
            "v": item.get("v") or item.get("volume"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # ensure numeric
    for col in ["o","h","l","c","v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dt_local"]).sort_values("dt_local").reset_index(drop=True)
    # VWAP (typical price variant)
    tp = (df["h"] + df["l"] + df["c"]).astype(float) / 3.0
    v = df["v"].astype(float).fillna(0.0)
    cum_v = v.cumsum().replace(0, np.nan)
    cum_pv = (tp * v).cumsum()
    df["vwap"] = (cum_pv / cum_v).ffill()
    return df

# =====================================================================================
# Plotting
# =====================================================================================
def _make_keylevels_figure(df: pd.DataFrame, local_tz, show_last_session=False, levels: dict | None = None):
    # Determine session bounds in user's local time
    day = df["dt_local"].iloc[-1].date() if not df.empty else _dt.datetime.now(tz=local_tz).date()
    ses_start, ses_end = _session_bounds_local(local_tz, day)

    fig = go.Figure()

    if not df.empty:
        # restrict to last regular session if requested
        if show_last_session:
            df = df[(df["dt_local"] >= ses_start) & (df["dt_local"] <= ses_end)].copy()

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df["dt_local"],
            open=df["o"], high=df["h"], low=df["l"], close=df["c"],
            name="Price"
        ))
        # VWAP line
        if "vwap" in df.columns and df["vwap"].notna().any():
            fig.add_trace(go.Scatter(
                x=df["dt_local"], y=df["vwap"], name="VWAP", mode="lines"
            ))
    else:
        # no data → draw empty session window
        xs = [ses_start, ses_end]
        ys = [np.nan, np.nan]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Price"))

        fig.add_annotation(
            x=ses_start + (ses_end - ses_start) / 2,
            yref="paper", y=0.5,
            text="Market closed",
            showarrow=False, font=dict(size=16)
        )

    # Horizontal levels from first chart (if present)
    levels = levels or {}
    for key, color in [("max_pos_gex", "#2FD06F"), ("max_neg_gex", "#FF3B30"), ("gflip", "#B0B8C5")]:
        val = levels.get(key)
        if val is None:
            continue
        fig.add_hline(y=val, line=dict(color=color, width=1, dash="dash"), opacity=0.8)

    # Layout
    fig.update_layout(
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis_title=_fmt_date_for_caption(ses_start.astimezone(local_tz)),
        yaxis_title="Price",
        xaxis=dict(
            showgrid=True,
            tickformat="%H:%M",
            range=[ses_start, ses_end]
        ),
        dragmode="pan"
    )
    return fig

# =====================================================================================
# Public section
# =====================================================================================
def render_key_levels_section(ticker: str):
    """
    Render Key Levels (intraday candles + VWAP + horizontal levels) using Polygon only.
    - API key from st.secrets['POLYGON_API_KEY'] or env.
    - Interval & Limit taken from st.session_state['kl_interval'/'kl_limit'] if present.
    - When no intraday data yet (before open), we still render the frame for the session.
    """
    local_tz = tz.gettz()  # user's local tz
    api_key = st.secrets.get("POLYGON_API_KEY", os.environ.get("POLYGON_API_KEY"))

    st.subheader("Key Levels")

    if not api_key:
        st.info("No POLYGON_API_KEY provided — cannot fetch intraday candles.")
        return

    # Read controls defined in app sidebar
    interval = st.session_state.get("kl_interval", "1m")
    limit = int(st.session_state.get("kl_limit", 640) or 640)

    # Toggle above chart (kept by spec)
    show_last_session = st.toggle("Last session", value=False, key="kl_last_session_hdr")

    candles = []
    try:
        candles = _fetch_polygon_candles(ticker, api_key, interval=interval, limit=limit)
    except Exception as e:
        st.warning(f"Polygon error: {e}")

    df = _normalize_candles_json(candles, local_tz=local_tz)

    # Horizontal levels from first chart (if available)
    levels = st.session_state.get("first_chart_max_levels", {}) if isinstance(st.session_state.get("first_chart_max_levels", {}), dict) else {}

    fig = _make_keylevels_figure(df, local_tz, show_last_session=show_last_session, levels=levels)
    st.plotly_chart(fig, use_container_width=True)
