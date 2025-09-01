
import os
import json
from typing import Optional, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .provider import fetch_stock_history, debug_meta

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_candles_cached(ticker: str, host: str, key: str, interval: str="1m", limit: int=640, dividend: Optional[bool]=None):
    data, content = fetch_stock_history(ticker, host, key, interval=interval, limit=int(limit), dividend=dividend)
    return data, content

def _normalize_candles_json(raw_json: Any) -> pd.DataFrame:
    """Return DataFrame with columns: ts, open, high, low, close, volume."""
    import pandas as _pd

    def to_dt(rec: Dict[str, Any]):
        # Try common timestamp fields
        if "timestamp_unix" in rec:
            return _pd.to_datetime(rec["timestamp_unix"], unit="s", utc=True)
        if "timestamp" in rec:
            try:
                return _pd.to_datetime(rec["timestamp"], utc=True)
            except Exception:
                return _pd.to_datetime(str(rec["timestamp"]), utc=True, errors="coerce")
        if "t" in rec:
            try:
                return _pd.to_datetime(rec["t"], unit="s", utc=True)
            except Exception:
                return _pd.to_datetime(str(rec["t"]), utc=True, errors="coerce")
        return _pd.NaT

    records = []
    if isinstance(raw_json, dict):
        if isinstance(raw_json.get("body"), list):
            records = raw_json["body"]
        elif isinstance(raw_json.get("data"), dict):
            for k in ("items", "body", "candles"):
                if isinstance(raw_json["data"].get(k), list):
                    records = raw_json["data"][k]
                    break
        elif isinstance(raw_json.get("result"), dict):
            for k in ("candles", "items", "body"):
                if isinstance(raw_json["result"].get(k), list):
                    records = raw_json["result"][k]
                    break
        elif isinstance(raw_json.get("candles"), list):
            records = raw_json["candles"]
    elif isinstance(raw_json, list):
        records = raw_json

    rows = []
    for r in (records or []):
        rows.append({
            "ts": to_dt(r),
            "open": r.get("open") or r.get("o"),
            "high": r.get("high") or r.get("h"),
            "low": r.get("low") or r.get("l"),
            "close": r.get("close") or r.get("c"),
            "volume": r.get("volume") or r.get("v"),
        })
    dfc = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    return dfc

def _take_last_session(dfc: pd.DataFrame, gap_minutes: int = 60) -> pd.DataFrame:
    """
    Keep only the last continuous session. A new session starts if the gap between
    neighboring candles is greater than gap_minutes.
    """
    if dfc.empty:
        return dfc
    d = dfc.sort_values("ts").copy()
    gaps = d["ts"].diff().dt.total_seconds().div(60).fillna(0)
    sess_id = (gaps > gap_minutes).cumsum()
    last_id = int(sess_id.iloc[-1])
    return d.loc[sess_id == last_id].reset_index(drop=True)

def render_key_levels_section(ticker: str, rapid_host: Optional[str], rapid_key: Optional[str]) -> None:
    """UI section 'Key Levels' (candles + debug)."""
    st.subheader("Key Levels")
    with st.container():
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","1h","1d"], index=0, key="kl_interval")
        with c2:
            limit = st.number_input("Limit", min_value=100, max_value=1000, value=640, step=10, key="kl_limit")
        with c3:
            dbg = st.toggle("Debug", value=False, key="kl_debug")
        with c4:
            uploader = st.file_uploader("JSON (optional)", type=["json","txt"], accept_multiple_files=False, label_visibility="collapsed", key="kl_uploader")

        candles_json, candles_bytes = None, None

        if uploader is not None:
            try:
                up_bytes = uploader.read()
                candles_json = json.loads(up_bytes.decode("utf-8"))
                candles_bytes = up_bytes
            except Exception as e:
                st.error(f"JSON read error: {e}")

        if candles_json is None:
            test_path = "/mnt/data/TEST ENDPOINT.TXT"
            if os.path.exists(test_path):
                try:
                    with open(test_path, "rb") as f:
                        tb = f.read()
                    candles_json = json.loads(tb.decode("utf-8"))
                    candles_bytes = tb
                    st.info("Using local test file: TEST ENDPOINT.TXT")
                except Exception:
                    pass

        if candles_json is None and rapid_host and rapid_key:
            try:
                candles_json, candles_bytes = _fetch_candles_cached(ticker, rapid_host, rapid_key, interval=interval, limit=int(limit))
                st.success("Candles fetched from provider")
            except Exception as e:
                st.error(f"Request error: {e}")

        if candles_json is None:
            st.warning("No data for Key Levels (upload JSON or set RAPIDAPI_HOST/RAPIDAPI_KEY).")
            return

        dfc = _normalize_candles_json(candles_json)
        if dfc.empty:
            st.warning("Candles are empty or not recognized.")
            return

        # Use only the last trading session and stretch it
        df_plot = _take_last_session(dfc, gap_minutes=60)
        if df_plot.empty:
            st.warning("Could not detect last session.")
            return

        fig = go.Figure(data=[
            go.Candlestick(
                x=df_plot["ts"],
                open=df_plot["open"],
                high=df_plot["high"],
                low=df_plot["low"],
                close=df_plot["close"],
                name="Price"
            )
        ])
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=30),
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            template="plotly_dark" if st.get_option("theme.base") == "dark" else None
        )
        # stretch x-axis to session range
        fig.update_xaxes(range=[df_plot["ts"].iloc[0], df_plot["ts"].iloc[-1]])

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.download_button(
            "Скачать JSON (Key Levels)",
            data=candles_bytes if isinstance(candles_bytes, (bytes, bytearray)) else json.dumps(candles_json, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{ticker}_{interval}_candles.json",
            mime="application/json",
            key="kl_download"
        )

        if dbg:
            with st.expander("Debug: provider meta & head"):
                st.json(debug_meta())
                st.write(df_plot.head(10))
