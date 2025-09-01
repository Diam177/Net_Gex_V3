
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
    def to_dt(rec: Dict[str, Any]):
        if "timestamp_unix" in rec:
            return pd.to_datetime(rec["timestamp_unix"], unit="s", utc=True)
        if "timestamp" in rec:
            try:
                return pd.to_datetime(rec["timestamp"], utc=True)
            except Exception:
                return pd.to_datetime(str(rec["timestamp"]), utc=True, errors="coerce")
        if "t" in rec:
            try:
                return pd.to_datetime(rec["t"], unit="s", utc=True)
            except Exception:
                return pd.to_datetime(str(rec["t"]), utc=True, errors="coerce")
        return pd.NaT

    records = []
    if isinstance(raw_json, dict):
        if isinstance(raw_json.get("body"), list):
            records = raw_json["body"]
        elif isinstance(raw_json.get("data"), dict):
            for k in ("items", "body", "candles"):
                if isinstance(raw_json["data"].get(k), list):
                    records = raw_json["data"][k]; break
        elif isinstance(raw_json.get("result"), dict):
            for k in ("candles","items","body"):
                if isinstance(raw_json["result"].get(k), list):
                    records = raw_json["result"][k]; break
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
            "low":  r.get("low")  or r.get("l"),
            "close":r.get("close")or r.get("c"),
            "volume": r.get("volume") or r.get("v"),
        })
    dfc = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    return dfc

def _take_last_session(dfc: pd.DataFrame, gap_minutes: int = 60) -> pd.DataFrame:
    """Keep only the last continuous session by time gaps > gap_minutes."""
    if dfc.empty:
        return dfc
    d = dfc.sort_values("ts").copy()
    gaps = d["ts"].diff().dt.total_seconds().div(60).fillna(0)
    sess_id = (gaps > gap_minutes).cumsum()
    last_id = int(sess_id.iloc[-1])
    return d.loc[sess_id == last_id].reset_index(drop=True)

def _build_rth_ticks_30m(df_plot: pd.DataFrame):
    """Build ET 09:30–16:00 ticks (30m). Return (tickvals_utc, ticktext)."""
    tz_et = "America/New_York"
    ts0 = df_plot["ts"].iloc[0]
    if ts0.tzinfo is None:
        ts0 = pd.to_datetime(ts0, utc=True)
    # convert first candle timestamp to ET to get that calendar date in New York
    ts0_et = ts0.tz_convert(tz_et)
    session_date_et = ts0_et.normalize()
    session_start_et = session_date_et + pd.Timedelta(hours=9, minutes=30)
    session_end_et   = session_date_et + pd.Timedelta(hours=16)
    ticks_et = pd.date_range(start=session_start_et, end=session_end_et, freq="30min")
    # Plotly x values are UTC in our data; keep tickvals in UTC for alignment
    tickvals = list(ticks_et.tz_convert("UTC"))
    ticktext = [t.strftime("%H:%M") for t in ticks_et]
    return tickvals, ticktext

def render_key_levels_section(ticker: str, rapid_host: Optional[str], rapid_key: Optional[str]) -> None:
    """UI section 'Key Levels' (candles + VWAP, no interactions)."""
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

        df_plot = _take_last_session(dfc, gap_minutes=60)
        if df_plot.empty:
            st.warning("Could not detect last session.")
            return

        # compute VWAP
        vol = pd.to_numeric(df_plot.get("volume", 0), errors="coerce").fillna(0.0)
        tp = (pd.to_numeric(df_plot["high"], errors="coerce") + pd.to_numeric(df_plot["low"], errors="coerce") + pd.to_numeric(df_plot["close"], errors="coerce")) / 3.0
        cum_vol = vol.cumsum()
        vwap = (tp.mul(vol)).cumsum() / cum_vol.replace(0, pd.NA)
        vwap = vwap.fillna(method="ffill")

        # build fixed RTH ticks
        tickvals, ticktext = _build_rth_ticks_30m(df_plot)

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
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=vwap, mode="lines", name="VWAP"))
        fig.update_layout(
            height=560,
            margin=dict(l=90, r=20, t=60, b=80),
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            dragmode=False,
            hovermode=False,
            plot_bgcolor="#161B22",
            paper_bgcolor="#161B22",
            font=dict(color="white"),
            template=None
        )
        # ----- Annotations: ticker (top-left) & session date (bottom-left) -----
        # Place ticker at the top-left in the paper coordinates (doesn't affect axes).
        fig.add_annotation(
    xref="paper", yref="paper",
    x=-0.06, y=1.08,
    text=str(ticker),
    showarrow=False, align="left"
)
        # Build ET session date label like 'Sep 1, 2025' from first candle timestamp
        try:
            _ts0 = pd.to_datetime(df_plot["ts"].iloc[0]).tz_convert("America/New_York")
        except Exception:
            _ts0 = pd.to_datetime(df_plot["ts"].iloc[0], utc=True).tz_convert("America/New_York")
        _date_text = f"{_ts0.strftime('%b')} {_ts0.day}, {_ts0.year}"
        fig.add_annotation(xref="paper", yref="paper", x=0.0, y=-0.10, text=_date_text, showarrow=False, align="left")
        # Move 'Time' axis title slightly up (closer to the axis)
        fig.update_xaxes(title_standoff=5)
    
        # fix ranges, remove interactions and rangeslider
        fig.update_xaxes(range=[tickvals[0], tickvals[-1]], fixedrange=True, tickmode="array", tickvals=tickvals, ticktext=ticktext)
        fig.update_yaxes(fixedrange=True)
        fig.update_layout(xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

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
