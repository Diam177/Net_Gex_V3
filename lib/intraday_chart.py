
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

def _et_session_bounds_for_ts(ts_utc: pd.Timestamp):
    """Return ET RTH bounds for the *calendar* date of ts_utc."""
    tz_et = "America/New_York"
    if ts_utc.tzinfo is None:
        ts_utc = pd.to_datetime(ts_utc, utc=True)
    ts_et = ts_utc.tz_convert(tz_et)
    d_et = ts_et.normalize()
    start_et = d_et + pd.Timedelta(hours=9, minutes=30)
    end_et   = d_et + pd.Timedelta(hours=16)
    return start_et.tz_convert("UTC"), end_et.tz_convert("UTC")

def _slice_current_session_or_skeleton(dfc: pd.DataFrame):
    """Return (df, (start,end), has_price) for *today's* ET session.
    If no candles yet, return a 2-row skeleton with only timestamps and has_price=False.
    """
    now_utc = pd.Timestamp.now(tz="UTC")
    start_utc, end_utc = _et_session_bounds_for_ts(now_utc)
    mask = (dfc["ts"] >= start_utc) & (dfc["ts"] <= end_utc)
    df_today = dfc.loc[mask].sort_values("ts").reset_index(drop=True)
    if not df_today.empty and {"open","high","low","close"}.issubset(df_today.columns):
        return df_today, (start_utc, end_utc), True
    # skeleton
    return pd.DataFrame({"ts":[start_utc, end_utc]}), (start_utc, end_utc), False


def render_key_levels_section(ticker: str, rapid_host: Optional[str], rapid_key: Optional[str]) -> None:
    """UI section 'Key Levels' (adds 'Last session' toggle and 'Market closed' state)."""
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
            uploader = st.file_uploader("JSON (optional)", type=["json"], accept_multiple_files=False, label_visibility="collapsed", key="kl_uploader")

        # New toggle
        last_session = st.toggle("Last session", value=False, key="kl_last_session")

        candles_json, candles_bytes = None, None
        if uploader is not None:
            try:
                candles_bytes = uploader.read()
                candles_json = json.loads(candles_bytes.decode("utf-8"))
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                return
        elif rapid_host and rapid_key:
            try:
                candles_json, candles_bytes = _fetch_candles_cached(ticker, rapid_host, rapid_key, interval=interval, limit=int(limit), dividend=None)
            except Exception as e:
                st.error(f"Provider error: {e}")
                return

        if candles_json is None:
            st.warning("No data for Key Levels (upload JSON or set RAPIDAPI_HOST/RAPIDAPI_KEY).")
            return

        dfc = _normalize_candles_json(candles_json)
        if dfc.empty:
            st.warning("Candles are empty or not recognized.")
            return

        if last_session:
            df_plot = _take_last_session(dfc, gap_minutes=60)
            has_price = (not df_plot.empty) and {"open","high","low","close"}.issubset(df_plot.columns)
        else:
            df_plot, sess_bounds, has_price = _slice_current_session_or_skeleton(dfc)

        # Build VWAP (only if price exists)
        vwap = None
        if has_price:
            # Typical price VWAP: ((H+L+C)/3 * Vol).cumsum() / Vol.cumsum()
            vol = df_plot.get("volume", df_plot.get("v"))
            if vol is None:
                vol = 0*df_plot["close"]
            tp = (pd.to_numeric(df_plot["high"], errors="coerce") +
                  pd.to_numeric(df_plot["low"], errors="coerce") +
                  pd.to_numeric(df_plot["close"], errors="coerce")) / 3.0
            vol = pd.to_numeric(vol, errors="coerce").fillna(0)
            cum_vol = vol.cumsum()
            vwap = (tp.mul(vol)).cumsum() / cum_vol.replace(0, pd.NA)
            vwap = vwap.fillna(method="ffill")

        # Build fixed RTH ticks (works also with skeleton)
        tickvals, ticktext = _build_rth_ticks_30m(df_plot)

        fig = go.Figure()
        if has_price:
            fig.add_trace(go.Candlestick(
                x=df_plot["ts"],
                open=df_plot["open"],
                high=df_plot["high"],
                low=df_plot["low"],
                close=df_plot["close"],
                name="Price"
            ))
            if vwap is not None:
                fig.add_trace(go.Scatter(x=df_plot["ts"], y=vwap, mode="lines", name="VWAP"))
        else:
            # Center label when market closed / pre-open
            fig.add_annotation(text="Market closed", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="white"))

        # Horizontal key levels from first chart (if present)
        try:
            levels = dict(st.session_state.get("first_chart_max_levels", {}))
        except Exception:
            levels = {}
        def _fmt_int(x):
            try:
                return f"{int(round(float(x)))}"
            except Exception:
                return str(x)
        try:
            from .plotting import LINE_STYLE, POS_COLOR, NEG_COLOR
            _cmap = {
                "max_pos_gex": POS_COLOR,
                "max_neg_gex": NEG_COLOR,
                "put_oi_max":  LINE_STYLE.get("Put OI", {}).get("line", "#AA3355"),
                "call_oi_max": LINE_STYLE.get("Call OI", {}).get("line", "#55AA55"),
                "put_vol_max": LINE_STYLE.get("Put Volume", {}).get("line", "#AF7AC5"),
                "call_vol_max":LINE_STYLE.get("Call Volume", {}).get("line", "#5DADE2"),
                "ag_max":      LINE_STYLE.get("AG", {}).get("line", "#F39C12"),
                "pz_max":      LINE_STYLE.get("PZ", {}).get("line", "#F4D03F"),
                "gflip":       "#AAAAAA",
            }
        except Exception:
            _cmap = {}

        if not df_plot.empty:
            x0, x1 = df_plot["ts"].iloc[0], df_plot["ts"].iloc[-1]
        else:
            # Fallback to tick range if any
            x0, x1 = (tickvals[0], tickvals[-1]) if tickvals else (None, None)

        def _add_line(tag, label):
            y = levels.get(tag)
            if y is None or x0 is None:
                return
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y, y], mode="lines",
                name=f"{label} ({_fmt_int(y)})",
                line=dict(dash="dot", width=2, color=_cmap.get(tag, "#BBBBBB")),
                hoverinfo="skip", showlegend=True
            ))
        for tag, label in [
            ("max_neg_gex", "Max Neg GEX"),
            ("max_pos_gex", "Max Pos GEX"),
            ("put_oi_max",  "Max Put OI"),
            ("call_oi_max", "Max Call OI"),
            ("put_vol_max", "Max Put Volume"),
            ("call_vol_max","Max Call Volume"),
            ("ag_max",      "AG"),
            ("pz_max",      "PZ"),
            ("gflip",       "G-Flip"),
        ]:
            _add_line(tag, label)

        # Layout
        fig.update_layout(
            height=560,
            margin=dict(l=90, r=20, t=50, b=50),
            xaxis_title="Time",
            yaxis_title="Price",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            dragmode=False,
            hovermode=False,
            plot_bgcolor="#161B22",
            paper_bgcolor="#161B22",
            font=dict(color="white"),
        )
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext,
                         range=[tickvals[0], tickvals[-1]] if tickvals else None)

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": False})

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
