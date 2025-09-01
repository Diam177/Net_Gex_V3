
import json
from typing import Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .provider import fetch_stock_history, debug_meta

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_candles_cached(ticker: str, host: str, key: str, interval: str="1m", limit: int=640, dividend: Optional[bool]=None) -> Tuple[dict, bytes]:
    data, content = fetch_stock_history(ticker, host, key, interval=interval, limit=int(limit), dividend=dividend)
    return data, content

def _normalize_history_payload(payload: dict) -> pd.DataFrame:
    """Accepts provider payload and returns tz-aware UTC dataframe with columns ts, open, high, low, close, volume."""
    # Expecting common shape: {"records":[{"t": "...", "o":..., "h":..., "l":..., "c":..., "v":...}, ...]}
    recs = None
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            recs = payload["records"]
        elif "data" in payload and isinstance(payload["data"], list):
            recs = payload["data"]
    if not recs:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])

    rows = []
    for r in recs:
        t = r.get("t") or r.get("time") or r.get("timestamp")
        if t is None:
            continue
        ts = pd.to_datetime(t, utc=True)
        rows.append({
            "ts": ts,
            "open": float(r.get("o") or r.get("open") or r.get("Open") or 0.0),
            "high": float(r.get("h") or r.get("high") or r.get("High") or 0.0),
            "low":  float(r.get("l") or r.get("low")  or r.get("Low")  or 0.0),
            "close":float(r.get("c") or r.get("close")or r.get("Close")or 0.0),
            "volume":int(r.get("v") or r.get("volume") or r.get("Volume") or 0),
        })
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    # ensure tz-aware
    if df["ts"].dt.tz is None:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df

def _build_session_ticks(df_plot: pd.DataFrame):
    """Return (tickvals_utc, ticktext_ET) for 09:30–16:00 ET with 30-min step, aligned to candles."""
    if df_plot.empty:
        return [], []
    tz_et = "America/New_York"
    ts0 = df_plot["ts"].iloc[0]
    if ts0.tzinfo is None:
        ts0 = pd.to_datetime(ts0, utc=True)
    ts0_et = ts0.tz_convert(tz_et)
    session_date_et = ts0_et.normalize()
    session_start_et = session_date_et + pd.Timedelta(hours=9, minutes=30)
    session_end_et   = session_date_et + pd.Timedelta(hours=16)
    ticks_et = pd.date_range(start=session_start_et, end=session_end_et, freq="30min")
    tickvals = list(ticks_et.tz_convert("UTC"))
    ticktext = [t.strftime("%H:%M") for t in ticks_et]
    return tickvals, ticktext

def render_key_levels_section(ticker: str, rapid_host: Optional[str], rapid_key: Optional[str]) -> None:
    """UI section 'Key Levels' (candles-only, static)."""
    st.subheader("Key Levels")
    with st.container():
        data, content = _fetch_candles_cached(ticker, rapid_host or "", rapid_key or "", interval="1m", limit=640, dividend=None)
        df_plot = _normalize_history_payload(data or {})
        if df_plot.empty:
            st.info("Нет данных для отрисовки свечей.")
            return

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_plot["ts"],
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            showlegend=False,
            hoverinfo="skip"
        ))

        # Build session ticks and date label (ET)
        tickvals, ticktext = _build_session_ticks(df_plot)
        fig.update_layout(
            margin=dict(l=40, r=20, t=10, b=40),
            xaxis=dict(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=False
            ),
            yaxis=dict(showgrid=False, automargin=True),
            height=360
        )

        # Render once
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

        # Add ET session date under x-axis (bottom-left) and re-render
        try:
            tz_et = "America/New_York"
            ts0 = df_plot["ts"].iloc[0]
            if getattr(ts0, "tzinfo", None) is None:
                ts0 = pd.to_datetime(ts0, utc=True)
            session_date_et = ts0.tz_convert(tz_et).normalize().date()
            session_str = session_date_et.strftime("%b %d, %Y")
            fig.add_annotation(x=0, y=-0.18, xref="paper", yref="paper", text=session_str,
                               showarrow=False, xanchor="left", yanchor="top",
                               font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": True}, key="kl_fig_with_date")
        except Exception:
            pass

        with st.expander("Debug: provider meta & head"):
            st.json(debug_meta())
            st.write(df_plot.head(10))
