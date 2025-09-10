import os
import json
from typing import Optional, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .provider_polygon import fetch_stock_history

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_candles_cached(ticker: str, host: str, key: str, interval: str="1m", limit: int=640, dividend: Optional[bool]=None):
    data, content = fetch_stock_history(ticker, host, key, interval=interval, limit=int(limit), dividend=dividend)
    return data, content


def _normalize_candles_json(raw_json: Any) -> pd.DataFrame:
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
    if not rows:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])
    dfc = pd.DataFrame(rows).dropna(subset=["ts"]).sort_values("ts")
    return dfc

def _take_last_session(dfc: pd.DataFrame, gap_minutes: int = 60) -> pd.DataFrame:
    if dfc.empty: return dfc
    d = dfc.sort_values("ts").copy()
    gaps = d["ts"].diff().dt.total_seconds().div(60).fillna(0)
    sess_id = (gaps > gap_minutes).cumsum()
    last_id = int(sess_id.iloc[-1])
    return d.loc[sess_id == last_id].reset_index(drop=True)

def _build_rth_ticks_30m(df_plot: pd.DataFrame):
    tz_et = "America/New_York"
    ts0 = df_plot["ts"].iloc[0]
    if ts0.tzinfo is None: ts0 = pd.to_datetime(ts0, utc=True)
    ts0_et = ts0.tz_convert(tz_et)
    session_date_et = ts0_et.normalize()
    session_start_et = session_date_et + pd.Timedelta(hours=9, minutes=30)
    session_end_et   = session_date_et + pd.Timedelta(hours=16)
    ticks_et = pd.date_range(start=session_start_et, end=session_end_et, freq="30min")
    tickvals = list(ticks_et.tz_convert("UTC"))
    ticktext = [t.strftime("%H:%M") for t in ticks_et]
    return tickvals, ticktext

def _build_rth_ticks_for_date(session_date_et: pd.Timestamp):
    if session_date_et.tz is None:
        session_date_et = session_date_et.tz_localize("America/New_York")
    start_et = session_date_et + pd.Timedelta(hours=9, minutes=30)
    end_et   = session_date_et + pd.Timedelta(hours=16)
    ticks_et = pd.date_range(start=start_et, end=end_et, freq="30min")
    tickvals = list(ticks_et.tz_convert("UTC"))
    ticktext = [t.strftime("%H:%M") for t in ticks_et]
    return tickvals, ticktext

def _filter_session_for_date(dfc: pd.DataFrame, session_date_et: pd.Timestamp) -> pd.DataFrame:
    if dfc.empty: return dfc
    if session_date_et.tz is None:
        session_date_et = session_date_et.tz_localize("America/New_York")
    start_et = session_date_et + pd.Timedelta(hours=9, minutes=30)
    end_et   = session_date_et + pd.Timedelta(hours=16)
    start_utc = start_et.tz_convert("UTC")
    end_utc   = end_et.tz_convert("UTC")
    d = dfc.sort_values("ts").copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    mask = (d["ts"] >= start_utc) & (d["ts"] <= end_utc)
    return d.loc[mask].reset_index(drop=True)

def render_key_levels_section(ticker: str, rapid_host: Optional[str], rapid_key: Optional[str]) -> None:
    st.subheader("Key Levels")

    # === Переключатель над чартом ===
    st.toggle("Last session", value=st.session_state.get("kl_last_session", False), key="kl_last_session")

    # Значения Interval/Limit берём из session_state (они в сайдбаре)
    interval = st.session_state.get("kl_interval", "1m")
    try:
        limit = int(st.session_state.get("kl_limit", 640))
    except Exception:
        limit = 640
    last_session = bool(st.session_state.get("kl_last_session", False))

    # --- Получение данных (без загрузчика файлов и debug) ---
    candles_json, candles_bytes = None, None

    test_path = "/mnt/data/TEST ENDPOINT.TXT"
    if os.path.exists(test_path):
        try:
            with open(test_path, "rb") as f:
                tb = f.read()
            candles_json = json.loads(tb.decode("utf-8"))
            candles_bytes = tb
            st.info("Using local test file: TEST ENDPOINT.TXT")
        except Exception:
            candles_json = None
            candles_bytes = None

    if candles_json is None and rapid_key:
        try:
            candles_json, candles_bytes = _fetch_candles_cached(ticker, rapid_host, rapid_key, interval=interval, limit=int(limit))
        except Exception as e:
            st.error(f"Request error: {e}")

    if candles_json is None:
        st.warning("No data for Key Levels (нужен API ключ для источника данных).")
        return

    dfc = _normalize_candles_json(candles_json)
    if dfc.empty:
        st.warning("Candles are empty or not recognized.")
        return

    session_date_et = pd.Timestamp.now(tz="America/New_York").normalize()

    if last_session:
        df_plot = _take_last_session(dfc, gap_minutes=60)
        has_candles = not df_plot.empty
        if not has_candles:
            st.warning("Could not detect last session.")
            return
    else:
        df_plot = _filter_session_for_date(dfc, session_date_et)
        has_candles = not df_plot.empty

    if has_candles:
        tickvals, ticktext = _build_rth_ticks_for_date(session_date_et)
        x0, x1 = tickvals[0], tickvals[-1]
        x_mid = tickvals[len(tickvals)//2]
    else:
        tickvals, ticktext = _build_rth_ticks_for_date(session_date_et)
        x0, x1 = tickvals[0], tickvals[-1]
        x_mid = tickvals[len(tickvals)//2]

    # VWAP
    if has_candles:
        vol = pd.to_numeric(df_plot.get("volume", 0), errors="coerce").fillna(0.0)
        tp = (pd.to_numeric(df_plot["high"], errors="coerce") + pd.to_numeric(df_plot["low"], errors="coerce") + pd.to_numeric(df_plot["close"], errors="coerce")) / 3.0
        cum_vol = vol.cumsum()
        vwap = (tp.mul(vol)).cumsum() / cum_vol.replace(0, pd.NA)
        vwap = vwap.fillna(method="ffill")
    else:
        vwap = None

    fig = go.Figure()
    if has_candles:
        fig.add_trace(go.Candlestick(
            x=df_plot["ts"], open=df_plot["open"], high=df_plot["high"],
            low=df_plot["low"], close=df_plot["close"], name="Price"
        ))
    if has_candles and vwap is not None:
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=vwap, mode="lines", name="VWAP"))

    
    # Secondary x-axis for overlayed Key Level lines (kept out of rangeslider)
    fig.update_layout(xaxis2=dict(overlaying='x', matches='x', visible=False))
# Key Levels
    levels = dict(st.session_state.get("first_chart_max_levels", {})) if isinstance(st.session_state.get("first_chart_max_levels", {}), dict) else {}

    def _fmt_int(x):
        try: return f"{int(round(float(x)))}"
        except Exception: return str(x)

    try:
        from .plotting import LINE_STYLE, POS_COLOR, NEG_COLOR
        _cmap = {
            "max_pos_gex": POS_COLOR,
            "max_pos_gex_2": POS_COLOR,
            "max_pos_gex_3": POS_COLOR,
            "max_neg_gex": NEG_COLOR,
            "max_neg_gex_2": NEG_COLOR,
            "max_neg_gex_3": NEG_COLOR,
            "call_oi_max": LINE_STYLE.get("Call OI", {}).get("line", "#55aa55"),
            "put_oi_max":  LINE_STYLE.get("Put OI", {}).get("line", "#aa3355"),
            "call_vol_max":LINE_STYLE.get("Call Volume", {}).get("line", "#2D83FF"),
            "put_vol_max": LINE_STYLE.get("Put Volume", {}).get("line", "#8C5A0A"),
            "ag_max":      LINE_STYLE.get("AG", {}).get("line", "#7D3C98"),
            "ag_max_2":    LINE_STYLE.get("AG", {}).get("line", "#7D3C98"),
            "ag_max_3":    LINE_STYLE.get("AG", {}).get("line", "#7D3C98"),
            "pz_max":      LINE_STYLE.get("PZ", {}).get("line", "#F4D03F"),
            "gflip":       "#AAAAAA",
        }
    except Exception:
        _cmap = {}
    DASHED_TAGS = set(['max_pos_gex_2','max_pos_gex_3','max_neg_gex_2','max_neg_gex_3','ag_max_2','ag_max_3'])

    def _add_line(tag, label):
        y = levels.get(tag)
        if y is None: return
        fig.add_trace(go.Scatter(xaxis='x2',
            x=[x0, x1], y=[y, y], mode="lines",
            name=f"{label} ({_fmt_int(y)})",
            line=dict(dash=("dot" if tag in DASHED_TAGS else "solid"), width=2, color=_cmap.get(tag, "#BBBBBB")),
            hoverinfo="skip", showlegend=True
        ))

    def _add_line_secondary(tag, label):
        y = levels.get(tag)
        if y is None: return
        fig.add_trace(go.Scatter(xaxis='x2',
            x=[x0, x1], y=[y, y], mode="lines",
            name=f"{label} ({_fmt_int(y)})",
            line=dict(dash=("dot" if tag in DASHED_TAGS else "solid"), width=2, color=_cmap.get(tag, "#BBBBBB")),
            hoverinfo="skip", showlegend=True
        ))

    _add_line("max_neg_gex", "Max Neg GEX")
    _add_line_secondary("max_neg_gex_2", "Neg Net GEX #2")
    _add_line_secondary("max_neg_gex_3", "Neg Net GEX #3")
    _add_line("max_pos_gex", "Max Pos GEX")
    _add_line_secondary("max_pos_gex_2", "Pos Net GEX #2")
    _add_line_secondary("max_pos_gex_3", "Pos Net GEX #3")
    _add_line("put_oi_max",  "Max Put OI")
    _add_line("call_oi_max", "Max Call OI")
    _add_line("put_vol_max", "Max Put Volume")
    _add_line("call_vol_max","Max Call Volume")
    _add_line("ag_max",      "AG")
    _add_line_secondary("ag_max_2", "AG #2")
    _add_line_secondary("ag_max_3", "AG #3")
    _add_line("pz_max",      "PZ")
    _add_line("gflip",       "G-Flip")

    # Сводные подписи при совпадении уровней
    try:
        order_pairs = [
            ("max_neg_gex", "Max Neg GEX"),
            ("max_neg_gex_2", "Neg Net GEX #2"),
            ("max_neg_gex_3", "Neg Net GEX #3"),
            ("max_pos_gex", "Max Pos GEX"),
            ("max_pos_gex_2", "Pos Net GEX #2"),
            ("max_pos_gex_3", "Pos Net GEX #3"),
            ("put_oi_max",  "Max Put OI"),
            ("call_oi_max", "Max Call OI"),
            ("put_vol_max", "Max Put Volume"),
            ("call_vol_max","Max Call Volume"),
            ("ag_max",      "AG"),
            ("ag_max_2",      "AG #2"),
            ("ag_max_3",      "AG #3"),
            ("pz_max",      "PZ"),
            ("gflip",       "G-Flip"),
        ]
        groups = {}
        for tag, label in order_pairs:
            y = levels.get(tag)
            if y is None: continue
            key = float(y)
            groups.setdefault(key, []).append(label)
        for y, labels in groups.items():
            if len(labels) >= 2:
                fig.add_annotation(
                    x=x_mid, y=y, xref="x", yref="y",
                    text=" + ".join(labels), showarrow=False,
                    xanchor="center", yshift=12, align="center",
                    bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.25)",
                    borderwidth=1, font=dict(size=11)
                )
    except Exception:
        pass
    # Подписи для одиночных линий в том же стиле
    try:
        for y, labels in groups.items():
            if len(labels) == 1:
                fig.add_annotation(
                    x=x_mid, y=y, xref="x", yref="y",
                    text=labels[0], showarrow=False,
                    xanchor="center", yshift=12, align="center",
                    bgcolor="rgba(0,0,0,0.35)", bordercolor="rgba(255,255,255,0.25)",
                    borderwidth=1, font=dict(size=11)
                )
    except Exception:
        pass


    # Надпись "Market closed"
    if not has_candles and not last_session:
        try: x_center = x0 + (x1 - x0) / 2
        except Exception: x_center = x_mid
        y_vals = []
        for tag in ("max_neg_gex","max_pos_gex","max_neg_gex_2","max_neg_gex_3","max_pos_gex_2","max_pos_gex_3","put_oi_max","call_oi_max","put_vol_max","call_vol_max","ag_max","ag_max_2","ag_max_3","pz_max","gflip"):
            v = levels.get(tag)
            if isinstance(v, (int,float)): y_vals.append(float(v))
        y_center = (min(y_vals)+max(y_vals))/2.0 if y_vals else 0.0
        fig.add_annotation(
            x=x_center, y=y_center, xref="x", yref="y",
            text="Market closed", showarrow=False,
            xanchor="center", yanchor="middle", align="center",
            font=dict(size=36, color="rgba(255,255,255,0.2)"),
            bgcolor="rgba(0,0,0,0)"
        )

    fig.update_layout(
        height=820, margin=dict(l=90, r=20, t=50, b=50),
        xaxis_title="Time", yaxis_title="Price",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode=False, hovermode=False,
        plot_bgcolor="#161B22", paper_bgcolor="#161B22",
        font=dict(color="white"), template=None
    )
    # Show rangeslider with candle-only content; set y-range to candle L/H
    if has_candles:
        try:
            _ymin = float(pd.to_numeric(df_plot['low'], errors='coerce').min())
            _ymax = float(pd.to_numeric(df_plot['high'], errors='coerce').max())
            fig.update_layout(xaxis_rangeslider=dict(visible=True, thickness=0.05, yaxis=dict(range=[_ymin, _ymax])))
        except Exception:
            fig.update_layout(xaxis_rangeslider_visible=True)
    # === Build Y ticks between min and max Key Levels with dynamic step (1/0.5/0.25) ===
    y_tickvals = None
    y_range = None
    try:
        level_keys = [
            "max_neg_gex","max_neg_gex_2","max_neg_gex_3",
            "max_pos_gex","max_pos_gex_2","max_pos_gex_3",
            "put_oi_max","call_oi_max","put_vol_max","call_vol_max",
            "ag_max","ag_max_2","ag_max_3","pz_max","gflip"
        ]
        _ys = []
        for _k in level_keys:
            _v = levels.get(_k)
            try:
                if _v is not None:
                    _ys.append(float(_v))
            except Exception:
                pass
        def _is_mult(x, step, eps=1e-6):
            return abs((x/step)-round(x/step)) < eps
        if _ys:
            step = 1.0
            if any(not _is_mult(v, 1.0) for v in _ys): step = 0.5
            if any(abs(v*4 - round(v*4)) < 1e-6 and not _is_mult(v, 0.5) for v in _ys): step = 0.25
            y_lo = min(_ys); y_hi = max(_ys)
            # snap to step
            import math
            y_lo = math.floor(y_lo/step)*step
            y_hi = math.ceil(y_hi/step)*step
            n_ticks = int(round((y_hi - y_lo)/step)) + 1
            y_tickvals = [round(y_lo + i*step, 10) for i in range(n_ticks)]
            y_range = [float(y_lo), float(y_hi)]
            if y_range[0] == y_range[1]: y_range = [y_range[0]-step, y_range[1]+step]
            # Values to highlight (where there are lines): snap each level to chosen step
            highlight_vals = []
            if _ys:
                _set = set()
                for _v in _ys:
                    _sv = round(_v/step)*step
                    # snap to nearest tick in y_tickvals to avoid FP drift
                    if 'y_tickvals' in locals() and y_tickvals:
                        # find nearest tick
                        _nearest = min(y_tickvals, key=lambda t: abs(t-_sv))
                        _set.add(round(_nearest,10))
                    else:
                        _set.add(round(_sv,10))
                highlight_vals = sorted(_set)

    except Exception:
        y_tickvals = None
        y_range = None
    fig.update_layout(xaxis_rangeslider_visible=True)
    
    fig.update_xaxes(range=[tickvals[0], tickvals[-1]], fixedrange=True, tickmode="array", tickvals=tickvals, ticktext=ticktext, tickfont=dict(size=10))
    fig.update_yaxes(fixedrange=True, range=(y_range if y_range is not None else None), tickmode=("array" if y_tickvals is not None else "auto"), tickvals=(y_tickvals if y_tickvals is not None else None), ticktext=([str(v) for v in y_tickvals] if y_tickvals is not None else None), tickfont=dict(size=10, color="#7d8590"))

    # Add white overlay labels at level lines (annotations) so they remain visible atop gray ticks
    try:
        if 'highlight_vals' in locals() and highlight_vals:
            _anns = list(fig.layout.annotations) if getattr(fig.layout, 'annotations', None) else []
            for _yv in highlight_vals:
                _anns.append(dict(
                    xref='paper', x=0, xanchor='right',
                    yref='y', y=float(_yv), yanchor='middle',
                    text=str(_yv),
                    showarrow=False,
                    align='right',
                    font=dict(size=10, color='#FFFFFF')
                ))
            fig.update_layout(annotations=_anns)
    except Exception:
        pass
    # Overlay y-axis for highlighted tick labels (white on top of gray base)
    try:
        _y2 = dict(
            overlaying='y', matches='y', side='left',
            anchor='free', position=0,  # lock to left edge
            showgrid=False, showline=False, zeroline=False,
            ticks='', ticklen=0,
            tickmode=('array' if 'highlight_vals' in locals() and highlight_vals else 'auto'),
            tickvals=(highlight_vals if 'highlight_vals' in locals() else None),
            ticktext=([str(v) for v in highlight_vals] if 'highlight_vals' in locals() and highlight_vals else None),
            tickfont=dict(size=10, color='#FFFFFF')
        )
        fig.update_layout(yaxis2=_y2)
    except Exception:
        pass


    

    # Дата под осью
    try:
        if has_candles:
            _ts0 = df_plot["ts"].iloc[0]
            _ts0 = pd.to_datetime(_ts0, utc=True) if getattr(_ts0, "tzinfo", None) is None else _ts0
            _date_text = _ts0.tz_convert("America/New_York").strftime("%b %d, %Y")
        else:
            _date_text = session_date_et.strftime("%b %d, %Y")
        fig.update_xaxes(title_text=f"Time<br><span style='font-size:10px;'>{_date_text}</span>", title_standoff=5)
    except Exception:
        pass

    fig.update_layout(legend=dict(itemclick='toggle', itemdoubleclick='toggleothers', font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "staticPlot": False})

    st.download_button(
        "Скачать JSON (Key Levels)",
        data=candles_bytes if isinstance(candles_bytes, (bytes,bytearray)) else json.dumps(candles_json, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{ticker}_{interval}_candles.json",
        mime="application/json",
        key="kl_download"
    )
