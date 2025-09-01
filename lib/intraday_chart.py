
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

        # ===== Horizontal lines from First Chart maxima/minima (robust discovery) =====
        # We try to discover levels in st.session_state even if keys vary between versions.
        import math

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                try:
                    return float(str(x).strip())
                except Exception:
                    return None

        # normalize key name: lower, remove spaces/underscores/hyphens
        def _norm(k: str) -> str:
            return str(k).replace("_","").replace("-","").replace(" ","").lower()

        # Deep scan any mapping/list structure for candidate values
        def _deep_find_levels(obj, want_keys):
            found = {}
            stack = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    for k, v in cur.items():
                        nk = _norm(k)
                        for tag, keyset in want_keys.items():
                            if nk in keyset and tag not in found:
                                val = None
                                if isinstance(v, dict):
                                    # nested dict with typical leaf names
                                    for leaf in ("price","strike","level","value","y","v","val"):
                                        if leaf in v:
                                            val = _to_float(v[leaf]); 
                                            if val is not None: break
                                if val is None:
                                    val = _to_float(v)
                                if val is not None:
                                    found[tag] = val
                        # continue scanning deeper
                        if isinstance(v, (dict, list, tuple)):
                            stack.append(v)
                elif isinstance(cur, (list, tuple)):
                    for it in cur:
                        if isinstance(it, (dict, list, tuple)):
                            stack.append(it)
            return found

        # define many aliases for each required level
        WANT = {
            "call_vol_max": set(_norm(x) for x in [
                "call_volume_max", "max_call_volume", "callVolMax", "call_volume_max_strike",
                "call_volume_peak", "call_volumemax", "call_volume_max_price"
            ]),
            "put_vol_max": set(_norm(x) for x in [
                "put_volume_max", "max_put_volume", "putVolMax", "put_volume_max_strike",
                "put_volume_peak", "put_volumemax", "put_volume_max_price"
            ]),
            "ag_max": set(_norm(x) for x in [
                "ag_max", "AG_max", "absolute_gamma_max", "abs_gamma_max", "gamma_abs_max", "AGMax"
            ]),
            "pz_max": set(_norm(x) for x in [
                "pz_max", "PZ_max", "power_zone_max", "powerzone_max", "pzPeak", "PZMax"
            ]),
            "gflip": set(_norm(x) for x in [
                "g_flip", "gflip", "gamma_flip", "gammaFlip", "gFlipLevel", "gflip_strike"
            ]),
            "call_oi_max": set(_norm(x) for x in [
                "call_oi_max", "max_call_oi", "callOiMax", "call_oi_max_strike", "call_oi_peak"
            ]),
            "put_oi_min": set(_norm(x) for x in [
                "put_oi_min", "min_put_oi", "putOiMin", "put_oi_min_strike", "put_oi_trough"
            ]),
        }

        # Gather candidate containers to look into
        containers = [st.session_state]
        for k in ("first_chart_max_levels", "opt_max_levels", "key_levels", "levels", "max_levels", "first_chart"):
            if isinstance(st.session_state.get(k), (dict, list)):
                containers.append(st.session_state[k])

        levels = {}
        for box in containers:
            got = _deep_find_levels(box, WANT)
            levels.update({k:v for k,v in got.items() if v is not None})

        # Draw helper
        def _hline(y, name):
            if y is None or (isinstance(y, float) and (math.isnan(y) or math.isinf(y))):
                return
            try:
                fig.add_hline(y=float(y), line_dash="dot", line_width=1, annotation_text=name, annotation_position="right")
            except Exception:
                x0, x1 = df_plot["ts"].iloc[0], df_plot["ts"].iloc[-1]
                fig.add_trace(go.Scatter(x=[x0,x1], y=[y,y], mode="lines", name=name, line=dict(dash="dot", width=1)))

        # Required by the task
        _hline(levels.get("call_vol_max"), "Call Vol max")
        _hline(levels.get("put_vol_max"), "Put Vol max")
        _hline(levels.get("ag_max"),      "AG max")
        _hline(levels.get("pz_max"),      "PZ max")
        _hline(levels.get("gflip"),       "G-Flip")

        # Additionally: Call OI max and Put OI min
        _hline(levels.get("call_oi_max"), "Call OI max")
        _hline(levels.get("put_oi_min"),  "Put OI min")

        # Optional debug
        if st.session_state.get("kl_debug"):
            st.write("Detected levels on intraday chart:", levels)

        # ----- Horizontal lines from the first chart (max levels) -----
        # We read precomputed price levels (strikes) from Streamlit session_state.
        # Expected keys (any of these dicts can be present): first_chart_max_levels, opt_max_levels,
        # key_levels, levels, max_levels. Within the dict we look for common field names.
        def _pick_level(dct, name_variants):
            # Return first numeric value found under any key in name_variants;
            # also support nested dicts containing fields like price/strike/value.
            for k, v in dct.items():
                kl = k.replace("_", "").replace("-", "").lower()
                for cand in name_variants:
                    if kl == cand:
                        try:
                            return float(v)
                        except Exception:
                            pass
                    # handle nested dicts where the outer key matches and inner has 'price/strike/value'
                    if kl == cand and isinstance(v, dict):
                        for leaf in ("price", "strike", "level", "value"):
                            if leaf in v:
                                try:
                                    return float(v[leaf])
                                except Exception:
                                    pass
            # try nested dicts one level down (without relying on exact outer key match)
            for v in dct.values():
                if isinstance(v, dict):
                    got = _pick_level(v, name_variants)
                    if got is not None:
                        return got
            return None

        # Candidate containers
        _containers = [st.session_state]
        for _k in ("first_chart_max_levels", "opt_max_levels", "key_levels", "levels", "max_levels"):
            if isinstance(st.session_state.get(_k), dict):
                _containers.append(st.session_state[_k])

        # Variants for each level name (normalized: lowercase, no underscores/hyphens)
        _variants = {
            "call": ["callvolumemaxstrike", "callvolumemax", "maxcallvolume", "callmax", "callmaxstrike", "callmaxlevel"],
            "put":  ["putvolumemaxstrike", "putvolumemax", "maxputvolume", "putmax", "putmaxstrike", "putmaxlevel"],
            "ag":   ["agmax", "aggammamax", "aggregategammamax", "aggmax"],
            "pz":   ["pzmax", "pressurezonemax", "maxpz"],
            "gflip":["gflip", "gflipstrike", "gammaflip", "gammagflip", "gfliplevel"]
        }

        _levels_found = {}
        for _name, _keys in _variants.items():
            val = None
            for _d in _containers:
                val = _pick_level(_d, _keys)
                if val is not None:
                    break
            if val is not None:
                _levels_found[_name] = val

        # Draw horizontal lines if levels are present.
        # We use shapes (add_hline) so we don't impact traces/legend or interactions.
        try:
            import math
            for _nm, _y in _levels_found.items():
                if _y is None or (isinstance(_y, float) and (math.isnan(_y) or math.isinf(_y))):
                    continue
                fig.add_hline(y=float(_y), line_dash="dot", line_width=1)
        except Exception:
            # Fallback via Scatter if add_hline is unavailable in the runtime
            try:
                _x0, _x1 = df_plot["ts"].iloc[0], df_plot["ts"].iloc[-1]
                for _nm, _y in _levels_found.items():
                    fig.add_trace(go.Scatter(x=[_x0, _x1], y=[_y, _y], mode="lines", name=f"{_nm}_lvl",
                                             line=dict(dash="dot", width=1)))
            except Exception:
                pass

        # Optional debug block
        if 'kl_debug' in st.session_state and st.session_state['kl_debug']:
            st.write("Max levels from first chart (detected):", _levels_found)
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
        fig.add_annotation(xref="paper", yref="paper", x=0.0, y=-0.11, text=_date_text, showarrow=False, align="left")
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
