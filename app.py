
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import requests
from typing import Any, Dict, List, Tuple

import streamlit as st



# Advanced metrics block
try:
    from advanced_analysis_block import render_advanced_analysis_block
except Exception:
    render_advanced_analysis_block = None
# --- Intraday price loader for Key Levels (Polygon v2 aggs 1-minute) ---
def _load_session_price_df_for_key_levels(ticker: str, session_date_str: str, api_key: str, timeout: int = 30):
    import pandas as pd
    import pytz
    t_raw = (ticker or '').strip()
    t = t_raw.upper()
    if t in {'SPX','NDX','VIX','RUT','DJX'} and not t_raw.startswith('I:'):
        t = f'I:{t}'
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
    
    # --- VWAP logic with SPX proxy ---
    # expose per-bar 'vw' for indices
    try:
        df["vw"] = pd.to_numeric(vwap_api, errors="coerce")
    except Exception:
        pass

    total_vol = float(df["volume"].fillna(0.0).sum())
    if "vw" in df.columns and df["vw"].notna().any() and total_vol == 0.0:
        # index case: use Polygon per-bar 'vw' directly
        df["vwap"] = df["vw"]
    elif any(v is not None for v in vwap_api) and total_vol > 0.0:
        vw = pd.Series(pd.to_numeric(vwap_api, errors="coerce"), index=df.index)
        vol = df["volume"].fillna(0.0)
        cum_vol = vol.cumsum().replace(0, pd.NA)
        df["vwap"] = (vw * vol).cumsum() / cum_vol
    else:
        df["vwap"] = pd.NA

    # SPY proxy only for SPX if vwap still NaN
    t_up = (ticker or "").upper()
    if t_up in {"SPX","^SPX","I:SPX"} and (df["vwap"].isna().all() or not df["vwap"].notna().any()):
        try:
            base = "https://api.polygon.io"
            url_spy = (f"{base}/v2/aggs/ticker/SPY/range/1/minute/"
                       f"{session_date_str}/{session_date_str}"
                       f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")
            r2 = requests.get(url_spy, timeout=timeout)
            r2.raise_for_status()
            js2 = r2.json() or {}
            res2 = js2.get("results") or []
            if res2:
                import pandas as pd, numpy as np, pytz
                tz = pytz.timezone("America/New_York")
                spy_time = pd.to_datetime([x.get("t") for x in res2], unit="ms", utc=True).tz_convert(tz)
                spy_df = pd.DataFrame({
                    "time": spy_time,
                    "spy_close": pd.Series(pd.to_numeric([x.get("c") for x in res2], errors="coerce")),
                    "spy_vol":   pd.Series(pd.to_numeric([x.get("v") for x in res2], errors="coerce")).fillna(0.0),
                    "spy_vw":    pd.Series(pd.to_numeric([x.get("vw") for x in res2], errors="coerce")),
                }).sort_values("time")
                # RTH filter
                spy_df = spy_df.set_index("time").between_time("09:30", "16:00").reset_index()
                px = spy_df["spy_vw"].where(spy_df["spy_vw"].notna(), spy_df["spy_close"])
                cumv = spy_df["spy_vol"].cumsum().replace(0, pd.NA)
                spy_df["spy_vwap"] = (px.mul(spy_df["spy_vol"])).cumsum() / cumv
                spy_df["minute"] = spy_df["time"].dt.floor("min")

                # SPX minutes in ET
                df_et = df.copy()
                try:
                    df_et["time"] = df_et["time"].dt.tz_convert(tz)
                except Exception:
                    df_et["time"] = pd.to_datetime(df_et["time"], utc=True).dt.tz_convert(tz)
                df_et["minute"] = df_et["time"].dt.floor("min")

                merged = pd.merge(df_et[["minute","price"]].rename(columns={"price":"spx_price"}),
                                  spy_df[["minute","spy_close"]], on="minute", how="inner").sort_values("minute")
                if not merged.empty and merged["spy_close"].iloc[0] and merged["spx_price"].iloc[0]:
                    alpha0 = float(merged["spx_price"].iloc[0]) / float(merged["spy_close"].iloc[0])
                else:
                    alpha0 = 10.0

                ratio_map = (merged.set_index("minute")["spx_price"] / merged.set_index("minute")["spy_close"]).dropna()
                minutes = spy_df["minute"].drop_duplicates().sort_values().reset_index(drop=True)
                alphas = []
                last_alpha = alpha0
                session_start = minutes.iloc[0] if len(minutes) else None
                for m in minutes:
                    w_start = m - pd.Timedelta(minutes=15)
                    r_win = ratio_map[(ratio_map.index > w_start) & (ratio_map.index <= m)]
                    if len(r_win) >= 5:
                        q1, q99 = r_win.quantile(0.01), r_win.quantile(0.99)
                        raw_alpha = r_win.clip(q1, q99).median()
                    else:
                        raw_alpha = last_alpha
                    if session_start is not None and (m - session_start) <= pd.Timedelta(minutes=30):
                        low = 0.98 * alpha0; high = 1.02 * alpha0
                        raw_alpha = min(max(raw_alpha, low), high)
                    delta = raw_alpha - last_alpha
                    max_step = 0.005 * last_alpha
                    if abs(delta) > abs(max_step):
                        raw_alpha = last_alpha + (max_step if delta > 0 else -max_step)
                    alphas.append(raw_alpha)
                    last_alpha = raw_alpha
                alpha_series = pd.Series(alphas, index=minutes, name="alpha")

                alpha_aligned = pd.merge(df_et[["minute"]], alpha_series, left_on="minute", right_index=True, how="left")["alpha"].fillna(method="ffill")
                spy_vwap_aligned = pd.merge(df_et[["minute"]], spy_df[["minute","spy_vwap"]], on="minute", how="left")["spy_vwap"].fillna(method="ffill")
                proxy_vwap = alpha_aligned.values * spy_vwap_aligned.values

                df["vwap"] = df["vwap"].where(df["vwap"].notna(), proxy_vwap)
        except Exception as e:
            try:
                import streamlit as st
                code = getattr(getattr(e, "response", None), "status_code", None)
                st.warning(f"SPY proxy VWAP недоступен ({code}) {type(e).__name__}: {e}. Использую TWAP.", icon="⚠️")
            except Exception:
                pass

    # Final fallback: TWAP
    if df["vwap"].isna().all():
        pr = df["price"].fillna(method="ffill")
        df["vwap"] = pr.expanding().mean()
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
    st.error("POLYGON_API_KEY is not set in Streamlit Secrets or environment variables.")
    st.stop()



# --- Input state helpers ------------------------------------------------------
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "SPY"

def _normalize_ticker():
    t = st.session_state.get("ticker", "")
    st.session_state["ticker"] = (t or "").strip().upper()

# --- Controls moved to sidebar ----------------------------------------------
with st.sidebar:
    st.text_input("Ticker", key="ticker", on_change=_normalize_ticker)
    ticker = st.session_state.get("ticker", "")

    # Получаем список будущих экспираций под выбранный тикер
    expirations: list[str] = st.session_state.get(f"expirations:{ticker}", [])
    if not expirations and ticker:
        try:
            expirations = list_future_expirations(ticker, api_key)
            st.session_state[f"expirations:{ticker}"] = expirations
        except Exception as e:
            st.error(f"Unable to retrieve expiration dates: {e}")
            expirations = []

    if expirations:
        # по умолчанию ближайшая дата — первая в списке
        default_idx = 0
        sel = st.selectbox("Expiration date", options=expirations, index=default_idx, key=f"exp_sel:{ticker}")
        expiration = sel

        # --- Режим агрегации экспираций ---
        mode_exp = st.radio("Expiration mode", ["Single","Multi"], index=0, horizontal=True)
        selected_exps = []
        weight_mode = "equal"
        if mode_exp == "Single":
            selected_exps = [expiration]
        else:
            selected_exps = st.multiselect("Select expiration", options=expirations, default=expirations[:2])
            weight_mode = st.selectbox("Weighing", ["equal","1/T","1/√T"], index=2)
    else:
        expiration = ""
        st.warning("Нет доступных дат экспираций для тикера.")



# --- Empty state guard for Multi with no selected expirations ---
try:
    if ('mode_exp' in locals()) and (mode_exp == 'Multi') and (not selected_exps):
        st.info('Select expiration date')
        st.stop()
except Exception:
    pass

# --- Data fetch ---------------------------------------------------------------
raw_records: List[Dict] | None = None
snapshot_js: Dict | None = None

if ticker and expiration:
    try:
        snapshot_js = download_snapshot_json(ticker, expiration, api_key)
        raw_records = _coerce_results(snapshot_js)
        st.session_state["raw_records"] = raw_records
    except PolygonError as e:
        st.error(f"Ошибка Polygon: {e}")
    except Exception as e:
        st.error(f"Ошибка при загрузке snapshot JSON: {e}")

# --- Sidebar: download raw provider JSON -------------------------------------
if snapshot_js:
    try:
        raw_bytes = json.dumps(snapshot_js, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        fname = f"{ticker}_{expiration}.json"
        st.sidebar.download_button(
            label="Download JSON",
            data=raw_bytes,
            file_name=fname,
            mime="application/json",
            use_container_width=True,
        )
    except Exception as e:
        st.sidebar.error(f"Failed to prepare JSON for download: {e}")
else:
    st.sidebar.info("Select the ticker and expiration date to download the snapshot and JSON.")

# --- Download tables button placeholder (below raw JSON) ---------------------
dl_tables_container = st.sidebar.empty()

# --- Spot price ---------------------------------------------------------------
S: float | None = None
if raw_records:
    S = _infer_spot_from_snapshot(raw_records)

S: float | None = None
# Источник истины для S — только сырые JSON-записи (snapshot/chain)
if raw_records:
    S = _infer_spot_from_snapshot(raw_records)
# Без иных источников. Если S не извлечён — позже отобразим явную ошибку.
                                    import pytz, pandas as pd
                                    _session_date_str = pd.Timestamp.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                                    _price_df = _load_session_price_df_for_key_levels(ticker, _session_date_str, st.secrets.get("POLYGON_API_KEY", ""))
                                    render_key_levels(df_final=df_final, ticker=ticker, g_flip=_gflip_val, price_df=_price_df, session_date=_session_date_str, toggle_key="key_levels_main")

                                    # --- Advanced Analysis Block (Single) — placed under Key Levels ---
                                    try:
                                        if render_advanced_analysis_block is not None:
                                            try:
                                                import pandas as pd, pytz
                                                _session_date_str = pd.Timestamp.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                                                _price_df = _load_session_price_df_for_key_levels(
                                                    ticker, _session_date_str, st.secrets.get("POLYGON_API_KEY", "")
                                                )
                                            except Exception:
                                                _price_df = None
                                            render_advanced_analysis_block(
                                                ticker=ticker,
                                                df_final=df_final,
                                                df_corr=df_corr if 'df_corr' in locals() else None,
                                                S=S if 'S' in locals() else None,
                                                price_df=_price_df,
                                                selected_exps=selected_exps if 'selected_exps' in locals() else None,
                                                weight_mode=weight_mode if 'weight_mode' in locals() else "1/√T",
                                                caption_suffix="Агрегировано по выбранной экспирации."
                                            )
                                    except Exception as _aabs_e:
                                        st.warning("Advanced block (Single) failed")
                                        st.exception(_aabs_e)
                                except Exception as _kl_e:
                                    st.error('Failed to display Key Levels chart')
                                    st.exception(_kl_e)

                            else:
                                st.info("The final table is empty for the selected expiration.")
            except Exception as _e:
                st.error("Error constructing final table.")
                st.exception(_e)
    except Exception as e:
            st.error("Pipeline sanitize/window error.")
            st.exception(e)
