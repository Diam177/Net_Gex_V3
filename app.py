import streamlit as st
import pandas as pd
import numpy as np
import time, json, datetime

import importlib, sys
from lib.provider import debug_meta as provider_debug_meta
import io, zipfile
from lib.intraday_chart import render_key_levels_section
from lib.compute import extract_core_from_chain, compute_series_metrics_for_expiry, aggregate_series
from lib.utils import choose_default_expiration, env_or_secret
from lib.plotting import make_figure, _select_atm_window

st.set_page_config(page_title="Net GEX / AG / PZ / PZ_FP", layout="wide")

# === Secrets / env ===
POLYGON_API_KEY = env_or_secret(st, "POLYGON_API_KEY", None)

# === Provider (early init for sidebar caption) ===
from lib import provider_polygon as _provider_module_early
fetch_option_chain = _provider_module_early.fetch_option_chain
_PROVIDER = "polygon"


with st.sidebar:
    # Основные поля
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    st.caption(f"Data provider: **{_PROVIDER}**")
    expiry_placeholder = st.empty()
    data_status_placeholder = st.empty()
    download_placeholder = st.empty()
    table_download_placeholder = st.empty()

    # ---- Debug & Download raw data ----
    with st.expander("Debug / Raw data"):
        if "last_chain_json" in st.session_state:
            raw_json = st.session_state["last_chain_json"]
            st.download_button("Download raw option chain JSON", data=json.dumps(raw_json, indent=2), file_name=f"{ticker}_option_chain.json", mime="application/json")
        if "last_debug_meta" in st.session_state:
            meta = st.session_state["last_debug_meta"]
            st.download_button("Download debug meta", data=json.dumps(meta, indent=2), file_name=f"{ticker}_debug_meta.json", mime="application/json")


    # ---- Контролы Key Levels (оставили только Interval/Limit) ----
    st.markdown("### Key Levels — Controls")
    st.selectbox("Interval", ["1m","2m","5m","15m","30m","1h","1d"], index=0, key="kl_interval")
    st.number_input("Limit", min_value=100, max_value=1000, value=640, step=10, key="kl_limit")
    # Last session убран из сайдбара — теперь он над чартом
    # -------------------------------------------------------------

raw_data = None
raw_bytes = None


# === Provider selection ===
provider_module = importlib.import_module("lib.provider_polygon")
if "lib.provider_polygon" in sys.modules:
    importlib.reload(sys.modules["lib.provider_polygon"])
fetch_option_chain = provider_module.fetch_option_chain
_PROVIDER = "polygon"

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_chain_cached(ticker, api_key, expiry_unix=None):
    data, content = fetch_option_chain(ticker, None, api_key, expiry_unix=expiry_unix)
    return data, content

# === Загрузка данных только из API ===
if (_PROVIDER=="polygon") or (_PROVIDER=="polygon"):
    try:
        base_json, base_bytes = _fetch_chain_cached(ticker, POLYGON_API_KEY, None)
        raw_data, raw_bytes = base_json, base_bytes
        data_status_placeholder.success("Data received")
    except Exception as e:
        data_status_placeholder.error(f"Ошибка запроса ({_PROVIDER}): {e}")
else:
    data_status_placeholder.warning("Укажите POLYGON_API_KEY (или POLYGON_API_KEY+POLYGON_API_KEY) в секретах/ENV.")

if raw_data is None:
    st.stop()

# === Parse core ===
try:
    quote, t0, S, expirations, blocks_by_date = extract_core_from_chain(raw_data)
except Exception as e:
    st.error(f"Неверная структура JSON: {e}")
    st.stop()

now_unix = int(time.time())
if not expirations:
    st.error("Список дат экспирации пуст. Проверьте источник.")
    st.stop()

# Безопасный выбор дефолтной экспирации
try:
    default_exp = choose_default_expiration(expirations, now_unix)
    if default_exp not in expirations:
        default_exp = expirations[0]
except Exception:
    default_exp = expirations[0]

def fmt_ts(ts):
    try:
        return datetime.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)

exp_labels = [fmt_ts(e) for e in expirations]
try:
    default_index = expirations.index(default_exp)
except ValueError:
    default_index = 0

sel_label = expiry_placeholder.selectbox("Expiration", options=exp_labels, index=default_index)
selected_exp = expirations[exp_labels.index(sel_label)]

# Если выбранной даты нет в блоках — дотягиваем конкретный expiry
if selected_exp not in blocks_by_date and ((_PROVIDER=="polygon") or (_PROVIDER=="polygon")):
    try:
        by_date_json, by_date_bytes = _fetch_chain_cached(ticker, POLYGON_API_KEY, selected_exp)
        _, _, _, expirations2, blocks_by_date2 = extract_core_from_chain(by_date_json)
        blocks_by_date.update(blocks_by_date2)
        raw_bytes = by_date_bytes
    except Exception as e:
        st.warning(f"Не удалось получить блок для выбранной даты: {e}")


# === Контекст для PZ/PZ_FP (all_series_ctx) ===
all_series_ctx = []
for e, block in blocks_by_date.items():
    try:
        strikes, call_oi, put_oi, call_vol, put_vol, iv_call, iv_put = aggregate_series(block)
        T = max((e - t0) / (365*24*3600), 1e-6)
        all_series_ctx.append({
            "strikes": strikes,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "call_vol": call_vol,
            "put_vol": put_vol,
            "iv_call": iv_call,
            "iv_put": iv_put,
            "T": T
        })
    except Exception:
        pass

day_high = quote.get("regularMarketDayHigh", None)
day_low  = quote.get("regularMarketDayLow", None)

# === Метрики для выбранной экспирации ===
metrics = compute_series_metrics_for_expiry(
    S=S, t0=t0, expiry_unix=selected_exp,
    block=blocks_by_date[selected_exp],
    day_high=day_high, day_low=day_low,
    all_series=all_series_ctx
)

# === Таблица по страйкам ===
df = pd.DataFrame({
    "Strike": metrics["strikes"],
    "Put OI": metrics["put_oi"],
    "Call OI": metrics["call_oi"],
    "Put Volume": metrics["put_vol"],
    "Call Volume": metrics["call_vol"],
    "Net Gex": metrics["net_gex"],
    "AG": metrics["ag"],
    "PZ": np.round(metrics["pz"], 6),
    "PZ_FP": np.round(metrics["pz_fp"], 6),
})

# === G-Flip (эвристика) ===
def compute_gflip(strikes_arr, gex_arr, spot=None, min_run=2, min_amp_ratio=0.12):
    strikes = np.asarray(strikes_arr, dtype=float)
    gex = np.asarray(gex_arr, dtype=float)
    if len(strikes) < 2: return None
    max_abs = float(np.nanmax(np.abs(gex))) if np.any(np.isfinite(gex)) else 0.0
    if max_abs <= 0: return None
    amp_thresh = max_abs * float(min_amp_ratio)
    crossings = []
    for i in range(len(gex)-1):
        gi, gj = gex[i], gex[i+1]
        if np.isfinite(gi) and np.isfinite(gj) and (gi*gj < 0):
            ki, kj = strikes[i], strikes[i+1]
            kflip = ki + (0.0 - gi) * (kj - ki) / (gj - gi)
            post_sign = np.sign(gj)
            j = i+1
            run_idx = []
            while j < len(gex) and np.sign(gex[j]) == post_sign:
                run_idx.append(j); j += 1
            run_len = len(run_idx)
            mean_abs = float(np.nanmean(np.abs(gex[run_idx]))) if run_idx else 0.0
            crossings.append({
                "k": float(kflip),
                "run_len": run_len,
                "mean_abs": mean_abs,
                "stable": (run_len >= int(min_run)) and (mean_abs >= amp_thresh)
            })
    if not crossings: return None
    if spot is None: spot = float(np.nanmedian(strikes))
    best = sorted(crossings, key=lambda c: (not c["stable"], abs(c["k"] - float(spot))))[0]
    return float(best["k"])

g_flip_val = compute_gflip(df["Strike"].values, df["Net Gex"].values, spot=S)

# === Plot ===
st.subheader("GammaStrat v6.5")
cols = st.columns(9)
toggles = {}
names = ["Net Gex","Put OI","Call OI","Put Volume","Call Volume","AG","PZ","PZ_FP","G-Flip"]
defaults = {"Net Gex": True, "Put OI": False, "Call OI": False, "Put Volume": False, "Call Volume": False, "AG": False, "PZ": False, "PZ_FP": False, "G-Flip": False}
for i, name in enumerate(names):
    with cols[i]:
        toggles[name] = st.toggle(name, value=defaults.get(name, False), key=f"tgl_{name}")

series_dict = {
    "Net Gex": df["Net Gex"].values,
    "Put OI": df["Put OI"].values,
    "Call OI": df["Call OI"].values,
    "Put Volume": df["Put Volume"].values,
    "Call Volume": df["Call Volume"].values,
    "AG": df["AG"].values,
    "PZ": df["PZ"].values,
    "PZ_FP": df["PZ_FP"].values,
}

# позиционный вызов — совместим с текущей сигнатурой
idx_keep = _select_atm_window(
    df["Strike"].values,
    df["Call OI"].values,
    df["Put OI"].values,
    S,
    1.5
)

fig = make_figure(
    strikes=df["Strike"].values,
    net_gex=df["Net Gex"].values,
    series_enabled=toggles,
    series_dict=series_dict,
    price=S,
    ticker=ticker,
    g_flip=g_flip_val
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# === Уровни для Key Levels ===
try:
    _strike = np.asarray(df["Strike"].values, dtype=float)

    def _nan_argmax(a):
        a = np.asarray(a, dtype=float)
        if not np.any(np.isfinite(a)): return None
        return int(np.nanargmax(a))

    _max_levels = {}
    # Call/Put Volume
    i_cv = _nan_argmax(df["Call Volume"].values[idx_keep])
    if i_cv is not None:
        _max_levels["call_vol_max"] = float(_strike[idx_keep[i_cv]])
    i_pv = _nan_argmax(df["Put Volume"].values[idx_keep])
    if i_pv is not None:
        _max_levels["put_vol_max"] = float(_strike[idx_keep[i_pv]])

    # AG / PZ
    i_ag = _nan_argmax(df["AG"].values[idx_keep])
    if i_ag is not None:
        _max_levels["ag_max"] = float(_strike[idx_keep[i_ag]])
    i_pz = _nan_argmax(df["PZ"].values[idx_keep])
    if i_pz is not None:
        _max_levels["pz_max"] = float(_strike[idx_keep[i_pz]])

    # Call OI / Put OI
    i_coi = _nan_argmax(df["Call OI"].values[idx_keep])
    if i_coi is not None:
        _max_levels["call_oi_max"] = float(_strike[idx_keep[i_coi]])
    i_poi = _nan_argmax(df["Put OI"].values[idx_keep])
    if i_poi is not None:
        _max_levels["put_oi_max"] = float(_strike[idx_keep[i_poi]])

    # Net GEX extrema
    gex_vals = np.asarray(df["Net Gex"].values, dtype=float)[idx_keep]
    if np.any(np.isfinite(gex_vals)):
        _max_levels["max_pos_gex"] = float(_strike[idx_keep[int(np.nanargmax(gex_vals))]])
        _max_levels["max_neg_gex"] = float(_strike[idx_keep[int(np.nanargmin(gex_vals))]])

    # G-Flip
    if g_flip_val is not None:
        _max_levels["gflip"] = float(g_flip_val)

    st.session_state['first_chart_max_levels'] = _max_levels
except Exception:
    pass

# === Key Levels chart ===
render_key_levels_section(ticker, POLYGON_API_KEY)
