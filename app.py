import streamlit as st
import pandas as pd
import numpy as np
import time, json, math, io, datetime

from lib.provider import fetch_option_chain, debug_meta
from lib.intraday_chart import render_key_levels_section
from lib.compute import extract_core_from_chain, compute_series_metrics_for_expiry, aggregate_series
from lib.utils import choose_default_expiration, env_or_secret
from lib.plotting import make_figure, _select_atm_window

st.set_page_config(page_title="Net GEX / AG / PZ / PZ_FP", layout="wide")
# === Secrets / env ===
RAPIDAPI_HOST = env_or_secret(st, "RAPIDAPI_HOST", None)
RAPIDAPI_KEY  = env_or_secret(st, "RAPIDAPI_KEY",  None)

with st.sidebar:
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    expiry_placeholder = st.empty()
    data_status_placeholder = st.empty()
    download_placeholder = st.empty()
    table_download_placeholder = st.empty()

col_f = st.container()
raw_data = None
raw_bytes = None

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_chain_cached(ticker, host, key, expiry_unix=None):
    data, content = fetch_option_chain(ticker, host, key, expiry_unix=expiry_unix)
    return data, content

with col_f:
    if RAPIDAPI_HOST and RAPIDAPI_KEY:
        try:
            base_json, base_bytes = _fetch_chain_cached(ticker, RAPIDAPI_HOST, RAPIDAPI_KEY, None)
            raw_data, raw_bytes = base_json, base_bytes
            data_status_placeholder.success("Данные получены")
        except Exception as e:
            data_status_placeholder.error(f"Ошибка запроса: {e}")

    uploader = st.file_uploader("JSON (опционная цепочка)", type=["json","txt"], accept_multiple_files=False)
    if uploader is not None:
        try:
            file_bytes = uploader.read()
            raw_data = json.loads(file_bytes.decode("utf-8"))
            raw_bytes = file_bytes
            data_status_placeholder.info("Загружен локальный JSON")
        except Exception as e:
            data_status_placeholder.error(f"Ошибка чтения JSON: {e}")

if raw_data is None:
    st.stop()

# === Core extract ===
quote, t0, S, expirations, blocks_by_date = extract_core_from_chain(raw_data)

def fmt_ts(e):
    try:
        return datetime.datetime.utcfromtimestamp(int(e)).strftime("%Y-%m-%d")
    except Exception:
        return str(e)

# --- УСИЛЕННАЯ ЗАЩИТА ВЫБОРА ЭКСПИРАЦИИ ---
# Приводим к списку
try:
    if expirations is None:
        expirations = []
    elif not isinstance(expirations, (list, tuple)):
        expirations = list(expirations)
    else:
        expirations = list(expirations)
except Exception:
    expirations = []

# Пусто — аккуратно выходим
if len(expirations) == 0:
    st.error("Не удалось извлечь список экспираций из JSON. Проверь входные данные.")
    st.stop()

# Дефолтная экспирация с безопасным fallback
try:
    default_exp = choose_default_expiration(expirations)
    if default_exp not in expirations:
        # на случай, если util вернул несуществующее значение
        default_exp = expirations[0]
except Exception:
    default_exp = expirations[0]

exp_labels = [fmt_ts(e) for e in expirations]
try:
    default_index = expirations.index(default_exp)
except ValueError:
    default_index = 0

sel_label = expiry_placeholder.selectbox("Expiration", options=exp_labels, index=default_index)
selected_exp = expirations[exp_labels.index(sel_label)]

# Если выбранной даты нет в уже пришедшем блоке — дотягиваем конкретный expiry
if selected_exp not in blocks_by_date and RAPIDAPI_HOST and RAPIDAPI_KEY:
    try:
        by_date_json, by_date_bytes = _fetch_chain_cached(ticker, RAPIDAPI_HOST, RAPIDAPI_KEY, selected_exp)
        quote2, t02, S2, expirations2, blocks_by_date2 = extract_core_from_chain(by_date_json)
        blocks_by_date.update(blocks_by_date2)
        raw_bytes = by_date_bytes
    except Exception as e:
        st.warning(f"Не удалось получить блок для выбранной даты: {e}")

# Кнопка "Скачать сырой JSON"
download_placeholder.download_button(
    "Скачать JSON",
    data=raw_bytes if raw_bytes is not None else json.dumps(raw_data, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name=f"{ticker}_{selected_exp}_raw.json",
    mime="application/json"
)

# Подготовка всех серий
all_series = []
for e, block in blocks_by_date.items():
    try:
        series = aggregate_series(block)
        metrics = compute_series_metrics_for_expiry(series, e, S)
        all_series.append((e, series, metrics))
    except Exception:
        pass

# Выбор серии по выбранной экспирации
series = aggregate_series(blocks_by_date[selected_exp])
metrics = compute_series_metrics_for_expiry(series, selected_exp, S)

# Таблица по страйкам
df = pd.DataFrame({
    "Strike": series["strikes"],
    "Put OI": series["put_oi"],
    "Call OI": series["call_oi"],
    "Put Volume": series.get("put_volume", np.zeros_like(series["strikes"], dtype=float)),
    "Call Volume": series.get("call_volume", np.zeros_like(series["strikes"], dtype=float)),
    "Net Gex": metrics["net_gex"],
    "AG": metrics["ag"],
    "PZ": np.round(metrics["pz"], 6),
    "PZ_FP": np.round(metrics["pz_fp"], 6),
})

def compute_gflip(strikes, gex, min_amp_ratio=0.12, min_run=2, spot=None):
    strikes = np.asarray(strikes, dtype=float)
    gex     = np.asarray(gex, dtype=float)
    if len(strikes) < 2:
        return None
    max_abs = float(np.nanmax(np.abs(gex))) if np.any(np.isfinite(gex)) else 0.0
    if max_abs <= 0:
        return None
    amp_thresh = max_abs * float(min_amp_ratio)

    crossings = []
    for i in range(len(gex)-1):
        gi, gj = gex[i], gex[i+1]
        if np.isfinite(gi) and np.isfinite(gj) and (gi * gj < 0):
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
                "i": i,
                "run_len": run_len,
                "mean_abs": mean_abs,
                "stable": (run_len >= int(min_run)) and (mean_abs >= amp_thresh)
            })

    zero_idx = [int(t) for t in np.where(np.isclose(gex, 0.0, atol=max_abs*1e-6))[0].tolist()]
    for zi in zero_idx:
        crossings.append({"k": strikes[zi], "i": zi, "run_len": 0, "mean_abs": 0.0, "stable": False})

    if not crossings:
        return None
    if spot is None:
        spot = float(np.nanmedian(strikes))
    best = sorted(crossings, key=lambda c: (not c["stable"], abs(c["k"] - float(spot))))[0]
    return float(best["k"])

g_flip_val = compute_gflip(df["Strike"].values, df["Net Gex"].values, spot=S)

table_csv = df.to_csv(index=False).encode('utf-8')
table_download_placeholder.download_button(
    'Скачать таблицу', data=table_csv,
    file_name=f"{ticker}_{selected_exp}_table.csv", mime='text/csv'
)

# === Plot ===
st.subheader("GammaStrat v4.5")
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

idx_keep = _select_atm_window(df["Strike"].values, df["Call OI"].values, df["Put OI"].values, price=S, widen=1.5)

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
    _np = np
    _strike = np.asarray(df["Strike"].values, dtype=float)
    _idx_keep = idx_keep

    def _nan_argmax(a):
        if not np.any(np.isfinite(a)):
            return None
        return int(np.nanargmax(a))

    _max_levels = {}
    # Call/Put Volume
    i_cv = _nan_argmax(_np.asarray(df["Call Volume"].values, dtype=float)[_idx_keep])
    if i_cv is not None:
        _max_levels["call_vol_max"] = float(_strike[_idx_keep[i_cv]])
    i_pv = _nan_argmax(_np.asarray(df["Put Volume"].values, dtype=float)[_idx_keep])
    if i_pv is not None:
        _max_levels["put_vol_max"] = float(_strike[_idx_keep[i_pv]])

    # AG / PZ
    i_ag = _nan_argmax(_np.asarray(df["AG"].values, dtype=float)[_idx_keep])
    if i_ag is not None:
        _max_levels["ag_max"] = float(_strike[_idx_keep[i_ag]])
    i_pz = _nan_argmax(_np.asarray(df["PZ"].values, dtype=float)[_idx_keep])
    if i_pz is not None:
        _max_levels["pz_max"] = float(_strike[_idx_keep[i_pz]])

    # Call/Put OI
    i_coi = _nan_argmax(_np.asarray(df["Call OI"].values, dtype=float)[_idx_keep])
    if i_coi is not None:
        _max_levels["call_oi_max"] = float(_strike[_idx_keep[i_coi]])
    i_poi_max = _nan_argmax(_np.asarray(df["Put OI"].values, dtype=float)[_idx_keep])
    if i_poi_max is not None:
        _max_levels["put_oi_max"] = float(_strike[_idx_keep[i_poi_max]])

    # Net GEX extrema
    try:
        gex_vals = _np.asarray(df["Net Gex"].values, dtype=float)[_idx_keep]
        i_ng_pos = _nan_argmax(gex_vals)
        if i_ng_pos is not None:
            _max_levels["max_pos_gex"] = float(_strike[_idx_keep[i_ng_pos]])
        if _np.any(_np.isfinite(gex_vals)):
            i_ng_neg = int(_np.nanargmin(gex_vals))
            _max_levels["max_neg_gex"] = float(_strike[_idx_keep[i_ng_neg]])
    except Exception:
        pass

    # G-Flip
    if g_flip_val is not None:
        try:
            _max_levels["gflip"] = float(g_flip_val)
        except Exception:
            pass

    st.session_state['first_chart_max_levels'] = _max_levels
except Exception:
    pass

# === Key Levels chart ===
render_key_levels_section(ticker, RAPIDAPI_HOST, RAPIDAPI_KEY)
