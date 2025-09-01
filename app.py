import streamlit as st
import pandas as pd
import numpy as np
import time, json, math, io, datetime

from lib.provider import fetch_option_chain, debug_meta
from lib.intraday_chart import render_key_levels_section
from lib.compute import extract_core_from_chain, compute_series_metrics_for_expiry, aggregate_series
from lib.utils import choose_default_expiration, env_or_secret
from lib.plotting import make_figure

st.set_page_config(page_title="Net GEX / AG / PZ / PZ_FP", layout="wide")
# === Secrets / env ===
RAPIDAPI_HOST = env_or_secret(st, "RAPIDAPI_HOST", None)
RAPIDAPI_KEY  = env_or_secret(st, "RAPIDAPI_KEY",  None)

with st.sidebar:
    # Controls in sidebar
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    expiry_placeholder = st.empty()
    data_status_placeholder = st.empty()
    download_placeholder = st.empty()
    table_download_placeholder = st.empty()
# === Inputs ===
# Перенесено в левый сайдбар
col_f = st.container()

raw_data = None
raw_bytes = None

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_chain_cached(ticker, host, key, expiry_unix=None):
    data, content = fetch_option_chain(ticker, host, key, expiry_unix=expiry_unix)
    return data, content

# === Fetch from RapidAPI ===
with col_f:
    if RAPIDAPI_HOST and RAPIDAPI_KEY:
        try:
            base_json, base_bytes = _fetch_chain_cached(ticker, RAPIDAPI_HOST, RAPIDAPI_KEY, None)
            raw_data, raw_bytes = base_json, base_bytes
            data_status_placeholder.success("Данные получены")
        except Exception as e:
            st.error(f"Ошибка запроса RapidAPI: {e}")
    else:
        st.warning("Не заданы RAPIDAPI_HOST/RAPIDAPI_KEY. Можно загрузить JSON вручную.")



if raw_data is None:
    st.stop()

# === Parse core ===
try:
    quote, t0, S, expirations, blocks_by_date = extract_core_from_chain(raw_data)
except Exception as e:
    st.error(f"Неверная структура JSON: {e}")
    st.info("Скачайте RAW/pretty JSON и debug-meta в блоке Debug выше и пришлите мне — подгоню адаптер.")
    st.stop()

now_unix = int(time.time())
if not expirations:
    st.error("Список дат экспирации пуст. Проверьте источник.")
    st.stop()

default_exp = choose_default_expiration(expirations, now_unix)

def fmt_ts(ts):
    return datetime.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")

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
        raw_bytes = by_date_bytes  # для кнопки "скачать"
    except Exception as e:
        st.warning(f"Не удалось получить блок для выбранной даты: {e}")

# Кнопка "Скачать сырой JSON"
download_placeholder.download_button(
    "Скачать JSON",
    data=raw_bytes if raw_bytes is not None else json.dumps(raw_data, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name=f"{ticker}_{selected_exp}_raw.json",
    mime="application/json"
)

# Подготовка всех серий (для PZ/PZ_FP нужен контекст всех экспираций)
all_series = []
for e in expirations:
    blk = blocks_by_date.get(int(e))
    if blk is None:
        continue
    strikes, call_oi, put_oi, call_vol, put_vol, iv_call, iv_put = aggregate_series(blk)
    T = max((int(e) - int(quote.get('regularMarketTime', now_unix))) / (365*24*3600), 1e-6)
    all_series.append({
        "E": int(e), "T": float(T), "strikes": strikes,
        "call_oi": call_oi, "put_oi": put_oi,
        "call_vol": call_vol, "put_vol": put_vol,
        "iv_call": iv_call, "iv_put": iv_put
    })

selected_block = blocks_by_date.get(selected_exp)
if selected_block is None:
    st.error("Не найден блок по выбранной экспирации.")
    st.stop()

day_high = quote.get("regularMarketDayHigh", None)
day_low  = quote.get("regularMarketDayLow", None)



def compute_gflip(strikes_arr, gex_arr, spot=None, min_run=3, min_amp_ratio=0.03):
    """
    Practical flip:
      - ищем ВСЕ точки пересечения нуля (строго)
      - для каждой точки проверяем «устойчивость» знака после пересечения:
          * знак сохраняется минимум на min_run соседних страйках
          * средняя |Net Gex| на этом отрезке >= min_amp_ratio * max(|Net Gex|) по ряду
      - возвращаем ближайшую к spot устойчивую точку (если spot не задан — первую устойчивую)
      - если нет устойчивых пересечений, используем ближайшее strict-пересечение
      - если весь ряд одного знака — возвращаем границу диапазона (внутри видимой области)
    """
    import numpy as np
    strikes = np.asarray(strikes_arr, dtype=float)
    gex = np.asarray(gex_arr, dtype=float)

    mask = np.isfinite(strikes) & np.isfinite(gex)
    if mask.sum() < 2:
        return None
    strikes = strikes[mask]; gex = gex[mask]
    order = np.argsort(strikes)
    strikes = strikes[order]; gex = gex[order]

    max_abs = float(np.nanmax(np.abs(gex))) if np.any(np.isfinite(gex)) else 0.0
    if max_abs <= 0:
        return None
    amp_thresh = max_abs * float(min_amp_ratio)

    # Найти строгие пересечения нуля (между соседними точками)
    crossings = []
    for i in range(len(gex)-1):
        gi, gj = gex[i], gex[i+1]
        if np.isfinite(gi) and np.isfinite(gj) and (gi * gj < 0):
            ki, kj = strikes[i], strikes[i+1]
            kflip = ki + (0.0 - gi) * (kj - ki) / (gj - gi)
            # оценка устойчивости после пересечения — берём «после» сторону знака gj
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

    # Точные нули тоже считаем кандидатами
    zero_idx = [int(t) for t in np.where(np.isclose(gex, 0.0, atol=max_abs*1e-6))[0].tolist()]
    for zi in zero_idx:
        # устойчивость определяем по «после» стороне (zi+1 …)
        j = zi+1
        post_sign = np.sign(gex[j]) if j < len(gex) else 0.0
        run_idx = []
        while j < len(gex) and np.sign(gex[j]) == post_sign:
            run_idx.append(j); j += 1
        run_len = len(run_idx)
        mean_abs = float(np.nanmean(np.abs(gex[run_idx]))) if run_idx else 0.0
        crossings.append({
            "k": float(strikes[zi]),
            "i": zi,
            "run_len": run_len,
            "mean_abs": mean_abs,
            "stable": (run_len >= int(min_run)) and (mean_abs >= amp_thresh)
        })

    if crossings:
        # Устойчивые кандидаты
        stable = [c for c in crossings if c["stable"]]
        if stable:
            if spot is None or not np.isfinite(spot):
                return float(stable[0]["k"])
            k_arr = np.array([c["k"] for c in stable], dtype=float)
            return float(k_arr[np.argmin(np.abs(k_arr - float(spot)))])
        # fallback: ближайшее strict-пересечение
        k_arr = np.array([c["k"] for c in crossings], dtype=float)
        if spot is None or not np.isfinite(spot):
            return float(k_arr[0])
        return float(k_arr[np.argmin(np.abs(k_arr - float(spot)))])
    # весь ряд одного знака
    if np.all(gex > 0):
        return float(strikes[-1])
    if np.all(gex < 0):
        return float(strikes[0])
    return None
S_used = float(quote.get('regularMarketPrice', S))
metrics = compute_series_metrics_for_expiry(
    S=S_used,
    t0=int(quote.get("regularMarketTime", t0)),
    expiry_unix=selected_exp,
    block=selected_block,
    day_high=day_high,
    day_low=day_low,
    all_series=all_series
)

# === Table ===
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

g_flip_val = compute_gflip(df["Strike"].values, df["Net Gex"].values, spot=S_used)

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

fig = make_figure(df["Strike"].values, df["Net Gex"].values, toggles, series_dict, price=S_used, ticker=ticker, g_flip=g_flip_val)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --- Key Levels under GammaStrat block ---

# ---- Export maxima/minima from first chart to session_state for intraday chart ----
try:
    import numpy as _np
    _strike = _np.asarray(df["Strike"].values, dtype=float)
    def _nan_argmax(a):
        a = _np.asarray(a, dtype=float)
        if not _np.any(_np.isfinite(a)): 
            return None
        return int(_np.nanargmax(a))
    def _nan_argmin(a):
        a = _np.asarray(a, dtype=float)
        if not _np.any(_np.isfinite(a)): 
            return None
        return int(_np.nanargmin(a))

    _max_levels = {}
    i_cv = _nan_argmax(df["Call Volume"].values)
    if i_cv is not None: _max_levels["call_volume_max"] = float(_strike[i_cv])
    i_pv = _nan_argmax(df["Put Volume"].values)
    if i_pv is not None: _max_levels["put_volume_max"] = float(_strike[i_pv])
    i_ag = _nan_argmax(df["AG"].values)
    if i_ag is not None: _max_levels["ag_max"] = float(_strike[i_ag])
    i_pz = _nan_argmax(df["PZ"].values)
    if i_pz is not None: _max_levels["pz_max"] = float(_strike[i_pz])
    # Call OI max / Put OI min
    i_coi = _nan_argmax(df["Call OI"].values)
    if i_coi is not None: _max_levels["call_oi_max"] = float(_strike[i_coi])
    i_poi_min = _nan_argmin(df["Put OI"].values)
    if i_poi_min is not None: _max_levels["put_oi_min"] = float(_strike[i_poi_min])
    # G-Flip level (already computed as g_flip_val if available)
    if 'g_flip_val' in locals() and g_flip_val is not None:
        try:
            _max_levels["gflip"] = float(g_flip_val)
        except Exception:
            pass
    \
# Net GEX extrema (by strike)
    try:
        gex_vals = _np.asarray(df["Net Gex"].values, dtype=float)
        i_ng_pos = _nan_argmax(gex_vals)
        if i_ng_pos is not None: _max_levels["max_pos_gex"] = float(_strike[i_ng_pos])
        # min (most negative)
        if _np.any(_np.isfinite(gex_vals)):
            i_ng_neg = int(_np.nanargmin(gex_vals))
            _max_levels["max_neg_gex"] = float(_strike[i_ng_neg])
        # abs max NG
        if _np.any(_np.isfinite(gex_vals)):
            i_ng_abs = int(_np.nanargmax(_np.abs(gex_vals)))
            _max_levels["ng"] = float(_strike[i_ng_abs])
    except Exception:
        pass
st.session_state['first_chart_max_levels'] = _max_levels
except Exception as _e:
    # Silent: don't break UI if something's missing
    pass
render_key_levels_section(ticker, RAPIDAPI_HOST, RAPIDAPI_KEY)

