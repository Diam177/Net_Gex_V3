import streamlit as st
import pandas as pd
import numpy as np
import time, json, datetime

import importlib, sys
from lib.intraday_chart import render_key_levels_section
from lib.compute import extract_core_from_chain, compute_series_metrics_for_expiry, aggregate_series
from lib.provider_polygon import fetch_stock_history
from lib.utils import choose_default_expiration, env_or_secret
from lib.plotting import make_figure, _select_atm_window
from lib.advanced_analysis import update_ao_summary, render_advanced_analysis_block

st.set_page_config(page_title="Net GEX / AG / PZ / PZ_FP", layout="wide")

# === Secrets / env ===
RAPIDAPI_HOST = env_or_secret(st, "RAPIDAPI_HOST", None)
RAPIDAPI_KEY  = env_or_secret(st, "RAPIDAPI_KEY",  None)
POLYGON_API_KEY = env_or_secret(st, "POLYGON_API_KEY", None)
# Provider flag for early UI
_PROVIDER = "polygon"

with st.sidebar:
    # Основные поля
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
    st.caption(f"Data provider: **{_PROVIDER}**")
    expiry_placeholder = st.empty()
    data_status_placeholder = st.empty()
    download_placeholder = st.empty()
    download_polygon_placeholder = st.empty()
    table_download_placeholder = st.empty()

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
def _fetch_chain_cached(ticker, host, key, expiry_unix=None):
    data, content = fetch_option_chain(ticker, host, key, expiry_unix=expiry_unix)
    return data, content

# === Загрузка данных только из API ===
if (_PROVIDER=="polygon" and POLYGON_API_KEY) or (_PROVIDER=="rapid" and RAPIDAPI_HOST and RAPIDAPI_KEY):
    try:
        base_json, base_bytes = _fetch_chain_cached(
            ticker,
            None if _PROVIDER=="polygon" else RAPIDAPI_HOST,
            POLYGON_API_KEY if _PROVIDER=="polygon" else RAPIDAPI_KEY,
            None
        )
        raw_data, raw_bytes = base_json, base_bytes
        data_status_placeholder.success("Data received")
    except Exception as e:
        data_status_placeholder.error(f"Ошибка запроса ({_PROVIDER}): {e}")
else:
    data_status_placeholder.warning("Укажите POLYGON_API_KEY (или RAPIDAPI_HOST+RAPIDAPI_KEY) в секретах/ENV.")

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

sel_labels = expiry_placeholder.multiselect(
    "Expiration",
    options=exp_labels,
    default=[exp_labels[default_index]] if exp_labels else []
)
# map labels back to epoch expirations
_lbl2ts = {lab: ts for lab, ts in zip(exp_labels, expirations)}
selected_exps = [ _lbl2ts.get(lab) for lab in sel_labels if lab in _lbl2ts ]
if not selected_exps and expirations:
    # ensure at least one selection
    selected_exps = [expirations[default_index]]
# keep primary (first) for places that expect single value (e.g., raw download name)
selected_exp = selected_exps[0]


# Если выбранных дат нет в блоках — дотягиваем недостающие expiry по одной
try:
    for _exp in selected_exps:
        if _exp not in blocks_by_date and ((_PROVIDER=="polygon" and POLYGON_API_KEY) or (_PROVIDER=="rapid" and RAPIDAPI_HOST and RAPIDAPI_KEY)):
            by_date_json, by_date_bytes = _fetch_chain_cached(
                ticker,
                None if _PROVIDER=="polygon" else RAPIDAPI_HOST,
                POLYGON_API_KEY if _PROVIDER=="polygon" else RAPIDAPI_KEY,
                _exp
            )
            _, _, _, expirations2, blocks_by_date2 = extract_core_from_chain(by_date_json)
            blocks_by_date.update(blocks_by_date2)
            raw_bytes = by_date_bytes  # last seen
except Exception as e:
    st.warning(f"Не удалось получить блок для выбранной даты: {e}")

# Кнопка "Скачать сырой JSON"
download_placeholder.download_button(
    "Download JSON",
    data=raw_bytes if raw_bytes is not None else json.dumps(raw_data, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name=f"{ticker}_{selected_exp}_raw.json",
    mime="application/json"
)

# Доп. кнопка: сырые данные от Polygon
if _PROVIDER == "polygon":
    download_polygon_placeholder.download_button(
        "Download Polygon JSON",
        data=raw_bytes if raw_bytes is not None else json.dumps(raw_data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{ticker}_{selected_exp}_polygon_raw.json",
        mime="application/json"
    )
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

# === Вспомогательная функция: суммирование метрик по нескольким экспирациям ===
def _sum_metrics_list(metrics_list):
    import numpy as _np
    # Собираем унифицированную шкалу страйков
    all_strikes = _np.array(sorted({float(x) for m in metrics_list for x in m.get("strikes", [])}), dtype=float)
    if all_strikes.size == 0:
        # Пусто — вернём структуру как у compute_series_metrics_for_expiry, но с нулями
        return {
            "strikes": _np.array([], dtype=float),
            "put_oi": _np.array([], dtype=float),
            "call_oi": _np.array([], dtype=float),
            "put_vol": _np.array([], dtype=float),
            "call_vol": _np.array([], dtype=float),
            "net_gex": _np.array([], dtype=float),
            "ag": _np.array([], dtype=float),
            "pz": _np.array([], dtype=float),
            "pz_fp": _np.array([], dtype=float),
        }
    # Инициализация аккумулирующих массивов
    acc = {
        "put_oi": _np.zeros_like(all_strikes, dtype=float),
        "call_oi": _np.zeros_like(all_strikes, dtype=float),
        "put_vol": _np.zeros_like(all_strikes, dtype=float),
        "call_vol": _np.zeros_like(all_strikes, dtype=float),
        "net_gex": _np.zeros_like(all_strikes, dtype=float),
        "ag": _np.zeros_like(all_strikes, dtype=float),
        "pz": _np.zeros_like(all_strikes, dtype=float),
        "pz_fp": _np.zeros_like(all_strikes, dtype=float),
    }
    # Быстрая индексация по страйку
    idx_map = {float(k): i for i, k in enumerate(all_strikes.tolist())}
    for m in metrics_list:
        s = _np.asarray(m.get("strikes", []), dtype=float)
        for key in ["put_oi","call_oi","put_vol","call_vol","net_gex","ag","pz","pz_fp"]:
            arr = _np.asarray(m.get(key, []), dtype=float)
            if s.size == 0 or arr.size == 0:
                continue
            # Сложить по страйкам
            for val_strike, val in zip(s, arr):
                j = idx_map.get(float(val_strike))
                if j is not None and _np.isfinite(val):
                    acc[key][j] += float(val)
    # Готово
    acc["strikes"] = all_strikes
    return acc
# === Метрики для выбранных экспираций (сумма по страйкам) ===
for _exp in selected_exps:
    if _exp not in blocks_by_date:
        st.error("Не найден блок выбранной экспирации у провайдера. Попробуйте другую дату или обновите страницу.")
        st.stop()
_metrics_list = []
for _exp in selected_exps:
    _metrics = compute_series_metrics_for_expiry(
        S=S, t0=t0, expiry_unix=_exp,
        block=blocks_by_date[_exp],
        day_high=day_high, day_low=day_low,
        all_series=all_series_ctx
    )
    _metrics_list.append(_metrics)

metrics = _sum_metrics_list(_metrics_list)
# --- Advanced Options summary (for the bottom block) ---
try:
    # Prepare lightweight DF with canonical column names expected by update_ao_summary
    _df_ao = pd.DataFrame({
        "put_oi": metrics.get("put_oi", []),
        "call_oi": metrics.get("call_oi", []),
        "put_volume": metrics.get("put_vol", []),
        "call_volume": metrics.get("call_vol", []),
        "net_gex": metrics.get("net_gex", []),
    })
    # Compute a simple Put/Call IV Skew across strikes from all_series_ctx (mean of (iv_put - iv_call))
    _iv_call_acc, _iv_put_acc, _cnt = {}, {}, {}
    for s in (all_series_ctx or []):
        for k, v in (s.get("iv_call") or {}).items():
            if v is None: continue
            _iv_call_acc[k] = _iv_call_acc.get(k, 0.0) + float(v)
            _cnt[k] = _cnt.get(k, 0) + 1
        for k, v in (s.get("iv_put") or {}).items():
            if v is None: continue
            _iv_put_acc[k] = _iv_put_acc.get(k, 0.0) + float(v)
            _cnt[k] = _cnt.get(k, 0) + 1
    _diffs = []
    for k in set(_iv_call_acc.keys()).intersection(_iv_put_acc.keys()):
        c = _cnt.get(k, 1)
        if c <= 0: 
            continue
        _diffs.append((_iv_put_acc[k]/c) - (_iv_call_acc[k]/c))
    _skew = float(np.nanmean(_diffs)) if len(_diffs)>0 else None
    
    # --- Compute ATM IV from nearest expiry (min positive DTE) and nearest strike to S ---
    _atm_iv = None
    try:
        best_dte = 1e9
        for s in (all_series_ctx or []):
            dte = s.get("days") or s.get("dte") or s.get("DTE")
            try:
                dte_val = float(dte)
            except Exception:
                # Fallback: compute from T if available (T in years)
                try:
                    dte_val = float(s.get("T")) * 365.0
                except Exception:
                    dte_val = None
            ivc = s.get("iv_call") or {}
            ivp = s.get("iv_put") or {}
            Ks = set(ivc.keys()) | set(ivp.keys())
            if not Ks:
                continue
            # choose strike closest to S
            def _to_f(x):
                try:
                    return float(x)
                except Exception:
                    try:
                        return float(str(x).replace("C","").replace("P",""))
                    except Exception:
                        return float("nan")
            k_near = min(Ks, key=lambda k: abs(_to_f(k) - float(S)))
            vals = []
            if k_near in ivc and ivc[k_near] is not None:
                vals.append(float(ivc[k_near]))
            if k_near in ivp and ivp[k_near] is not None:
                vals.append(float(ivp[k_near]))
            if vals:
                if (dte_val is not None and dte_val >= 0 and dte_val < best_dte) or (best_dte == 1e9 and dte_val is not None):
                    best_dte = dte_val if dte_val is not None else best_dte
                    _atm_iv = float(np.nanmean(vals))
    except Exception:
        _atm_iv = None

    _extra_iv = {"skew": _skew, "atm_iv": _atm_iv}
    update_ao_summary(ticker, _df_ao, S, selected_exps, extra_iv=_extra_iv)

except Exception:
    pass

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
        try:
            _k = float(g_flip_val)
            _i = int(np.argmin(np.abs(_strike - _k)))
            _max_levels["gflip"] = float(_strike[_i])
        except Exception:
            _max_levels["gflip"] = float(g_flip_val)

    st.session_state['first_chart_max_levels'] = _max_levels
except Exception:
    pass

# === Key Levels chart ===
render_key_levels_section(ticker, None, POLYGON_API_KEY)
# === Advanced Options Market Analysis block ===

# Pre-compute simple intraday VWAP series from minute candles
_vwap_series = None
try:
    candles_json, _ = fetch_stock_history(ticker, None, POLYGON_API_KEY, interval=str(st.session_state.get("kl_interval","1m")), limit=int(st.session_state.get("kl_limit",640)))
    recs = candles_json.get("candles", []) if isinstance(candles_json, dict) else []
    if recs:
        sums_pv, sums_v = 0.0, 0.0
        _vwap_series = []
        for r in recs:
            h = float(r.get("high", 0.0)); l = float(r.get("low", 0.0)); c = float(r.get("close", 0.0))
            v = float(r.get("volume", 0.0))
            tp = (h + l + c) / 3.0
            sums_pv += tp * v
            sums_v  += v
            if sums_v > 0:
                _vwap_series.append(sums_pv / sums_v)
except Exception:
    _vwap_series = None

render_advanced_analysis_block(vwap_series=_vwap_series, fallback_ticker=ticker)
