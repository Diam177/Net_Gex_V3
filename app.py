import streamlit as st
import pandas as pd
import numpy as np
import time, json, datetime

import importlib, sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.intraday_chart import render_key_levels_section
from lib.compute import extract_core_from_chain, compute_series_metrics_for_expiry, aggregate_series
from lib.provider_polygon import fetch_stock_history
from lib.utils import choose_default_expiration, env_or_secret
from lib.plotting import make_figure, _select_atm_window
from lib.ui_final_table import render_final_table
from lib.advanced_analysis import update_ao_summary, render_advanced_analysis_block

# Update page title to reflect new Power Zone and Easy Reach metrics
st.set_page_config(page_title="Net GEX / AG / Power Zone / Easy Reach", layout="wide")

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

# --- Auto populate session_state for final table ---
try:
    # Build raw_records from blocks_by_date for selected expirations
    _exp_sig = tuple(sorted(selected_exps)) if isinstance(selected_exps, (list, tuple)) else ()
    _need = (
        "raw_records" not in st.session_state or
        st.session_state.get("_last_exp_sig") != _exp_sig or
        st.session_state.get("_last_ticker") != ticker
    )
    if _need:
        _raw_records = []
        for _exp in (selected_exps or []):
            blk = blocks_by_date.get(_exp, {}) or {}
            for _side, _key in (("C","calls"), ("P","puts")):
                for r in (blk.get(_key, []) or []):
                    _raw_records.append({
                        "side": "call" if _side=="C" else "put",
                        "strike": r.get("strike"),
                        "expiration": _exp,
                        "open_interest": r.get("openInterest") if r.get("openInterest") is not None else r.get("open_interest"),
                        "volume": r.get("volume"),
                        "implied_volatility": r.get("impliedVolatility") if r.get("impliedVolatility") is not None else r.get("implied_volatility"),
                        "delta": r.get("delta"),
                        "gamma": r.get("gamma"),
                        "vega": r.get("vega"),
                        "contract_size": r.get("contractSize") if r.get("contractSize") is not None else r.get("contract_size") or 100,
                    })
        if _raw_records:
            st.session_state["raw_records"] = _raw_records
            st.session_state["spot"] = float(S)
            st.session_state["_last_exp_sig"] = _exp_sig
            st.session_state["_last_ticker"] = ticker
            # Precompute df_corr/windows (non-fatal if fails)
            try:
                from lib.sanitize_window import SanitizerConfig, sanitize_and_window_pipeline
                _res = sanitize_and_window_pipeline(_raw_records, S=float(S), cfg=SanitizerConfig())
                st.session_state["df_corr"] = _res["df_corr"]
                st.session_state["windows"] = _res["windows"]
            except Exception:
                pass
except Exception:
    pass
# --- End auto populate ---


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
        # Compute per-series metrics to obtain gamma exposures and OI/vol per strike
        series_metrics = compute_series_metrics_for_expiry(
            S=S, t0=t0, expiry_unix=e, block=block,
            day_high=day_high, day_low=day_low, all_series=None
        )
        strikes = series_metrics.get("strikes", [])
        call_oi_arr = series_metrics.get("call_oi", [])
        put_oi_arr  = series_metrics.get("put_oi",  [])
        call_vol_arr= series_metrics.get("call_vol", [])
        put_vol_arr = series_metrics.get("put_vol", [])
        gamma_abs_share_arr = series_metrics.get("gamma_abs_share", [])
        gamma_net_share_arr = series_metrics.get("gamma_net_share", [])
        # Convert OI and volume arrays back to dicts keyed by strike for downstream use
        call_oi = {float(k): float(v) for k, v in zip(strikes, call_oi_arr)}
        put_oi  = {float(k): float(v) for k, v in zip(strikes, put_oi_arr)}
        call_vol= {float(k): float(v) for k, v in zip(strikes, call_vol_arr)}
        put_vol = {float(k): float(v) for k, v in zip(strikes, put_vol_arr)}
        # Obtain IV dictionaries via aggregate_series to avoid computing twice
        strikes_tmp, call_oi_tmp, put_oi_tmp, call_vol_tmp, put_vol_tmp, iv_call, iv_put = aggregate_series(block)
        # Time to expiry in years
        T = max((e - t0) / (365*24*3600), 1e-6)
        # Append context for this expiry.  Note: gamma_abs_share and gamma_net_share
        # are kept as arrays (aligned with strikes) rather than converting to dict.
        all_series_ctx.append({
            "strikes": strikes,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "call_vol": call_vol,
            "put_vol": put_vol,
            "gamma_abs_share": gamma_abs_share_arr,
            "gamma_net_share": gamma_net_share_arr,
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
            # placeholders for new profiles; will be computed separately
            "power_zone": _np.array([], dtype=float),
            "er_up": _np.array([], dtype=float),
            "er_down": _np.array([], dtype=float),
        }
    # Инициализация аккумулирующих массивов
    acc = {
        "put_oi": _np.zeros_like(all_strikes, dtype=float),
        "call_oi": _np.zeros_like(all_strikes, dtype=float),
        "put_vol": _np.zeros_like(all_strikes, dtype=float),
        "call_vol": _np.zeros_like(all_strikes, dtype=float),
        "net_gex": _np.zeros_like(all_strikes, dtype=float),
        "ag": _np.zeros_like(all_strikes, dtype=float),
        # initialize new profiles with zeros; they will be overridden after aggregation
        "power_zone": _np.zeros_like(all_strikes, dtype=float),
        "er_up": _np.zeros_like(all_strikes, dtype=float),
        "er_down": _np.zeros_like(all_strikes, dtype=float),
    }
    # Быстрая индексация по страйку
    idx_map = {float(k): i for i, k in enumerate(all_strikes.tolist())}
    for m in metrics_list:
        s = _np.asarray(m.get("strikes", []), dtype=float)
        for key in ["put_oi","call_oi","put_vol","call_vol","net_gex","ag","power_zone","er_up","er_down"]:
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
# --- Compute new Power Zone and Easy Reach metrics across aggregated strikes ---
try:
    # Use compute_power_zone_and_er from lib.compute on aggregated strike grid
    from lib.compute import compute_power_zone_and_er
    pz_new, er_up, er_down = compute_power_zone_and_er(
        S=float(S),
        strikes_eval=metrics.get("strikes", []),
        all_series_ctx=all_series_ctx,
        day_high=day_high,
        day_low=day_low
    )
    # Assign into metrics structure
    if len(pz_new) == len(metrics.get("strikes", [])):
        metrics["power_zone"] = pz_new
    else:
        metrics["power_zone"] = np.zeros_like(metrics.get("strikes", []), dtype=float)
    if len(er_up) == len(metrics.get("strikes", [])):
        metrics["er_up"] = er_up
    else:
        metrics["er_up"] = np.zeros_like(metrics.get("strikes", []), dtype=float)
    if len(er_down) == len(metrics.get("strikes", [])):
        metrics["er_down"] = er_down
    else:
        metrics["er_down"] = np.zeros_like(metrics.get("strikes", []), dtype=float)
except Exception as _e:
    # On failure set to zeros
    metrics["power_zone"] = np.zeros_like(metrics.get("strikes", []), dtype=float)
    metrics["er_up"] = np.zeros_like(metrics.get("strikes", []), dtype=float)
    metrics["er_down"] = np.zeros_like(metrics.get("strikes", []), dtype=float)
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
    "Strike": metrics.get("strikes", []),
    "Put OI": metrics.get("put_oi", []),
    "Call OI": metrics.get("call_oi", []),
    "Put Volume": metrics.get("put_vol", []),
    "Call Volume": metrics.get("call_vol", []),
    "Net Gex": metrics.get("net_gex", []),
    "AG": metrics.get("ag", []),
    "Power Zone": np.round(metrics.get("power_zone", np.zeros_like(metrics.get("strikes", []))), 6),
    "ER Up": np.round(metrics.get("er_up", np.zeros_like(metrics.get("strikes", []))), 6),
    "ER Down": np.round(metrics.get("er_down", np.zeros_like(metrics.get("strikes", []))), 6),
})

# Remove legacy PZ metrics; compute_power_zone_v2 no longer used

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
cols = st.columns(10)
toggles = {}
# Define toggle names for each series to be displayed. Legacy PZ metrics are removed and replaced by the new Power Zone and Easy Reach metrics.
names = [
    "Net Gex", "Put OI", "Call OI", "Put Volume", "Call Volume",
    "AG", "Power Zone", "ER Up", "ER Down", "G-Flip"
]
defaults = {
    "Net Gex": True,
    "Put OI": False,
    "Call OI": False,
    "Put Volume": False,
    "Call Volume": False,
    "AG": False,
    "Power Zone": False,
    "ER Up": False,
    "ER Down": False,
    "G-Flip": False
}
for i, name in enumerate(names):
    with cols[i]:
        toggles[name] = st.toggle(name, value=defaults.get(name, False), key=f"tgl_{name}")

# Build the series dictionary mapping names to data arrays for plotting.
# It includes the new Power Zone and Easy Reach metrics instead of legacy PZ variants.
series_dict = {
    "Net Gex": df["Net Gex"].values,
    "Put OI": df["Put OI"].values,
    "Call OI": df["Call OI"].values,
    "Put Volume": df["Put Volume"].values,
    "Call Volume": df["Call Volume"].values,
    "AG": df["AG"].values,
    "Power Zone": df.get("Power Zone", pd.Series(dtype=float)).values,
    "ER Up": df.get("ER Up", pd.Series(dtype=float)).values,
    "ER Down": df.get("ER Down", pd.Series(dtype=float)).values,
}

# позиционный вызов — совместим с текущей сигнатурой
idx_keep = _select_atm_window(df["Strike"].values, df["Call OI"].values, df["Put OI"].values, S)

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
    # AG secondary/tertiary peaks
    try:
        _ag_vals = np.asarray(df["AG"].values, dtype=float)[idx_keep]
        if np.any(np.isfinite(_ag_vals)):
            order = np.argsort(_ag_vals)[::-1]  # desc
            uniq = []
            for j in order:
                k = int(j)
                x = float(_strike[idx_keep[k]])
                if x not in uniq:
                    uniq.append(x)
                if len(uniq) >= 3:
                    break
            if len(uniq) >= 2:
                _max_levels["ag_max_2"] = float(uniq[1])
            if len(uniq) >= 3:
                _max_levels["ag_max_3"] = float(uniq[2])
    except Exception:
        pass
        # Locate the maximum of the Power Zone profile among kept strikes
        try:
            i_pz = _nan_argmax(df["Power Zone"].values[idx_keep])
        except Exception:
            i_pz = None
        if i_pz is not None:
            _max_levels["power_zone_max"] = float(_strike[idx_keep[i_pz]])

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
    # Secondary extremes (top-3 by sign)
    try:
        order_desc = np.argsort(gex_vals)[::-1]
        order_asc  = np.argsort(gex_vals)
        uniq_pos, uniq_neg = [], []
        for j in order_desc:
            x = float(_strike[idx_keep[int(j)]])
            val = gex_vals[int(j)]
            if np.isfinite(val) and val > 0 and x not in uniq_pos:
                uniq_pos.append(x)
            if len(uniq_pos) >= 3:
                break
        for j in order_asc:
            x = float(_strike[idx_keep[int(j)]])
            val = gex_vals[int(j)]
            if np.isfinite(val) and val < 0 and x not in uniq_neg:
                uniq_neg.append(x)
            if len(uniq_neg) >= 3:
                break
        if len(uniq_pos) >= 2:
            _max_levels["max_pos_gex_2"] = float(uniq_pos[1])
        if len(uniq_pos) >= 3:
            _max_levels["max_pos_gex_3"] = float(uniq_pos[2])
        if len(uniq_neg) >= 2:
            _max_levels["max_neg_gex_2"] = float(uniq_neg[1])
        if len(uniq_neg) >= 3:
            _max_levels["max_neg_gex_3"] = float(uniq_neg[2])
    except Exception:
        pass

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

# === Финальная таблица (окно, NetGEX/AG, PZ/ER) ===
try:
    render_final_table()
except Exception as _e:
    try:
        import streamlit as st
        st.warning(f"Final table section error: {_e}")
    except Exception:
        pass

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



# --- TIKER DATA BLOCK (independent source) ---
import importlib
import streamlit as st
def _lazy_imports():
    td_block = None
    sanitize_entry = None
    SanitizerCfg = None
    # tiker_data block
    try:
        td_mod = importlib.import_module("lib.tiker_data")
        td_block = getattr(td_mod, "render_tiker_data_block", None)
    except Exception:
        pass
    # sanitize entry
    try:
        sw_mod = importlib.import_module("lib.sanitize_window")
        sanitize_entry = getattr(sw_mod, "sanitize_from_tiker_data", None)
        SanitizerCfg = getattr(sw_mod, "SanitizerConfig", None)
    except Exception:
        pass
    return td_block, sanitize_entry, SanitizerCfg

td_block, sanitize_entry, SanitizerCfg = _lazy_imports()
if td_block is not None and sanitize_entry is not None and SanitizerCfg is not None:
    with st.container():

        st.subheader("Тикер/Экспирации (источник сырых данных: tiker_data.py)")
        td = td_block("Тикер/Экспирации (сырьё + свечи)")
        if getattr(td, "selected", None):
            spot, all_exps, bundles = sanitize_entry(td.ticker, selected_exps=td.selected, cfg=SanitizerCfg())
            # Persist for downstream modules
            st.session_state["spot"] = spot
            exp0 = td.selected[0]
            bun = bundles.get(exp0)
            if bun:
                st.session_state["df_corr"] = bun.df_corr
                # windows expected as dict[exp] -> [K]
                win = bun.windows if isinstance(bun.windows, dict) else {}
                st.session_state["windows"] = {exp0: win.get(exp0, [])}
                st.session_state["_last_ticker"] = td.ticker
                st.session_state["_last_exp_sig"] = exp0
                # OHLC for Key Levels (optional use by intraday_chart)
                if getattr(td, "ohlc", None) is not None:
                    st.session_state["ohlc_df"] = td.ohlc
                    st.session_state["day_high"] = td.day_high
                    st.session_state["day_low"] = td.day_low
