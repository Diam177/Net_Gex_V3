
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Tuple

import streamlit as st

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
st.sidebar.markdown("### Действия")
st.markdown("## Главная · Выбор тикера/экспирации и просмотр df_raw")

api_key = _get_api_key()
if not api_key:
    st.error("POLYGON_API_KEY не задан в Streamlit Secrets или переменных окружения.")
    st.stop()

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    ticker = st.text_input("Тикер", value=st.session_state.get("ticker", "SPY")).strip().upper()
    st.session_state["ticker"] = ticker

with col2:
    # Получаем список будущих экспираций под выбранный тикер
    expirations: list[str] = st.session_state.get(f"expirations:{ticker}", [])
    if not expirations and ticker:
        try:
            expirations = list_future_expirations(ticker, api_key)
            st.session_state[f"expirations:{ticker}"] = expirations
        except Exception as e:
            st.error(f"Не удалось получить даты экспираций: {e}")
            expirations = []

    if expirations:
        # по умолчанию ближайшая дата — первая в списке
        default_idx = 0
        sel = st.selectbox("Дата экспирации", options=expirations, index=default_idx, key=f"exp_sel:{ticker}")
        expiration = sel

    else:
        expiration = ""
        st.warning("Нет доступных дат экспираций для тикера.")



# --- Режим экспираций: Single / Multi ---
mode_exp = st.radio("Режим экспираций", ["Single","Multi"], index=0, horizontal=True)
selected_exps = []
weight_mode = "равные"
if mode_exp == "Single":
    selected_exps = [expiration] if expiration else []
else:
    selected_exps = st.multiselect("Выберите экспирации", options=expirations, default=(expirations[:2] if expirations else []))
    weight_mode = st.selectbox("Взвешивание", ["равные","1/T","1/√T"], index=2)

with col3:
    spot_manual = st.text_input("Spot (необязательно)", value=st.session_state.get("spot_manual",""))
    st.session_state["spot_manual"] = spot_manual

st.divider()

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
            label="Скачать сырой JSON (Polygon)",
            data=raw_bytes,
            file_name=fname,
            mime="application/json",
            use_container_width=True,
        )
    except Exception as e:
        st.sidebar.error(f"Не удалось подготовить JSON к скачиванию: {e}")
else:
    st.sidebar.info("Выберите тикер и дату экспирации, чтобы загрузить snapshot и скачать JSON.")

# --- Spot price ---------------------------------------------------------------
S: float | None = None
# 1) ручной ввод
if spot_manual:
    try:
        S = float(spot_manual)
    except Exception:
        st.warning("Spot введён некорректно. Будет использована автоматическая оценка.")
# 2) из Polygon (если нет ручного)
if S is None and ticker:
    try:
        S, ts_ms, src = get_spot_price(ticker, api_key)
        st.caption(f"Spot {S} (источник: {src})")
    except Exception:
        S = None
# 3) из snapshot (fallback)
if S is None and raw_records:
    S = _infer_spot_from_snapshot(raw_records)
    if S:
        st.caption(f"Spot ~{S:.2f} (оценка по ATM-страйкам из snapshot)")

# --- Run sanitize/window + show df_raw ---------------------------------------
if raw_records:
    try:
        res = sanitize_and_window_pipeline(raw_records, S=S)
        df_raw = res.get("df_raw")
        if df_raw is None or getattr(df_raw, "empty", True):
            st.warning("df_raw пуст. Проверьте формат данных.")
        else:
            
            # --- Скачивание промежуточных таблиц ---
            try:
                import io, zipfile, pandas as pd, numpy as np
                def _csv_bytes(df: "pd.DataFrame") -> bytes:
                    buf = io.StringIO()
                    df.to_csv(buf, index=False)
                    return buf.getvalue().encode("utf-8")

                if mode_exp == "Single":
                    st.markdown("**Скачать промежуточные таблицы (Single):**")
                    df_marked = res.get("df_marked")
                    df_corr = res.get("df_corr")
                    df_weights = res.get("df_weights")
                    windows = res.get("windows")
                    window_raw = res.get("window_raw")
                    window_corr = res.get("window_corr")

                    df_windows = None
                    if windows and isinstance(windows, dict) and (df_weights is not None):
                        try:
                            rows = []
                            for exp_key, idxs in windows.items():
                                g = df_weights[df_weights["exp"] == exp_key].sort_values("K").reset_index(drop=True)
                                for i in list(idxs):
                                    i = int(i)
                                    if 0 <= i < len(g):
                                        rows.append({"exp": exp_key, "row_index": i, "K": float(g.loc[i, "K"]),
                                                     "w_blend": float(g.loc[i, "w_blend"]) if "w_blend" in g.columns else np.nan})
                            if rows:
                                df_windows = pd.DataFrame(rows, columns=["exp","row_index","K","w_blend"]).sort_values(["exp","K"])
                        except Exception:
                            df_windows = None

                    if df_raw is not None and not getattr(df_raw, "empty", True):
                        st.download_button("Скачать df_raw.csv", _csv_bytes(df_raw), file_name=f"{ticker}_{expiration}_df_raw.csv", mime="text/csv")
                    if df_marked is not None and not getattr(df_marked, "empty", True):
                        st.download_button("Скачать df_marked.csv", _csv_bytes(df_marked), file_name=f"{ticker}_{expiration}_df_marked.csv", mime="text/csv")
                    if df_corr is not None and not getattr(df_corr, "empty", True):
                        st.download_button("Скачать df_corr.csv", _csv_bytes(df_corr), file_name=f"{ticker}_{expiration}_df_corr.csv", mime="text/csv")
                    if df_weights is not None and not getattr(df_weights, "empty", True):
                        st.download_button("Скачать df_weights.csv", _csv_bytes(df_weights), file_name=f"{ticker}_{expiration}_df_weights.csv", mime="text/csv")
                    if df_windows is not None and not getattr(df_windows, "empty", True):
                        st.download_button("Скачать windows.csv", _csv_bytes(df_windows), file_name=f"{ticker}_{expiration}_windows.csv", mime="text/csv")
                    if window_raw is not None and not getattr(window_raw, "empty", True):
                        st.download_button("Скачать window_raw.csv", _csv_bytes(window_raw), file_name=f"{ticker}_{expiration}_window_raw.csv", mime="text/csv")
                    if window_corr is not None and not getattr(window_corr, "empty", True):
                        st.download_button("Скачать window_corr.csv", _csv_bytes(window_corr), file_name=f"{ticker}_{expiration}_window_corr.csv", mime="text/csv")

                else:
                    if selected_exps:
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                            for _exp in selected_exps:
                                try:
                                    js_e = download_snapshot_json(ticker, _exp, api_key)
                                    recs_e = _coerce_results(js_e)
                                    res_e = sanitize_and_window_pipeline(recs_e, S=S)
                                    def _wcsv(name, df):
                                        if df is None or getattr(df, "empty", True):
                                            return
                                        s = io.StringIO(); df.to_csv(s, index=False)
                                        zf.writestr(f"{_exp}/{name}.csv", s.getvalue())
                                    _wcsv("df_raw", res_e.get("df_raw"))
                                    _wcsv("df_marked", res_e.get("df_marked"))
                                    _wcsv("df_corr", res_e.get("df_corr"))
                                    _wcsv("df_weights", res_e.get("df_weights"))
                                    win_e = res_e.get("windows") or {}
                                    dfe_w = res_e.get("df_weights")
                                    dfw = None
                                    if win_e and (dfe_w is not None):
                                        rows = []
                                        for ek, idxs in win_e.items():
                                            g = dfe_w[dfe_w["exp"] == ek].sort_values("K").reset_index(drop=True)
                                            for i in list(idxs):
                                                i = int(i)
                                                if 0 <= i < len(g):
                                                    rows.append({"exp": ek, "row_index": i, "K": float(g.loc[i, "K"]),
                                                                 "w_blend": float(g.loc[i, "w_blend"]) if "w_blend" in g.columns else np.nan})
                                        if rows:
                                            dfw = pd.DataFrame(rows, columns=["exp","row_index","K","w_blend"]).sort_values(["exp","K"])
                                    _wcsv("windows", dfw)
                                    _wcsv("window_raw", res_e.get("window_raw"))
                                    _wcsv("window_corr", res_e.get("window_corr"))
                                except Exception:
                                    pass
                        st.download_button(
                            "Скачать ZIP (все exp)",
                            data=zip_buf.getvalue(),
                            file_name=f"{ticker}_multi_intermediate_tables.zip",
                            mime="application/zip"
                        )
            except Exception as _edl:
                st.warning("Не удалось подготовить файлы для скачивания.")
                st.exception(_edl)
            st.dataframe(df_raw, use_container_width=True)
    except Exception as e:
        st.error("Ошибка пайплайна sanitize/window.")
        st.exception(e)
else:
    st.info("Задайте тикер и экспирацию, чтобы загрузить snapshot и посмотреть df_raw.")
