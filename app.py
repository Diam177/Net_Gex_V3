
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
            st.dataframe(df_raw, use_container_width=True)
            # --- Ниже показываем остальные массивы по степени создания ---
            # 1) df_marked (df_raw + флаги аномалий)
            df_marked = res.get("df_marked")
            if df_marked is not None and not getattr(df_marked, "empty", True):
                st.subheader("df_marked")
                st.dataframe(df_marked, use_container_width=True, hide_index=True)

            # 2) df_corr (восстановленные IV/Greeks)
            df_corr = res.get("df_corr")
            if df_corr is not None and not getattr(df_corr, "empty", True):
                st.subheader("df_corr")
                st.dataframe(df_corr, use_container_width=True, hide_index=True)

            # 3) df_weights (веса окна по страйку)
            df_weights = res.get("df_weights")
            if df_weights is not None and not getattr(df_weights, "empty", True):
                st.subheader("df_weights")
                st.dataframe(df_weights, use_container_width=True, hide_index=True)

            # 4) windows (выбранные страйки каждого окна) — преобразуем в таблицу
            windows = res.get("windows")
            if windows and isinstance(windows, dict) and df_weights is not None:
                try:
                    rows = []
                    for exp, idx in windows.items():
                        # df_weights на эту экспирацию, отсортируем по K как в select_windows
                        g = df_weights[df_weights["exp"] == exp].sort_values("K").reset_index(drop=True)
                        for i in list(idx):
                            if 0 <= int(i) < len(g):
                                rows.append({"exp": exp, "row_index": int(i), "K": float(g.loc[int(i), "K"]),
                                             "w_blend": float(g.loc[int(i), "w_blend"])})
                    import pandas as pd  # локальный импорт безопасен
                    df_windows = pd.DataFrame(rows, columns=["exp","row_index","K","w_blend"]).sort_values(["exp","K"])
                    st.subheader("windows (табличный вид)")
                    st.dataframe(df_windows, use_container_width=True, hide_index=True)
                except Exception as _e:
                    st.warning("Не удалось отобразить windows в табличном виде.")
                    st.exception(_e)

            # 5) window_raw (строки окна из исходных + флаги)
            window_raw = res.get("window_raw")
            if window_raw is not None and not getattr(window_raw, "empty", True):
                st.subheader("window_raw")
                st.dataframe(window_raw, use_container_width=True, hide_index=True)

            # 6) window_corr (строки окна из исправленных данных)
            window_corr = res.get("window_corr")
            if window_corr is not None and not getattr(window_corr, "empty", True):
                st.subheader("window_corr")
                st.dataframe(window_corr, use_container_width=True, hide_index=True)

            # 7) Финальная таблица (NetGEX/AG + PZ/ER) для выбранной экспирации
            try:

                from lib.final_table import build_final_tables_from_corr, FinalTableConfig
                if df_corr is not None and windows:
                    final_tables = build_final_tables_from_corr(df_corr, windows)
                    exps = list(final_tables.keys())
                    if exps:
                        exp_to_show = expiration if 'expiration' in locals() and expiration in final_tables else exps[0]
                        df_final = final_tables.get(exp_to_show)
                        if df_final is not None and not getattr(df_final, "empty", True):
                            st.subheader("Final Table")
                            st.dataframe(df_final, use_container_width=True, hide_index=True)
                        else:
                            st.info("Финальная таблица пуста для выбранной экспирации.")
            except Exception as _e:
                st.error("Ошибка построения финальной таблицы.")
                st.exception(_e)
                    # 7) Финальная таблица (по df_corr + windows)
            try:
                from lib.final_table import build_final_tables_from_corr, FinalTableConfig
                df_corr = res.get("df_corr")
                windows = res.get("windows")
                if df_corr is not None and windows is not None:
                    final_tables = build_final_tables_from_corr(df_corr, windows, cfg=FinalTableConfig())
                    for exp, df_final in final_tables.items():
                        if df_final is not None and not getattr(df_final, "empty", True):
                            st.subheader(f"Финальная таблица · {exp}")
                            st.dataframe(df_final, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error("Ошибка при построении финальной таблицы")
                st.exception(e)


    except Exception as e:
        st.error("Ошибка пайплайна sanitize/window.")
        st.exception(e)
else:
    st.info("Задайте тикер и экспирацию, чтобы загрузить snapshot и посмотреть df_raw.")
