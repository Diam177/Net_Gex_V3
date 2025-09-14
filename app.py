
# -*- coding: utf-8 -*-
"""
Append-only section to render df_raw on the main page.
This block is safe: it only adds UI on top of your current app.
If you already have app content, keep it and append this block to the end of your existing app.py.
"""

from typing import Any, Iterable, Dict, List
import json

try:
    import streamlit as st
    from lib.sanitize_window import sanitize_and_window_pipeline
except Exception:
    # If Streamlit or project modules aren't available during static checks,
    # simply skip the df_raw section.
    st = None


def _coerce_results(data: Any) -> List[Dict]:
    """
    Accepts many possible JSON layouts and returns a list[dict] of option records.
    - If data is a list -> return as-is
    - If data is a dict and has "results" -> return that
    - If data is a dict and has "options" -> return that
    Otherwise -> empty list
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "options", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def render_df_raw_section() -> None:
    if st is None:
        return

    st.markdown("### Сырые данные (df_raw)")
    st.caption("Источник — провайдерский JSON (Polygon snapshot или эквивалент).")

    # 1) Пытаемся взять 'сырье' из session_state (не мешает существующим потокам данных)
    raw = None
    for key in ("raw_records", "provider_raw", "snapshot_results", "options_raw", "raw"):
        if key in st.session_state and st.session_state[key]:
            raw = st.session_state[key]
            break

    # 2) Фолбэк: файл загрузки JSON (без замены вашей логики выбора тикера/экспирации)
    up = st.file_uploader("Загрузите JSON (если данных нет в состоянии)", type=["json"], key="dfraw_uploader")
    if raw is None and up is not None:
        try:
            raw = _coerce_results(json.load(up))
            st.session_state["raw_records"] = raw
        except Exception as e:
            st.error(f"Не удалось прочитать JSON: {e}")
            return

    if raw is None:
        st.info("Нет входных данных. Загрузите JSON или заполните st.session_state['raw_records'].")
        return

    # 3) Параметры спота (если есть в состоянии — используем; иначе не навязываем ввод)
    S = None
    if "spot" in st.session_state and isinstance(st.session_state["spot"], (int, float)):
        S = float(st.session_state["spot"])

    # 4) Запуск пайплайна sanitize → window
    try:
        res = sanitize_and_window_pipeline(raw, S=S)
    except Exception as e:
        st.error("Ошибка в sanitize_and_window_pipeline. Проверьте формат входных данных и значения S.")
        st.exception(e)
        return

    # 5) Вывод df_raw
    df_raw = res.get("df_raw")
    if df_raw is None or getattr(df_raw, "empty", True):
        st.warning("df_raw пуст. Проверьте, что JSON содержит корректные поля (details/greeks и т.д.).")
        return

    st.dataframe(df_raw, use_container_width=True)


# Вызов рендера. Этот вызов не мешает вашей логике и просто добавляет секцию на главной.
try:
    render_df_raw_section()
except Exception:
    # Никогда не падаем из-за декоративной секции
    pass
