
# -*- coding: utf-8 -*-
"""
ui_final_table.py — Streamlit UI блок для показа финальной таблицы (окно страйков + NetGEX/AG + PZ/ER)
Минимальное вмешательство в app.py: достаточно импортировать и вызвать render_final_table().
"""

from __future__ import annotations

import io
import json
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

# При условии, что файлы лежат в папке lib/
from .final_table import (
    FinalTableConfig,
    process_from_raw,
    build_final_tables_from_corr,
)


@st.cache_data(show_spinner=False)
def _compute_from_raw_cached(raw_records, S: float, final_cfg: FinalTableConfig):
    return process_from_raw(raw_records, S=S, final_cfg=final_cfg)


@st.cache_data(show_spinner=False)
def _build_from_corr_cached(df_corr: pd.DataFrame, windows: Dict[str, Any], final_cfg: FinalTableConfig):
    return build_final_tables_from_corr(df_corr, windows, cfg=final_cfg)


def render_final_table(section_title: str = "Финальная таблица (окно, NetGEX/AG, PZ/ER)") -> None:
    """
    Рендерит блок на главной странице:
      - выбираем источник данных (из session_state df_corr/windows или из raw_records)
      - считаем финальные таблицы по экспирациям
      - даём выбор экспирации, показываем таблицу, кнопки для скачивания (CSV/Parquet)
    """
    st.header(section_title)

    # Настройки
    scale_millions = st.number_input("Масштаб (млн $ на 1% движения)", value=1_000_000, min_value=1, step=100_000)
    final_cfg = FinalTableConfig(scale_millions=scale_millions)

    # Попытка использовать данные из session_state проекта, если они уже где-то были посчитаны:
    have_corr = ("df_corr" in st.session_state) and ("windows" in st.session_state)
    have_raw  = ("raw_records" in st.session_state) and ("spot" in st.session_state)

    source = st.radio(
        "Источник данных",
        options=["Из session_state (df_corr/windows)", "Из session_state (raw_records + S)", "Загрузить сырой JSON вручную"],
        index=0 if have_corr else (1 if have_raw else 2),
        horizontal=True,
    )

    tables_by_exp = None

    if source == "Из session_state (df_corr/windows)":
        if not have_corr:
            st.warning("В session_state нет df_corr/windows — загрузите данные или выберите другой источник.")
        else:
            df_corr = st.session_state["df_corr"]
            windows = st.session_state["windows"]
            try:
                tables_by_exp = _build_from_corr_cached(df_corr, windows, final_cfg)
            except Exception as e:
                st.error(f"Ошибка построения из df_corr/windows: {e}")

    elif source == "Из session_state (raw_records + S)":
        if not have_raw:
            st.warning("В session_state нет raw_records/spot — загрузите данные или выберите другой источник.")
        else:
            raw_records = st.session_state["raw_records"]
            S = float(st.session_state["spot"])
            try:
                tables_by_exp = _compute_from_raw_cached(raw_records, S, final_cfg)
            except Exception as e:
                st.error(f"Ошибка расчёта из raw_records: {e}")

    else:
        # Ручная загрузка сырых данных
        S = st.number_input("Spot (S)", value=0.0, min_value=0.0, step=0.1, format="%.2f")
        up = st.file_uploader("Загрузите сырой JSON (список записей опционной цепочки)", type=["json"], accept_multiple_files=False)
        if up is not None and S > 0:
            try:
                raw_records = json.load(up)
                if not isinstance(raw_records, list):
                    st.error("Ожидался JSON-массив записей.")
                else:
                    tables_by_exp = _compute_from_raw_cached(raw_records, S, final_cfg)
            except Exception as e:
                st.error(f"Не удалось прочитать JSON: {e}")

    if not tables_by_exp:
        st.stop()

    # Выбор экспирации и показ таблицы
    exps = list(tables_by_exp.keys())
    if not exps:
        st.info("Нет доступных экспираций для отображения.")
        st.stop()

    exp = st.selectbox("Экспирация", exps, index=0)
    df_show = tables_by_exp.get(exp)
    if df_show is None or df_show.empty:
        st.info("Таблица пуста.")
        st.stop()

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Кнопки скачать
    csv_bytes = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать CSV", data=csv_bytes, file_name=f"final_table_{exp}.csv", mime="text/csv")

    try:
        import pyarrow as pa  # optional
        import pyarrow.parquet as pq
        buf = io.BytesIO()
        table = pa.Table.from_pandas(df_show)
        pq.write_table(table, buf)
        st.download_button("Скачать Parquet", data=buf.getvalue(), file_name=f"final_table_{exp}.parquet", mime="application/octet-stream")
    except Exception:
        pass
