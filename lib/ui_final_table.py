
# -*- coding: utf-8 -*-
"""
ui_final_table.py — упрощённый UI-блок:
Показывает финальную таблицу (окно страйков + NetGEX/AG + PZ/ER) и даёт кнопки скачивания.
Источник данных выбирается автоматически:
  1) df_corr + windows из st.session_state (если есть)
  2) raw_records + spot из st.session_state (если есть)
Если данных нет — выводит лаконичное уведомление.
"""

from __future__ import annotations

import io
from typing import Dict, Any

import pandas as pd
import streamlit as st

from .final_table import (
    FinalTableConfig,
    process_from_raw,
    build_final_tables_from_corr,
)
from .final_table_chart import figure_from_final_df


@st.cache_data(show_spinner=False)
def _compute_from_raw_cached(raw_records, S: float, final_cfg: FinalTableConfig):
    return process_from_raw(raw_records, S=S, final_cfg=final_cfg)


@st.cache_data(show_spinner=False)
def _build_from_corr_cached(df_corr: pd.DataFrame, windows: Dict[str, Any], final_cfg: FinalTableConfig):
    return build_final_tables_from_corr(df_corr, windows, cfg=final_cfg)


def render_final_table(section_title: str = "Финальная таблица (окно, NetGEX/AG, PZ/ER)") -> None:
    st.header(section_title)

    # Настройки масштаба для *_M
    scale_millions = st.number_input("Масштаб (млн $ на 1% движения)", value=1_000_000, min_value=1, step=100_000)
    final_cfg = FinalTableConfig(scale_millions=scale_millions)

    tables_by_exp = None

    # 1) Пытаемся использовать конвейер df_corr + windows (быстрее)
    if ("df_corr" in st.session_state) and ("windows" in st.session_state):
        try:
            tables_by_exp = _build_from_corr_cached(st.session_state["df_corr"], st.session_state["windows"], final_cfg)
        except Exception as e:
            st.warning(f"Не удалось собрать из df_corr/windows: {e}")

    # 2) Фолбэк: из raw_records + spot
    if not tables_by_exp and ("raw_records" in st.session_state) and ("spot" in st.session_state):
        try:
            tables_by_exp = _compute_from_raw_cached(st.session_state["raw_records"], float(st.session_state["spot"]), final_cfg)
        except Exception as e:
            st.warning(f"Не удалось собрать из raw_records: {e}")

    if not tables_by_exp:
        st.info("Нет данных для финальной таблицы. Выберите экспирацию/получите сырые данные.")
        return

    exps = list(tables_by_exp.keys())
    if not exps:
        st.info("Нет доступных экспираций для отображения.")
        return

    exp = st.selectbox("Экспирация", exps, index=0)
    df_show = tables_by_exp.get(exp)
    if df_show is None or df_show.empty:
        st.info("Таблица пуста.")
        return

    # Показ таблицы и кнопки скачать
    st.dataframe(df_show, use_container_width=True, hide_index=True)

# --- Чарт «профиль по страйкам» из финальной таблицы ---
try:
    # Только те серии, которые реально присутствуют
    present = {
        "Put OI": "put_oi" in df_show.columns,
        "Call OI": "call_oi" in df_show.columns,
        "Put Volume": "put_vol" in df_show.columns,
        "Call Volume": "call_vol" in df_show.columns,
        "AG": ("AG_1pct_M" in df_show.columns) or ("AG_1pct" in df_show.columns),
        "Power Zone": "PZ" in df_show.columns,
        "ER Up": "ER_Up" in df_show.columns,
        "ER Down": "ER_Down" in df_show.columns,
    }
    cols = st.columns(4)
    enabled = {}
    i = 0
    for label, exists in present.items():
        if not exists:
            continue
        with cols[i % 4]:
            enabled[label] = st.checkbox(label, value=True, key=f"ftc_{label.replace(' ','_')}")
        i += 1

    fig = figure_from_final_df(df_show, enabled=enabled)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Не удалось построить чарт по финальной таблице: {e}")


    # CSV (включая call_vol/put_vol, если присутствуют в df_show)
    csv_bytes = df_show.to_csv(index=False).encode("utf-8")
    st.download_button("Скачать CSV", data=csv_bytes, file_name=f"final_table_{exp}.csv", mime="text/csv")

    # Parquet (если pyarrow установлен)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        buf = io.BytesIO()
        table = pa.Table.from_pandas(df_show)
        pq.write_table(table, buf)
        st.download_button("Скачать Parquet", data=buf.getvalue(), file_name=f"final_table_{exp}.parquet", mime="application/octet-stream")
    except Exception:
        pass
