
# -*- coding: utf-8 -*-
"""
ui_final_table.py — UI для финальной таблицы.

Особенности:
- НЕ дублирует выбор экспирации, если он уже сделан в блоке tiker_data (флаг exp_locked_by_tiker_data).
- Источник данных:
    (A) df_corr + windows из st.session_state, если есть
    (B) иначе raw_records + spot из st.session_state
- Сами расчёты выполняют функции из lib.final_table (если доступны).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import io
import streamlit as st

# Попробуем подхватить pandas (для красивого отображения); если нет — обойдёмся
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _import_final_table_module():
    """
    Ленивая загрузка вычислительных функций. Ничего не импортируем на уровне модуля,
    чтобы файл был безопасен при отсутствующих зависимостях.
    """
    import importlib
    ft = importlib.import_module("lib.final_table")
    FinalTableConfig = getattr(ft, "FinalTableConfig", None)
    build_from_corr = getattr(ft, "build_final_tables_from_corr", None)
    process_from_raw = getattr(ft, "process_from_raw", None)
    return FinalTableConfig, build_from_corr, process_from_raw


def _choose_exps_for_table() -> List[str]:
    """
    Возвращает список экспираций, которые следует отрисовать в таблице.
    Если tiker_data «заблокировал» выбор — берём его.
    Иначе пробуем извлечь из df_corr или raw_records.
    """
    if st.session_state.get("exp_locked_by_tiker_data"):
        exps = st.session_state.get("tiker_selected_exps") or []
        return list(exps)

    # извлечь из df_corr/windows
    df_corr = st.session_state.get("df_corr")
    if df_corr is not None and pd is not None:
        try:
            if "expiration" in df_corr.columns:
                exps = sorted(list(map(str, pd.unique(df_corr["expiration"]))))
                if exps:
                    return exps[:1]  # по умолчанию — первая (чтобы не дублировать логику сайдбара)
        except Exception:
            pass

    # извлечь из raw_records
    raw_records = st.session_state.get("raw_records") or []
    if isinstance(raw_records, list) and raw_records:
        try:
            exps = sorted({str(r.get("expiration")) for r in raw_records if r.get("expiration")})
            if exps:
                return exps[:1]
        except Exception:
            pass

    # если совсем ничего
    return []


def _download_buttons(df_show, exp: str):
    csv_bytes = None
    try:
        csv_bytes = df_show.to_csv(index=False).encode("utf-8")
    except Exception:
        pass

    if csv_bytes:
        st.download_button("Скачать CSV", data=csv_bytes, file_name=f"final_table_{exp}.csv", mime="text/csv")

    # Parquet — по возможности
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        buf = io.BytesIO()
        table = pa.Table.from_pandas(df_show)
        pq.write_table(table, buf)
        st.download_button("Скачать Parquet", data=buf.getvalue(), file_name=f"final_table_{exp}.parquet",
                           mime="application/octet-stream")
    except Exception:
        pass


def render_final_table(*args, section_title: str = "Финальная таблица") -> None:
    """
    Поддерживает вызов как render_final_table(st), так и render_final_table().
    Аргумент st игнорируем — модуль сам импортирует streamlit как st.
    """
    st.header(section_title)

    # Пояснение про «замок» экспираций из tiker_data
    if st.session_state.get("exp_locked_by_tiker_data"):
        chosen = st.session_state.get("tiker_selected_exps") or []
        st.caption("Экспирация задана в блоке Ticker/Expiration: " + (", ".join(chosen) if chosen else "—"))

    # Импорт вычислительных функций (если нет — покажем ошибку)
    try:
        FinalTableConfig, build_from_corr, process_from_raw = _import_final_table_module()
    except Exception as e:
        st.error(f"Не удалось импортировать lib.final_table: {e}")
        return

    # Конфигурация: используем конструктор по умолчанию, если доступен
    final_cfg = None
    try:
        if FinalTableConfig is not None:
            final_cfg = FinalTableConfig()  # без аргументов, чтобы не ломать совместимость
    except Exception:
        final_cfg = None

    # Выбор экспираций для отображения
    exps_to_show = _choose_exps_for_table()
    if not exps_to_show:
        st.info("Нет выбранной экспирации. Задайте её в блоке Ticker/Expiration.")
        return

    # Источники данных
    df_corr = st.session_state.get("df_corr")
    windows = st.session_state.get("windows")
    raw_records = st.session_state.get("raw_records")
    S = st.session_state.get("spot")

    for exp in exps_to_show:
        st.subheader(f"Экспирация: {exp}")
        df_show = None

        # Вариант A: из df_corr + windows
        if df_corr is not None and windows is not None and callable(build_from_corr):
            try:
                result = build_from_corr(df_corr, windows, cfg=final_cfg) if final_cfg is not None else build_from_corr(df_corr, windows)
                # Ожидаем, что результат — dict {exp: DataFrame} или DataFrame
                if isinstance(result, dict):
                    df_show = result.get(exp)
                else:
                    df_show = result
            except Exception as e:
                st.warning(f"Сборка таблицы из df_corr/windows не удалась: {e}")

        
        # Вариант B: из raw_records + S
        if df_show is None and raw_records is not None and S is not None and callable(process_from_raw):
            try:
                # Пытаемся несколько сигнатур (в разных версиях проекта имя параметра отличается)
                result = None
                try:
                    result = process_from_raw(raw_records, S=S, exp=exp, final_cfg=final_cfg) if final_cfg is not None else process_from_raw(raw_records, S=S, exp=exp)
                except TypeError:
                    try:
                        result = process_from_raw(raw_records, S=S, expiration=exp, final_cfg=final_cfg) if final_cfg is not None else process_from_raw(raw_records, S=S, expiration=exp)
                    except TypeError:
                        try:
                            result = process_from_raw(raw_records, S=S, final_cfg=final_cfg) if final_cfg is not None else process_from_raw(raw_records, S=S)
                        except TypeError:
                            try:
                                result = process_from_raw(raw_records, S, exp)  # positional fallback
                            except Exception:
                                result = None

                # Нормализуем результат
                if result is not None:
                    if isinstance(result, tuple) and len(result) >= 1:
                        result = result[0]
                    if isinstance(result, dict):
                        df_show = result.get(exp) or next(iter(result.values()), None)
                    else:
                        df_show = result
            except Exception as e:
                st.error(f"Ошибка расчёта таблицы из raw_records/S: {e}")
                return

        if df_show is None:
            st.info("Нет данных для построения таблицы.")
            continue


        # Отображение
        try:
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        except Exception:
            # если не DataFrame — просто покажем как есть
            st.write(df_show)

        _download_buttons(df_show, exp)
