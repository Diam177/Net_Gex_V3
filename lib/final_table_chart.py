
# -*- coding: utf-8 -*-
"""
final_table_chart.py — построение профиля по страйкам на основе ФИНАЛЬНОЙ таблицы проекта.

Что делает:
- Берёт DataFrame финальной таблицы (на одну экспирацию) с колонками:
  ['K','NetGEX_1pct' (или 'NetGEX_1pct_M'),
   'AG_1pct' (или 'AG_1pct_M'),
   'call_oi','put_oi','call_vol','put_vol','PZ','ER_Up','ER_Down', 'S'(опц.)]
- Формирует словари серий в формате, совместимом с plotting.make_figure()
- Строит Plotly‑фигуру идентичную внешнему проекту (использует присланный plotting.py)

Public API:
    figure_from_final_df(df, price=None, ticker=None, g_flip=None, enabled=None) -> go.Figure
        df       : pandas.DataFrame c колонками, перечисленными выше
        price    : текущая цена базового актива (если None — возьмём df['S'] или st.session_state['spot'])
        ticker   : строка тикера (если None — возьмём st.session_state['ticker'] при наличии)
        g_flip   : уровень gamma‑flip для вертикальной линии (опционально)
        enabled  : dict[str,bool] включение/выключение серий ('Put OI','Call OI','Put Volume','Call Volume','AG','Power Zone','ER Up','ER Down')

Зависимости:
- plotly
- numpy, pandas
- dateutil (требуется plotting.py)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Используем присланный файл plotting.py (лежит в корне проекта рядом с app.py)
try:
    from plotting import make_figure  # type: ignore
except Exception as e:
    raise ImportError("Не найден plotting.py с функцией make_figure. Поместите его в корень проекта.") from e


_SERIES_ORDER = [
    "Put OI", "Call OI", "Put Volume", "Call Volume",
    "AG", "Power Zone", "ER Up", "ER Down",
]


def _detect_price(df: pd.DataFrame, price: Optional[float]) -> Optional[float]:
    """Выбрать цену: приоритет — аргумент price, потом колонка S, потом st.session_state['spot']."""
    if price is not None and np.isfinite(price):
        return float(price)
    if "S" in df.columns:
        s_vals = pd.to_numeric(df["S"], errors="coerce").dropna()
        if not s_vals.empty:
            return float(s_vals.iloc[0])
    # Ленивая подтяжка из Streamlit session_state (если библиотека доступна)
    try:
        import streamlit as st  # локальный импорт
        v = st.session_state.get("spot")
        if v is not None:
            return float(v)
    except Exception:
        pass
    return None


def _detect_ticker(ticker: Optional[str]) -> Optional[str]:
    if ticker:
        return str(ticker)
    try:
        import streamlit as st
        t = st.session_state.get("ticker")
        if t:
            return str(t)
    except Exception:
        pass
    return None


def _extract_series_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Построить (strikes, net_gex, series_dict) из финальной таблицы."""
    # Базовые поля
    if "K" not in df.columns:
        raise ValueError("В финальной таблице отсутствует колонка 'K' (страйк).")
    strikes = pd.to_numeric(df["K"], errors="coerce").to_numpy(dtype=float)

    # Net GEX: предпочитаем '_M' если он есть, иначе обычный
    if "NetGEX_1pct_M" in df.columns:
        net_gex = pd.to_numeric(df["NetGEX_1pct_M"], errors="coerce").to_numpy(dtype=float)
    elif "NetGEX_1pct" in df.columns:
        net_gex = pd.to_numeric(df["NetGEX_1pct"], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError("Не найдена колонка NetGEX_1pct(_M) в финальной таблице.")

    # Дополнительные серии
    series: Dict[str, np.ndarray] = {}

    if "put_oi" in df.columns:
        series["Put OI"] = pd.to_numeric(df["put_oi"], errors="coerce").to_numpy(dtype=float)
    if "call_oi" in df.columns:
        series["Call OI"] = pd.to_numeric(df["call_oi"], errors="coerce").to_numpy(dtype=float)

    if "put_vol" in df.columns:
        series["Put Volume"] = pd.to_numeric(df["put_vol"], errors="coerce").to_numpy(dtype=float)
    if "call_vol" in df.columns:
        series["Call Volume"] = pd.to_numeric(df["call_vol"], errors="coerce").to_numpy(dtype=float)

    # AG
    if "AG_1pct_M" in df.columns:
        series["AG"] = pd.to_numeric(df["AG_1pct_M"], errors="coerce").to_numpy(dtype=float)
    elif "AG_1pct" in df.columns:
        series["AG"] = pd.to_numeric(df["AG_1pct"], errors="coerce").to_numpy(dtype=float)

    # PZ/ER
    if "PZ" in df.columns:
        series["Power Zone"] = pd.to_numeric(df["PZ"], errors="coerce").to_numpy(dtype=float)
    if "ER_Up" in df.columns:
        series["ER Up"] = pd.to_numeric(df["ER_Up"], errors="coerce").to_numpy(dtype=float)
    if "ER_Down" in df.columns:
        series["ER Down"] = pd.to_numeric(df["ER_Down"], errors="coerce").to_numpy(dtype=float)

    return strikes, net_gex, series


def figure_from_final_df(
    df: pd.DataFrame,
    price: Optional[float] = None,
    ticker: Optional[str] = None,
    g_flip: Optional[float] = None,
    enabled: Optional[Dict[str, bool]] = None,
) -> go.Figure:
    """
    Построить фигуру «один‑в‑один» как в plotting.py, но из нашей финальной таблицы.
    """
    strikes, net_gex, series_dict = _extract_series_from_df(df)

    # Опциональное включение/выключение серий
    if enabled is None:
        # По умолчанию включаем все присутствующие в таблице
        enabled = {k: True for k in series_dict.keys()}
    else:
        # Оставим только те ключи, которые реально существуют в таблице
        enabled = {k: bool(v) for k, v in enabled.items() if k in series_dict}

    use_price = _detect_price(df, price)
    use_ticker = _detect_ticker(ticker)

    # Собираем фигуру ровно тем же билдером
    fig = make_figure(
        strikes=strikes,
        net_gex=net_gex,
        series_enabled=enabled,
        series_dict=series_dict,
        price=use_price,
        ticker=use_ticker,
        g_flip=g_flip,
    )
    return fig
