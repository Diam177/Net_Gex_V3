# -*- coding: utf-8 -*-
"""
final_table_chart.py — самодостаточный модуль построения интерактивного чарта
на основе ФИНАЛЬНОЙ таблицы проекта. НЕ требует plotting.py.

Визуализация:
- Бар-серия Net GEX по страйкам (левая ось Y)
- Линейные серии (правая ось Y): Put/Call OI, Put/Call Volume, AG, Power Zone, ER Up/Down
- Вертикальная линия текущей цены (если доступна)
- Вертикальная линия G‑Flip (если передан параметр)
- Тумблеры серий формируются на стороне вызывающего кода (UI)

API:
    figure_from_final_df(df, price=None, ticker=None, g_flip=None, enabled=None) -> plotly.graph_objects.Figure

Ожидаемые колонки df:
    K (страйк) — обязательно
    NetGEX_1pct_M или NetGEX_1pct — обязательно
    put_oi, call_oi, put_vol, call_vol — опционально
    AG_1pct_M или AG_1pct — опционально
    PZ, ER_Up, ER_Down — опционально
    S — опционально (цена спот; при отсутствии можно передать price)
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------- helpers ----------

def _detect_price(df: pd.DataFrame, price: Optional[float]) -> Optional[float]:
    if price is not None and np.isfinite(price):
        return float(price)
    if "S" in df.columns:
        s_vals = pd.to_numeric(df["S"], errors="coerce").dropna()
        if not s_vals.empty:
            return float(s_vals.iloc[0])
    try:
        import streamlit as st  # локально; не требуем streamlit как зависимость
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


def _extract_series_from_df(df: pd.DataFrame):
    if "K" not in df.columns:
        raise ValueError("В финальной таблице отсутствует колонка 'K' (страйк).")
    strikes = pd.to_numeric(df["K"], errors="coerce").to_numpy(dtype=float)

    if "NetGEX_1pct_M" in df.columns:
        net_gex = pd.to_numeric(df["NetGEX_1pct_M"], errors="coerce").to_numpy(dtype=float)
    elif "NetGEX_1pct" in df.columns:
        net_gex = pd.to_numeric(df["NetGEX_1pct"], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError("Не найдена колонка NetGEX_1pct(_M) в финальной таблице.")

    series: Dict[str, np.ndarray] = {}

    if "put_oi" in df.columns:
        series["Put OI"] = pd.to_numeric(df["put_oi"], errors="coerce").to_numpy(dtype=float)
    if "call_oi" in df.columns:
        series["Call OI"] = pd.to_numeric(df["call_oi"], errors="coerce").to_numpy(dtype=float)
    if "put_vol" in df.columns:
        series["Put Volume"] = pd.to_numeric(df["put_vol"], errors="coerce").to_numpy(dtype=float)
    if "call_vol" in df.columns:
        series["Call Volume"] = pd.to_numeric(df["call_vol"], errors="coerce").to_numpy(dtype=float)

    if "AG_1pct_M" in df.columns:
        series["AG"] = pd.to_numeric(df["AG_1pct_M"], errors="coerce").to_numpy(dtype=float)
    elif "AG_1pct" in df.columns:
        series["AG"] = pd.to_numeric(df["AG_1pct"], errors="coerce").to_numpy(dtype=float)

    if "PZ" in df.columns:
        series["Power Zone"] = pd.to_numeric(df["PZ"], errors="coerce").to_numpy(dtype=float)
    if "ER_Up" in df.columns:
        series["ER Up"] = pd.to_numeric(df["ER_Up"], errors="coerce").to_numpy(dtype=float)
    if "ER_Down" in df.columns:
        series["ER Down"] = pd.to_numeric(df["ER_Down"], errors="coerce").to_numpy(dtype=float)

    return strikes, net_gex, series


def _build_figure(
    strikes: np.ndarray,
    net_gex: np.ndarray,
    series_enabled: Dict[str, bool],
    series_dict: Dict[str, np.ndarray],
    price: Optional[float],
    ticker: Optional[str],
    g_flip: Optional[float],
) -> go.Figure:
    fig = go.Figure()

    # Бар-серия для Net GEX (левая ось)
    fig.add_trace(go.Bar(
        x=strikes, y=net_gex, name="Net GEX",
        hovertemplate="K=%{x}<br>Net GEX=%{y:.0f}<extra></extra>",
        opacity=0.9,
    ))

    # Правую ось используем под линии
    for label, arr in series_dict.items():
        if not series_enabled.get(label, True):
            continue
        fig.add_trace(go.Scatter(
            x=strikes, y=arr, name=label, mode="lines",
            hovertemplate="K=%{x}<br>" + label + "=%{y:.0f}<extra></extra>",
            yaxis="y2",
        ))

    # Вертикальные линии цены и G‑Flip
    shapes = []
    annotations = []

    if price is not None and np.isfinite(price):
        shapes.append(dict(type="line", x0=price, x1=price, y0=0, y1=1,
                           xref="x", yref="paper", line=dict(width=2, dash="dash")))
        annotations.append(dict(
            x=price, y=1.02, xref="x", yref="paper",
            text=f"Price: {price:.2f}",
            showarrow=False
        ))

    if g_flip is not None and np.isfinite(g_flip):
        shapes.append(dict(type="line", x0=g_flip, x1=g_flip, y0=0, y1=1,
                           xref="x", yref="paper", line=dict(width=2, dash="dot")))
        annotations.append(dict(
            x=g_flip, y=1.02, xref="x", yref="paper",
            text=f"G-Flip: {g_flip:.0f}",
            showarrow=False
        ))

    title = "Финальная таблица · Профиль по страйкам"
    if ticker:
        title = f"{ticker} · " + title

    fig.update_layout(
        title=title,
        barmode="relative",
        xaxis=dict(title="Strike (K)", tickformat=""),
        yaxis=dict(title="Net GEX", rangemode="tozero"),
        yaxis2=dict(title="OI / Volume / AG / PZ / ER", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=60, b=60),
        shapes=shapes,
        annotations=annotations,
    )
    return fig


# ---------- public API ----------

def figure_from_final_df(
    df: pd.DataFrame,
    price: Optional[float] = None,
    ticker: Optional[str] = None,
    g_flip: Optional[float] = None,
    enabled: Optional[Dict[str, bool]] = None,
) -> go.Figure:
    strikes, net_gex, series_dict = _extract_series_from_df(df)

    if enabled is None:
        series_enabled = {k: True for k in series_dict.keys()}
    else:
        series_enabled = {k: bool(v) for k, v in enabled.items() if k in series_dict}

    use_price = _detect_price(df, price)
    use_ticker = _detect_ticker(ticker)

    fig = _build_figure(
        strikes=strikes,
        net_gex=net_gex,
        series_enabled=series_enabled,
        series_dict=series_dict,
        price=use_price,
        ticker=use_ticker,
        g_flip=g_flip,
    )
    return fig
