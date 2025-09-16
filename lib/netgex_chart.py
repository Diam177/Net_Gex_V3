# -*- coding: utf-8 -*-
"""
netgex_chart.py — бар‑чарт Net GEX для главной страницы (Streamlit + Plotly).

Функция:
    render_netgex_bars(df_final, ticker, spot=None, toggle_key=None)

Поведение:
- Отрисовывает бары Net GEX (в млн $ на 1%) по страйкам без пропусков.
- Подписи на нижней оси — ВСЕ страйки из данных (в порядке возрастания), без «пустых» промежутков.
- Наведение курсора на столбец показывает таблицу вида:
    Strike, Call OI, Put OI, Call Volume, Put Volume, Net GEX (M$)
- Вертикальная линия «Price» ставится по текущей цене, с интерполяцией позиции между ближайшими страйками.
- Никаких побочных зависимостей, только pandas/numpy/plotly/streamlit.
"""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as _np
import pandas as _pd
import streamlit as st
import plotly.graph_objects as go

# Цвета (оставьте как в вашем проекте при необходимости)
COLOR_NEG = '#D9493A'    # отрицательные столбцы
COLOR_POS = '#60A5E7'    # положительные столбцы
COLOR_PRICE = '#E4A339'  # линия цены
BG_COLOR = '#111111'

def _inject_no_crosshair_css() -> None:
    # Глобально переопределяем курсор для Plotly-чартов в Streamlit
    st.markdown(
        """
        <style>
        /* Отключаем crosshair-курсор у Plotly в Streamlit */
        div[data-testid="stPlotlyChart"] .draglayer,
        div[data-testid="stPlotlyChart"] .hoverlayer,
        div[data-testid="stPlotlyChart"] .zoomlayer,
        div[data-testid="stPlotlyChart"] .nsewdrag,
        div[data-testid="stPlotlyChart"] .neswdrag,
        div[data-testid="stPlotlyChart"] .ewdrag,
        div[data-testid="stPlotlyChart"] .nsdrag {
            cursor: default !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _to_float_array(a: Sequence) -> _np.ndarray:
    return _np.array(_pd.to_numeric(a, errors='coerce'), dtype=float)

def render_netgex_bars(
    df_final: _pd.DataFrame,
    ticker: str,
    spot: Optional[float] = None,
    toggle_key: Optional[str] = None,
) -> None:
    """Рендерит бар‑чарт Net GEX. Требуется DataFrame с колонками:
       K, NetGEX_1pct_M (или NetGEX_1pct), (необязательно: S, call_oi, put_oi, call_vol, put_vol).
    """
    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для графика Net GEX.")
        return

    if "K" not in df_final.columns:
        st.warning("В финальной таблице отсутствует столбец 'K'.")
        return

    # Выбор метрики Net GEX в млн $ / 1%
    if "NetGEX_1pct_M" in df_final.columns:
        y_col = "NetGEX_1pct_M"
    elif "NetGEX_1pct" in df_final.columns:
        df_final = df_final.copy()
        df_final["NetGEX_1pct_M"] = _pd.to_numeric(df_final["NetGEX_1pct"], errors="coerce") / 1e6
        y_col = "NetGEX_1pct_M"
    else:
        st.warning("Нет столбцов NetGEX_1pct_M / NetGEX_1pct — нечего рисовать.")
        return

    # Цена БА
    if spot is None and "S" in df_final.columns and df_final["S"].notna().any():
        spot = float(_pd.to_numeric(df_final["S"], errors="coerce").dropna().iloc[0])

    # Тумблер
    show = st.toggle("Net GEX", value=True, key=(toggle_key or f"netgex_toggle_{ticker}"))
    if not show:
        return

    # Приведение типов и подготовка агрегатов
    base = df_final.copy()
    base["K"] = _pd.to_numeric(base["K"], errors="coerce")
    base[y_col] = _pd.to_numeric(base[y_col], errors="coerce")

    # Гарантируем наличие числовых колонок для ховера
    for c in ("call_oi", "put_oi", "call_vol", "put_vol"):
        if c in base.columns:
            base[c] = _pd.to_numeric(base[c], errors="coerce")
        else:
            base[c] = 0.0

    g = (base
         .dropna(subset=["K"])
         .groupby("K", as_index=False)
         .agg({y_col: "sum",
               "call_oi": "sum",
               "put_oi": "sum",
               "call_vol": "sum",
               "put_vol": "sum"})
         .sort_values("K")
         .reset_index(drop=True))

    Ks = g["K"].to_numpy(dtype=float)
    Ys = g[y_col].to_numpy(dtype=float)

    # Ось X — последовательные индексы без пропусков, подписи — реальные страйки
    x_idx = _np.arange(len(Ks), dtype=float)
    bar_width = 0.9
    colors = _np.where(Ys >= 0.0, COLOR_POS, COLOR_NEG)

    # customdata для ховера
    k_str = _np.array([str(int(k)) if float(k).is_integer() else f"{k:.2f}" for k in Ks], dtype=object)
    custom = _np.column_stack([
        k_str,
        g["call_oi"].to_numpy(dtype=float),
        g["put_oi"].to_numpy(dtype=float),
        g["call_vol"].to_numpy(dtype=float),
        g["put_vol"].to_numpy(dtype=float),
    ])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_idx,
        y=Ys,
        name="Net GEX (M$ / 1%)",
        marker_color=colors.tolist(),
        width=bar_width,
        customdata=custom,
        hovertemplate=(
            "<b>Strike</b>: %{customdata[0]}<br>"
            "<b>Call OI</b>: %{customdata[1]:,.0f}<br>"
            "<b>Put OI</b>: %{customdata[2]:,.0f}<br>"
            "<b>Call Vol</b>: %{customdata[3]:,.0f}<br>"
            "<b>Put Vol</b>: %{customdata[4]:,.0f}<br>"
            "<b>Net GEX</b>: %{y:.3f}M"
            "<extra></extra>"
        ),
    ))

    # Вертикальная линия текущей цены (интерполяция между ближайшими страйками)
    if spot is not None and _np.isfinite(spot) and len(Ks) > 0:
        try:
            if len(Ks) >= 2:
                j = int(_np.searchsorted(Ks, spot))
                if j <= 0:
                    x_price = 0.0
                elif j >= len(Ks):
                    x_price = float(len(Ks) - 1)
                else:
                    k0, k1 = Ks[j - 1], Ks[j]
                    frac = 0.0 if (k1 - k0) == 0 else (spot - k0) / (k1 - k0)
                    x_price = (j - 1) + float(_np.clip(frac, 0.0, 1.0))
            else:
                x_price = 0.0
        except Exception:
            x_price = 0.0

        y0 = min(0.0, float(_np.nanmin(Ys))) * 1.05
        y1 = max(0.0, float(_np.nanmax(Ys))) * 1.05
        fig.add_shape(type="line", x0=x_price, x1=x_price, y0=y0, y1=y1,
                      line=dict(color=COLOR_PRICE, width=2))
        fig.add_annotation(x=x_price, y=y1, text=f"Price: {spot:.2f}",
                           showarrow=False, yshift=8,
                           font=dict(color=COLOR_PRICE, size=12), xanchor="center")

    # Подписи страйков по индексам
    fig.update_layout(
        hoverlabel=dict(font=dict(size=10)),  # smaller hover font
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        dragmode=False,
        xaxis=dict(
            fixedrange=True,
            title=None,
            tickmode="array",
            tickvals=x_idx.tolist(),
            ticktext=k_str.tolist(),
            tickangle=0,
            tickfont=dict(size=10),
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            fixedrange=True,
            title="Net GEX",
            showgrid=False,
            zeroline=False,
        ),
    )

    # Автомасштаб
    fig.update_yaxes(autorange=True)
    fig.update_xaxes(autorange=True)

    # ВАЖНО: включаем hover (НЕ staticPlot)
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme='streamlit',
        config={'displayModeBar': False, 'scrollZoom': False, 'doubleClick': False}
    )
