
# -*- coding: utf-8 -*-
"""
netgex_chart.py — бар‑чарт Net GEX для главной страницы.

Функция render_netgex_bars(df_final, ticker, spot=None, toggle_key=None):
  • df_final: DataFrame по одной экспирации (или агрегированной multi‑финалке)
  • ticker: строка для подписи в левом верхнем углу
  • spot: текущая цена БА; если None — берётся из df_final['S']
  • toggle_key: уникальный ключ для st.toggle

Зависимости: plotly>=5, pandas, streamlit
"""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as _np
import pandas as _pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception as e:
    raise RuntimeError("Требуется пакет 'plotly' (plotly>=5.22.0)") from e

# --- Цвета/оформление ---
COLOR_NEG = '#ff2d2d'    # красный
COLOR_POS = '#22ccff'    # бирюзовый
COLOR_PRICE = '#ff9900'  # оранжевая линия цены
BG_COLOR = '#111111'
FG_COLOR = '#e0e0e0'
GRID_COLOR = 'rgba(255,255,255,0.10)'

def _to_num(a: Sequence) -> _np.ndarray:
    return _np.array(_pd.to_numeric(a, errors='coerce'), dtype=float)

def render_netgex_bars(
    df_final: _pd.DataFrame,
    ticker: str,
    spot: Optional[float] = None,
    toggle_key: Optional[str] = None,
) -> None:
    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для графика Net GEX.")
        return
    if "K" not in df_final.columns:
        st.warning("В финальной таблице отсутствует столбец 'K'.")
        return

    # Выбор колонки Net GEX (в млн $/1% приоритетно)
    if "NetGEX_1pct_M" in df_final.columns:
        y_col = "NetGEX_1pct_M"
    elif "NetGEX_1pct" in df_final.columns:
        df_final = df_final.copy()
        df_final["NetGEX_1pct_M"] = df_final["NetGEX_1pct"] / 1e6
        y_col = "NetGEX_1pct_M"
    else:
        st.warning("Нет столбцов NetGEX_1pct_M / NetGEX_1pct — нечего рисовать.")
        return

    # spot
    if spot is None and "S" in df_final.columns and df_final["S"].notna().any():
        spot = float(df_final["S"].dropna().iloc[0])

    # Тумблер
    show = st.toggle("Net GEX", value=True, key=(toggle_key or f"netgex_toggle_{ticker}"))
    if not show:
        return

    # Подготовка данных и ширины бара
    df = df_final[["K", y_col]].dropna().copy()
    df["K"] = _pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["K"]).sort_values("K").reset_index(drop=True)
    df = df.groupby("K", as_index=False)[y_col].sum()

    Ks = df["K"].to_numpy(dtype=float)
    Ys = df[y_col].to_numpy(dtype=float)

    if len(Ks) >= 2:
        diffs = _np.diff(_np.unique(Ks))
        step = float(_np.nanmin(diffs)) if diffs.size else 1.0
    else:
        step = 1.0
    # Фиксированная визуальная ширина ~30px (аппроксимация по диапазону X)
    # Предполагаем рабочую ширину области графика ~1200px (минус поля). Для адаптивности берём максимум с 0.2*step, чтобы не схлопывалось на малом окне.
    plot_px = 1200.0
    x_range = float(Ks.max() - Ks.min()) if Ks.size else 1.0
    px_target = 28.0
    # ширина из пикселей и безопасная ширина как доля шага; берём МИН(пиксели, 0.7*step), чтобы не было наложения
    width_px_based = (x_range * (px_target / plot_px))
    bar_width = min(step * 0.7, max(width_px_based, step * 0.2))
    colors = _np.where(Ys >= 0.0, COLOR_POS, COLOR_NEG)
    
    # Фигура
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=Ks,
        y=Ys,
        name="Net GEX (M$ / 1%)",
        marker_color=colors,
        width=bar_width,
        hovertemplate="K=%{x}<br>Net GEX=%{y:.3f}M<extra></extra>",
    ))

    # Вертикальная линия цены
    if spot is not None and _np.isfinite(spot):
        y0 = min(0.0, float(_np.nanmin(Ys))) * 1.05
        y1 = max(0.0, float(_np.nanmax(Ys))) * 1.05
        fig.add_shape(type="line", x0=spot, x1=spot, y0=y0, y1=y1, line=dict(color=COLOR_PRICE, width=2))
        fig.add_annotation(x=spot, y=y1, text=f"Price: {spot:.2f}", showarrow=False, yshift=8,
                           font=dict(color=COLOR_PRICE, size=12), xanchor="center")

    # Тикер
    if ticker:
        fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.12, text=str(ticker),
                           showarrow=False, font=dict(size=16, color=FG_COLOR), xanchor="left", yanchor="bottom")

    # Подписи страйков: все значения, горизонтально, шрифт 10
    tick_vals = Ks.tolist()
    tick_text = [str(int(k)) if float(k).is_integer() else f"{k:.2f}" for k in Ks]

    # Динамический подбор размера шрифта подписей страйков так,
    # чтобы ширина текста не превышала ширину столбца
    try:
        max_chars = max(len(t) for t in tick_text) if tick_text else 1
        # оценка ширины символа ~0.6 от размера шрифта
        CHAR_COEF = 0.6
        MAX_FONT = 10
        MIN_FONT = 6
        # оценка ширины бара в пикселях
        bar_px = plot_px * (bar_width / max(x_range, 1e-9))
        est_label_px = MAX_FONT * CHAR_COEF * max_chars
        if est_label_px > bar_px:
            tick_font_size = max(MIN_FONT, int(bar_px / (CHAR_COEF * max_chars)))
        else:
            tick_font_size = MAX_FONT
    except Exception:
        tick_font_size = 10



    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        dragmode=False,
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=0,
            tickfont=dict(size=tick_font_size),   # <<< фиксированный размер 10
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Net GEX",
            showgrid=False,
            zeroline=False,
        ),
    )

    # Автомасштаб
    fig.update_yaxes(autorange=True)
    fig.update_xaxes(autorange=True)

    # Статичный график без зума/панорамы и без панели управления
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={'displayModeBar': False, 'staticPlot': True})
