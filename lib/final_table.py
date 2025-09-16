
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
COLOR_NEG = '#EB382E'    # красный
COLOR_POS = '#44A7ED'    # бирюзовый
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
    
    # Последовательные позиции без "пустых" промежутков между страйками
    x_idx = _np.arange(len(Ks), dtype=float)
    bar_width = 0.9
    colors = _np.where(Ys >= 0.0, COLOR_POS, COLOR_NEG)
    
    # Фигура
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_idx,
        y=Ys,
        name="Net GEX (M$ / 1%)",
        marker_color=colors,
        width=bar_width,
        hovertemplate="K=%{x}<br>Net GEX=%{y:.3f}M<extra></extra>",
    ))

    # Вертикальная линия цены
    
    if spot is not None and _np.isfinite(spot):
    # интерполируем позицию цены между ближайшими страйками на оси индексов
        try:
            if len(Ks) >= 2:
                j = int(_np.searchsorted(Ks, spot))
                if j <= 0:
                    x_price = 0.0
                elif j >= len(Ks):
                    x_price = float(len(Ks) - 1)
                else:
                    k0, k1 = Ks[j-1], Ks[j]
                    frac = 0.0 if (k1 - k0) == 0 else (spot - k0) / (k1 - k0)
                    x_price = (j - 1) + float(_np.clip(frac, 0.0, 1.0))
            else:
                x_price = 0.0
        except Exception:
            x_price = 0.0
        
        y0 = min(0.0, float(_np.nanmin(Ys))) * 1.05
        y1 = max(0.0, float(_np.nanmax(Ys))) * 1.05
        fig.add_shape(type="line", x0=x_price, x1=x_price, y0=y0, y1=y1, line=dict(color=COLOR_PRICE, width=2))
        fig.add_annotation(x=x_price, y=y1, text=f"Price: {spot:.2f}", showarrow=False, yshift=8,
                           font=dict(color=COLOR_PRICE, size=12), xanchor="center")
    
    # Тикер
    if ticker:
        fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.12, text=str(ticker),
                           showarrow=False, font=dict(size=16, color=FG_COLOR), xanchor="left", yanchor="bottom")

    # Подписи страйков: все значения, горизонтально, шрифт 10
    tick_vals = x_idx.tolist()
    tick_text = [str(int(k)) if float(k).is_integer() else f"{k:.2f}" for k in Ks]

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
            tickfont=dict(size=10),   # <<< фиксированный размер 10
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
