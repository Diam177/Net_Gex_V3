
# -*- coding: utf-8 -*-
"""
netgex_chart.py — компактный чарт Net GEX для главной страницы.

Функция render_netgex_bars(df_final, ticker, spot=None, toggle_key=None):
  • df_final: DataFrame по ОДНОЙ выбранной экспирации (как показывает ваша финальная таблица)
  • ticker: строка для подписи в левом верхнем углу
  • spot: текущая цена БА; если None — берётся из df_final['S'] (первое доступное значение)
  • toggle_key: уникальный ключ для st.toggle (чтобы избежать конфликтов ключей в Streamlit)

Требует: plotly>=5, pandas, streamlit
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception as e:  # защитная ветка, если plotly не установлен
    raise RuntimeError("Для отрисовки графика требуется пакет 'plotly'. Добавьте 'plotly>=5.22.0' в requirements.txt") from e


# === Цвета и оформление (легко подстраиваются) ===============================
COLOR_NEG = '#ff2d2d'   # красный для отрицательного Net GEX
COLOR_POS = '#22ccff'   # бирюзовый/голубой для положительного Net GEX
COLOR_PRICE = '#ff9900' # оранжевая линия цены
BG_COLOR = '#111111'    # фон
FG_COLOR = '#e0e0e0'    # цвет подписей/осей
GRID_COLOR = 'rgba(255,255,255,0.10)'

def _coerce_numeric(a: Sequence) -> np.ndarray:
    return np.array(pd.to_numeric(a, errors='coerce'), dtype=float)


def render_netgex_bars(
    df_final: pd.DataFrame,
    ticker: str,
    spot: Optional[float] = None,
    toggle_key: Optional[str] = None,
) -> None:
    """Рисует чарт Net GEX под финальной таблицей с тумблером отображения."""
    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для графика Net GEX.")
        return

    # Проверяем наличие ключевых столбцов
    if "K" not in df_final.columns:
        st.warning("В финальной таблице отсутствует столбец 'K' (strike).")
        return

    # Выбор колонки с Net GEX (в млн $)
    y_col = None
    if "NetGEX_1pct_M" in df_final.columns:
        y_col = "NetGEX_1pct_M"
    elif "NetGEX_1pct" in df_final.columns:
        # приводим к млн $ на 1%
        df_final = df_final.copy()
        df_final["NetGEX_1pct_M"] = df_final["NetGEX_1pct"] / 1e6
        y_col = "NetGEX_1pct_M"
    else:
        st.warning("В финальной таблице нет столбцов NetGEX_1pct_M / NetGEX_1pct — нечего рисовать.")
        return

    # Определяем spot
    if spot is None:
        if "S" in df_final.columns and df_final["S"].notna().any():
            spot = float(df_final["S"].dropna().iloc[0])
        else:
            spot = None

    # Гарантируем, что на оси X будут ВСЕ страйки (как требовалось)
    df = df_final.copy()
    df = df[["K", y_col]].dropna()
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["K"])
    df = df.sort_values("K").reset_index(drop=True)

    # Аггрегируем по страйку (на всякий случай)
    df = df.groupby("K", as_index=False)[y_col].sum()

    Ks = df["K"].to_numpy(dtype=float)
    Ys = df[y_col].to_numpy(dtype=float)

    # Применим ширину бара по минимальному шагу страйков
    if len(Ks) >= 2:
        diffs = np.diff(np.unique(Ks))
        step = float(np.nanmin(diffs)) if diffs.size else 1.0
    else:
        step = 1.0
    bar_width = step * 0.8

    # Палитра по знаку
    colors = np.where(Ys >= 0.0, COLOR_POS, COLOR_NEG)

    # Тумблер отображения
    show = st.toggle("Показать Net GEX", value=True, key=(toggle_key or f"netgex_toggle_{ticker}"))
    if not show:
        return

    # Создаём фигуру
    fig = go.Figure()

    # Столбики
    fig.add_trace(go.Bar(
        x=Ks,
        y=Ys,
        name="Net GEX (M$ / 1%)",
        marker_color=colors,
        width=bar_width,
        hovertemplate="K=%{x}<br>Net GEX=%{y:.3f}M<extra></extra>",
    ))

    # Вертикальная линия цены + подпись
    if spot is not None and np.isfinite(spot):
        fig.add_shape(
            type="line",
            x0=spot, x1=spot,
            y0=min(0.0, float(np.nanmin(Ys))) * 1.05,
            y1=max(0.0, float(np.nanmax(Ys))) * 1.05,
            line=dict(color=COLOR_PRICE, width=2),
        )
        # подпись у верхней части линии
        fig.add_annotation(
            x=spot, y=max(0.0, float(np.nanmax(Ys))) * 1.05,
            text=f"Price: {spot:.2f}",
            showarrow=False,
            yshift=8,
            font=dict(color=COLOR_PRICE, size=12),
            xanchor="center",
        )

    # Тикер в левом верхнем углу
    if ticker:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.0, y=1.12,
            text=str(ticker),
            showarrow=False,
            font=dict(size=16, color=FG_COLOR),
            xanchor="left", yanchor="bottom",
        )

    # Оформление осей
    tick_vals = Ks.tolist()  # все страйки на шкале
    tick_text = [str(int(k)) if float(k).is_integer() else f"{k:.2f}" for k in Ks]

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Net GEX (M)",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=True,
            zerolinecolor=GRID_COLOR,
        ),
    )

    # Автомасштаб
    fig.update_yaxes(autorange=True)
    fig.update_xaxes(autorange=True)

    st.plotly_chart(fig, use_container_width=True, theme=None)
