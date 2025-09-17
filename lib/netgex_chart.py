# -*- coding: utf-8 -*-
"""
netgex_chart.py — визуализация финальной таблицы: основной бар‑чарт Net GEX
и вспомогательные серии (Call/Put OI, Call/Put Volume, AG, PZ, ER_Up, ER_Down).

Главная функция:
    render_netgex_bars(df_final, ticker: str, spot: float | None = None, toggle_key: str | None = None)

Требования к df_final (выход из final_table.py):
    обязательные колонки: ["K", "NetGEX_1pct", "call_oi", "put_oi"]
    опциональные:          ["call_vol", "put_vol", "AG_1pct", "PZ", "ER_Up", "ER_Down", "S"]

Особенности:
- Подсказка при наведении на БАР Net GEX: табличка со значениями
  Strike, Call OI, Put OI, Call Volume, Put Volume, Net GEX.
  Цвет всплывашки совпадает с цветом бара: красная для отрицательных значений,
  бирюзовая для положительных.
- Подсказка при наведении на ТОЧКУ для остальных серий (линия+точки):
  та же табличка; цвет всплывашки = цвету серии.
- Выравнивание по всем фактическим страйкам (категориальная ось без пропусков).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# Цвета
COLOR_NEG = "#D9493A"   # отрицательные столбцы Net GEX
COLOR_POS = "#60A5E7"   # положительные столбцы Net GEX
COLOR_PRICE = "#E4A339"
FG_COLOR = "#EAEAEA"
BG_COLOR = "#0F1115"

# Серии
SERIES_COLORS = {
    "put_oi":   "#800020",
    "call_oi":  "#2ECC71",
    "put_vol":  "#FF8C00",
    "call_vol": "#1E88E5",
    "AG_1pct":  "#9A7DF7",
    "PZ":       "#E4C51E",
    "ER_Up":    "#1FCE54",
    "ER_Down":  "#D21717",
}

SERIES_LABELS = {
    "put_oi":   "Put OI",
    "call_oi":  "Call OI",
    "put_vol":  "Put Volume",
    "call_vol": "Call Volume",
    "AG_1pct":  "AG",
    "PZ":       "PZ",
    "ER_Up":    "ER_Up",
    "ER_Down":  "ER_Down",
}


def _aggregate_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Гарантированно получить по одному значению на страйк:
    суммируем по K для всех колонок, где это уместно.
    """
    cols = [c for c in [
        "K", "NetGEX_1pct", "call_oi", "put_oi", "call_vol", "put_vol",
        "AG_1pct", "PZ", "ER_Up", "ER_Down"
    ] if c in df.columns]

    g = (df[cols]
         .groupby("K", as_index=False)
         .sum(numeric_only=True))
    g = g.sort_values("K").reset_index(drop=True)
    return g


def _make_customdata(dfK: pd.DataFrame) -> np.ndarray:
    """
    Собрать customdata на каждый K в фиксированном порядке полей:
    [K, call_oi, put_oi, call_vol, put_vol, NetGEX_1pct]
    Отсутствующие колонки заменяем на 0.
    """
    def get(col: str) -> np.ndarray:
        return dfK[col].to_numpy(dtype=float) if col in dfK.columns else np.zeros(len(dfK), dtype=float)

    cd = np.c_[
        dfK["K"].to_numpy(dtype=float),
        get("call_oi"),
        get("put_oi"),
        get("call_vol"),
        get("put_vol"),
        get("NetGEX_1pct"),
    ]
    return cd


def _hovertemplate() -> str:
    """
    Единый шаблон всплывашки для всех серий.
    """
    return (
        "<b>Strike: %{customdata[0]:.0f}</b><br>"
        "Call OI: %{customdata[1]:.0f}<br>"
        "Put OI: %{customdata[2]:.0f}<br>"
        "Call Volume: %{customdata[3]:.1f}<br>"
        "Put Volume: %{customdata[4]:.1f}<br>"
        "Net GEX: %{customdata[5]:.1f}"
        "<extra></extra>"
    )


def _compute_gflip_approx(K: np.ndarray, netgex: np.ndarray) -> Optional[float]:
    """
    Линейная интерполяция нулевого перехода NetGEX (между соседними страйками).
    Возвращает координату x (в категориальной шкале используем индекс, не K),
    т.к. ось X категориальная.
    """
    for i in range(1, len(netgex)):
        y0, y1 = netgex[i-1], netgex[i]
        if np.isnan(y0) or np.isnan(y1):
            continue
        if y0 == 0:
            return float(i-1)
        if y1 == 0:
            return float(i)
        if (y0 < 0 and y1 > 0) or (y0 > 0 and y1 < 0):
            # линейная интерполяция по индексу
            t = abs(y0) / (abs(y0) + abs(y1))
            return float((i-1) + t)
    return None


def render_netgex_bars(df_final: pd.DataFrame, ticker: str, spot: float | None = None, toggle_key: str | None = None) -> None:
    """
    Основной рендер чартов. Строго ничего вне визуализации не меняет.
    """
    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для отображения.")
        return

    dfK = _aggregate_final(df_final)

    # Категориальная ось из всех фактических страйков
    cat_x = [str(int(k)) for k in dfK["K"].to_list()]
    x_idx = list(range(len(cat_x)))
    mapper_idx_from_k = {k: i for i, k in enumerate(dfK["K"].to_list())}

    # Данные серий
    Y_net = dfK["NetGEX_1pct"].to_numpy(dtype=float) if "NetGEX_1pct" in dfK.columns else np.zeros(len(dfK))
    colors_bars = [COLOR_POS if y >= 0 else COLOR_NEG for y in Y_net]

    customdata = _make_customdata(dfK)

    # Тумблеры
    key_prefix = (toggle_key or "main") + "_"
    show_call_oi = st.toggle(SERIES_LABELS["call_oi"], value=False, key=key_prefix+"call_oi")
    show_put_oi  = st.toggle(SERIES_LABELS["put_oi"],  value=False, key=key_prefix+"put_oi")
    show_call_v  = st.toggle(SERIES_LABELS["call_vol"],value=False, key=key_prefix+"call_vol")
    show_put_v   = st.toggle(SERIES_LABELS["put_vol"], value=False, key=key_prefix+"put_vol")
    show_ag      = st.toggle(SERIES_LABELS["AG_1pct"], value=False, key=key_prefix+"ag")
    show_pz      = st.toggle(SERIES_LABELS["PZ"],      value=False, key=key_prefix+"pz")
    show_er_up   = st.toggle(SERIES_LABELS["ER_Up"],   value=False, key=key_prefix+"er_up")
    show_er_dn   = st.toggle(SERIES_LABELS["ER_Down"], value=False, key=key_prefix+"er_dn")
    show_gflip   = st.toggle("G-Flip", value=True, key=key_prefix+"gflip")  # можно отключать

    fig = go.Figure()

    # Бары Net GEX с массивом цветов для hoverlabel
    fig.add_trace(go.Bar(
        x=x_idx, y=Y_net,
        name="Net GEX (per 1%)",
        marker_color=colors_bars,
        customdata=customdata,
        hovertemplate=_hovertemplate(),
        hoverlabel=dict(bgcolor=colors_bars, bordercolor="white", font=dict(color="white", size=13)),
    ))

    # Вспомогательная функция для добавления линий+точек
    def add_line_series(col: str, label: str):
        if col not in dfK.columns:
            return
        fig.add_trace(go.Scatter(
            x=x_idx,
            y=dfK[col].to_numpy(dtype=float),
            name=label,
            mode="lines+markers",
            line=dict(width=2, color=SERIES_COLORS[col]),
            marker=dict(size=6, color=SERIES_COLORS[col]),
            customdata=customdata,
            hovertemplate=_hovertemplate(),
            hoverlabel=dict(bgcolor=SERIES_COLORS[col], bordercolor="white", font=dict(color="white", size=13)),
        ))

    # Добавляем серии согласно тумблерам
    if show_call_oi: add_line_series("call_oi", SERIES_LABELS["call_oi"])
    if show_put_oi:  add_line_series("put_oi",  SERIES_LABELS["put_oi"])
    if show_call_v:  add_line_series("call_vol",SERIES_LABELS["call_vol"])
    if show_put_v:   add_line_series("put_vol", SERIES_LABELS["put_vol"])
    if show_ag:      add_line_series("AG_1pct", SERIES_LABELS["AG_1pct"])
    if show_pz:      add_line_series("PZ",      SERIES_LABELS["PZ"])
    if show_er_up:   add_line_series("ER_Up",   SERIES_LABELS["ER_Up"])
    if show_er_dn:   add_line_series("ER_Down", SERIES_LABELS["ER_Down"])

    # Вертикальная линия цены (если есть)
    if spot is None:
        if "S" in df_final.columns and not pd.isna(df_final["S"]).all():
            try:
                spot = float(df_final["S"].iloc[0])
            except Exception:
                spot = None
    if spot is not None:
        # Ищем ближайший страйк и ставим линию по индексу
        # Если strike шаг не равен 1, индексная позиция надёжнее для категориальной оси
        try:
            # ближайший индекс
            diffs = np.abs(dfK["K"].to_numpy(dtype=float) - float(spot))
            x_price = int(np.argmin(diffs))
        except Exception:
            x_price = 0
        y0 = min(0.0, float(np.nanmin(Y_net))) * 1.05
        y1 = max(0.0, float(np.nanmax(Y_net))) * 1.05
        fig.add_shape(type="line", x0=x_price, x1=x_price, y0=y0, y1=y1, line=dict(color=COLOR_PRICE, width=2))
        fig.add_annotation(x=x_price, y=y1, text=f"Price: {spot:.2f}", showarrow=False, yshift=8,
                           font=dict(color=COLOR_PRICE, size=12), xanchor="center")

    # G-Flip (по индексу между столбцами)
    if show_gflip and len(Y_net) >= 2:
        x_g = _compute_gflip_approx(dfK["K"].to_numpy(dtype=float), Y_net)
        if x_g is not None:
            fig.add_shape(type="line", x0=x_g, x1=x_g, y0=0, y1=1, xref="x", yref="paper",
                          line=dict(color="#AAAAAA", width=1, dash="dash"))
            fig.add_annotation(x=x_g, xref="x", y=1.02, yref="paper", text="G-Flip",
                               showarrow=False, font=dict(size=12, color="#AAAAAA"),
                               xanchor="center", yanchor="bottom")

    # Оси и оформление
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=FG_COLOR, size=12),
        margin=dict(l=60, r=30, t=40, b=60),
        showlegend=True,
        hovermode="closest",
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=cat_x,
            tickmode="array",
            tickvals=x_idx,
            ticktext=cat_x,
            tickangle=0,
            title_text="Strike",
        ),
        yaxis=dict(title_text="Net GEX"),
    )

    # Заголовок (тикер слева сверху)
    if ticker:
        fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.12, text=str(ticker),
                           showarrow=False, font=dict(size=16, color=FG_COLOR),
                           xanchor="left", yanchor="bottom")

    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
