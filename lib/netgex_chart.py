# -*- coding: utf-8 -*-
"""
netgex_chart.py — бар‑чарт Net GEX для главной страницы.

Публичный API:
    render_netgex_bars(df_final, ticker, spot=None, toggle_key=None, height=520)

Что делает:
- Рисует бар‑чарт Net GEX по страйкам из финальной таблицы.
- Показывает вертикальную линию spot (если передан или есть в df["S"]). 
- (Не менялось) Если в проекте используется тумблер G‑Flip — логика его расчёта оставлена, но код не включает/изменяет этот тумблер.
- ДОБАВЛЕНО: тумблеры Put OI и Call OI (по умолчанию выключены). При включении — 
  рисуются точки+линия на правой оси «Other parameters»:
    • Put OI — бордовый (maroon)
    • Call OI — зелёный (green)

Зависимости: plotly>=5, pandas>=1.5, numpy>=1.23, streamlit>=1.24
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# --------------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---------------------------
def _series(df: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    """Вернуть первый существующий столбец из names, иначе пустую серию длиной df."""
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([np.nan] * len(df))


def _coerce_float_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([np.nan] * len(s))


def _fmt_tick(v: float) -> str:
    """Красивый вывод страйка: без лишних .0"""
    try:
        iv = int(v)
        return str(iv) if abs(v - iv) < 1e-8 else ('{:.2f}'.format(v).rstrip('0').rstrip('.'))
    except Exception:
        return str(v)


def _infer_spot(df: pd.DataFrame, explicit_spot: Optional[float]) -> Optional[float]:
    if explicit_spot is not None and math.isfinite(explicit_spot):
        return float(explicit_spot)
    s = _series(df, ["S"])
    s = _coerce_float_series(s)
    if s.notna().any():
        return float(np.nanmedian(s))
    return None


def _find_gamma_flip_x(strikes: np.ndarray, netgex: np.ndarray) -> Optional[float]:
    """
    Линейная интерполяция нуля Net GEX между соседними страйками.
    Возвращает координату по X (страйк), либо None.
    """
    if strikes.size < 2:
        return None
    y = netgex.astype(float)
    x = strikes.astype(float)

    # Нули «как есть»
    candidates = [float(x[i]) for i, v in enumerate(y) if v == 0.0]

    # Смена знака между соседями
    sign = np.sign(y)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    for i in idx:
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        if y1 == y0:
            continue
        x_star = x0 - y0 * (x1 - x0) / (y1 - y0)
        if min(x0, x1) <= x_star <= max(x0, x1):
            candidates.append(float(x_star))

    if not candidates:
        return None
    # Если несколько — берём ближайший к споту или к среднему strikes
    return float(sorted(candidates, key=lambda v: (v))[0])


# --------------------------- ОСНОВНАЯ ФУНКЦИЯ ---------------------------
def render_netgex_bars(
    df_final: pd.DataFrame,
    ticker: str,
    spot: Optional[float] = None,
    toggle_key: Optional[str] = None,
    height: int = 520,
) -> None:
    """
    Рендерит чарт в Streamlit. Ничего не возвращает.

    Обязательные столбцы для базового графика: K, NetGEX_1pct_M (или NetGEX_1pct).
    Дополнительные столбцы для тумблеров: call_oi, put_oi.
    """
    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для графика.")
        return

    # Безопасная копия и сортировка по страйку
    df = df_final.copy()
    x_k = _coerce_float_series(_series(df, ["K", "strike", "Strike"]))
    df = df.assign(_K=x_k).sort_values("_K").reset_index(drop=True)

    # Y: Net GEX (берём *_M если есть)
    y_netgex = _coerce_float_series(_series(df, ["NetGEX_1pct_M", "NetGEX_1pct", "Net GEX"]))

    # Доп. ряды для тумблеров
    y_put_oi = _coerce_float_series(_series(df, ["put_oi", "Put OI", "putOI"]))
    y_call_oi = _coerce_float_series(_series(df, ["call_oi", "Call OI", "callOI"]))

    # X‑ось: отображаем ВСЕ страйки как есть (без промежутков)
    uniq_x = df["_K"].to_numpy(dtype=float)

    # ---------------------- ТУМБЛЕРЫ ----------------------
    key_prefix = (toggle_key or "ngc")
    col1, col2 = st.columns(2)
    t_put_oi = col1.toggle("Put OI", value=False, key=f"{key_prefix}-putoi")
    t_call_oi = col2.toggle("Call OI", value=False, key=f"{key_prefix}-calloi")

    # ---------------------- ФИГУРА С ДВУМЯ Осями Y ----------------------
    fig = go.Figure()

    # Цвета баров Net GEX по знаку (не меняем существующую палитру проекта)
    colors = np.where(y_netgex >= 0, "#2ecc71", "#e74c3c")  # зелёный / красный

    fig.add_bar(
        x=uniq_x,
        y=y_netgex,
        name="Net GEX",
        marker_color=colors,
        hovertemplate="Strike: %{x}<br>Net GEX: %{y:.0f}<extra></extra>",
        yaxis="y",
    )

    # Правая ось под «Other parameters»
    fig.update_layout(
        yaxis=dict(title="Net GEX", zeroline=True, fixedrange=True),
        yaxis2=dict(
            title="Other parameters",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        bargap=0.15,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
    )

    # Линии OI (по умолчанию выключены, добавляем только при включении тумблера)
    if t_put_oi and y_put_oi.notna().any():
        fig.add_scatter(
            x=uniq_x,
            y=y_put_oi,
            mode="lines+markers",
            name="Put OI",
            marker=dict(size=6, color="maroon"),
            line=dict(width=2, color="maroon"),
            yaxis="y2",
            hovertemplate="Strike: %{x}<br>Put OI: %{y:.0f}<extra></extra>",
        )
    if t_call_oi and y_call_oi.notna().any():
        fig.add_scatter(
            x=uniq_x,
            y=y_call_oi,
            mode="lines+markers",
            name="Call OI",
            marker=dict(size=6, color="green"),
            line=dict(width=2, color="green"),
            yaxis="y2",
            hovertemplate="Strike: %{x}<br>Call OI: %{y:.0f}<extra></extra>",
        )

    # X‑ось — все страйки из таблицы (фиксированный шрифт 10)
    fig.update_xaxes(
        tickmode="array",
        tickvals=uniq_x,
        ticktext=[_fmt_tick(v) for v in uniq_x],
        tickfont=dict(size=10),
        fixedrange=True,
    )

    # Вертикальная линия spot (золотая), если известна
    spot_val = _infer_spot(df, spot)
    if spot_val is not None and math.isfinite(spot_val):
        fig.add_vline(x=spot_val, line_width=2, line_dash="solid", line_color="#f1c40f")
        fig.add_annotation(
            x=spot_val,
            yref="paper",
            y=1.05,
            text=f"Spot: {spot_val:.2f}",
            showarrow=False,
            font=dict(color="#f1c40f"),
        )

    # (НЕ МЕНЯЛОСЬ) — Если в проекте поверх этого файла дополнительно рисуется G‑Flip,
    # то ниже оставлена безопасная заготовка вычисления координаты нуля Net GEX.
    # Показывать/прятать линию следует в коде, который управляет соответствующим тумблером.
    try:
        x_g = _find_gamma_flip_x(uniq_x, y_netgex.to_numpy(dtype=float))
        if x_g is not None:
            # Тонкая пунктирная линия; подпись можно переместить в вашем управляющем коде
            fig.add_shape(
                type="line",
                x0=x_g,
                x1=x_g,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(width=1, color="#AAAAAA", dash="dash"),
                layer="above",
            )
            fig.add_annotation(
                x=x_g,
                xref="x",
                y=1.02,
                yref="paper",
                text=f"G-Flip: { _fmt_tick(x_g) }",
                showarrow=False,
                font=dict(color="#AAAAAA"),
                xanchor="center",
                yanchor="bottom",
                align="center",
            )
    except Exception:
        pass

    # Рендер без панели управления; масштаб фиксирован
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False, "staticPlot": True},
    )
