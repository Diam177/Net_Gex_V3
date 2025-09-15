# -*- coding: utf-8 -*-
"""
chart_final_table.py
--------------------
Стримлит-рендер чарта для финальной таблицы.
Зависимости: plotly>=5.18, pandas, numpy, streamlit

Публичный API:
    render_final_chart(df, title=None, spot=None, show_toggles=True, height=520)

Вход:
    df: pandas.DataFrame с колонками (любое подмножество — отсутствие будет игнорироваться):
        ["K","S","F","NetGEX_1pct","AG_1pct","NetGEX_1pct_M","AG_1pct_M",
         "call_oi","put_oi","call_vol","put_vol","PZ","ER_Up","ER_Down"]
    spot: float|None — текущая цена (вертикальная линия). Если None — возьмём медиану df["S"].
    title: str|None — заголовок (тикер).
    show_toggles: bool — показать переключатели видимости серий.
    height: int — высота фигуры.

Поведение (см. ориентир по скриншоту):
    - Столбцы (левая ось): Net GEX (красные отрицательные, голубые положительные).
    - Линия+заливка (правая ось): AG (фиолетовая волна).
    - Переключатели: Put OI, Call OI, Put Volume, Call Volume, PZ, ER (Up/Down). AG и Net GEX включены по умолчанию.
    - Вертикальная золотая линия со значением spot.
"""
from __future__ import annotations

import math
from typing import Optional, Dict

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception as e:  # pragma: no cover
    raise RuntimeError("plotly is required: pip install plotly>=5.18") from e

# streamlit может отсутствовать в тестовом окружении
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    class _ST:
        def toggle(self, *a, **k): return True
        def columns(self, n): return [self]*n
        def markdown(self, *a, **k): pass
        def write(self, *a, **k): pass
        def plotly_chart(self, *a, **k): pass
        def caption(self, *a, **k): pass
    st = _ST()  # type: ignore


def _coerce_series(df: pd.DataFrame, col_candidates) -> Optional[pd.Series]:
    for c in col_candidates:
        if c in df.columns:
            return df[c]
    return None


def _split_pos_neg(y: pd.Series) -> Dict[str, pd.Series]:
    y_pos = y.clip(lower=0)
    y_neg = y.clip(upper=0)
    return {"pos": y_pos, "neg": y_neg}



def _safe_float_median(series: Optional[pd.Series]) -> float:
    """Return float(median(series)) or NaN if series is None/empty/non-numeric."""
    if series is None:
        return float("nan")
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return float("nan")
        return float(np.nanmedian(s.values))
    except Exception:
        return float("nan")

def render_final_chart(
    df: pd.DataFrame,
    title: Optional[str] = None,
    spot: Optional[float] = None,
    show_toggles: bool = True,
    height: int = 520,
) -> None:
    if df is None or len(df) == 0:
        st.caption("Нет данных для отображения.")
        return

    # Базовая ось X
    x_raw = df["K"].astype(float)
    # Квантование страйков на сетку, чтобы исключить накопление ошибок с плавающей точкой
    import numpy as _np
    uniq_sorted = _np.sort(_np.unique(x_raw.values.astype(float)))
    if uniq_sorted.size >= 2:
        step = _np.median(_np.diff(uniq_sorted))
        if _np.isfinite(step) and step > 0:
            x = _np.round(x_raw.values.astype(float) / step) * step
            x = _np.array(x, dtype=float)
            uniq_sorted = _np.sort(_np.unique(x))
        else:
            x = x_raw.values.astype(float)
    else:
        x = x_raw.values.astype(float)
    # Преобразуем назад в pandas.Series для совместимости
    import pandas as _pd
    x = _pd.Series(x)

    # Найдём серии
    y_netgex = _coerce_series(df, ["NetGEX_1pct", "NetGEX", "NetGEX_1pct_M"])  # M тоже ок — масштаб правой оси не привязан
    y_ag     = _coerce_series(df, ["AG_1pct", "AG", "AG_1pct_M"])
    y_call_oi = _coerce_series(df, ["call_oi", "Call OI", "callOI"])
    y_put_oi  = _coerce_series(df, ["put_oi", "Put OI", "putOI"])
    y_call_vol = _coerce_series(df, ["call_vol", "Call Volume", "callVolume","call_vol_sum"])
    y_put_vol  = _coerce_series(df, ["put_vol", "Put Volume", "putVolume","put_vol_sum"])
    y_pz     = _coerce_series(df, ["PZ","PZ_FP","PZ_norm"])
    y_er_up  = _coerce_series(df, ["ER_Up","ER_up","ERup"])
    y_er_dn  = _coerce_series(df, ["ER_Down","ER_down","ERdn"])

    # Переключатели
    if show_toggles:
        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10 = st.columns(10)
        t_netgex = c1.toggle("Net Gex", value=True)
        t_put_oi = c2.toggle("Put OI", value=False, key="put_oi_t")
        t_call_oi= c3.toggle("Call OI", value=False, key="call_oi_t")
        t_put_v  = c4.toggle("Put Volume", value=False, key="put_v_t")
        t_call_v = c5.toggle("Call Volume", value=False, key="call_v_t")
        t_ag     = c6.toggle("AG", value=False, key="ag_t")
        t_pz     = c7.toggle("PZ", value=False, key="pz_t")
        t_er_up  = c8.toggle("ER Up", value=False, key="er_up_t")
        t_er_dn  = c9.toggle("ER Down", value=False, key="er_dn_t")
        t_gflip  = c10.toggle("G-Flip", value=False, key="gflip_t")  # placeholder
    else:
        t_netgex=True; t_ag=False
        t_put_oi=t_call_oi=t_put_v=t_call_v=t_pz=t_er_up=t_er_dn=t_gflip=False

# Фигура с двумя Y-осями
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=40,r=40,t=60,b=50),
        legend=dict(orientation="h", y=1.12, x=0.01),
    )

    # Net GEX как бары: отрицательные — красные, положительные — голубые
    if t_netgex and y_netgex is not None:
        split = _split_pos_neg(y_netgex.astype(float))
        fig.add_bar(
            x=x, y=split["neg"],
            name="Net GEX (neg)", marker_color="crimson", opacity=0.9, yaxis="y1"
        )
        fig.add_bar(
            x=x, y=split["pos"],
            name="Net GEX (pos)", marker_color="lightskyblue", opacity=0.9, yaxis="y1"
        )

    # AG как фиолетовая линия с заливкой на вторую ось
    if t_ag and y_ag is not None:
        fig.add_trace(go.Scatter(
            x=x, y=y_ag.astype(float),
            name="AG",
            mode="lines+markers",
            line=dict(width=2, color="#9b59b6"),
            marker=dict(size=4),
            yaxis="y2",
            fill="tozeroy",
            fillcolor="rgba(155,89,182,0.25)"
        ))

    # Доп.слои — OI и Volume (как линии на левой оси, чтобы не перегружать правую)
    if t_put_oi and y_put_oi is not None:
        fig.add_trace(go.Scatter(x=x, y=y_put_oi.astype(float), name="Put OI", mode="lines", line=dict(width=1), yaxis="y1"))
    if t_call_oi and y_call_oi is not None:
        fig.add_trace(go.Scatter(x=x, y=y_call_oi.astype(float), name="Call OI", mode="lines", line=dict(width=1), yaxis="y1"))
    if t_put_v and y_put_vol is not None:
        fig.add_trace(go.Scatter(x=x, y=y_put_vol.astype(float), name="Put Vol", mode="lines", line=dict(width=1), yaxis="y1"))
    if t_call_v and y_call_vol is not None:
        fig.add_trace(go.Scatter(x=x, y=y_call_vol.astype(float), name="Call Vol", mode="lines", line=dict(width=1), yaxis="y1"))

    # PZ / ER (тонкие линии на вторую ось, чтобы масштаб совпадал с AG)
    if t_pz and y_pz is not None:
        fig.add_trace(go.Scatter(x=x, y=y_pz.astype(float), name="PZ", mode="lines", line=dict(width=1, dash="dot"), yaxis="y2"))
    if t_pz and y_er_up is not None:
        fig.add_trace(
            go.Scatter(x=x, y=y_er_up.astype(float), name="ER Up", mode="lines", line=dict(width=1, dash="dash"), yaxis="y2")
        )
    if t_pz and y_er_dn is not None:
        fig.add_trace(
            go.Scatter(x=x, y=y_er_dn.astype(float), name="ER Down", mode="lines", line=dict(width=1, dash="dash"), yaxis="y2")
        )

    # Оси
    fig.update_layout(
        xaxis=dict(title="Strike"),
        yaxis=dict(title="Net GEX", rangemode="tozero"),
        yaxis2=dict(title="Other parameters (AG)", overlaying="y", side="right", rangemode="tozero"),
    )

    # Вертикальная линия spot (золотая)
    spot_val = float(spot) if spot is not None else _safe_float_median(_coerce_series(df, ["S"]))
    if math.isfinite(spot_val):
        fig.add_vline(x=spot_val, line_width=2, line_dash="solid", line_color="#f1c40f")
        fig.add_annotation(x=spot_val, yref="paper", y=1.05, text=f"Price: {spot_val:.2f}", showarrow=False, font=dict(color="#f1c40f"))

    # Заголовок
    if title:
        fig.update_layout(title=dict(text=title, x=0.02, xanchor="left"))

    st.plotly_chart(fig, use_container_width=True)
