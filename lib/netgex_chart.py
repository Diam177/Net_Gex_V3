# -*- coding: utf-8 -*-
"""
chart_final_table.py
--------------------
Рендер совмещённого чарта по финальной таблице.

Публичный API:
    render_final_chart(df, title=None, spot=None, show_toggles=True, height=520)

Ожидаемые колонки df (используется подмножество — отсутствующие игнорируются):
    ["K","S","F","NetGEX_1pct","AG_1pct","NetGEX_1pct_M","AG_1pct_M",
     "call_oi","put_oi","call_vol","put_vol","PZ","ER_Up","ER_Down"]

Все дополнительные кривые (Call OI, Put/Call Volume, AG, PZ, ER_Up/Down) рисуются
поверх основного графика по правой оси Y. Каждая кривая отображается точками,
соединёнными сглаженной линией (spline) с заливкой до нуля. Прозрачность заливки — 70%.
"""
from __future__ import annotations

from typing import Dict, Optional, Iterable, Tuple

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import streamlit as st  # type: ignore
except Exception:  # на случай оффлайн-тестов
    class _ST:  # pragma: no cover
        def toggle(self, *a, **k): return True
        def columns(self, n): return [self]*n
        def markdown(self, *a, **k): pass
        def write(self, *a, **k): pass
        def plotly_chart(self, *a, **k): pass
        def caption(self, *a, **k): pass
    st = _ST()  # type: ignore

__all__ = ["render_final_chart"]


# ----------------------------- helpers ---------------------------------------
def _num_series(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    """Вернуть первую найденную колонку как числовую серию (float) либо None."""
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    return None


def _x_strikes(df: pd.DataFrame) -> pd.Series:
    if "K" not in df.columns:
        raise ValueError("В финальной таблице отсутствует колонка 'K' со страйками.")
    return pd.to_numeric(df["K"], errors="coerce")


def _fmt_ticks(values: Iterable[float]) -> Tuple[Iterable[float], Iterable[str]]:
    uniq = []
    seen = set()
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv in seen:
            continue
        seen.add(fv)
        uniq.append(fv)
    uniq.sort()
    ticktext = []
    for v in uniq:
        iv = int(round(v))
        ticktext.append(str(iv) if abs(v - iv) < 1e-8 else ("{:.2f}".format(v).rstrip("0").rstrip(".")))
    return uniq, ticktext


def _rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(r,g,b,alpha)'."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _add_line_with_fill(fig: go.Figure, x, y, name: str, color: str, showlegend: bool = True):
    """Добавить линию+точки с заливкой до нуля (правый Y)."""
    if y is None:
        return
    y = pd.to_numeric(y, errors="coerce")
    if not y.notna().any():
        return
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="lines+markers",
            line=dict(shape="spline", width=2, color=color),
            marker=dict(size=5, color=color),
            fill="tozeroy",
            fillcolor=_rgba(color, 0.30),  # 70% прозрачность заливки
            yaxis="y2",
            hovertemplate="%{x}<br>%{y:.0f}<extra>"+name+"</extra>",
            showlegend=showlegend,
        )
    )


# --------------------------- main renderer -----------------------------------
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

    # Базовая ось X (числовая, без искусственных промежутков по тикам)
    x = _x_strikes(df)
    tickvals, ticktext = _fmt_ticks(x)

    # Доступные серии (поддерживаем разные синонимы колонок)
    y_netgex = _num_series(df, ["NetGEX_1pct", "NetGEX_1pct_M", "NetGEX"])
    y_call_oi = _num_series(df, ["call_oi", "Call OI", "callOI"])
    y_put_oi  = _num_series(df, ["put_oi", "Put OI", "putOI"])
    y_call_vol = _num_series(df, ["call_vol", "Call Volume", "callVolume","call_vol_sum"])
    y_put_vol  = _num_series(df, ["put_vol", "Put Volume", "putVolume","put_vol_sum"])
    y_ag       = _num_series(df, ["AG_1pct", "AG_1pct_M", "AG"])
    y_pz       = _num_series(df, ["PZ","PZ_FP","PZ_norm"])
    y_er_up    = _num_series(df, ["ER_Up","ER_up","ERup"])
    y_er_dn    = _num_series(df, ["ER_Down","ER_down","ERdn"])

    # Переключатели UI
    if show_toggles:
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
        t_netgex = c1.toggle("Net Gex", value=True, key="ft_netgex")
        t_put_oi = c2.toggle("Put OI", value=False, key="ft_put_oi")
        t_call_oi= c3.toggle("Call OI", value=False, key="ft_call_oi")
        t_put_v  = c4.toggle("Put Volume", value=False, key="ft_put_vol")
        t_call_v = c5.toggle("Call Volume", value=False, key="ft_call_vol")
        t_ag     = c6.toggle("AG", value=False, key="ft_ag")
        t_pz     = c7.toggle("PZ", value=False, key="ft_pz")
        t_er_up  = c8.toggle("ER_Up", value=False, key="ft_er_up")
        t_er_dn  = c9.toggle("ER_Down", value=False, key="ft_er_dn")
    else:
        t_netgex = True
        t_put_oi = t_call_oi = t_put_v = t_call_v = t_ag = t_pz = t_er_up = t_er_dn = False

    # --- Фигура с двумя осями Y ---
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=40, r=40, t=60, b=50),
        legend=dict(orientation="h", y=1.12, x=0.01),
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            fixedrange=True,
            title=None,
        ),
        yaxis=dict(
            title="Net GEX (1% $)",
            zeroline=True,
            zerolinewidth=1,
            showgrid=True,
        ),
        yaxis2=dict(
            title="Доп. метрики (правая ось)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
        dragmode=False,
    )

    # --- NetGEX бар-чарт (левая ось) ---
    if t_netgex and y_netgex is not None and y_netgex.notna().any():
        y = y_netgex.fillna(0.0)
        y_pos = y.clip(lower=0)
        y_neg = y.clip(upper=0)
        fig.add_trace(
            go.Bar(
                x=x, y=y_pos, name="Net GEX +",
                marker_color="#2ecc71", opacity=0.95,
                hovertemplate="%{x}<br>%{y:.0f}<extra>Net GEX +</extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=x, y=y_neg, name="Net GEX -",
                marker_color="#e74c3c", opacity=0.95,
                hovertemplate="%{x}<br>%{y:.0f}<extra>Net GEX -</extra>",
            )
        )

    # --- Право-осевые кривые с заполнением ---
    # Цвета подобраны стабильные и читаемые. Прозрачность заливки 70% (alpha=0.30).
    if t_call_oi: _add_line_with_fill(fig, x, y_call_oi, "Call OI", "#3498db")
    if t_put_oi:  _add_line_with_fill(fig, x, y_put_oi,  "Put OI",  "#95a5a6")
    if t_call_v:  _add_line_with_fill(fig, x, y_call_vol,"Call Volume", "#1abc9c")
    if t_put_v:   _add_line_with_fill(fig, x, y_put_vol, "Put Volume",  "#e67e22")
    if t_ag:      _add_line_with_fill(fig, x, y_ag,      "AG (1%)", "#9b59b6")
    if t_pz:      _add_line_with_fill(fig, x, y_pz,      "PZ", "#f39c12")
    if t_er_up:   _add_line_with_fill(fig, x, y_er_up,   "ER_Up", "#2ecc71")
    if t_er_dn:   _add_line_with_fill(fig, x, y_er_dn,   "ER_Down", "#e74c3c")

    # --- Вертикальная линия spot ---
    spot_val = None
    if spot is not None:
        try:
            spot_val = float(spot)
        except Exception:
            spot_val = None
    if spot_val is None:
        s_series = _num_series(df, ["S"])
        if s_series is not None and s_series.notna().any():
            spot_val = float(np.nanmedian(s_series.values))
    if spot_val is not None and math.isfinite(spot_val):
        fig.add_vline(x=spot_val, line_width=2, line_dash="solid", line_color="#f1c40f")
        fig.add_annotation(
            x=spot_val, yref="paper", y=1.05, showarrow=False,
            text=f"Price: {spot_val:.2f}", font=dict(color="#f1c40f")
        )

    # Заголовок
    if title:
        fig.update_layout(title=dict(text=title, x=0.02, xanchor="left"))

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})
