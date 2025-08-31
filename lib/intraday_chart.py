# lib/intraday_chart.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    # df: ['datetime','high','low','close','volume']
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"].clip(lower=0)
    cum_pv = pv.cumsum()
    cum_v  = df["volume"].clip(lower=0).cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def _add_hline(fig: go.Figure, y: float, name: str, dash: str = "solid", width: float = 2.0):
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return
    fig.add_hline(y=y, line_width=width, line_dash=dash,
                  annotation_text=name, annotation_position="top left")

def _add_zone(fig: go.Figure, lo: float, hi: float, name: str, fillcolor: str, opacity: float = 0.18):
    if lo is None or hi is None or not (hi > lo):
        return
    fig.add_shape(type="rect", xref="x", yref="y",
                  x0=fig.layout.xaxis.range[0] if fig.layout.xaxis.range else None,
                  x1=fig.layout.xaxis.range[1] if fig.layout.xaxis.range else None,
                  y0=lo, y1=hi, line=dict(width=0),
                  fillcolor=fillcolor, opacity=opacity, layer="below")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color=fillcolor), name=name, showlegend=True))

def make_intraday_levels_figure(
    df: pd.DataFrame,
    levels: Dict[str, Optional[float]] | None = None,
    zones: Dict[str, Optional[Tuple[float, float]]] | None = None,
    title: str = "",
) -> go.Figure:
    """
    Ожидает df с колонками: ['datetime','open','high','low','close','volume'] — одна сессия.
    levels: ключи (опционально): 'main_resistance','main_support','max_neg_gex','max_pos_gex',
            'max_ag','g_flip','global_call_vol_level','global_put_vol_level'
    zones:  ключи (опционально): 'resistance_zone'=(lo,hi), 'support_zone'=(lo,hi), 'buffer_zone'=(lo,hi)
    """
    levels = levels or {}
    zones  = zones  or {}
    df = df.copy().sort_values("datetime")
    df["datetime"] = pd.to_datetime(df["datetime"])

    fig = go.Figure()
    # Свечи
    fig.add_trace(go.Candlestick(
        x=df["datetime"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price", increasing_line_width=1.2, decreasing_line_width=1.2
    ))
    # VWAP
    vwap = compute_vwap(df)
    fig.add_trace(go.Scatter(x=df["datetime"], y=vwap, mode="lines", name="VWAP", line=dict(width=2)))

    # Линии
    _add_hline(fig, levels.get("main_resistance"), "Main Resistance", "solid", 3)
    _add_hline(fig, levels.get("main_support"),    "Main Support",    "solid", 3)
    _add_hline(fig, levels.get("global_call_vol_level"), "Global Call Vol", "dot", 2)
    _add_hline(fig, levels.get("global_put_vol_level"),  "Global Put Vol",  "dot", 2)
    _add_hline(fig, levels.get("max_neg_gex"), "Max Neg GEX", "dash", 2)
    _add_hline(fig, levels.get("max_pos_gex"), "Max Pos GEX", "dash", 2)
    _add_hline(fig, levels.get("max_ag"),      "Max AG",      "dash", 2)
    _add_hline(fig, levels.get("g_flip"),      "G-Flip Zone", "dash", 2)

    # Зоны
    if zones.get("resistance_zone"): _add_zone(fig, *zones["resistance_zone"], "Resistance Zone", "#3DAF6A")
    if zones.get("support_zone"):    _add_zone(fig, *zones["support_zone"],    "Support Zone",    "#FF6A3D")
    if zones.get("buffer_zone"):     _add_zone(fig, *zones["buffer_zone"],     "Buffer Zone",     "#3066BE")

    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Time", yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02),
        margin=dict(l=40, r=20, t=50, b=40), hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig
