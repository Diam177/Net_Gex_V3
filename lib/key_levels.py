
# -*- coding: utf-8 -*-
"""
Key Levels chart for GammaStrat.

Renders vertical reference lines for important price/strike levels
derived from the final table (df_final). Designed to be side‑effect
friendly in Streamlit (renders with st.plotly_chart) but also returns
a Plotly Figure when Streamlit is unavailable.

Public API
----------
render_key_levels(df_final: pandas.DataFrame, S: float | None = None) -> None | plotly.graph_objects.Figure
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# Optional Streamlit; keep import lazy-safe
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

import plotly.graph_objects as go


def _get_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col in df.columns:
        s = df[col]
        if s.notna().any():
            return s
    return None


def _max_at(df: pd.DataFrame, col: str) -> Optional[float]:
    s = _get_series(df, col)
    if s is None:
        return None
    try:
        idx = s.astype(float).idxmax()
        return float(df.loc[idx, 'K'])
    except Exception:
        return None


def _min_at(df: pd.DataFrame, col: str) -> Optional[float]:
    s = _get_series(df, col)
    if s is None:
        return None
    try:
        idx = s.astype(float).idxmin()
        return float(df.loc[idx, 'K'])
    except Exception:
        return None


def _gflip_from_zero_cross(df: pd.DataFrame) -> Optional[float]:
    """
    Estimate gamma flip as zero crossing of NetGEX_1pct across strikes.
    Takes the crossing closest to zero magnitude.
    """
    if 'K' not in df.columns or 'NetGEX_1pct' not in df.columns:
        return None
    d = df[['K', 'NetGEX_1pct']].dropna().sort_values('K')
    if d.empty:
        return None
    y = d['NetGEX_1pct'].to_numpy(dtype=float)
    x = d['K'].to_numpy(dtype=float)
    s = np.sign(y)
    changes = np.where(np.diff(s) != 0)[0]
    if changes.size == 0:
        # Fallback: nearest to zero
        j = int(np.argmin(np.abs(y)))
        return float(x[j])
    # Choose the sign-change segment with minimal |y| near boundary
    best_x = None
    best_val = float('inf')
    for i in changes:
        # Linear interpolation between (x[i], y[i]) and (x[i+1], y[i+1])
        x0, x1, y0, y1 = x[i], x[i+1], y[i], y[i+1]
        if y1 == y0:
            xz = (x0 + x1) / 2.0
        else:
            xz = x0 - y0 * (x1 - x0) / (y1 - y0)
        val = min(abs(y0), abs(y1))
        if val < best_val:
            best_val = val
            best_x = float(xz)
    return best_x


def _vwap_strike(df: pd.DataFrame) -> Optional[float]:
    """
    Approximate 'VWAP of strikes' using available weights.
    Preference: AG_1pct -> |NetGEX_1pct| -> (call_oi+put_oi) -> None
    """
    if 'K' not in df.columns:
        return None
    k = df['K'].astype(float)
    if 'AG_1pct' in df.columns and df['AG_1pct'].notna().any():
        w = df['AG_1pct'].abs().astype(float)
    elif 'NetGEX_1pct' in df.columns and df['NetGEX_1pct'].notna().any():
        w = df['NetGEX_1pct'].abs().astype(float)
    elif {'call_oi', 'put_oi'}.issubset(df.columns):
        w = (df['call_oi'].fillna(0).astype(float) + df['put_oi'].fillna(0).astype(float))
    else:
        return None
    if not (w > 0).any():
        return None
    vw = float((k * w).sum() / w.sum())
    return vw


def _collect_levels(df: pd.DataFrame, S: Optional[float]) -> Dict[str, float]:
    levels: Dict[str, float] = {}

    # Price
    if S is None:
        # Try from column S (single value per row)
        if 'S' in df.columns and df['S'].notna().any():
            try:
                S = float(df['S'].dropna().iloc[0])
            except Exception:
                S = None
    if S is not None and np.isfinite(S):
        levels['Price'] = float(S)

    # G-Flip
    for cand in ('G_Flip', 'g_flip', 'GFlip', 'gflip'):
        if cand in df.columns and df[cand].notna().any():
            try:
                levels['G-Flip'] = float(df[cand].dropna().iloc[0])
                break
            except Exception:
                pass
    if 'G-Flip' not in levels:
        gf = _gflip_from_zero_cross(df)
        if gf is not None:
            levels['G-Flip'] = gf

    # VWAP-like strike
    vwap = _vwap_strike(df)
    if vwap is not None:
        levels['VWAP'] = vwap

    # Net GEX extrema
    mx = _max_at(df, 'NetGEX_1pct')
    mn = _min_at(df, 'NetGEX_1pct')
    if mn is not None:
        levels['Max Neg GEX'] = mn
    if mx is not None:
        levels['Max Pos GEX'] = mx

    # OI & Volume
    m_put_oi = _max_at(df, 'put_oi')
    if m_put_oi is not None:
        levels['Max Put OI'] = m_put_oi
    m_call_oi = _max_at(df, 'call_oi')
    if m_call_oi is not None:
        levels['Max Call OI'] = m_call_oi
    m_put_vol = _max_at(df, 'put_vol')
    if m_put_vol is not None:
        levels['Max Put Volume'] = m_put_vol
    m_call_vol = _max_at(df, 'call_vol')
    if m_call_vol is not None:
        levels['Max Call Volume'] = m_call_vol

    # AG & PZ
    ag = _max_at(df, 'AG_1pct')
    if ag is not None:
        levels['AG'] = ag
    pz = _max_at(df, 'PZ')
    if pz is not None:
        levels['PZ'] = pz

    return levels


def _x_range_padding(xmin: float, xmax: float) -> tuple[float, float]:
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return xmin, xmax
    dx = xmax - xmin
    pad = 0.03 * dx if dx > 0 else 1.0
    return xmin - pad, xmax + pad


def render_key_levels(df_final: pd.DataFrame, S: Optional[float] = None):
    """
    Render the Key Levels chart under the main chart.

    If Streamlit is present, renders directly and returns None.
    Otherwise returns a Plotly Figure.
    """
    if df_final is None or getattr(df_final, 'empty', True):
        return None

    df = df_final.copy()
    if 'K' not in df.columns:
        return None
    df['K'] = pd.to_numeric(df['K'], errors='coerce')
    df = df.dropna(subset=['K']).sort_values('K')
    if df.empty:
        return None

    levels = _collect_levels(df, S)

    # Base figure (empty area) with x-axis as strikes
    xmin, xmax = float(df['K'].min()), float(df['K'].max())
    xmin, xmax = _x_range_padding(xmin, xmax)

    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=16, r=16, t=8, b=24),
        height=220,
        xaxis=dict(title='', showgrid=False, zeroline=False, range=[xmin, xmax]),
        yaxis=dict(title='', showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
    )

    # Draw lines + labels
    order = [
        'Price', 'G-Flip', 'VWAP',
        'Max Neg GEX', 'Max Pos GEX',
        'Max Put OI', 'Max Call OI',
        'Max Put Volume', 'Max Call Volume',
        'AG', 'PZ',
    ]
    for name in order:
        if name not in levels:
            continue
        try:
            x = float(levels[name])
        except Exception:
            continue
        # vertical line
        fig.add_shape(type='line', x0=x, x1=x, y0=0.0, y1=0.92, line=dict(width=2))
        # label above
        fig.add_annotation(
            x=x, y=0.96, text=f"{name}: {x:g}",
            showarrow=False, yanchor='bottom', textangle=0
        )

    # Render
    if st is not None:
        try:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            return None
        except Exception:
            # Fallthrough to returning figure on any Streamlit issues
            pass
    return fig
