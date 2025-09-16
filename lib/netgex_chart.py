
# -*- coding: utf-8 -*-
"""
netgex_chart.py — бар‑чарт Net GEX + опциональные линии Put OI / Call OI.
СТРОГО минимальные правки: добавлены два тумблера и правый Y (Other parameters).
Существующие Net GEX и G‑Flip НЕ трогаются.
"""

from __future__ import annotations
from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------- helpers ---------------------------------

def _pick_col(df, candidates: Sequence[str]) -> Optional[str]:
    """Вернёт имя первого существующего столбца из candidates или None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def _prepare_series(df, k_col: str, y_col: str):
    """Агрегирует по страйку и сортирует по возрастанию страйка."""
    out = df[[k_col, y_col]].copy().dropna()
    out[k_col] = pd.to_numeric(out[k_col], errors="coerce")
    out = out.dropna(subset=[k_col])
    out = out.groupby(k_col, as_index=False)[y_col].sum().sort_values(k_col, kind="mergesort")
    return out


def _compute_gamma_flip_from_table(sorted_k, sorted_y, spot: Optional[float]) -> Optional[float]:
    """Линейная интерполяция нуля Net GEX между соседними страйками."""
    if len(sorted_k) < 2:
        return None
    roots = []
    for i in range(len(sorted_k) - 1):
        y0, y1 = sorted_y[i], sorted_y[i + 1]
        if y0 == 0:
            roots.append(float(sorted_k[i]))
            continue
        if y0 * y1 < 0:
            k0, k1 = float(sorted_k[i]), float(sorted_k[i + 1])
            # K* = K0 - y0*(K1-K0)/(y1-y0)
            try:
                k_star = k0 - y0 * (k1 - k0) / (y1 - y0)
                roots.append(float(k_star))
            except ZeroDivisionError:
                continue
    if not roots:
        return None
    if spot is None:
        # ближе к середине диапазона
        mid = 0.5 * (float(sorted_k[0]) + float(sorted_k[-1]))
        return min(roots, key=lambda r: abs(r - mid))
    return min(roots, key=lambda r: abs(r - float(spot)))


# ----------------------------- main API ---------------------------------

def render_netgex_bars(
    df_final,
    ticker: str,
    spot: Optional[float] = None,
    toggle_key: Optional[str] = None,
):
    """
    Отрисовывает базовый Net GEX бар‑чарт + (по тумблерам) линии Put OI / Call OI справа.
    Существующие элементы (Net GEX, G‑Flip) не меняем.
    """
    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для отображения.")
        return

    # ---- колонки
    k_col = _pick_col(df_final, ["K", "strike", "Strike"])
    ng_col = _pick_col(df_final, ["NetGEX_1pct_M", "NetGEX_1pct", "net_gex", "NetGEX"])
    call_oi_col = _pick_col(df_final, ["call_oi", "Call OI", "call_open_interest", "callOI"])
    put_oi_col  = _pick_col(df_final, ["put_oi",  "Put OI",  "put_open_interest",  "putOI"])

    if k_col is None or ng_col is None:
        st.error("В таблице нет обязательных столбцов для графика (K / Net GEX).")
        return

    # ---- подготовка базовой серии Net GEX
    ng_df = _prepare_series(df_final, k_col, ng_col)
    Ks = ng_df[k_col].astype(float).tolist()
    Y  = ng_df[ng_col].astype(float).tolist()

    # ---- существующие тумблеры (сохраняем): Net GEX / G‑Flip
    key_prefix = toggle_key if toggle_key else f"netgex:{ticker}"
    show_bars  = st.toggle("Net GEX", value=True,  key=f"{key_prefix}:netgex_visible")
    show_gflip = st.toggle("G-Flip",  value=False, key=f"{key_prefix}:gflip_visible")

    fig = go.Figure()

    if show_bars:
        fig.add_bar(
            x=Ks,
            y=Y,
            name="Net GEX",
            hovertemplate="Strike: %{x}<br>Net GEX: %{y:.0f}<extra></extra>",
        )

    # ---- новые тумблеры справа (строго по ТЗ)
    show_put_oi  = st.toggle("Put OI",  value=False, key=f"{key_prefix}:put_oi")
    show_call_oi = st.toggle("Call OI", value=False, key=f"{key_prefix}:call_oi")

    any_right = False

    if show_put_oi and put_oi_col is not None:
        put_df = _prepare_series(df_final, k_col, put_oi_col).set_index(k_col).reindex(Ks).fillna(0.0).reset_index()
        fig.add_trace(go.Scatter(
            x=Ks,
            y=put_df[put_oi_col].astype(float),
            mode="lines+markers",
            name="Put OI",
            marker=dict(size=6, symbol="circle", color="#800000"),
            line=dict(width=1.6, color="#800000"),
            yaxis="y2",
            hovertemplate="Strike: %{x}<br>Put OI: %{y:.0f}<extra></extra>",
        ))
        any_right = True

    if show_call_oi and call_oi_col is not None:
        call_df = _prepare_series(df_final, k_col, call_oi_col).set_index(k_col).reindex(Ks).fillna(0.0).reset_index()
        fig.add_trace(go.Scatter(
            x=Ks,
            y=call_df[call_oi_col].astype(float),
            mode="lines+markers",
            name="Call OI",
            marker=dict(size=6, symbol="circle", color="#00A000"),
            line=dict(width=1.6, color="#00A000"),
            yaxis="y2",
            hovertemplate="Strike: %{x}<br>Call OI: %{y:.0f}<extra></extra>",
        ))
        any_right = True

    if any_right:
        fig.update_layout(
            yaxis2=dict(
                title="Other parameters",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                rangemode="tozero",
            )
        )

    # ---- G‑Flip (вертикальная линия) — оставляем, как есть, только если просят
    if show_gflip:
        # попытка вычислить из Net GEX
        gflip = _compute_gamma_flip_from_table(Ks, Y, spot)
        if gflip is not None:
            fig.add_vline(x=gflip, line_width=1.5, line_dash="dash", line_color="#cccccc")
            fig.add_annotation(
                x=gflip, y=max(Y) if Y else 0,
                xref="x", yref="y",
                showarrow=False, text="G-Flip", yanchor="bottom",
                bgcolor="rgba(0,0,0,0)", font=dict(size=12),
            )

    # ---- оси/легенда — не меняем внешний вид, только гарантируем корректность
    fig.update_layout(
        yaxis=dict(title="Net GEX"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)
