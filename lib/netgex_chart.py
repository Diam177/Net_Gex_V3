
# -*- coding: utf-8 -*-
"""
netgex_chart.py — чарт Net GEX с возможностью поверх наносить Put OI / Call OI
(по требованию пользователя — строго без изменений остальной логики).

Функция:
    render_netgex_bars(df_final, ticker, spot=None, toggle_key=None)

Требования:
    - df_final содержит столбцы со страйком и Net GEX; названия столбцов
      могут отличаться — модуль подберёт первый подходящий из списка.
    - Для Put/Call OI используются столбцы put_oi / call_oi (если их нет —
      соответствующий тумблер ничего не нарисует).

Правки относительно базовой версии:
    • Добавлены тумблеры Put OI / Call OI (по умолчанию выключены).
    • Линии и маркеры:
        Put OI — бордовый  (#800000)
        Call OI — зелёный   (#00A000)
    • Эти ряды рисуются по правой оси y2 с заголовком "Other parameters".
    • НИКАКИХ других изменений поведения графика.
"""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------- helpers ---------------------------------

def _pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Вернёт имя первого существующего столбца из candidates или None."""
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def _prepare_series(df: pd.DataFrame,
                    k_col: str,
                    y_col: str) -> pd.DataFrame:
    """Агрегирует по страйку и сортирует по возрастанию страйка."""
    out = (
        df[[k_col, y_col]]
        .copy()
        .dropna()
    )
    # приведение типов страйка
    out[k_col] = pd.to_numeric(out[k_col], errors="coerce")
    out = out.dropna(subset=[k_col])
    out = (
        out.groupby(k_col, as_index=False)[y_col]
           .sum()
           .sort_values(k_col, kind="mergesort")
    )
    return out


# ----------------------------- main API ---------------------------------

def render_netgex_bars(
    df_final: pd.DataFrame,
    ticker: str,
    spot: Optional[float] = None,
    toggle_key: Optional[str] = None,
):
    """Рендерит бар‑чарт Net GEX и (по тумблерам) линии Put OI / Call OI справа.
    НИЧЕГО другого в макете не меняет.
    """
    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для отображения.")
        return

    # Колонки
    k_col = _pick_col(df_final, ["K", "strike", "Strike"])
    ng_col = _pick_col(df_final, ["NetGEX_1pct_M", "NetGEX_1pct", "net_gex", "NetGEX"])
    call_oi_col = _pick_col(df_final, ["call_oi", "Call OI", "call_open_interest", "callOI"])
    put_oi_col  = _pick_col(df_final, ["put_oi",  "Put OI",  "put_open_interest",  "putOI"])

    if k_col is None or ng_col is None:
        st.error("В таблице нет обязательных столбцов для графика (K / NetGEX).")
        return

    # Подготовка Net GEX (левый Y)
    ng_df = _prepare_series(df_final, k_col, ng_col)
    Ks = ng_df[k_col].astype(float).tolist()
    Y  = ng_df[ng_col].astype(float).tolist()

    # Базовый бар‑чарт Net GEX (цвета оставляем как есть — единый цвет,
    # чтобы не вносить несанкционированных изменений стиля).
    fig = go.Figure()
    fig.add_bar(
        x=Ks,
        y=Y,
        name="Net GEX",
        marker=dict(line=dict(width=0)),
        hovertemplate="Strike: %{x}<br>Net GEX: %{y:.0f}<extra></extra>",
    )

    # --------------------- тумблеры справа ---------------------
    # Стабильные ключи, чтобы не пересекаться с другими toggles
    key_prefix = toggle_key if toggle_key else f"netgex:{ticker}"
    show_put_oi  = st.toggle("Put OI",  value=False, key=f"{key_prefix}:put_oi")
    show_call_oi = st.toggle("Call OI", value=False, key=f"{key_prefix}:call_oi")

    any_right = False

    # Put OI — бордовый, правая ось
    if show_put_oi and put_oi_col is not None:
        put_df = _prepare_series(df_final, k_col, put_oi_col)
        # выравниваем по тем же страйкам (если где-то нет значений — 0)
        put_df = put_df.set_index(k_col).reindex(Ks).fillna(0.0).reset_index()
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

    # Call OI — зелёный, правая ось
    if show_call_oi and call_oi_col is not None:
        call_df = _prepare_series(df_final, k_col, call_oi_col)
        call_df = call_df.set_index(k_col).reindex(Ks).fillna(0.0).reset_index()
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

    # Правая ось — только если есть что рисовать
    if any_right:
        fig.update_layout(
            yaxis2=dict(
                title="Other parameters",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
                rangemode="tozero",
                exponentformat="none",
            )
        )

    # Оси и базовый лайаут — максимально нейтрально,
    # чтобы не менять существующий вид графика
    fig.update_layout(
        xaxis=dict(
            title=None,
            type="category",
            categoryorder="array",
            categoryarray=Ks,
            tickmode="array",
            tickvals=Ks,
        ),
        yaxis=dict(title="Net GEX"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=10, b=10),
    )

    # Выводим без панели управления и без интерактива — как было
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displayModeBar": False, "staticPlot": True})
