
# -*- coding: utf-8 -*-
"""
netgex_chart.py — бар‑чарт Net GEX для главной страницы + опциональные оверлеи Put OI / Call OI.

Публичный API:
    render_netgex_bars(df_final, ticker, spot=None, toggle_key=None)

Вход:
    df_final: pandas.DataFrame финальной таблицы по одной экспирации (или агрегированной)
              Ожидаемые колонки (подмножество): ["K","S","NetGEX_1pct_M","NetGEX_1pct","put_oi","call_oi"]
    ticker:   str — подпись графика
    spot:     float|None — цена БА; если None — берём медиану столбца "S"
    toggle_key: str|None — ключ для уникальности st.toggle

Поведение:
    • Базы (Net GEX) — столбики по оси Y (левая шкала).
    • При включении тумблеров "Put OI" / "Call OI" рисуются точки+линии поверх по правой шкале (Other parameters).
    • По умолчанию тумблеры выключены.
    • Никаких изменений в остальной логике: ось X — ровно по страйкам из df, без пропусков; фиксированный шрифт 10.
"""

from __future__ import annotations

from typing import Optional, Iterable, Dict, Any
import math

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit as st


# ------------------------- маленькие утилиты -------------------------

def _coerce_series(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    """Вернёт первый существующий столбец из candidates, иначе None."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None

def _safe_float_median(s: Optional[pd.Series]) -> float:
    if s is None:
        return float("nan")
    s = pd.to_numeric(s, errors="coerce")
    if s.empty:
        return float("nan")
    v = float(s.median())
    return v

def _split_pos_neg(y: pd.Series) -> Dict[str, pd.Series]:
    y = y.fillna(0.0).astype(float)
    neg = y.where(y < 0.0, other=0.0)
    pos = y.where(y > 0.0, other=0.0)
    return {"neg": neg, "pos": pos}

def _fmt_ticks(vals: Iterable[float]) -> Dict[str, Any]:
    """Вернём (tickvals, ticktext) с форматированием без лишних нулей."""
    out_vals = []
    out_txt  = []
    for v in vals:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        out_vals.append(fv)
        iv = int(round(fv))
        if abs(fv - iv) < 1e-8:
            out_txt.append(str(iv))
        else:
            out_txt.append(("{:.2f}".format(fv)).rstrip("0").rstrip("."))
    return {"tickvals": out_vals, "ticktext": out_txt}


# ------------------------- основной рендер -------------------------

def render_netgex_bars(df_final: pd.DataFrame,
                       ticker: str,
                       spot: Optional[float] = None,
                       toggle_key: Optional[str] = None) -> None:
    """
    Рендер бар‑чарта Net GEX + опциональные оверлеи Put OI / Call OI.
    Ничего не возвращает — рисует через st.plotly_chart(...).
    """

    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для отрисовки.")
        return

    df = df_final.copy()

    # ось X — все страйки из финальной таблицы, отсортированные, без пропусков/интерполяций
    if "K" not in df.columns:
        st.error("В финальной таблице отсутствует колонка 'K' (страйк).")
        return
    df = df.sort_values("K", kind="mergesort").reset_index(drop=True)
    x = pd.to_numeric(df["K"], errors="coerce").astype(float).tolist()

    # Базовая серия Net GEX (приоритет *_M)
    y_netgex = _coerce_series(df, ["NetGEX_1pct_M", "NetGEX_1pct"])
    if y_netgex is None:
        st.error("В финальной таблице отсутствуют колонки NetGEX_1pct_M/NetGEX_1pct.")
        return
    y_netgex = pd.to_numeric(y_netgex, errors="coerce").fillna(0.0)

    # Правые оверлеи: Put OI / Call OI
    y_put_oi  = _coerce_series(df, ["put_oi", "Put OI", "putOI"])
    y_call_oi = _coerce_series(df, ["call_oi", "Call OI", "callOI"])
    if y_put_oi is not None:
        y_put_oi = pd.to_numeric(y_put_oi, errors="coerce")
    if y_call_oi is not None:
        y_call_oi = pd.to_numeric(y_call_oi, errors="coerce")

    # --- UI переключатели (по умолчанию выключены) ---
    # Стараемся не трогать остальные элементы UI — добавляем только 2 нужных тумблера
    c1, c2 = st.columns(2)
    t_put_oi  = c1.toggle("Put OI", value=False, key=(f"{toggle_key}:put_oi" if toggle_key else "put_oi"))
    t_call_oi = c2.toggle("Call OI", value=False, key=(f"{toggle_key}:call_oi" if toggle_key else "call_oi"))

    # --- Фигура ---
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=40, t=60, b=50),
        legend=dict(orientation="h", y=1.12, x=0.01),
        bargap=0.05,
    )

    # Net GEX — бары: отрицательные (красные), положительные (голубые)
    split = _split_pos_neg(y_netgex)
    fig.add_bar(
        x=x, y=split["neg"], name="Net GEX (neg)", marker_color="crimson", opacity=0.9, yaxis="y1"
    )
    fig.add_bar(
        x=x, y=split["pos"], name="Net GEX (pos)", marker_color="lightskyblue", opacity=0.9, yaxis="y1"
    )

    # Правая ось появится только если включён хотя бы один оверлей
    add_secondary = (t_put_oi and y_put_oi is not None) or (t_call_oi and y_call_oi is not None)
    if add_secondary:
        fig.update_layout(
            yaxis2=dict(
                title="Other parameters",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

        if t_put_oi and y_put_oi is not None:
            fig.add_scatter(
                x=x, y=y_put_oi, mode="lines+markers",
                name="Put OI", line=dict(width=2, color="maroon"),
                marker=dict(size=6, color="maroon"),
                yaxis="y2"
            )
        if t_call_oi and y_call_oi is not None:
            fig.add_scatter(
                x=x, y=y_call_oi, mode="lines+markers",
                name="Call OI", line=dict(width=2, color="green"),
                marker=dict(size=6, color="green"),
                yaxis="y2"
            )

    # Оформление осей X (все страйки, шрифт 10)
    ticks = _fmt_ticks(x)
    fig.update_xaxes(tickmode="array", tickvals=ticks["tickvals"], ticktext=ticks["ticktext"],
                     fixedrange=True, tickfont=dict(size=10))

    # Автомасштаб обеих осей
    fig.update_yaxes(autorange=True, fixedrange=True)         # левая
    if add_secondary:
        fig.update_yaxes(autorange=True, fixedrange=True, secondary_y=True)

    # Вертикальная линия spot (золотая), если передана/можно извлечь из df["S"]
    spot_val = float(spot) if (spot is not None and math.isfinite(float(spot))) else _safe_float_median(_coerce_series(df, ["S"]))
    if math.isfinite(spot_val):
        fig.add_vline(x=spot_val, line_width=2, line_dash="solid", line_color="#f1c40f")
        fig.add_annotation(x=spot_val, yref="paper", y=1.05,
                           text=f"Spot: {spot_val:.2f}",
                           showarrow=False, font=dict(color="#f1c40f"))

    # Заголовок
    if ticker:
        fig.update_layout(title=dict(text=str(ticker), x=0.02, xanchor="left"))

    # Без лишних контролов
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "displayModeBar": False, "scrollZoom": False, "doubleClick": False, "staticPlot": True}
    )
