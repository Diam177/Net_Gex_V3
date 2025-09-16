# -*- coding: utf-8 -*-
"""
netgex_chart.py — бар‑чарт Net GEX для главной страницы (Streamlit + Plotly).

Функция:
    render_netgex_bars(df_final, ticker, spot=None, toggle_key=None)

Особенности:
- ВСЕ страйки на оси X без «пустых» промежутков.
- Ховер: Strike, Call OI, Put OI, Call Volume, Put Volume, Net GEX (M$).
- Взаимодействия (зум/пан/скролл/даблклик) отключены; ховер остаётся.
- Фон графика прозрачный.
- ОДНА аннотация с тикером — в верхнем левом углу внутри области графика, сразу справа от подписей оси Y.
- Легенда Net GEX — в правом верхнем углу.
"""

from __future__ import annotations
from typing import Optional

import numpy as _np
import pandas as _pd
import streamlit as st
import plotly.graph_objects as go

def _compute_gamma_flip_from_table(df_final, y_col: str, spot: float | None) -> float | None:
    """
    Вычисляет уровень *G-Flip* (страйк K*, где Net GEX(K) ≈ 0) по финальной таблице.

    Методика (приближённая, но строго определённая):
    - Рассматриваем дискретную функцию y(K) = NetGEX(K) (в млн $ / 1%), агрегированную по страйкам.
    - Строим кусочно‑линейную интерполяцию между соседними страйками.
    - Ищем корень y(K*) = 0 в интервалах, где знаки соседних значений отличаются.
    - Если корней несколько, выбираем тот, который ближе всего к текущей цене spot (если известна),
      иначе — ближайший к медиане диапазона страйков.

    Формула нулевой точки на отрезке [K0, K1] с y0 = y(K0), y1 = y(K1):
        K* = K0 - y0 * (K1 - K0) / (y1 - y0)        (линейная интерполяция)

    Замечание по "науке":
    Строгое определение гамма‑флипа — корень агрегированной функции NetGamma(S) по цене БА.
    При отсутствии сырых серий (IV, τ, Δ, Γ) мы используем устойчивую proxy‑оценку
    через NetGEX(K) по страйкам вокруг ATM. На практике этот подход совпадает
    с визуальным нулевым пересечением Net GEX‑баров у ATM и стабильно работает
    для интрадей‑индикации.
    """
    import numpy as _np
    import pandas as _pd

    if df_final is None or len(df_final) == 0 or y_col not in df_final.columns or "K" not in df_final.columns:
        return None

    base = df_final.copy()
    base["K"] = _pd.to_numeric(base["K"], errors="coerce")
    base[y_col] = _pd.to_numeric(base[y_col], errors="coerce")
    base = base.dropna(subset=["K", y_col])
    if base.empty:
        return None

    g = base.groupby("K", as_index=False).agg({y_col: "sum"}).sort_values("K").reset_index(drop=True)
    Ks = g["K"].to_numpy(dtype=float)
    Ys = g[y_col].to_numpy(dtype=float)

    # Вычисляем G-Flip по финальной таблице
    g_flip = _compute_gamma_flip_from_table(base, y_col, spot)

    n = len(Ks)
    if n < 2:
        return None

    # Индексы, где есть смена знака
    sign = _np.sign(Ys)
    zero_idx = _np.where(Ys == 0.0)[0]
    candidates = []
    for zi in zero_idx:
        candidates.append(float(Ks[zi]))

    prod = sign[:-1] * sign[1:]
    change_idx = _np.where(prod < 0)[0]
    for i in change_idx:
        K0, K1 = Ks[i], Ks[i+1]
        y0, y1 = Ys[i], Ys[i+1]
        if y1 == y0:
            continue
        K_star = K0 - y0 * (K1 - K0) / (y1 - y0)
        if (K_star >= min(K0, K1)) and (K_star <= max(K0, K1)):
            candidates.append(float(K_star))

    if not candidates:
        return None

    if spot is not None and _np.isfinite(spot):
        diffs = _np.abs(_np.array(candidates, dtype=float) - float(spot))
        j = int(_np.argmin(diffs))
        return float(candidates[j])
    else:
        mid = 0.5 * (float(Ks[0]) + float(Ks[-1]))
        diffs = _np.abs(_np.array(candidates, dtype=float) - mid)
        j = int(_np.argmin(diffs))
        return float(candidates[j])


# Цвета
COLOR_NEG = '#D9493A'    # отрицательные столбцы
COLOR_POS = '#60A5E7'    # положительные столбцы
COLOR_PRICE = '#E4A339'  # линия цены

def render_netgex_bars(
    df_final: _pd.DataFrame,
    ticker: str,
    spot: Optional[float] = None,
    toggle_key: Optional[str] = None,
) -> None:
    if df_final is None or len(df_final) == 0:
        st.info("Нет данных для графика Net GEX.")
        return
    if "K" not in df_final.columns:
        st.warning("В финальной таблице отсутствует столбец 'K'.")
        return

    # Выбор метрики Net GEX в млн $ / 1%
    if "NetGEX_1pct_M" in df_final.columns:
        y_col = "NetGEX_1pct_M"
    elif "NetGEX_1pct" in df_final.columns:
        df_final = df_final.copy()
        df_final["NetGEX_1pct_M"] = _pd.to_numeric(df_final["NetGEX_1pct"], errors="coerce") / 1e6
        y_col = "NetGEX_1pct_M"
    else:
        st.warning("Нет столбцов NetGEX_1pct_M / NetGEX_1pct — нечего рисовать.")
        return

    # Цена БА
    if spot is None and "S" in df_final.columns and df_final["S"].notna().any():
        spot = float(_pd.to_numeric(df_final["S"], errors="coerce").dropna().iloc[0])

    # Тумблер
    show = st.toggle("Net GEX", value=True, key=(toggle_key or f"netgex_toggle_{ticker}"))
    show_gflip = st.toggle("G-Flip", value=True, key=(toggle_key or f"gflip_toggle_{ticker}"))
    if not show:
        return

    # Приведение типов и агрегаты
    base = df_final.copy()
    base["K"] = _pd.to_numeric(base["K"], errors="coerce")
    base[y_col] = _pd.to_numeric(base[y_col], errors="coerce")

    for c in ("call_oi", "put_oi", "call_vol", "put_vol"):
        if c in base.columns:
            base[c] = _pd.to_numeric(base[c], errors="coerce")
        else:
            base[c] = 0.0

    g = (base
         .dropna(subset=["K"])
         .groupby("K", as_index=False)
         .agg({y_col: "sum",
               "call_oi": "sum",
               "put_oi": "sum",
               "call_vol": "sum",
               "put_vol": "sum"})
         .sort_values("K")
         .reset_index(drop=True))

    Ks = g["K"].to_numpy(dtype=float)
    Ys = g[y_col].to_numpy(dtype=float)

    # Вычисляем G-Flip по финальной таблице
    g_flip = _compute_gamma_flip_from_table(base, y_col, spot)

    # Ось X — последовательные индексы без пропусков, подписи — реальные страйки
    x_idx = _np.arange(len(Ks), dtype=float)
    bar_width = 0.9
    colors = _np.where(Ys >= 0.0, COLOR_POS, COLOR_NEG)

    # customdata для ховера
    k_str = _np.array([str(int(k)) if float(k).is_integer() else f"{k:.2f}" for k in Ks], dtype=object)
    custom = _np.column_stack([
        k_str,
        g["call_oi"].to_numpy(dtype=float),
        g["put_oi"].to_numpy(dtype=float),
        g["call_vol"].to_numpy(dtype=float),
        g["put_vol"].to_numpy(dtype=float),
    ])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_idx,
        y=Ys,
        name="Net GEX",
        marker_color=colors.tolist(),
        width=bar_width,
        customdata=custom,
        hovertemplate=(
            "<b>Strike</b>: %{customdata[0]}<br>"
            "<b>Call OI</b>: %{customdata[1]:,.0f}<br>"
            "<b>Put OI</b>: %{customdata[2]:,.0f}<br>"
            "<b>Call Vol</b>: %{customdata[3]:,.0f}<br>"
            "<b>Put Vol</b>: %{customdata[4]:,.0f}<br>"
            "<b>Net GEX</b>: %{y:.3f}M"
            "<extra></extra>"
        ),
    ))

    # Вертикальная линия текущей цены (интерполяция между ближайшими страйками)
    if spot is not None and _np.isfinite(spot) and len(Ks) > 0:
        try:
            if len(Ks) >= 2:
                j = int(_np.searchsorted(Ks, spot))
                if j <= 0:
                    x_price = 0.0
                elif j >= len(Ks):
                    x_price = float(len(Ks) - 1)
                else:
                    k0, k1 = Ks[j - 1], Ks[j]
                    frac = 0.0 if (k1 - k0) == 0 else (spot - k0) / (k1 - k0)
                    x_price = (j - 1) + float(_np.clip(frac, 0.0, 1.0))
            else:
                x_price = 0.0
        except Exception:
            x_price = 0.0

        y0 = min(0.0, float(_np.nanmin(Ys))) * 1.05
        y1 = max(0.0, float(_np.nanmax(Ys))) * 1.05
        fig.add_shape(type="line", x0=x_price, x1=x_price, y0=y0, y1=y1,
                      line=dict(color=COLOR_PRICE, width=2))
        fig.add_annotation(x=x_price, y=y1, text=f"Price: {spot:.2f}",
                           showarrow=False, yshift=8,
                           font=dict(color=COLOR_PRICE, size=12), xanchor="center")

    # Компоновка
    fig.update_layout(
        template=None,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(font=dict(size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(
            x=1, y=1, xanchor='right', yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        dragmode=False,
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=x_idx.tolist(),
            ticktext=k_str.tolist(),
            tickangle=0,
            tickfont=dict(size=10),
            showgrid=False,
            showline=False,
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            title="Net GEX",
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
    )

    # --- ЕДИНСТВЕННАЯ аннотация с тикером ---
    if isinstance(ticker, str) and ticker.strip():
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0, y=1,
            xanchor='left', yanchor='top',
            xshift=12,     # вправо от левого края области графика
            yshift=6,      # немного ниже верхней кромки
            text=ticker.upper(),
            showarrow=False,
            align='left',
            font=dict(size=14)
        )

    
    # --- G-Flip marker (optional, dashed) ---
    try:
        if 'show_gflip' in locals() and show_gflip and (g_flip is not None) and (len(Ks) > 0):
            import numpy as _np
            j = int(_np.searchsorted(Ks, float(g_flip)))
            if j <= 0:
                x_g = 0.0
                k_snap = float(Ks[0])
            elif j >= len(Ks):
                x_g = float(len(Ks) - 1)
                k_snap = float(Ks[-1])
            else:
                k0, k1 = Ks[j - 1], Ks[j]
                frac = 0.0 if (k1 - k0) == 0 else (float(g_flip) - k0) / (k1 - k0)
                x_g = (j - 1) + float(_np.clip(frac, 0.0, 1.0))
                k_snap = float(k0 if abs(float(g_flip)-k0) <= abs(float(g_flip)-k1) else k1)

            fig.add_shape(type="line", x0=x_g, x1=x_g, y0=0, y1=1, xref="x", yref="paper",
                          line=dict(width=1, color="#AAAAAA", dash="dash"), layer="above")
            fig.add_annotation(x=x_g, xref="x", y=1, yref="paper",
                               text=f"G-Flip: {k_snap:g}", showarrow=False,
                               yshift=6, font=dict(size=12, color="#AAAAAA"), xanchor="center")
    except Exception:
        pass

    # Автомасштаб
    fig.update_yaxes(autorange=True)
    fig.update_xaxes(autorange=True)

    # Прозрачность контейнера Plotly
    st.markdown(
        """
        <style>
        div[data-testid="stPlotlyChart"] .js-plotly-plot,
        div[data-testid="stPlotlyChart"] .plot-container,
        div[data-testid="stPlotlyChart"] .svg-container {
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Рендер
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={
            'displayModeBar': False,
            'scrollZoom': False,
            'doubleClick': False
        }
    )
