
# -*- coding: utf-8 -*-
"""
netgex_chart.py — бар‑чарт Net GEX для главной страницы.

Функция render_netgex_bars(df_final, ticker, spot=None, toggle_key=None):
  • df_final: DataFrame по одной экспирации (или агрегированной multi‑финалке)
  • ticker: строка для подписи в левом верхнем углу
  • spot: текущая цена БА; если None — берётся из df_final['S']
  • toggle_key: уникальный ключ для st.toggle

Зависимости: plotly>=5, pandas, streamlit
"""

from __future__ import annotations
from typing import Optional, Sequence
import pandas as _pd
import streamlit as st

import numpy as _np
def _compute_gamma_flip_from_table(df_final, y_col: str, spot: float | None) -> float | None:
    """
    G-Flip (K*): страйк, где агрегированный Net GEX(K) меняет знак.
    Метод: кусочно-линейная интерполяция между соседними страйками и поиск корня.
    K* = K0 - y0*(K1-K0)/(y1-y0)
    Если несколько корней — выбираем ближайший к spot (если известен), иначе к середине диапазона.
    """
    import pandas as _pd
    if df_final is None or len(df_final) == 0 or "K" not in df_final.columns or y_col not in df_final.columns:
        return None
    base = df_final.copy()
    base["K"] = _pd.to_numeric(base["K"], errors="coerce")
    base[y_col] = _pd.to_numeric(base[y_col], errors="coerce")
    base = base.dropna(subset=["K", y_col])
    if base.empty:
        return None
    g = base.groupby("K", as_index=False)[y_col].sum().sort_values("K").reset_index(drop=True)
    Ks = g["K"].to_numpy(dtype=float)
    Ys = g[y_col].to_numpy(dtype=float)
    if len(Ks) < 2:
        return None
    # прямые нули и смена знака
    cand = [float(Ks[i]) for i,v in enumerate(Ys) if v == 0.0]
    sign = _np.sign(Ys)
    idx = _np.where(sign[:-1]*sign[1:] < 0)[0]
    for i in idx:
        K0, K1 = Ks[i], Ks[i+1]
        y0, y1 = Ys[i], Ys[i+1]
        if y1 == y0: 
            continue
        Kstar = K0 - y0*(K1-K0)/(y1-y0)
        if min(K0,K1) <= Kstar <= max(K0,K1):
            cand.append(float(Kstar))
    if not cand:
        return None
    if spot is not None and _np.isfinite(spot):
        j = int(_np.argmin(_np.abs(_np.array(cand) - float(spot))))
        return float(cand[j])
    mid = 0.5*(float(Ks[0]) + float(Ks[-1]))
    j = int(_np.argmin(_np.abs(_np.array(cand) - mid)))
    return float(cand[j])


try:
    import plotly.graph_objects as go
except Exception as e:
    raise RuntimeError("Требуется пакет 'plotly' (plotly>=5.22.0)") from e

# --- Цвета/оформление ---
COLOR_NEG = '#D9493A'    # красный
COLOR_POS = '#60A5E7'    # бирюзовый
COLOR_PRICE = '#E4A339'  # оранжевая линия цены
BG_COLOR = '#111111'
FG_COLOR = '#e0e0e0'
GRID_COLOR = 'rgba(255,255,255,0.10)'

def _to_num(a: Sequence) -> _np.ndarray:
    return _np.array(_pd.to_numeric(a, errors='coerce'), dtype=float)

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

    # Выбор колонки Net GEX (в млн $/1% приоритетно)
    if "NetGEX_1pct_M" in df_final.columns:
        y_col = "NetGEX_1pct_M"
    elif "NetGEX_1pct" in df_final.columns:
        df_final = df_final.copy()
        df_final["NetGEX_1pct_M"] = df_final["NetGEX_1pct"] / 1e6
        y_col = "NetGEX_1pct_M"
    else:
        st.warning("Нет столбцов NetGEX_1pct_M / NetGEX_1pct — нечего рисовать.")
        return

    # spot
    if spot is None and "S" in df_final.columns and df_final["S"].notna().any():
        spot = float(df_final["S"].dropna().iloc[0])

    # Тумблер
    show = st.toggle("Net GEX", value=True, key=(toggle_key or f"netgex_toggle_{ticker}"))
    show_gflip = st.toggle("G-Flip", value=True, key=(f"{toggle_key}__gflip" if toggle_key else f"gflip_toggle_{ticker}"))
    if not show:
        return

    # Подготовка данных и ширины бара
    df = df_final[["K", y_col]].dropna().copy()
    df["K"] = _pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["K"]).sort_values("K").reset_index(drop=True)
    df = df.groupby("K", as_index=False)[y_col].sum()

    Ks = df["K"].to_numpy(dtype=float)
    Ys = df[y_col].to_numpy(dtype=float)
    g_flip = _compute_gamma_flip_from_table(df, y_col, spot)
    
    # Последовательные позиции без "пустых" промежутков между страйками
    x_idx = _np.arange(len(Ks), dtype=float)
    bar_width = 0.9
    colors = _np.where(Ys >= 0.0, COLOR_POS, COLOR_NEG)
    
    # Фигура
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_idx,
        y=Ys,
        name="Net GEX (M$ / 1%)",
        marker_color=colors,
        width=bar_width,
        hovertemplate="K=%{x}<br>Net GEX=%{y:.3f}M<extra></extra>",
    ))

    # Вертикальная линия цены
    
    if spot is not None and _np.isfinite(spot):
    # интерполируем позицию цены между ближайшими страйками на оси индексов
        try:
            if len(Ks) >= 2:
                j = int(_np.searchsorted(Ks, spot))
                if j <= 0:
                    x_price = 0.0
                elif j >= len(Ks):
                    x_price = float(len(Ks) - 1)
                else:
                    k0, k1 = Ks[j-1], Ks[j]
                    frac = 0.0 if (k1 - k0) == 0 else (spot - k0) / (k1 - k0)
                    x_price = (j - 1) + float(_np.clip(frac, 0.0, 1.0))
            else:
                x_price = 0.0
        except Exception:
            x_price = 0.0
        
        y0 = min(0.0, float(_np.nanmin(Ys))) * 1.05
        y1 = max(0.0, float(_np.nanmax(Ys))) * 1.05
        fig.add_shape(type="line", x0=x_price, x1=x_price, y0=y0, y1=y1, line=dict(color=COLOR_PRICE, width=2))
        fig.add_annotation(x=x_price, y=y1, text=f"Price: {spot:.2f}", showarrow=False, yshift=8,
                           font=dict(color=COLOR_PRICE, size=12), xanchor="center")
    
    # Тикер
    if ticker:
        fig.add_annotation(xref="paper", yref="paper", x=0.0, y=1.12, text=str(ticker),
                           showarrow=False, font=dict(size=16, color=FG_COLOR), xanchor="left", yanchor="bottom")

    # Подписи страйков: все значения, горизонтально, шрифт 10
    tick_vals = x_idx.tolist()
    tick_text = [str(int(k)) if float(k).is_integer() else f"{k:.2f}" for k in Ks]

    fig.update_layout(

        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        dragmode=False,
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=0,
            tickfont=dict(size=10),   # <<< фиксированный размер 10
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Net GEX",
            showgrid=False,
            zeroline=False,
        ),
    )

    
    
    
    # --- G-Flip marker (optional) ---
    try:
        if 'show_gflip' in locals() and show_gflip and (g_flip is not None) and (len(Ks) > 0):
            # Всегда привязываем линию к центру ближайшего страйка (визуальная консистентность с подписью)
            k_arr = Ks.astype(float)
            g_val = float(g_flip)
            import numpy as _np
            snap_idx = int(_np.argmin(_np.abs(k_arr - g_val)))
            x_g = float(snap_idx)
            k_snap = float(k_arr[snap_idx])

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

    # Статичный график без зума/панорамы и без панели управления
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={'displayModeBar': False, 'staticPlot': True})
