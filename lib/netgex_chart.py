# -*- coding: utf-8 -*-
"""
netgex_chart.py — бар‑чарт Net GEX для главной страницы с улучшенными hover-подсказками.
"""

from __future__ import annotations
from typing import Optional, Sequence
import pandas as _pd
import streamlit as st
import numpy as _np

def _compute_gamma_flip_from_table(df_final, y_col: str, spot: float | None) -> float | None:
    """G-Flip (K*): страйк, где агрегированный Net GEX(K) меняет знак."""
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

def _to_num(a: Sequence) -> _np.ndarray:
    return _np.array(_pd.to_numeric(a, errors='coerce'), dtype=float)

def _create_hover_template(series_name: str = "") -> str:
    """Создает единый шаблон hover для всех серий"""
    return (
        "<b style='color:white; font-size:14px'>Strike: %{customdata[0]:.0f}</b><br>" +
        "<span style='font-size:12px'>" +
        "Call OI: <b>%{customdata[1]:,.0f}</b><br>" +
        "Put OI: <b>%{customdata[2]:,.0f}</b><br>" +
        "Call Volume: <b>%{customdata[3]:,.0f}</b><br>" +
        "Put Volume: <b>%{customdata[4]:,.0f}</b><br>" +
        f"{series_name}: <b>" + "%{customdata[5]:,.1f}</b>" +
        "</span><extra></extra>"
    )

def _create_hover_data(Ks, df_final, y_values, series_name=""):
    """Создает customdata для hover"""
    hover_data = {}
    for k in Ks:
        k_data = df_final[df_final["K"] == k]
        if not k_data.empty:
            hover_data[k] = {
                "call_oi": k_data["call_oi"].sum() if "call_oi" in k_data.columns else 0,
                "put_oi": k_data["put_oi"].sum() if "put_oi" in k_data.columns else 0,
                "call_vol": k_data["call_vol"].sum() if "call_vol" in k_data.columns else 0,
                "put_vol": k_data["put_vol"].sum() if "put_vol" in k_data.columns else 0,
            }
        else:
            hover_data[k] = {"call_oi": 0, "put_oi": 0, "call_vol": 0, "put_vol": 0}
    
    customdata_list = []
    for i, k in enumerate(Ks):
        hd = hover_data.get(k, {})
        customdata_list.append([
            k,  # Strike
            hd.get("call_oi", 0),  # Call OI
            hd.get("put_oi", 0),   # Put OI  
            hd.get("call_vol", 0),  # Call Volume
            hd.get("put_vol", 0),   # Put Volume
            y_values[i] if i < len(y_values) else 0  # Series value
        ])
    return customdata_list

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

    # Выбор колонки Net GEX
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

    # --- Toggles: single horizontal row ---
    st.markdown("<style>div[data-testid='column']{padding-left:0px!important;padding-right:2px!important}</style>", unsafe_allow_html=True)
    st.markdown("""<style>div[data-testid="stWidgetLabel"] * {font-size: 0.80rem !important;line-height: 1.1 !important;}
    label:has(> div[data-baseweb="switch"]) * {font-size: 0.80rem !important;line-height: 1.1 !important;}</style>""", unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10, gap="small")
    with col1:
        show = st.toggle("Net GEX", value=True, key=(toggle_key or f"netgex_toggle_{ticker}"))
    with col2:
        show_gflip = st.toggle("G-Flip", value=False, key=(f"{toggle_key}__gflip" if toggle_key else f"gflip_toggle_{ticker}"))
    with col3:
        show_put_oi = st.toggle("Put OI", value=False, key=(f"{toggle_key}__put_oi" if toggle_key else f"putoi_toggle_{ticker}"))
    with col4:
        show_call_oi = st.toggle("Call OI", value=False, key=(f"{toggle_key}__call_oi" if toggle_key else f"calloi_toggle_{ticker}"))
    with col5:
        show_put_vol = st.toggle("Put Vol", value=False, key=(f"{toggle_key}__put_vol" if toggle_key else f"putvol_toggle_{ticker}"))
    with col6:
        show_call_vol = st.toggle("Call Vol", value=False, key=(f"{toggle_key}__call_vol" if toggle_key else f"callvol_toggle_{ticker}"))
    with col7:
        show_ag = st.toggle("AG", value=False, key=(f"{toggle_key}__ag" if toggle_key else f"ag_toggle_{ticker}"))
    with col8:
        show_pz = st.toggle("PZ", value=False, key=(f"{toggle_key}__pz" if toggle_key else f"pz_toggle_{ticker}"))
    with col9:
        show_er_up = st.toggle("ER_Up", value=False, key=(f"{toggle_key}__er_up" if toggle_key else f"erup_toggle_{ticker}"))
    with col10:
        show_er_down = st.toggle("ER_Down", value=False, key=(f"{toggle_key}__er_down" if toggle_key else f"erdown_toggle_{ticker}"))

    if not show:
        return

    # Подготовка данных
    df = df_final[["K", y_col]].dropna().copy()
    df["K"] = _pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["K"]).sort_values("K").reset_index(drop=True)
    df = df.groupby("K", as_index=False)[y_col].sum()

    Ks = df["K"].to_numpy(dtype=float)
    Ys = df[y_col].to_numpy(dtype=float)
    g_flip = _compute_gamma_flip_from_table(df, y_col, spot)
    
    x_idx = _np.arange(len(Ks), dtype=float)
    bar_width = 0.9
    colors = _np.where(Ys >= 0.0, COLOR_POS, COLOR_NEG)
    
    # Фигура
    fig = go.Figure()
    
    # Net GEX столбики с hover
    customdata_netgex = _create_hover_data(Ks, df_final, Ys, "Net GEX")
    fig.add_trace(go.Bar(
        x=x_idx, y=Ys, name="Net GEX (M$ / 1%)", marker_color=colors, width=bar_width,
        customdata=customdata_netgex, hovertemplate=_create_hover_template("Net GEX"),
        hoverlabel=dict(bgcolor=colors, bordercolor="white", font=dict(size=12, color="white", family="Arial")),
    ))

    # Функция для добавления серий
    def add_series(show_flag, col_name, color, name, yaxis="y2"):
        if show_flag and col_name in df_final.columns:
            df_series = df_final.groupby("K", as_index=False)[col_name].sum().sort_values("K").reset_index(drop=True)
            _map_series = {float(k): float(v) for k, v in zip(df_series["K"].to_numpy(), df_series[col_name].to_numpy())}
            y_series = [_map_series.get(float(k), 0) for k in Ks]
            customdata_series = _create_hover_data(Ks, df_final, y_series, name)
            
            fig.add_trace(go.Scatter(
                x=x_idx, y=y_series, customdata=customdata_series, yaxis=yaxis,
                mode="lines+markers", line=dict(shape="spline", smoothing=1.0, width=1.5, color=color),
                marker=dict(size=6, color=color),                 fill="tozeroy", fillcolor=f"rgba({color[1:3]}, {color[3:5]}, {color[5:7]}, 0.3)" if color.startswith("#") and len(color)==7 else "rgba(128,128,128,0.3)",
                name=name, hovertemplate=_create_hover_template(name),
                hoverlabel=dict(bgcolor=color, bordercolor="white", font=dict(size=12, color="white", family="Arial")),
            ))

    # Добавляем все серии
    add_series(show_put_oi, "put_oi", "#800020", "Put OI")
    add_series(show_call_oi, "call_oi", "#2ECC71", "Call OI")
    add_series(show_put_vol, "put_vol", "#FF8C00", "Put Volume")
    add_series(show_call_vol, "call_vol", "#1E88E5", "Call Volume")
    
    # AG с правильным выбором колонки
    if show_ag:
        ag_col = "AG_1pct" if "AG_1pct" in df_final.columns else ("AG_1pct_M" if "AG_1pct_M" in df_final.columns else None)
        if ag_col:
            add_series(True, ag_col, "#9A7DF7", "AG")
    
    add_series(show_pz, "PZ", "#E4C51E", "PZ")
    add_series(show_er_up, "ER_Up", "#1FCE54", "ER_Up")
    add_series(show_er_down, "ER_Down", "#D21717", "ER_Down")

    # (Invisible) dummy trace для правой оси
    try:
        fig.add_trace(go.Scatter(
            x=[x_idx[0] if len(x_idx) > 0 else 0], y=[0], yaxis="y2",
            mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip",
        ))
    except Exception:
        pass

    # Вертикальная линия цены
    if spot is not None and _np.isfinite(spot):
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
                           showarrow=False, font=dict(size=16, color="#e0e0e0"), xanchor="left", yanchor="bottom")

    # Подписи страйков
    tick_vals = x_idx.tolist()
    tick_text = [str(int(k)) if float(k).is_integer() else f"{k:.2f}" for k in Ks]

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#111111", plot_bgcolor="#111111",
        margin=dict(l=40, r=60, t=40, b=40), showlegend=False, dragmode=False,
        xaxis=dict(title=None, tickmode="array", tickvals=tick_vals, ticktext=tick_text, tickangle=0, tickfont=dict(size=10), showgrid=False, showline=False, zeroline=False,),
        yaxis=dict(title="Net GEX", showgrid=False, zeroline=False,),
        yaxis2=dict(title="Other parameters", overlaying="y", side="right", showgrid=False, zeroline=False, showline=True, ticks="outside", tickfont=dict(size=10),),
    )

    # --- G-Flip marker ---
    try:
        if show_gflip and (g_flip is not None) and (len(Ks) > 0):
            k_arr = Ks.astype(float)
            g_val = float(g_flip)
            snap_idx = int(_np.argmin(_np.abs(k_arr - g_val)))
            x_g = float(snap_idx)
            k_snap = float(k_arr[snap_idx])
            fig.add_shape(type="line", x0=x_g, x1=x_g, y0=0, y1=1, xref="x", yref="paper",
                          line=dict(width=1, color="#AAAAAA", dash="dash"), layer="above")
            fig.add_annotation(x=x_g, xref="x", y=1.02, yref="paper", text=f"G-Flip: {k_snap:g}", showarrow=False, yshift=0, 
                               font=dict(size=12, color="#AAAAAA"), xanchor="center", yanchor="bottom", align="center")
    except Exception:
        pass

    # Автомасштаб
    fig.update_yaxes(autorange=True)
    fig.update_xaxes(autorange=True)

    # Статичный график
    st.plotly_chart(fig, use_container_width=True, theme=None, config={'displayModeBar': False, 'staticPlot': True})
