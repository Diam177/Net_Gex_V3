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
    G-Flip (K*): граница текущего режима гаммы около spot.
    Алгоритм (без порогов):
      1) Агрегируем NetGEX по страйку и сортируем (K, G).
      2) Определяем знак в точке spot (линейной интерполяцией по соседним узлам).
      3) Находим непрерывный сегмент R0 текущего знака, содержащий spot.
      4) С каждой стороны R0 берём прилегающий сегмент противоположного знака (непрерывный),
         считаем его массу M = sum(|G|) по этому сегменту.
      5) Выбираем сторону с бóльшей массой (при равенстве — ближайшую к spot).
      6) Возвращаем корень между последним узлом R0 и первым узлом противоположного сегмента на выбранной стороне.
    """
    import pandas as _pd, numpy as _np
    if df_final is None or len(df_final) == 0 or "K" not in df_final.columns or y_col not in df_final.columns:
        return None

    g = (df_final
         .assign(K=_pd.to_numeric(df_final["K"], errors="coerce"),
                 V=_pd.to_numeric(df_final[y_col], errors="coerce"))[["K","V"]]
         .dropna()
         .groupby("K", as_index=False)["V"].sum()
         .sort_values("K")
         .reset_index(drop=True))
    if g.empty or len(g) < 2:
        return None
    Ks = g["K"].to_numpy(dtype=float)
    Ys = g["V"].to_numpy(dtype=float)
    n = len(Ks)

    # --- знак в точке spot ---
    def _interp_y_at(x: float) -> float:
        pos = _np.searchsorted(Ks, x, side="left")
        if pos <= 0:
            return float(Ys[0])
        if pos >= n:
            return float(Ys[-1])
        K0, K1 = Ks[pos-1], Ks[pos]
        y0, y1 = Ys[pos-1], Ys[pos]
        if K1 == K0:
            return float(y0)
        t = (float(x) - K0) / (K1 - K0)
        return float(y0 + t*(y1 - y0))

    if spot is None or not _np.isfinite(spot):
        # fallback: середина диапазона
        spot = 0.5*(Ks[0] + Ks[-1])

    yspot = _interp_y_at(float(spot))
    s0 = _np.sign(yspot)
    if s0 == 0:
        # берём знак ближайшего незерового узла
        idx_near = int(_np.argmin(_np.abs(Ks - float(spot))))
        left = idx_near
        right = idx_near
        while left >= 0 and Ys[left] == 0:
            left -= 1
        while right < n and Ys[right] == 0:
            right += 1
        cand_signs = []
        if left >= 0 and Ys[left] != 0:
            cand_signs.append(_np.sign(Ys[left]))
        if right < n and Ys[right] != 0:
            cand_signs.append(_np.sign(Ys[right]))
        s0 = cand_signs[0] if cand_signs else 1.0  # по умолчанию +

    # --- сегмент R0 (содержит spot) ---
    pos = int(_np.searchsorted(Ks, float(spot), side="left"))
    i = max(0, min(n-1, pos))
    # включаем нули в текущий режим
    def _same_reg(val: float) -> bool:
        return (_np.sign(val) == s0) or (val == 0.0)

    L = i
    while L-1 >= 0 and _same_reg(Ys[L-1]):
        L -= 1
    R = i
    while R+1 < n and _same_reg(Ys[R+1]):
        R += 1

    # --- масса противоположного сегмента слева ---
    M_left = 0.0
    left_pair = None  # (i0_in_R0, i1_in_opp) для интерполяции: (R0_index, Opp_index)
    j = L-1
    if j >= 0 and _np.sign(Ys[j]) == -s0:
        k = j
        while k-1 >= 0 and (_np.sign(Ys[k-1]) == -s0 or Ys[k-1] == 0.0):
            k -= 1
        # сегмент противоположного знака: [k .. j]
        M_left = float(_np.sum(_np.abs(Ys[k:j+1])))
        left_pair = (L, j)  # граница между j (opp) и L (R0)

    # --- масса противоположного сегмента справа ---
    M_right = 0.0
    right_pair = None
    j = R+1
    if j < n and _np.sign(Ys[j]) == -s0:
        k = j
        while k+1 < n and (_np.sign(Ys[k+1]) == -s0 or Ys[k+1] == 0.0):
            k += 1
        # сегмент противоположного знака: [j .. k]
        M_right = float(_np.sum(_np.abs(Ys[j:k+1])))
        right_pair = (R, j)  # граница между R (R0) и j (opp)

    # если нет противоположных сегментов — вернём стандартный корень ближайший к spot
    if left_pair is None and right_pair is None:
        # классический поиск корней
        cand = [float(Ks[t]) for t,v in enumerate(Ys) if v == 0.0]
        sign = _np.sign(Ys)
        idx = _np.where(sign[:-1]*sign[1:] < 0)[0]
        for t in idx:
            K0, K1 = Ks[t], Ks[t+1]
            y0, y1 = Ys[t], Ys[t+1]
            if y1 == y0:
                continue
            Kstar = K0 - y0*(K1-K0)/(y1-y0)
            if min(K0, K1) <= Kstar <= max(K0, K1):
                cand.append(float(Kstar))
        if not cand:
            return None
        j = int(_np.argmin(_np.abs(_np.array(cand) - float(spot))))
        return float(cand[j])

    # выбор стороны
    pick = None
    if left_pair and right_pair:
        if abs(M_left - M_right) > 1e-12:
            pick = left_pair if (M_left > M_right) else right_pair
        else:
            # равные массы: ближе к spot
            kL = Ks[left_pair[0]]  # R0 index near boundary
            kR = Ks[right_pair[0]]
            pick = left_pair if abs(kL - float(spot)) <= abs(kR - float(spot)) else right_pair
    else:
        pick = left_pair or right_pair

    i0, i1 = pick  # i0 in R0, i1 in opposite
    # точные нули на границе
    if Ys[i0] == 0.0:
        return float(Ks[i0])
    if Ys[i1] == 0.0:
        return float(Ks[i1])
    # линейная интерполяция
    K0, K1 = Ks[min(i0, i1)], Ks[max(i0, i1)]
    y0, y1 = Ys[min(i0, i1)], Ys[max(i0, i1)]
    if K1 == K0 or y1 == y0:
        return float(Ks[i0])
    Kstar = K0 - y0*(K1-K0)/(y1-y0)
    # ограничим корень интервалом
    if Kstar < min(K0, K1):
        Kstar = min(K0, K1)
    elif Kstar > max(K0, K1):
        Kstar = max(K0, K1)
    return float(Kstar)
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

# --- Toggles: single horizontal row ---
    # --- Toggles: single horizontal row ---

    # компактный зазор между колонками с тумблерами
    st.markdown(
        "<style>div[data-testid='column']{padding-left:0px!important;padding-right:2px!important}</style>",
        unsafe_allow_html=True,
    )
    # уменьшить шрифт подписей тумблеров (чуть меньше)
# уменьшить шрифт подписей тумблеров
    st.markdown("""
    <style>
    /* уменьшить шрифт подписи у st.toggle (надёжные селекторы) */
    div[data-testid="stWidgetLabel"] * {
      font-size: 0.80rem !important;
      line-height: 1.1 !important;
    }
    label:has(> div[data-baseweb="switch"]) * {
      font-size: 0.80rem !important;
      line-height: 1.1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8, gap="small")
    with col1:
        show = st.toggle("Net GEX", value=True,
                         key=(toggle_key or f"netgex_toggle_{ticker}"))
    with col2:
        show_gflip = st.toggle("G-Flip", value=False,
                               key=(f"{toggle_key}__gflip" if toggle_key else f"gflip_toggle_{ticker}"))
    with col3:
        show_put_oi = st.toggle("Put OI", value=False,
                      key=(f"{toggle_key}__put_oi" if toggle_key else f"putoi_toggle_{ticker}"))
    with col4:
        show_call_oi = st.toggle("Call OI", value=False,
                      key=(f"{toggle_key}__call_oi" if toggle_key else f"calloi_toggle_{ticker}")
)
    with col5:
        show_put_vol = st.toggle("Put Vol", value=False,
                      key=(f"{toggle_key}__put_vol" if toggle_key else f"putvol_toggle_{ticker}")
)
    with col6:
        show_call_vol = st.toggle("Call Vol", value=False,
                      key=(f"{toggle_key}__call_vol" if toggle_key else f"callvol_toggle_{ticker}")
)
    with col7:
        show_ag = st.toggle("AG", value=False,
                      key=(f"{toggle_key}__ag" if toggle_key else f"ag_toggle_{ticker}")
)
    with col8:
        show_pz = st.toggle("PZ", value=False,
                      key=(f"{toggle_key}__pz" if toggle_key else f"pz_toggle_{ticker}")
)

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
    
    # Подготовка данных для hover
    # Собираем данные по страйкам для всплывающей подсказки
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
    
    # Создаем customdata для hover
    customdata_list = []
    for i, k in enumerate(Ks):
        hd = hover_data.get(k, {})
        customdata_list.append([
            k,  # Strike
            hd.get("call_oi", 0),  # Call OI
            hd.get("put_oi", 0),   # Put OI  
            hd.get("call_vol", 0),  # Call Volume
            hd.get("put_vol", 0),   # Put Volume
            Ys[i]  # Net GEX value
        ])
    
    
# Фигура
    fig = go.Figure()

    # Определяем единицы измерения для подписи Net GEX в ховере
    _unit_suffix = "M" if str(y_col).endswith("_M") else ""

    # Разделяем положительные и отрицательные значения на два трека,
    # чтобы цвет hover-таблички соответствовал цвету столбца (как на примерах).
    pos_mask = (Ys >= 0)
    neg_mask = ~pos_mask

    def _subset(arr, mask, fill=None):
        out = []
        for a, m in zip(arr, mask):
            out.append(a if m else fill)
        return out

    # customdata по точкам: [Strike, Call OI, Put OI, Call Vol, Put Vol, Net GEX]
    # Для треков используем одну и ту же customdata, Plotly игнорирует элементы с y=None.
    hover_tmpl = (
        "<b>Strike: %{customdata[0]:.0f}</b><br>"
        "Call OI: %{customdata[1]:,.0f}<br>"
        "Put OI: %{customdata[2]:,.0f}<br>"
        "Call Volume: %{customdata[3]:,.0f}<br>"
        "Put Volume: %{customdata[4]:,.0f}<br>"
        "Net GEX: %{customdata[5]:,.1f}"+ _unit_suffix +
        "<extra></extra>"
    )

    # Положительные бары (синие)
    fig.add_trace(go.Bar(
        x=x_idx,
        y=_subset(Ys, pos_mask),
        name="Net GEX (>0)",
        marker_color=COLOR_POS,
        width=bar_width,
        customdata=customdata_list,
        hovertemplate=hover_tmpl,
        hoverlabel=dict(bgcolor=COLOR_POS, bordercolor="white",
                        font=dict(size=13, color="white")),
    ))
    # Отрицательные бары (красные)
    fig.add_trace(go.Bar(
        x=x_idx,
        y=_subset(Ys, neg_mask),
        name="Net GEX (<0)",
        marker_color=COLOR_NEG,
        width=bar_width,
        customdata=customdata_list,
        hovertemplate=hover_tmpl,
        hoverlabel=dict(bgcolor=COLOR_NEG, bordercolor="white",
                        font=dict(size=13, color="white")),
    ))
# --- Put OI markers (toggle-controlled) ---
    try:
        if 'show_put_oi' in locals() and show_put_oi:
            # Суммируем финальный put_oi по страйкам и выравниваем по Ks
            if ("K" in df_final.columns) and ("put_oi" in df_final.columns):
                df_put = df_final.groupby("K", as_index=False)["put_oi"].sum().sort_values("K").reset_index(drop=True)
                _map_put = {float(k): float(v) for k, v in zip(df_put["K"].to_numpy(), df_put["put_oi"].to_numpy())}
                y_put = [_map_put.get(float(k), None) for k in Ks]
                # Линия + точки, плавная, заливка к нулю правой оси (y2). Прозрачность заливки ~70% (alpha=0.3)
                fig.add_trace(go.Scatter(
                    x=x_idx,
                    y=y_put,
                    customdata=Ks,
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(shape="spline", smoothing=1.0, width=1.5, color="#800020"),
                    marker=dict(size=6, color="#800020"),
                    fill="tozeroy",
                    fillcolor="rgba(128, 0, 32, 0.3)",
                    name="Put OI",
                    hovertemplate="Strike: %{customdata}<br>Put OI: %{y:.0f}<extra></extra>",
                ))
    except Exception:
        pass
    # --- Call OI markers (toggle-controlled) ---
    try:
        if 'show_call_oi' in locals() and show_call_oi:
            if ("K" in df_final.columns) and ("call_oi" in df_final.columns):
                df_call = df_final.groupby("K", as_index=False)["call_oi"].sum().sort_values("K").reset_index(drop=True)
                _map_call = {float(k): float(v) for k, v in zip(df_call["K"].to_numpy(), df_call["call_oi"].to_numpy())}
                y_call = [_map_call.get(float(k), None) for k in Ks]
                fig.add_trace(go.Scatter(
                    x=x_idx,
                    y=y_call,
                    customdata=Ks,
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(shape="spline", smoothing=1.0, width=1.5, color="#2ECC71"),
                    marker=dict(size=6, color="#2ECC71"),
                    fill="tozeroy",
                    fillcolor="rgba(46, 204, 113, 0.3)",
                    name="Call OI",
                    hovertemplate="Strike: %{customdata}<br>Call OI: %{y:.0f}<extra></extra>",
                ))
    except Exception:
        pass

    # --- Put Volume markers (toggle-controlled) ---
    try:
        if 'show_put_vol' in locals() and show_put_vol:
            if ("K" in df_final.columns) and ("put_vol" in df_final.columns):
                df_pv = df_final.groupby("K", as_index=False)["put_vol"].sum().sort_values("K").reset_index(drop=True)
                _map_pv = {float(k): float(v) for k, v in zip(df_pv["K"].to_numpy(), df_pv["put_vol"].to_numpy())}
                y_pv = [_map_pv.get(float(k), None) for k in Ks]
                fig.add_trace(go.Scatter(
                    x=x_idx,
                    y=y_pv,
                    customdata=Ks,
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(shape="spline", smoothing=1.0, width=1.5, color="#FF8C00"),
                    marker=dict(size=6, color="#FF8C00"),
                    fill="tozeroy",
                    fillcolor="rgba(255, 140, 0, 0.3)",
                    name="Put Volume",
                    hovertemplate="Strike: %{customdata}<br>Put Volume: %{y:.0f}<extra></extra>",
                ))
    except Exception:
        pass

    # --- Call Volume markers (toggle-controlled) ---
    try:
        if 'show_call_vol' in locals() and show_call_vol:
            if ("K" in df_final.columns) and ("call_vol" in df_final.columns):
                df_cv = df_final.groupby("K", as_index=False)["call_vol"].sum().sort_values("K").reset_index(drop=True)
                _map_cv = {float(k): float(v) for k, v in zip(df_cv["K"].to_numpy(), df_cv["call_vol"].to_numpy())}
                y_cv = [_map_cv.get(float(k), None) for k in Ks]
                fig.add_trace(go.Scatter(
                    x=x_idx,
                    y=y_cv,
                    customdata=Ks,
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(shape="spline", smoothing=1.0, width=1.5, color="#1E88E5"),
                    marker=dict(size=6, color="#1E88E5"),
                    fill="tozeroy",
                    fillcolor="rgba(30, 136, 229, 0.3)",
                    name="Call Volume",
                    hovertemplate="Strike: %{customdata}<br>Call Volume: %{y:.0f}<extra></extra>",
                ))
    except Exception:
        pass

    # --- AG markers (toggle-controlled) ---
    try:
        if 'show_ag' in locals() and show_ag:
            # предпочитаем AG_1pct, иначе AG_1pct_M
            ag_col = "AG_1pct" if "AG_1pct" in df_final.columns else ("AG_1pct_M" if "AG_1pct_M" in df_final.columns else None)
            if ("K" in df_final.columns) and (ag_col is not None):
                df_ag = df_final.groupby("K", as_index=False)[ag_col].sum().sort_values("K").reset_index(drop=True)
                _map_ag = {float(k): float(v) for k, v in zip(df_ag["K"].to_numpy(), df_ag[ag_col].to_numpy())}
                y_ag = [_map_ag.get(float(k), None) for k in Ks]
                fig.add_trace(go.Scatter(
                    x=x_idx,
                    y=y_ag,
                    customdata=Ks,
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(shape="spline", smoothing=1.0, width=1.5, color="#9A7DF7"),
                    marker=dict(size=6, color="#9A7DF7"),
                    fill="tozeroy",
                    fillcolor="rgba(154, 125, 247, 0.3)",
                    name="AG",
                    hovertemplate="Strike: %{customdata}<br>AG: %{y:.0f}<extra></extra>",
                ))
    except Exception:
        pass

    # --- PZ markers (toggle-controlled) ---
    try:
        if 'show_pz' in locals() and show_pz:
            if ("K" in df_final.columns) and ("PZ" in df_final.columns):
                df_pz = df_final.groupby("K", as_index=False)["PZ"].sum().sort_values("K").reset_index(drop=True)
                _map_pz = {float(k): float(v) for k, v in zip(df_pz["K"].to_numpy(), df_pz["PZ"].to_numpy())}
                y_pz = [_map_pz.get(float(k), None) for k in Ks]
                fig.add_trace(go.Scatter(
                    x=x_idx,
                    y=y_pz,
                    customdata=Ks,
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(shape="spline", smoothing=1.0, width=1.5, color="#E4C51E"),
                    marker=dict(size=6, color="#E4C51E"),
                    fill="tozeroy",
                    fillcolor="rgba(228, 197, 30, 0.3)",
                    name="PZ",
                    hovertemplate="Strike: %{customdata}<br>PZ: %{y:.3f}<extra></extra>",
                ))
    except Exception:
        pass



    # (Invisible) dummy trace to expose right-side secondary y-axis without drawing anything
    try:
        fig.add_trace(go.Scatter(
            x=[x_idx[0] if len(x_idx) > 0 else 0],
            y=[0],
            yaxis="y2",
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        ))
    except Exception:
        pass

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
        hovermode='closest',
        barmode="overlay",

        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=40, r=60, t=40, b=40),
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
        yaxis2=dict(
            title="Other parameters",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="outside",
            tickfont=dict(size=10),
        ),
    )

    
    
    
    # --- G-Flip marker (optional) ---
    try:
        if 'show_gflip' in locals() and show_gflip and (g_flip is not None) and (len(Ks) > 0):
            # Всегда привязываем линию к центру ближайшего страйка (визуальная консистентность с подписью)
            k_arr = Ks.astype(float)
            g_val = float(g_flip)
            snap_idx = int(_np.argmin(_np.abs(k_arr - g_val)))
            x_g = float(snap_idx)
            k_snap = float(k_arr[snap_idx])

            fig.add_shape(type="line", x0=x_g, x1=x_g, y0=0, y1=1, xref="x", yref="paper",
                          line=dict(width=1, color="#AAAAAA", dash="dash"), layer="above")
            fig.add_annotation(x=x_g, xref="x", y=1.02, yref="paper", text=f"G-Flip: {k_snap:g}", showarrow=False, yshift=0, font=dict(size=12, color="#AAAAAA"), xanchor="center", yanchor="bottom", align="center")
    except Exception:
        pass

    # Автомасштаб
    fig.update_yaxes(autorange=True, fixedrange=True)
    fig.update_xaxes(autorange=True, fixedrange=True)

    # График без панели и без зума/панорамы, но с hover-подсказками
    fig.update_layout(xaxis_title='Strikes')
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.0, y=1.0, xanchor='left', yanchor='top',
        text=str(ticker), showarrow=False,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={'displayModeBar': False})
