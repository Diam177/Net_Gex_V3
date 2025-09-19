
# -*- coding: utf-8 -*-
"""
lib/key_levels.py — интрадей‑чарт «Key Levels» с горизонтальными уровнями из финальной таблицы.

Функция render_key_levels(
    df_final,               # pandas.DataFrame финальной таблицы по страйкам (одна экспирация или агрегат multi)
    ticker,                 # строка для заголовка/легенды
    g_flip=None,            # опционально: значение G‑Flip, уже вычисленное в netgex_chart
    price_df=None,          # опционально: DataFrame с колонками ['time', 'price', 'vwap'] (time — tz‑aware или naive UTC)
    session_date=None,      # опционально: дата торгового дня (date/datetime/str 'YYYY-MM-DD'); по умолчанию — сегодня в ET
    toggle_key=None,        # опциональный уникальный ключ для st.toggle
):
    - Берёт уровни из df_final: Max Neg/Pos GEX, Max Put/Call OI, Max Put/Call Volume, пик AG, пик PZ.
    - G‑Flip берёт из параметра g_flip; если None — пытается вычислить как корень NetGEX(K)
      через кусочно‑линейную интерполяцию (совместимо с netgex_chart._compute_gamma_flip_from_table).
    - Рисует горизонтальные линии уровней поперёк интрадей оси времени (09:30—16:00 ET),
      поверх простых серий Price (линия) и VWAP (линия), если price_df задан.
    - Слева рядом со шкалой рисует белыми те значения, где есть уровень; остальные тики серые.
    - Справа у правого края выводит подписи для совпадающих уровней («Max Pos GEX + Max Call OI»).
    - Внизу под осью времени выводит дату (как на референс‑скриншоте).
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import streamlit as st

# --- Цвета (совместимые с netgex_chart.py) ---
COLOR_PUT_OI    = "#800020"
COLOR_CALL_OI   = "#2ECC71"
COLOR_PUT_VOL   = "#FF8C00"
COLOR_CALL_VOL  = "#1E88E5"
COLOR_AG        = "#9A7DF7"
COLOR_PZ        = "#E4C51E"
COLOR_GFLIP     = "#AAAAAA"

COLOR_MAX_POS_GEX = "#60A5E7"  # голубой
COLOR_MAX_NEG_GEX = "#D9493A"  # красный
COLOR_PRICE       = "#FFFFFF"  # белая линия цены
COLOR_VWAP        = "#E4A339"  # оранжевая VWAP
BACKGROUND        = "#0E1117"
AXIS_GRAY         = "#AAAAAA"
GRID_COLOR        = "rgba(255,255,255,0.05)"

def _to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def _group_max_level(df: pd.DataFrame, value_col: str) -> Optional[float]:
    if df is None or df.empty or "K" not in df.columns or value_col not in df.columns:
        return None
    g = (
        df[["K", value_col]]
        .copy()
        .assign(K=lambda x: pd.to_numeric(x["K"], errors="coerce"),
                V=lambda x: pd.to_numeric(x[value_col], errors="coerce"))
        .dropna()
        .groupby("K", as_index=False)["V"].sum()
        .sort_values("K")
        .reset_index(drop=True)
    )
    if g.empty:
        return None
    i = int(np.nanargmax(g["V"].to_numpy()))
    return float(g.iloc[i]["K"])

def _group_min_level(df: pd.DataFrame, value_col: str) -> Optional[float]:
    if df is None or df.empty or "K" not in df.columns or value_col not in df.columns:
        return None
    g = (
        df[["K", value_col]]
        .copy()
        .assign(K=lambda x: pd.to_numeric(x["K"], errors="coerce"),
                V=lambda x: pd.to_numeric(x[value_col], errors="coerce"))
        .dropna()
        .groupby("K", as_index=False)["V"].sum()
        .sort_values("K")
        .reset_index(drop=True)
    )
    if g.empty:
        return None
    i = int(np.nanargmin(g["V"].to_numpy()))
    return float(g.iloc[i]["K"])

def _compute_gflip_piecewise(df_final: pd.DataFrame, y_col: str = "NetGEX_1pct", spot: Optional[float] = None) -> Optional[float]:
    if df_final is None or df_final.empty or "K" not in df_final.columns or y_col not in df_final.columns:
        return None
    base = df_final[["K", y_col]].copy()
    base["K"] = pd.to_numeric(base["K"], errors="coerce")
    base[y_col] = pd.to_numeric(base[y_col], errors="coerce")
    base = base.dropna()
    if base.empty:
        return None
    g = base.groupby("K", as_index=False)[y_col].sum().sort_values("K").reset_index(drop=True)
    Ks = g["K"].to_numpy(dtype=float)
    Ys = g[y_col].to_numpy(dtype=float)
    if len(Ks) < 2:
        return None
    cand: List[float] = [float(Ks[i]) for i, v in enumerate(Ys) if v == 0.0]
    sign = np.sign(Ys)
    idx = np.where(sign[:-1] * sign[1:] < 0)[0]
    for i in idx:
        K0, K1 = float(Ks[i]), float(Ks[i+1])
        y0, y1 = float(Ys[i]), float(Ys[i+1])
        if y1 != y0:
            root = K0 - y0 * (K1 - K0) / (y1 - y0)
            root = max(min(root, K1), K0) if K1 >= K0 else max(min(root, K0), K1)
            cand.append(float(root))
    if not cand:
        return None
    if spot is not None and np.isfinite(spot):
        j = int(np.argmin(np.abs(np.array(cand) - float(spot))))
        return float(cand[j])
    mid = 0.5 * (float(Ks[0]) + float(Ks[-1]))
    j = int(np.argmin(np.abs(np.array(cand) - mid)))
    return float(cand[j])

def _session_timerange(session_date: Optional[Union[str, pd.Timestamp]]) -> Tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    try:
        import pytz
        tz = pytz.timezone("America/New_York")
    except Exception:
        tz = None
    if session_date is None:
        session_date = pd.Timestamp.utcnow().tz_localize("UTC") if tz is None else pd.Timestamp.now(tz)
    session_date = pd.to_datetime(session_date).date()
    t0_naive = pd.Timestamp(year=session_date.year, month=session_date.month, day=session_date.day, hour=9, minute=30)
    t1_naive = pd.Timestamp(year=session_date.year, month=session_date.month, day=session_date.day, hour=16, minute=0)
    if tz is not None:
        t0 = tz.localize(t0_naive)
        t1 = tz.localize(t1_naive)
    else:
        t0 = t0_naive
        t1 = t1_naive
    idx = pd.date_range(t0, t1, freq="1min")
    return t0, t1, idx

def _format_date_for_footer(dt: pd.Timestamp) -> str:
    return dt.strftime("%b %d, %Y")

def _make_tick_annotations(fig, y_vals: Iterable[float], x_left):
    used = set()
    for y in y_vals:
        if not np.isfinite(y):
            continue
        y_ = float(y)
        key = round(y_, 2)
        if key in used:
            continue
        used.add(key)
        fig.add_annotation(
            x=x_left, xref="x", y=y_, yref="y",
            text=f"{y_:g}",
            showarrow=False,
            xanchor="right", yanchor="middle",
            font=dict(size=10, color="#FFFFFF"),
            align="right",
        )

def render_key_levels(
    df_final: pd.DataFrame,
    ticker: str,
    g_flip: Optional[float] = None,
    price_df: Optional[pd.DataFrame] = None,
    session_date: Optional[Union[str, pd.Timestamp]] = None,
    toggle_key: Optional[str] = None,
) -> None:
    import plotly.graph_objects as go

    if df_final is None or getattr(df_final, "empty", True):
        st.info("Нет данных для графика Key Levels.")
        return
    if "K" not in df_final.columns:
        st.warning("В финальной таблице отсутствует столбец 'K'.")
        return

    # --- Вычисляем уровни ---
    cols = df_final.columns
    y_ag  = "AG_1pct_M"     if "AG_1pct_M"     in cols else ("AG_1pct"     if "AG_1pct"     in cols else None)
    y_ng  = "NetGEX_1pct_M" if "NetGEX_1pct_M" in cols else ("NetGEX_1pct" if "NetGEX_1pct" in cols else None)
    has_call_oi = "call_oi" in cols
    has_put_oi  = "put_oi"  in cols
    has_call_vol= "call_vol" in cols
    has_put_vol = "put_vol"  in cols
    has_pz      = "PZ" in cols

    level_map: Dict[str, Optional[float]] = {}
    if y_ng:
        level_map["Max Pos GEX"] = _group_max_level(df_final, y_ng)
        level_map["Max Neg GEX"] = _group_min_level(df_final, y_ng)
    if has_put_oi:
        level_map["Max Put OI"] = _group_max_level(df_final, "put_oi")
    if has_call_oi:
        level_map["Max Call OI"] = _group_max_level(df_final, "call_oi")
    if has_put_vol:
        level_map["Max Put Volume"] = _group_max_level(df_final, "put_vol")
    if has_call_vol:
        level_map["Max Call Volume"] = _group_max_level(df_final, "call_vol")
    if y_ag:
        level_map["AG"] = _group_max_level(df_final, y_ag)
    if has_pz:
        level_map["PZ"] = _group_max_level(df_final, "PZ")

    # G‑Flip
    spot = float(pd.to_numeric(df_final.get("S"), errors="coerce").median()) if "S" in cols else None
    if g_flip is None:
        g_flip = _compute_gflip_piecewise(df_final, y_col=(y_ng or "NetGEX_1pct"), spot=spot)
    if g_flip is not None and np.isfinite(g_flip):
        level_map["G-Flip"] = float(g_flip)

    # --- Временная ось ---
    t0, t1, t_idx = _session_timerange(session_date)
    x_left, x_right = t0, t1

    # --- Готовим фигуру ---
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        margin=dict(l=60, r=110, t=40, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0, font=dict(size=10)),
    )
    fig.update_yaxes(
        tickfont=dict(color=AXIS_GRAY, size=10),
        gridcolor=GRID_COLOR,
        zeroline=False,
        title=dict(text="Price", font=dict(color="#FFFFFF", size=11)),
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        tickfont=dict(color=AXIS_GRAY, size=10),
        gridcolor=GRID_COLOR,
        zeroline=False,
        type="date",
        title=dict(text="Time", font=dict(color="#FFFFFF", size=11)),
        range=[x_left, x_right],
    )

    # Принудительно закрепим тип оси X как временной: добавим «пустой» временной трейс
    fig.add_trace(go.Scatter(x=[x_left, x_right], y=[None, None], mode="lines",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))

    # --- Price / VWAP (если есть) ---
    if price_df is not None and not price_df.empty:
        pdf = price_df.copy()
        if "time" not in pdf.columns:
            pdf = pdf.reset_index().rename(columns={pdf.columns[0]: "time"})
        pdf["time"] = pd.to_datetime(pdf["time"])
        pdf = pdf[(pdf["time"] >= x_left) & (pdf["time"] <= x_right)]
        # Price
        if "price" in pdf.columns:
            fig.add_trace(go.Scatter(
                x=pdf["time"], y=pd.to_numeric(pdf["price"], errors="coerce"),
                mode="lines",
                line=dict(width=1.2, color=COLOR_PRICE),
                name="Price",
                hovertemplate="Time: %{x|%H:%M}<br>Price: %{y:.2f}<extra></extra>",
            ))
        # VWAP
        if "vwap" in pdf.columns:
            vwap_series = pd.to_numeric(pdf["vwap"], errors="coerce")
        elif set(["price","volume"]).issubset(set(pdf.columns)):
            vol = pd.to_numeric(pdf["volume"], errors="coerce").fillna(0.0)
            pr  = pd.to_numeric(pdf["price"], errors="coerce").fillna(np.nan)
            cum_vol = vol.cumsum().replace(0, np.nan)
            vwap_series = (pr.mul(vol)).cumsum() / cum_vol
        else:
            vwap_series = None
        if vwap_series is not None:
            fig.add_trace(go.Scatter(
                x=pdf["time"], y=vwap_series,
                mode="lines",
                line=dict(width=1.0, color=COLOR_VWAP, dash="solid"),
                name="VWAP",
                hovertemplate="Time: %{x|%H:%M}<br>VWAP: %{y:.2f}<extra></extra>",
            ))
    # --- Горизонтальные уровни ---
    color_map = {
        "Max Pos GEX": COLOR_MAX_POS_GEX,
        "Max Neg GEX": COLOR_MAX_NEG_GEX,
        "Max Put OI": COLOR_PUT_OI,
        "Max Call OI": COLOR_CALL_OI,
        "Max Put Volume": COLOR_PUT_VOL,
        "Max Call Volume": COLOR_CALL_VOL,
        "AG": COLOR_AG,
        "PZ": COLOR_PZ,
        "G-Flip": COLOR_GFLIP,
    }

    # Плейсхолдеры легенды удалены
    # Сгруппируем совпадающие значения (±0.05) для подписи справа
    eps = 0.05
    groups: Dict[float, List[str]] = {}
    for name, val in level_map.items():
        if val is None or not np.isfinite(val):
            continue
        y = float(val)
        found_key = None
        for key in list(groups.keys()):
            if abs(y - key) <= eps:
                found_key = key
                break
        if found_key is None:
            groups[y] = [name]
        else:
            groups[found_key].append(name)

    # Диапазон Y
    all_levels = [float(v) for v in groups.keys()]
    if all_levels:
        y_min = min(all_levels) - 3
        y_max = max(all_levels) + 3
    else:
        S_vals = pd.to_numeric(df_final.get("S"), errors="coerce").dropna().tolist() if "S" in df_final.columns else []
        if S_vals:
            s = float(np.median(S_vals))
            y_min, y_max = s - 10.0, s + 10.0
        else:
            y_min, y_max = 0.0, 1.0
    fig.update_yaxes(range=[y_min, y_max])

    # Линии и подписи справа
    for y, members in sorted(groups.items(), key=lambda kv: kv[0]):
        labels_sorted = sorted(members, key=lambda s: s)
        if "G-Flip" in labels_sorted:
            color = COLOR_GFLIP
        else:
            prio = ["Max Neg GEX", "Max Pos GEX", "AG", "PZ", "Max Put OI", "Max Call OI", "Max Put Volume", "Max Call Volume"]
            pick = next((p for p in prio if p in labels_sorted), labels_sorted[0])
            color = color_map.get(pick, "#CCCCCC")

        for _lbl in labels_sorted:
            if _lbl in ["G-Flip", "Max Put OI", "Max Call OI", "Max Put Volume", "Max Call Volume", "AG", "PZ"]:
                _clr = COLOR_GFLIP if _lbl == "G-Flip" else color_map.get(_lbl, color)
                fig.add_trace(go.Scatter(
                x=[x_left, x_right], y=[float(y), float(y)],
                mode="lines", line=dict(color=_clr, width=1.4, dash="solid"),
                name=_lbl, showlegend=True,
            ))
        fig.add_annotation(
            x=x_right, xref="x",
            y=float(y), yref="y",
            text=" + ".join(labels_sorted),
            showarrow=False,
            xanchor="right", yanchor="bottom",
            align="right",
            yshift=6,
            font=dict(size=10, color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=0.5,
        )

    # Белые подписи‑тиков слева
    _make_tick_annotations(fig, groups.keys(), x_left)

    # Заголовок (тикер)
    fig.add_annotation(
        x=0, xref="paper", y=1.12, yref="paper",
        text=str(ticker),
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(size=12, color="#FFFFFF"),
    )

    # Дата под осью
    fig.add_annotation(
        x=0.5, xref="paper", y=-0.18, yref="paper",
        text=_format_date_for_footer(pd.to_datetime(x_left)),
        showarrow=False, xanchor="center", yanchor="top",
        font=dict(size=10, color="#FFFFFF"),
    )

    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
