# -*- coding: utf-8 -*-
"""
key_levels.py — Key Levels чарт для отображения ключевых уровней.

Отображает серии: Price, VWAP, Max Neg GEX, Max Pos GEX, Max Put OI, Max Call OI,
Max Put Volume, Max Call Volume, AG, PZ, G-Flip

Зависимости: plotly>=5, pandas, streamlit, numpy
"""

from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime

# Import G-Flip calculation from netgex_chart
from lib.netgex_chart import _compute_gamma_flip_from_table

try:
    import plotly.graph_objects as go
except Exception as e:
    raise RuntimeError("Требуется пакет 'plotly' (plotly>=5.22.0)") from e

# Цвета серий (копируем из netgex_chart для консистентности)
SERIES_COLORS = {
    "Price": "#E4A339",           # оранжевый
    "VWAP": "#60A5E7",           # бирюзовый
    "Max Neg GEX": "#D9493A",    # красный
    "Max Pos GEX": "#60A5E7",    # бирюзовый
    "Max Put OI": "#800020",     # темно-красный
    "Max Call OI": "#2ECC71",    # зеленый
    "Max Put Volume": "#FF8C00",  # темно-оранжевый
    "Max Call Volume": "#1E88E5", # синий
    "AG": "#9A7DF7",             # фиолетовый
    "PZ": "#E4C51E",             # желтый
    "G-Flip": "#AAAAAA",         # серый
    "Neg Net GEX #2": "#D9493A", # красный
    "Neg Net GEX #3": "#D9493A", # красный
    "AG #2": "#9A7DF7",          # фиолетовый
    "AG #3": "#9A7DF7",          # фиолетовый
}

# Символы для легенды
SERIES_SYMBOLS = {
    "Price": "⬤",
    "VWAP": "―",
    "Max Neg GEX": "······",
    "Max Pos GEX": "⬤",
    "Max Put OI": "―",
    "Max Call OI": "―",
    "Max Put Volume": "······",
    "Max Call Volume": "······",
    "AG": "―",
    "PZ": "······",
    "G-Flip": "- - -"
}

# Цвета фона
BG_COLOR = '#0E1117'
FG_COLOR = '#FFFFFF'
GRID_COLOR = 'rgba(255,255,255,0.05)'
AXIS_INACTIVE = '#666666'
AXIS_ACTIVE = '#FFFFFF'

def _generate_trading_session_timeline() -> Tuple[np.ndarray, List[str]]:
    """Генерирует временную шкалу торговой сессии от 9:30 до 16:00"""
    # Массив временных точек в минутах от начала дня
    times = []
    labels = []
    
    # От 9:30 (570 минут) до 16:00 (960 минут) с шагом 30 минут
    for minutes in range(570, 961, 30):
        hours = minutes // 60
        mins = minutes % 60
        times.append(minutes)
        labels.append(f"{hours:02d}:{mins:02d}")
    
    return np.array(times), labels

def _extract_key_levels(df_final: pd.DataFrame) -> Dict[str, float]:
    """Извлекает ключевые уровни из финальной таблицы"""
    levels = {}
    
    if df_final is None or df_final.empty:
        return levels
    
    # Price - текущая цена
    if "S" in df_final.columns and df_final["S"].notna().any():
        levels["Price"] = float(df_final["S"].dropna().iloc[0])
    
    # VWAP - взвешенная средняя цена
    if "K" in df_final.columns:
        # Пробуем с volume
        if "call_vol" in df_final.columns and "put_vol" in df_final.columns:
            total_vol = df_final["call_vol"].fillna(0) + df_final["put_vol"].fillna(0)
            if total_vol.sum() > 0:
                levels["VWAP"] = float((df_final["K"] * total_vol).sum() / total_vol.sum())
        # Иначе используем OI
        elif "call_oi" in df_final.columns and "put_oi" in df_final.columns:
            total_oi = df_final["call_oi"].fillna(0) + df_final["put_oi"].fillna(0)
            if total_oi.sum() > 0:
                levels["VWAP"] = float((df_final["K"] * total_oi).sum() / total_oi.sum())
    
    # Max Neg GEX и Max Pos GEX
    col = "NetGEX_1pct" if "NetGEX_1pct" in df_final.columns else ("NetGEX_1pct_M" if "NetGEX_1pct_M" in df_final.columns else None)
    if col:
        neg_data = df_final[df_final[col] < 0]
        pos_data = df_final[df_final[col] > 0]
        
        # Max Neg GEX (наиболее отрицательное значение)
        if not neg_data.empty:
            min_idx = neg_data[col].idxmin()
            levels["Max Neg GEX"] = float(df_final.loc[min_idx, "K"])
            
            # Дополнительные уровни Neg GEX
            sorted_neg = neg_data.nsmallest(3, col)
            if len(sorted_neg) > 1:
                levels["Neg Net GEX #2"] = float(sorted_neg.iloc[1]["K"])
            if len(sorted_neg) > 2:
                levels["Neg Net GEX #3"] = float(sorted_neg.iloc[2]["K"])
        
        # Max Pos GEX (наиболее положительное значение)
        if not pos_data.empty:
            max_idx = pos_data[col].idxmax()
            levels["Max Pos GEX"] = float(df_final.loc[max_idx, "K"])
    
    # Max Put OI
    if "put_oi" in df_final.columns and df_final["put_oi"].sum() > 0:
        max_idx = df_final["put_oi"].idxmax()
        levels["Max Put OI"] = float(df_final.loc[max_idx, "K"])
    
    # Max Call OI
    if "call_oi" in df_final.columns and df_final["call_oi"].sum() > 0:
        max_idx = df_final["call_oi"].idxmax()
        levels["Max Call OI"] = float(df_final.loc[max_idx, "K"])
    
    # Max Put Volume
    if "put_vol" in df_final.columns and df_final["put_vol"].sum() > 0:
        max_idx = df_final["put_vol"].idxmax()
        levels["Max Put Volume"] = float(df_final.loc[max_idx, "K"])
    
    # Max Call Volume
    if "call_vol" in df_final.columns and df_final["call_vol"].sum() > 0:
        max_idx = df_final["call_vol"].idxmax()
        levels["Max Call Volume"] = float(df_final.loc[max_idx, "K"])
    
    # AG - абсолютная гамма
    ag_col = "AG_1pct" if "AG_1pct" in df_final.columns else ("AG_1pct_M" if "AG_1pct_M" in df_final.columns else None)
    if ag_col and df_final[ag_col].sum() > 0:
        sorted_ag = df_final.nlargest(3, ag_col)
        if len(sorted_ag) > 0:
            levels["AG"] = float(sorted_ag.iloc[0]["K"])
        if len(sorted_ag) > 1:
            levels["AG #2"] = float(sorted_ag.iloc[1]["K"])
        if len(sorted_ag) > 2:
            levels["AG #3"] = float(sorted_ag.iloc[2]["K"])
    
    # PZ - Power Zone
    if "PZ" in df_final.columns and df_final["PZ"].sum() > 0:
        max_idx = df_final["PZ"].idxmax()
        levels["PZ"] = float(df_final.loc[max_idx, "K"])
    
    # G-Flip
    y_col = "NetGEX_1pct_M" if "NetGEX_1pct_M" in df_final.columns else "NetGEX_1pct"
    if y_col in df_final.columns:
        spot = levels.get("Price")
        g_flip = _compute_gamma_flip_from_table(df_final, y_col, spot)
        if g_flip is not None:
            levels["G-Flip"] = float(g_flip)
    
    return levels

def _group_overlapping_levels(levels: Dict[str, float], threshold: float = 0.5) -> Dict[float, List[str]]:
    """Группирует серии с близкими значениями для отображения меток"""
    grouped = {}
    
    # Пропускаем вспомогательные уровни (с # в имени)
    main_levels = {k: v for k, v in levels.items() if '#' not in k}
    
    for name, value in main_levels.items():
        found = False
        for group_val in grouped:
            if abs(value - group_val) <= threshold:
                grouped[group_val].append(name)
                found = True
                break
        
        if not found:
            grouped[value] = [name]
    
    return grouped

def render_key_levels(
    df_final: pd.DataFrame,
    ticker: str,
    spot: Optional[float] = None,
    toggle_key: Optional[str] = None
) -> None:
    """Рендерит Key Levels чарт"""
    
    if df_final is None or df_final.empty:
        st.info("Нет данных для графика Key Levels.")
        return
    
    # Получаем ключевые уровни
    levels = _extract_key_levels(df_final)
    if not levels:
        st.warning("Не удалось извлечь ключевые уровни из данных.")
        return
    
    # Группируем близкие уровни
    grouped = _group_overlapping_levels(levels)
    
    # Временная шкала
    times, time_labels = _generate_trading_session_timeline()
    
    # Создаем фигуру
    fig = go.Figure()
    
    # Уникальные значения уровней для настройки осей
    all_values = list(levels.values())
    if all_values:
        y_min = min(all_values) * 0.995
        y_max = max(all_values) * 1.005
    else:
        y_min, y_max = 640, 660
    
    # Активные уровни (те, для которых есть линии)
    active_levels = set(grouped.keys())
    
    # Добавляем линии для каждого уровня
    legend_entries = []
    for level_value, series_names in grouped.items():
        # Цвет первой серии в группе
        main_series = series_names[0]
        color = SERIES_COLORS.get(main_series, '#FFFFFF')
        
        # Определяем стиль линии
        if main_series in ["Price", "VWAP", "AG", "Max Call OI", "Max Put OI"]:
            dash_style = 'solid'
        elif main_series == "G-Flip":
            dash_style = 'dash'
        else:
            dash_style = 'dot'
        
        # Основная линия
        fig.add_trace(go.Scatter(
            x=times,
            y=[level_value] * len(times),
            mode='lines',
            line=dict(color=color, width=1.5, dash=dash_style),
            name=', '.join(series_names),
            showlegend=True,
            hovertemplate=f"{'<br>'.join(series_names)}: {level_value:.2f}<extra></extra>"
        ))
        
        # Метка справа для групп с несколькими сериями
        if len(series_names) > 1:
            # Сокращаем имена для компактности
            short_names = []
            for s in series_names:
                if "Max" in s:
                    parts = s.split()
                    short_names.append(f"{parts[1]} {parts[2]}" if len(parts) > 2 else s)
                else:
                    short_names.append(s)
            
            fig.add_annotation(
                x=times[-1],
                y=level_value,
                text=' + '.join(short_names),
                xanchor='left',
                yanchor='middle',
                xshift=5,
                font=dict(size=9, color=color),
                showarrow=False
            )
    
    # Настройка layout
    fig.update_layout(
        title=dict(
            text=f"<b>Key Levels</b>",
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            font=dict(size=14, color=FG_COLOR)
        ),
        hovermode='closest',
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        margin=dict(l=60, r=100, t=30, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="left",
            x=0,
            font=dict(size=10, color=FG_COLOR),
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0
        ),
        xaxis=dict(
            title="Time",
            tickmode='array',
            tickvals=times[::2],  # Каждая вторая метка
            ticktext=time_labels[::2],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            tickfont=dict(size=10, color=FG_COLOR),
            titlefont=dict(size=11, color=FG_COLOR)
        ),
        yaxis=dict(
            title="Price",
            range=[y_min, y_max],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            tickfont=dict(size=10),
            titlefont=dict(size=11, color=FG_COLOR),
            # Цвет меток в зависимости от активности
            tickmode='auto'
        ),
        height=400
    )
    
    # Форматируем цвет меток оси Y
    # К сожалению, Plotly не позволяет напрямую задавать разные цвета для разных tick labels
    # Но мы можем использовать аннотации для имитации этого эффекта
    
    # Добавляем текущую дату под графиком
    current_date = datetime.now().strftime("Sep %d, %Y")
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.08,
        text=current_date,
        showarrow=False,
        font=dict(size=10, color=FG_COLOR),
        xanchor="center"
    )
    
    # Отображаем график
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={'displayModeBar': False})

def render_key_levels_with_controls(
    df_final: pd.DataFrame,
    ticker: str,
    spot: Optional[float] = None
) -> None:
    """Отображает Key Levels чарт с контролами в сайдбаре"""
    
    # Заголовок секции
    st.header("Key Levels")
    
    # Контролы в сайдбаре
    with st.sidebar:
        st.subheader("Key Levels — Controls")
        
        # Интервал
        interval = st.selectbox(
            "Interval",
            ["1m", "5m", "15m", "30m", "1h"],
            index=0,
            key="key_levels_interval"
        )
        
        # Лимиты цены
        st.text("Limit")
        col1, col2 = st.columns(2)
        with col1:
            limit_min = st.number_input(
                "",
                min_value=0,
                value=640,
                step=10,
                key="key_levels_limit_min",
                label_visibility="collapsed"
            )
        with col2:
            limit_max = st.number_input(
                "",
                min_value=0,
                value=660,
                step=10,
                key="key_levels_limit_max",
                label_visibility="collapsed"
            )
    
    # Рендерим чарт
    render_key_levels(df_final, ticker, spot)
