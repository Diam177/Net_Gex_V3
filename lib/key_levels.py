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
from datetime import datetime, time

# Import G-Flip calculation from netgex_chart
from lib.netgex_chart import _compute_gamma_flip_from_table

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    raise RuntimeError("Требуется пакет 'plotly' (plotly>=5.22.0)") from e

# Цвета серий (соответствуют скриншоту)
SERIES_COLORS = {
    "Price": "#5A9FD4",           # голубой
    "VWAP": "#FFA500",           # оранжевый
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

# Цвета фона и осей
BG_COLOR = '#0E1117'
FG_COLOR = '#FFFFFF'
GRID_COLOR = 'rgba(255,255,255,0.05)'
AXIS_COLOR_INACTIVE = '#666666'  # серый для неактивных значений
AXIS_COLOR_ACTIVE = '#FFFFFF'    # белый для активных значений

def _generate_intraday_timeline() -> Tuple[List[float], List[str]]:
    """Генерирует временную шкалу внутридневной торговой сессии"""
    # Создаем временные точки с 9:30 до 16:00 с интервалом 30 минут
    time_points = []
    time_labels = []
    
    # Начало сессии 9:30
    current_hour = 9
    current_min = 30
    
    while current_hour < 16 or (current_hour == 16 and current_min == 0):
        time_val = current_hour + current_min / 60.0
        time_points.append(time_val)
        time_labels.append(f"{current_hour:02d}:{current_min:02d}")
        
        # Добавляем 30 минут
        current_min += 30
        if current_min >= 60:
            current_min -= 60
            current_hour += 1
    
    # Добавляем конец сессии если его нет
    if time_points[-1] != 16.0:
        time_points.append(16.0)
        time_labels.append("16:00")
    
    return time_points, time_labels

def _find_key_levels_from_data(df_final: pd.DataFrame) -> Dict[str, float]:
    """Извлекает ключевые уровни из финальной таблицы"""
    levels = {}
    
    if df_final is None or df_final.empty:
        return levels
    
    # Price (текущая цена спота)
    if "S" in df_final.columns and df_final["S"].notna().any():
        levels["Price"] = float(df_final["S"].dropna().iloc[0])
    
    # VWAP - взвешенная по объему средняя цена
    if "K" in df_final.columns:
        total_vol = 0
        vwap_sum = 0
        
        if "call_vol" in df_final.columns and "put_vol" in df_final.columns:
            vol = df_final["call_vol"].fillna(0) + df_final["put_vol"].fillna(0)
            if vol.sum() > 0:
                levels["VWAP"] = float((df_final["K"] * vol).sum() / vol.sum())
        elif "call_oi" in df_final.columns and "put_oi" in df_final.columns:
            # Фоллбэк на OI если нет volume
            oi = df_final["call_oi"].fillna(0) + df_final["put_oi"].fillna(0)
            if oi.sum() > 0:
                levels["VWAP"] = float((df_final["K"] * oi).sum() / oi.sum())
    
    # Max Neg GEX (самая отрицательная точка)
    if "NetGEX_1pct" in df_final.columns:
        neg_mask = df_final["NetGEX_1pct"] < 0
        if neg_mask.any():
            min_idx = df_final.loc[neg_mask, "NetGEX_1pct"].idxmin()
            levels["Max Neg GEX"] = float(df_final.loc[min_idx, "K"])
            
            # Дополнительные Neg GEX уровни
            sorted_neg = df_final.loc[neg_mask].nsmallest(3, "NetGEX_1pct")
            if len(sorted_neg) > 1:
                levels["Neg Net GEX #2"] = float(sorted_neg.iloc[1]["K"])
            if len(sorted_neg) > 2:
                levels["Neg Net GEX #3"] = float(sorted_neg.iloc[2]["K"])
    
    # Max Pos GEX (самая положительная точка)
    if "NetGEX_1pct" in df_final.columns:
        pos_mask = df_final["NetGEX_1pct"] > 0
        if pos_mask.any():
            max_idx = df_final.loc[pos_mask, "NetGEX_1pct"].idxmax()
            levels["Max Pos GEX"] = float(df_final.loc[max_idx, "K"])
    
    # Max Put OI
    if "put_oi" in df_final.columns and df_final["put_oi"].notna().any():
        max_idx = df_final["put_oi"].idxmax()
        levels["Max Put OI"] = float(df_final.loc[max_idx, "K"])
    
    # Max Call OI
    if "call_oi" in df_final.columns and df_final["call_oi"].notna().any():
        max_idx = df_final["call_oi"].idxmax()
        levels["Max Call OI"] = float(df_final.loc[max_idx, "K"])
    
    # Max Put Volume
    if "put_vol" in df_final.columns and df_final["put_vol"].notna().any():
        max_idx = df_final["put_vol"].idxmax()
        levels["Max Put Volume"] = float(df_final.loc[max_idx, "K"])
    
    # Max Call Volume
    if "call_vol" in df_final.columns and df_final["call_vol"].notna().any():
        max_idx = df_final["call_vol"].idxmax()
        levels["Max Call Volume"] = float(df_final.loc[max_idx, "K"])
    
    # AG - абсолютная гамма
    if "AG_1pct" in df_final.columns and df_final["AG_1pct"].notna().any():
        sorted_ag = df_final.nlargest(3, "AG_1pct")
        if len(sorted_ag) > 0:
            levels["AG"] = float(sorted_ag.iloc[0]["K"])