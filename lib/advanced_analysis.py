
# -*- coding: utf-8 -*-
"""
Advanced Options Market Analysis — calculations & rendering

Задача:
- Собрать сводные метрики по выбранным экспирациям (используем уже агрегированный df).
- Аккуратно отрисовать блок под чартом Key Levels.
- Ничего не ломать в остальной логике приложения.

Файл экспортирует две функции:

    from lib.advanced_analysis import update_ao_summary, render_advanced_analysis_block

1) update_ao_summary(ticker, spot, df, all_series_ctx=None):
   • Считает:
       - Put/Call Ratio (OI‑based и Volume‑based),
       - Aggregated Net GEX (сумма по всем страйкам; если выбрано несколько экспираций — по суммарному df),
       - Skew (Put/Call IV Skew) как среднее по выбранным экспирациям: IV_put(ATM) − IV_call(ATM),
         где ATM берётся как страйк, ближайший к spot.
   • Пытается взять VWAP последней сессии из st.session_state["kl_vwap_last"], если он уже рассчитан в Key Levels.
   • Сохраняет всё в st.session_state["ao_summary"].
   • Без вывода в UI.

2) render_advanced_analysis_block(ticker):
   • Рисует блок «Advanced Options Market Analysis: «Тикер»» внизу страницы.
   • Безопасен к отсутствующим данным: пропускает поля, где данных нет.

Примечания:
- IV Rank / IV Percentile стандартно требуют истории волатильности. В текущей версии
  модуль НЕ генерирует эти значения без исторических данных; вместо них показывается «—».
- Никаких сетевых запросов здесь НЕТ.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import math
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------- helpers -------------------------

def _nan_sum(x) -> float:
    try:
        return float(np.nansum(np.asarray(x, dtype=float)))
    except Exception:
        return float('nan')

def _safe_ratio(num: float, den: float) -> float:
    try:
        den = float(den)
        if not math.isfinite(den) or abs(den) < 1e-12:
            return float('nan')
        val = float(num) / den
        return val
    except Exception:
        return float('nan')

def _nearest_strike_iv(iv_map: Dict[float, float], spot: float) -> Optional[float]:
    """Берём IV по ближайшему к spot страйку из словаря {strike: iv}."""
    try:
        if not iv_map:
            return None
        ks = np.asarray(list(iv_map.keys()), dtype=float)
        ivs = np.asarray([iv_map[k] for k in iv_map.keys()], dtype=float)
        idx = int(np.nanargmin(np.abs(ks - float(spot))))
        v = float(ivs[idx])
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


# ------------------------- public API -------------------------

def update_ao_summary(
    ticker: str,
    spot: Optional[float],
    df: pd.DataFrame,
    all_series_ctx: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Подготовка агрегированных метрик и сохранение их в session_state["ao_summary"].

    Параметры:
    - ticker: строка тикера.
    - spot: текущая цена базового актива (из quote в app.py).
    - df: уже агрегированный по выбранным экспирациям датафрейм со столбцами:
        ["Strike","Put OI","Call OI","Put Volume","Call Volume","Net Gex","AG","PZ","PZ_FP"]
    - all_series_ctx: список по экспирациям из app.py с полями:
        {"strikes","call_oi","put_oi","call_vol","put_vol","iv_call","iv_put","T"}.
        Используется только для оценки Skew (IV_put^ATM − IV_call^ATM).

    Возвращает: None (пишет в st.session_state).
    """
    try:
        summary: Dict[str, Any] = {"ticker": ticker}

        # 1) Current price (spot)
        summary["spot"] = float(spot) if spot is not None and math.isfinite(float(spot or np.nan)) else None

        # 2) VWAP — возьмём из Key Levels, если уже посчитан
        vwap_last = st.session_state.get("kl_vwap_last", None)
        try:
            summary["vwap"] = float(vwap_last) if vwap_last is not None and math.isfinite(float(vwap_last)) else None
        except Exception:
            summary["vwap"] = None

        # 3) P/C Ratios (by OI and by Volume) из агрегированного df
        def _col(x): 
            return np.asarray(df[x].values, dtype=float) if (isinstance(df, pd.DataFrame) and x in df.columns) else np.array([np.nan])
        sum_put_oi  = _nan_sum(_col("Put OI"))
        sum_call_oi = _nan_sum(_col("Call OI"))
        pc_oi = _safe_ratio(sum_put_oi, sum_call_oi)

        sum_put_vol  = _nan_sum(_col("Put Volume"))
        sum_call_vol = _nan_sum(_col("Call Volume"))
        pc_vol = _safe_ratio(sum_put_vol, sum_call_vol)

        summary["pc_ratio_oi"] = pc_oi
        summary["pc_ratio_vol"] = pc_vol

        # 4) Aggregated Net GEX
        summary["net_gex_sum"] = _nan_sum(_col("Net Gex"))

        # 5) IV Rank / Percentile — нет истории → n/a
        summary["iv_rank"] = None
        summary["iv_percentile"] = None

        # 6) Skew (Put/Call IV) — по ATM, усреднённый по выбранным экспирациям
        skew_vals: List[float] = []
        if all_series_ctx and (spot is not None) and math.isfinite(float(spot)):
            s = float(spot)
            for ctx in all_series_ctx:
                iv_call = (ctx or {}).get("iv_call") or {}
                iv_put  = (ctx or {}).get("iv_put")  or {}
                ivc = _nearest_strike_iv(iv_call, s)
                ivp = _nearest_strike_iv(iv_put,  s)
                if (ivc is not None) and (ivp is not None) and math.isfinite(ivc) and math.isfinite(ivp):
                    skew_vals.append(float(ivp) - float(ivc))
        if skew_vals:
            summary["skew_pc_iv"] = float(np.nanmean(np.asarray(skew_vals, dtype=float)))
        else:
            summary["skew_pc_iv"] = None

        st.session_state["ao_summary"] = summary

    except Exception as e:
        # Не ломаем страницу в случае любой ошибки
        st.session_state["ao_summary"] = {"ticker": ticker, "error": str(e)}


def _colorize_ratio(val: Optional[float]) -> str:
    """
    Возвращает HTML-span с раскраской по правилам пользователя.
    Пороги: < 0.8 — зелёный, > 1.2 — красный, иначе — жёлтый.
    Если val = None/NaN → '—'.
    """
    try:
        if val is None or not math.isfinite(float(val)):
            return "<span>—</span>"
        v = float(val)
        if v < 0.8:
            color = "green"
        elif v > 1.2:
            color = "red"
        else:
            color = "goldenrod"
        return f"<span style='color:{color}; font-weight:600;'>{v:.2f}</span>"
    except Exception:
        return "<span>—</span>"


def _colorize_net_gex(val: Optional[float]) -> str:
    """
    Зеленый — если > 0; красный — если < 0; иначе — обычный.
    """
    try:
        if val is None or not math.isfinite(float(val)):
            return "<span>—</span>"
        v = float(val)
        if v > 0:
            color = "green"
        elif v < 0:
            color = "red"
        else:
            color = "inherit"
        return f"<span style='color:{color}; font-weight:600;'>{v:,.1f}</span>"
    except Exception:
        return "<span>—</span>"


def render_advanced_analysis_block(ticker: str) -> None:
    """
    Рендер блока под чартом Key Levels.
    Читает st.session_state["ao_summary"] и ничего не пересчитывает.
    """
    try:
        sm = st.session_state.get("ao_summary", {}) or {}
        if sm.get("ticker") != ticker and "ticker" in sm and ticker:
            # допускаем, но сигналим юзеру
            st.caption(f"⚠️ Данные собраны для: {sm.get('ticker')}, а выбран: {ticker}")

        st.subheader(f'Advanced Options Market Analysis: «{ticker}»')

        # --- значения/тексты ---
        spot = sm.get("spot", None)
        vwap = sm.get("vwap", None)
        pc_oi = sm.get("pc_ratio_oi", None)
        pc_vol = sm.get("pc_ratio_vol", None)
        net_gex_sum = sm.get("net_gex_sum", None)
        iv_rank = sm.get("iv_rank", None)
        iv_pct  = sm.get("iv_percentile", None)
        skew    = sm.get("skew_pc_iv", None)

        price_text = ("—" if spot is None or not math.isfinite(float(spot)) 
                      else f"{float(spot):,.2f}")
        vwap_text  = ("—" if vwap is None or not math.isfinite(float(vwap))
                      else f"{float(vwap):,.2f}")
        iv_rank_text = ("—" if iv_rank is None or not math.isfinite(float(iv_rank))
                        else f"{float(iv_rank):.1f}")
        iv_pct_text  = ("—" if iv_pct is None or not math.isfinite(float(iv_pct))
                        else f"{float(iv_pct):.1f}")
        skew_text    = ("—" if skew is None or not math.isfinite(float(skew))
                        else f"{float(skew):.4f}")

        pc_oi_html  = _colorize_ratio(pc_oi)
        pc_vol_html = _colorize_ratio(pc_vol)
        ng_html     = _colorize_net_gex(net_gex_sum)

        # --- layout ---
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.3])
        with c1:
            st.metric("Текущая цена", price_text)
            st.markdown(f"**VWAP:** {vwap_text}")
        with c2:
            st.markdown("**P/C Ratio (OI):** " + pc_oi_html, unsafe_allow_html=True)
            st.markdown("**P/C Ratio (Volume):** " + pc_vol_html, unsafe_allow_html=True)
        with c3:
            st.markdown("**Net GEX (sum):** " + ng_html, unsafe_allow_html=True)
            st.caption("Агрегировано по выбранным экспирациям.")
        with c4:
            st.markdown(f"**IV Rank / Percentile:** {iv_rank_text} / {iv_pct_text}")
            st.markdown(f"**Skew (Put/Call IV):** {skew_text}")

    except Exception as e:  # never break the page
        st.caption(f"Advanced analysis block error: {e}")
