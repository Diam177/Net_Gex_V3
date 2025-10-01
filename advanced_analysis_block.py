
# -*- coding: utf-8 -*-
"""
advanced_analysis_block.py — компактный инфо‑блок «Advanced Options Market Analysis».

Показывает:
- Текущая цена (S)
- VWAP (интрадей)
- P/C Ratio (OI)
- P/C Ratio (Volume)
- Net GEX (sum)
- ATM IV
- Skew (Put/Call IV)
- Expected Move (1d) и (1w) + диапазоны

Зависимости: streamlit, pandas, numpy
Входные данные берутся из уже рассчитанных таблиц проекта:
- df_final: финальная таблица (single или multi), содержит по K: call_oi, put_oi, call_vol, put_vol, NetGEX_1pct[_M], S.
- df_corr: «исправленная» таблица из sanitize_window (по всем выбранным экспирациям), содержит: exp, side (C/P), K, S, T, iv_corr, delta_corr.
- price_df: опционально, DataFrame с колонками ['time','price','vwap'] для VWAP.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st


# ---------- Utility computations ----------

def _fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:,.{digits}f}".replace(",", " ")

def _compute_vwap_value(price_df: Optional[pd.DataFrame]) -> Optional[float]:
    if price_df is None or price_df.empty:
        return None
    cand = None
    for col in ["vwap", "VWAP", "Vwap"]:
        if col in price_df.columns:
            cand = pd.to_numeric(price_df[col], errors="coerce").dropna()
            break
    if cand is None or cand.empty:
        # если нет готового VWAP, посчитаем кумулятивно
        if {"price","volume"}.issubset(set(price_df.columns)):
            pr  = pd.to_numeric(price_df["price"], errors="coerce")
            vol = pd.to_numeric(price_df["volume"], errors="coerce").fillna(0.0)
            cum_vol = vol.cumsum().replace(0, np.nan)
            c = (pr.mul(vol)).cumsum() / cum_vol
            cand = c.dropna()
        else:
            return None
    return float(cand.iloc[-1]) if not cand.empty else None


def _per_exp_atm_iv(df_corr: pd.DataFrame, S: float) -> Dict[str, Dict[str, float]]:
    """
    Возвращает по каждой экспирации словарь:
      {"atm_call_iv": ..., "atm_put_iv": ..., "T_med": ...}
    Метод: выбираем по каждой стороне контракт с |abs(delta_corr) - 0.5| минимальным.
    """
    out: Dict[str, Dict[str, float]] = {}
    if df_corr is None or df_corr.empty or "exp" not in df_corr.columns:
        return out
    req_cols = {"side","iv_corr","delta_corr","T"}
    have = set(df_corr.columns)
    if not req_cols.issubset(have):
        return out

    for exp, g in df_corr.groupby("exp", sort=False):
        res = {}
        T_med = float(pd.to_numeric(g.get("T"), errors="coerce").median()) if "T" in g.columns else float("nan")
        for side, key in [("C","atm_call_iv"), ("P","atm_put_iv")]:
            gs = g[g.get("side")==side].copy()
            if gs.empty:
                res[key] = float("nan")
                continue
            # ближайший к |delta|=0.5
            d = pd.to_numeric(gs.get("delta_corr"), errors="coerce")
            iv = pd.to_numeric(gs.get("iv_corr"), errors="coerce")
            idx = (d.abs() - 0.5).abs().idxmin() if d.notna().any() else None
            val = float(iv.loc[idx]) if (idx in iv.index and np.isfinite(iv.loc[idx])) else float("nan")
            res[key] = val
        res["T_med"] = T_med if np.isfinite(T_med) else float("nan")
        out[str(exp)] = res
    return out


def _blend(values: List[Tuple[float, float]], mode: str = "1/√T") -> Optional[float]:
    """
    Смешивание по T: values = [(val, T_med), ...].
    """
    vals = [(float(v), float(T)) for v, T in values if np.isfinite(v) and np.isfinite(T) and T >= 1/252.0]
    if not vals:
        # без T — возьмём простую медиану доступных значений
        vals = [(float(v), 1.0) for v, T in values if np.isfinite(v)]
    if not vals:
        return None
    if mode == "1/T":
        w = [1.0 / max(T, 1/252.0) for _, T in vals]
    elif mode == "1/√T":
        w = [1.0 / math.sqrt(max(T, 1/252.0)) for _, T in vals]
    else:
        w = [1.0 for _ in vals]
    w_sum = sum(w) or 1.0
    num = sum(v * wi for (v, _), wi in zip(vals, w))
    return num / w_sum


def compute_metrics(
    df_final: Optional[pd.DataFrame],
    df_corr: Optional[pd.DataFrame],
    S: Optional[float],
    price_df: Optional[pd.DataFrame] = None,
    selected_exps: Optional[List[str]] = None,
    weight_mode: str = "1/√T"
) -> Dict[str, Optional[float]]:
    """
    Возвращает словарь метрик для блока.
    """
    metrics: Dict[str, Optional[float]] = {
        "S": float(S) if S is not None and np.isfinite(S) else None,
        "VWAP": _compute_vwap_value(price_df),
        "pc_oi": None,
        "pc_vol": None,
        "netgex_sum": None,
        "netgex_is_millions": False,
        "atm_iv": None,
        "skew": None,
        "em_1d": None,
        "em_1w": None,
    }

    # Put/Call ratios и Net GEX — из df_final
    if df_final is not None and not getattr(df_final, "empty", True):
        cols = set(df_final.columns)
        # P/C (OI)
        if {"put_oi","call_oi"}.issubset(cols):
            p = pd.to_numeric(df_final["put_oi"], errors="coerce").fillna(0.0).sum()
            c = pd.to_numeric(df_final["call_oi"], errors="coerce").fillna(0.0).sum()
            metrics["pc_oi"] = (float(p)/float(c)) if c > 0 else None
        # P/C (Volume)
        if {"put_vol","call_vol"}.issubset(cols):
            pv = pd.to_numeric(df_final["put_vol"], errors="coerce").fillna(0.0).sum()
            cv = pd.to_numeric(df_final["call_vol"], errors="coerce").fillna(0.0).sum()
            metrics["pc_vol"] = (float(pv)/float(cv)) if cv > 0 else None
        # Net GEX sum
        if "NetGEX_1pct_M" in cols:
            metrics["netgex_sum"] = float(pd.to_numeric(df_final["NetGEX_1pct_M"], errors="coerce").fillna(0.0).sum())
            metrics["netgex_is_millions"] = True
        elif "NetGEX_1pct" in cols:
            metrics["netgex_sum"] = float(pd.to_numeric(df_final["NetGEX_1pct"], errors="coerce").fillna(0.0).sum())
            metrics["netgex_is_millions"] = False

    # ATM IV и Skew — из df_corr
    if df_corr is not None and not getattr(df_corr, "empty", True):
        per = _per_exp_atm_iv(df_corr, S if S is not None else float("nan"))
        # фильтруем по выбранным экспирациям, если переданы
        if selected_exps:
            per = {k: v for k, v in per.items() if k in set(map(str, selected_exps))}
        # blend ATM IV (среднее call/put per expiry, затем взвешивание по T)
        pairs_iv_T = []
        pairs_skew_T = []
        for e, d in per.items():
            call_iv = d.get("atm_call_iv")
            put_iv  = d.get("atm_put_iv")
            T_med   = d.get("T_med")
            if (call_iv is not None) and (put_iv is not None) and np.isfinite(call_iv) and np.isfinite(put_iv):
                pairs_iv_T.append( (0.5*(float(call_iv)+float(put_iv)), float(T_med)) )
                pairs_skew_T.append( (float(put_iv) - float(call_iv), float(T_med)) )
            elif call_iv is not None and np.isfinite(call_iv):
                pairs_iv_T.append( (float(call_iv), float(T_med)) )
            elif put_iv is not None and np.isfinite(put_iv):
                pairs_iv_T.append( (float(put_iv), float(T_med)) )
        atm_iv_blend = _blend(pairs_iv_T, mode=weight_mode) if pairs_iv_T else None
        skew_blend   = _blend(pairs_skew_T, mode=weight_mode) if pairs_skew_T else None
        metrics["atm_iv"] = atm_iv_blend
        metrics["skew"] = skew_blend

    # Expected Move
    S_val = metrics["S"]
    iv_val = metrics["atm_iv"]
    if (S_val is not None) and (iv_val is not None):
        em_1d = float(S_val) * float(iv_val) * math.sqrt(1.0/252.0)
        em_1w = float(S_val) * float(iv_val) * math.sqrt(5.0/252.0)
        metrics["em_1d"] = em_1d
        metrics["em_1w"] = em_1w

    return metrics


# ---------- Streamlit renderer ----------

def render_advanced_analysis_block(
    ticker: str,
    df_final: Optional[pd.DataFrame],
    df_corr: Optional[pd.DataFrame],
    S: Optional[float],
    price_df: Optional[pd.DataFrame] = None,
    selected_exps: Optional[List[str]] = None,
    weight_mode: str = "1/√T",
    caption_suffix: str = "Агрегировано по выбранным экспирациям."
) -> Dict[str, Optional[float]]:
    """
    Рисует инфо‑блок и возвращает словарь метрик (для логов/тестов).
    """
    m = compute_metrics(
        df_final=df_final, df_corr=df_corr, S=S, price_df=price_df,
        selected_exps=selected_exps, weight_mode=weight_mode
    )

    st.markdown(f"### Advanced Options Market Analysis: {ticker}")
    # Сетка 4x2 как в примере
    c1, c2, c3, c4 = st.columns(4)

    # Левая колонка
    with c1:
        st.markdown(f"**Текущая цена:** {_fmt_num(m['S'], 2)}")
        st.markdown(f"**VWAP:** {_fmt_num(m['VWAP'], 2)}")
    # Вторая колонка
    with c2:
        st.markdown(f"**P/C Ratio (OI):** {_fmt_num(m['pc_oi'], 2)}")
        st.markdown(f"**P/C Ratio (Volume):** {_fmt_num(m['pc_vol'], 2)}")
    # Третья колонка
    with c3:
        if m["netgex_sum"] is None:
            ng = "—"
        else:
            if m["netgex_is_millions"]:
                ng = f"{_fmt_num(m['netgex_sum'], 0)}M"
            else:
                ng = f"{_fmt_num(m['netgex_sum'], 0)}"
        st.markdown(f"**Net GEX (sum):** {ng}")
        st.caption(caption_suffix)
    # Четвёртая колонка
    with c4:
        st.markdown(f"**ATM IV:** {_fmt_num(m['atm_iv'], 2)}")
        st.markdown(f"**Skew (Put/Call IV):** {_fmt_num(m['skew'], 2)}")

    # Нижняя строка: Expected Move
    c5, c6 = st.columns(2)
    S_val = m["S"]
    if m["em_1d"] is not None and S_val is not None:
        em = m["em_1d"]; pct = 100.0 * em / float(S_val) if float(S_val)!=0 else float("nan")
        lo = float(S_val) - em; hi = float(S_val) + em
        st.session_state["__em_1d_vals__"] = (em, pct, lo, hi)  # для возможной отладки
        with c5:
            st.markdown(f"**Expected Move (1d):** ±{_fmt_num(em, 2)} ({_fmt_num(pct, 2)}%) — диапазон [{_fmt_num(lo,2)}; {_fmt_num(hi,2)}]")
    else:
        with c5:
            st.markdown("**Expected Move (1d):** —")

    if m["em_1w"] is not None and S_val is not None:
        em = m["em_1w"]; pct = 100.0 * em / float(S_val) if float(S_val)!=0 else float("nan")
        lo = float(S_val) - em; hi = float(S_val) + em
        st.session_state["__em_1w_vals__"] = (em, pct, lo, hi)
        with c6:
            st.markdown(f"**Expected Move (1w):** ±{_fmt_num(em, 2)} ({_fmt_num(pct, 2)}%) — диапазон [{_fmt_num(lo,2)}; {_fmt_num(hi,2)}]")
    else:
        with c6:
            st.markdown("**Expected Move (1w):** —")

    return m
