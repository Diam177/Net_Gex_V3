
# -*- coding: utf-8 -*-
"""
advanced_analysis_block.py — инфо‑блок «Advanced Options Market Analysis»
с окраской метрик и нормализацией Net GEX; EM(1d)/(1w) в левой колонке без переноса строк.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st


# ---------- Utility computations ----------

def _fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        xv = float(x)
    except Exception:
        return "—"
    if not np.isfinite(xv):
        return "—"
    return f"{xv:,.{digits}f}".replace(",", " ")

def _compute_vwap_value(price_df: Optional[pd.DataFrame]) -> Optional[float]:
    if price_df is None or getattr(price_df, "empty", True):
        return None
    cand = None
    for col in ("vwap", "VWAP", "Vwap", "vw"):
        if col in price_df.columns:
            cand = pd.to_numeric(price_df[col], errors="coerce").dropna()
            break
    if cand is None or cand.empty:
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
    По каждой экспирации возвращает:
      {"atm_call_iv": ..., "atm_put_iv": ..., "T_med": ...}
    Метод: ближайший к |Δ|=0.5 для C и P.
    """
    out: Dict[str, Dict[str, float]] = {}
    if df_corr is None or getattr(df_corr, "empty", True) or "exp" not in df_corr.columns:
        return out

def _per_exp_atm_iv_raw(df_raw: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if df_raw is None or getattr(df_raw, "empty", True):
        return out
    cols = set(map(str, df_raw.columns))
    need = {"exp","side","delta","iv","T"}
    if not need.issubset(cols):
        return out
    df = df_raw.copy()
    df = df[(df["delta"].notna()) & (df["iv"].notna())]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta","iv"])
    for exp, g in df.groupby("exp"):
        res = {"atm_call_iv": float("nan"), "atm_put_iv": float("nan"), "T_med": float("nan")}
        try:
            res["T_med"] = float(np.nanmedian(pd.to_numeric(g.get("T"), errors="coerce")))
        except Exception:
            res["T_med"] = float("nan")
        for side, key in (("C","atm_call_iv"), ("P","atm_put_iv")):
            gs = g[g["side"]==side]
            if gs.empty:
                continue
            d  = pd.to_numeric(gs["delta"], errors="coerce")
            iv = pd.to_numeric(gs["iv"],    errors="coerce")
            if d.notna().any():
                idx = (d.abs() - 0.5).abs().idxmin()
                val = iv.loc[idx] if idx in iv.index else float("nan")
                res[key] = float(val) if np.isfinite(val) else float("nan")
        out[str(exp)] = res
    return out

    req_cols = {"side","iv_corr","delta_corr","T"}
    if not req_cols.issubset(set(df_corr.columns)):
        return out

    for exp, g in df_corr.groupby("exp", sort=False):
        res = {}
        T_med = float(pd.to_numeric(g.get("T"), errors="coerce").median()) if "T" in g.columns else float("nan")
        for side, key in (("C","atm_call_iv"), ("P","atm_put_iv")):
            gs = g[g.get("side")==side].copy()
            if gs.empty:
                res[key] = float("nan")
                continue
            d  = pd.to_numeric(gs.get("delta_corr"), errors="coerce")
            iv = pd.to_numeric(gs.get("iv_corr"), errors="coerce")
            if d.notna().any():
                idx = (d.abs() - 0.5).abs().idxmin()
                val = iv.loc[idx] if idx in iv.index else float("nan")
            else:
                val = float("nan")
            res[key] = float(val) if np.isfinite(val) else float("nan")
        res["T_med"] = T_med if np.isfinite(T_med) else float("nan")
        out[str(exp)] = res
    return out


def _blend(values: List[Tuple[float, float]], mode: str = "1/√T") -> Optional[float]:
    vals = [(float(v), float(T)) for v, T in values if np.isfinite(v) and np.isfinite(T) and T >= 1/252.0]
    if not vals:
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



# ---------- Bracket helpers (threshold labels) ----------
def _bracket_pc(val, *, vol=False):
    try:
        v = float(val)
    except Exception:
        return ""
    if vol:
        if v < 0.90: return "[<0.90]"
        if v > 1.10: return "[>1.10]"
        return "[0.90–1.10]"
    else:
        if v < 0.90: return "[<0.90]"
        if v > 1.20: return "[>1.20]"
        return "[0.90–1.20]"

def _bracket_iv(val, *, asset_class="ETF"):
    try:
        v = float(val)
    except Exception:
        return ""
    if str(asset_class).upper() == "EQUITY":
        if v < 0.30: return "[<0.30]"
        if v > 0.50: return "[>0.50]"
        return "[0.30–0.50]"
    if v < 0.15: return "[<0.15]"
    if v > 0.25: return "[>0.25]"
    return "[0.15–0.25]"

def _bracket_skew(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v <= 0.01: return "[≤0.01]"
    if v >= 0.03: return "[≥0.03]"
    return "[0.01–0.03]"

def _bracket_netgex(sum_val, g_norm):
    try:
        if g_norm is not None and np.isfinite(float(g_norm)):
            g = float(g_norm)
            if g >= 0.15: return "[≥+0.15]"
            if g <= -0.15: return "[≤−0.15]"
            return "[−0.15…+0.15]"
    except Exception:
        pass
    # fallback by sign if no g_norm
    try:
        s = float(sum_val)
    except Exception:
        return ""
    return "[>0]" if s > 0 else ("[<0]" if s < 0 else "[]")
# ---------- Color helpers ----------
def _color_for_value(val, *, lower, upper):
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    if v < lower:
        return "green"
    if v > upper:
        return "red"
    return "orange"

def _color_for_pc(val, *, vol=False):
    return _color_for_value(val, lower=0.90, upper=1.10 if vol else 1.20)

def _color_for_iv(val, *, asset_class="ETF"):
    try:
        v = float(val)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    if asset_class and str(asset_class).upper() == "EQUITY":
        if v < 0.30: return "green"
        if v > 0.50: return "red"
        return "orange"
    if v < 0.15: return "green"
    if v > 0.25: return "red"
    return "orange"

def _color_for_skew(val):
    try:
        v = float(val)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    if v <= 0.01: return "green"
    if v >= 0.03: return "red"
    return "orange"

def _color_for_netgex(sum_val, g_norm):
    try:
        if g_norm is not None and np.isfinite(float(g_norm)):
            g = float(g_norm)
            if g >= 0.15: return "green"
            if g <= -0.15: return "red"
            return "orange"
    except Exception:
        pass
    try:
        s = float(sum_val)
    except Exception:
        return None
    if not np.isfinite(s):
        return None
    if s > 0: return "green"
    if s < 0: return "red"
    return "orange"

def _colored_line(label, value_html, color):
    if color:
        st.markdown(f"<b>{label}:</b> <span style='color:{color};'>{value_html}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"**{label}:** {value_html}")


# ---------- Metrics ----------

def compute_metrics(
    df_final: Optional[pd.DataFrame],
    df_corr: Optional[pd.DataFrame],
    S: Optional[float],
    price_df: Optional[pd.DataFrame] = None,
    selected_exps: Optional[List[str]] = None,
    weight_mode: str = "1/√T", df_raw: Optional[pd.DataFrame] = None) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "S": float(S) if S is not None and np.isfinite(float(S)) else None,
        "VWAP": _compute_vwap_value(price_df),
        "pc_oi": None,
        "pc_vol": None,
        "netgex_sum": None,
        "netgex_is_millions": False,
        "netgex_sum_base": None,
        "ag_sum_base": None,
        "g_norm": None,
        "atm_iv": None,
        "skew": None,
        "em_1d": None,
        "em_1w": None,
    }

    # Put/Call ratios и Net GEX — из df_final
    if df_final is not None and not getattr(df_final, "empty", True):
        cols = set(df_final.columns)
        if {"put_oi","call_oi"}.issubset(cols):
            p = pd.to_numeric(df_final["put_oi"], errors="coerce").fillna(0.0).sum()
            c = pd.to_numeric(df_final["call_oi"], errors="coerce").fillna(0.0).sum()
            metrics["pc_oi"] = (float(p)/float(c)) if c > 0 else None
        if {"put_vol","call_vol"}.issubset(cols):
            pv = pd.to_numeric(df_final["put_vol"], errors="coerce").fillna(0.0).sum()
            cv = pd.to_numeric(df_final["call_vol"], errors="coerce").fillna(0.0).sum()
            metrics["pc_vol"] = (float(pv)/float(cv)) if cv > 0 else None

        if "NetGEX_1pct_M" in cols:
            ngs = pd.to_numeric(df_final["NetGEX_1pct_M"], errors="coerce").fillna(0.0)
            metrics["netgex_sum"] = float(ngs.sum())
            metrics["netgex_is_millions"] = True
            metrics["netgex_sum_base"] = float(ngs.sum() * 1e6)
        elif "NetGEX_1pct" in cols:
            ngs = pd.to_numeric(df_final["NetGEX_1pct"], errors="coerce").fillna(0.0)
            metrics["netgex_sum"] = float(ngs.sum())
            metrics["netgex_is_millions"] = False
            metrics["netgex_sum_base"] = float(ngs.sum())

        if "AG_1pct_M" in cols:
            ags = pd.to_numeric(df_final["AG_1pct_M"], errors="coerce")
            metrics["ag_sum_base"] = float(ags.fillna(0.0).abs().sum() * 1e6)
        elif "AG_1pct" in cols:
            ags = pd.to_numeric(df_final["AG_1pct"], errors="coerce")
            metrics["ag_sum_base"] = float(ags.fillna(0.0).abs().sum())

        agb = metrics["ag_sum_base"]
        ngb = metrics["netgex_sum_base"]
        if agb is not None and ngb is not None and agb > 0:
            gnorm = float(ngb) / float(agb)
            metrics["g_norm"] = max(-1.0, min(1.0, gnorm))

    # ATM IV и Skew — из df_corr
    if df_corr is not None and not getattr(df_corr, "empty", True):
        per = _per_exp_atm_iv(df_corr, S if S is not None else float("nan"))
        if selected_exps:
            per = per or {}
        per = {k: v for k, v in per.items() if k in set(map(str, selected_exps))}
        pairs_iv_T, pairs_skew_T = [], []
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
        metrics["atm_iv"] = _blend(pairs_iv_T, mode=weight_mode) if pairs_iv_T else None
        # Skew only from df_raw
        metrics["skew"] = None
        if df_raw is not None and not getattr(df_raw, "empty", True):
            per_raw = _per_exp_atm_iv_raw(df_raw)
            if selected_exps:
                per_raw = {k: v for k, v in per_raw.items() if k in set(map(str, selected_exps))}
            pairs_skew_T_raw = []
            for d in per_raw.values():
                call_iv = d.get("atm_call_iv"); put_iv = d.get("atm_put_iv"); T_med = d.get("T_med")
                if np.isfinite(call_iv) and np.isfinite(put_iv) and np.isfinite(T_med):
                    pairs_skew_T_raw.append((float(put_iv) - float(call_iv), float(T_med)))
            if pairs_skew_T_raw:
                metrics["skew"] = _blend(pairs_skew_T_raw, mode=weight_mode)
        metrics["_skew_from_corr"] = _blend(pairs_skew_T, mode=weight_mode) if pairs_skew_T else None

    # Expected Moves
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
    asset_class: str = "ETF",
    caption_suffix: str = "Агрегировано по выбранным экспирациям."
, df_raw: Optional[pd.DataFrame] = None) -> Dict[str, Optional[float]]:
    m = compute_metrics(
        df_final=df_final, df_corr=df_corr, S=S, price_df=price_df,
        selected_exps=selected_exps, weight_mode=weight_mode
    , df_raw=df_raw)

    st.markdown(f"### Advanced Options Market Analysis: {ticker}")
    c1, c2, c3, c4 = st.columns(4)

    # Левая колонка + EM без переноса
    with c1:
        _colored_line("Текущая цена", _fmt_num(m["S"], 2), None)
        _colored_line("VWAP", _fmt_num(m["VWAP"], 2), None)
        S_val = m.get('S')
        if m.get('em_1d') is not None and S_val is not None:
            em = float(m['em_1d']); pct = 100.0 * em / float(S_val) if float(S_val)!=0 else float('nan')
            lo = float(S_val) - em; hi = float(S_val) + em
            st.markdown(f"<div style='white-space:nowrap;'><b>Expected Move (1d):</b> ±{_fmt_num(em, 2)} ({_fmt_num(pct, 2)}%) — диапазон [{_fmt_num(lo,2)}; {_fmt_num(hi,2)}]</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='white-space:nowrap;'><b>Expected Move (1d):</b> —</div>", unsafe_allow_html=True)
        if m.get('em_1w') is not None and S_val is not None:
            emw = float(m['em_1w']); pctw = 100.0 * emw / float(S_val) if float(S_val)!=0 else float('nan')
            low = float(S_val) - emw; hiw = float(S_val) + emw
            st.markdown(f"<div style='white-space:nowrap;'><b>Expected Move (1w):</b> ±{_fmt_num(emw, 2)} ({_fmt_num(pctw, 2)}%) — диапазон [{_fmt_num(low,2)}; {_fmt_num(hiw,2)}]</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='white-space:nowrap;'><b>Expected Move (1w):</b> —</div>", unsafe_allow_html=True)

    with c2:
        _colored_line('P/C Ratio (OI)', f"{_fmt_num(m['pc_oi'], 2)} {_bracket_pc(m.get('pc_oi'), vol=False)}", _color_for_pc(m.get('pc_oi'), vol=False))
        _colored_line('P/C Ratio (Volume)', f"{_fmt_num(m['pc_vol'], 2)} {_bracket_pc(m.get('pc_vol'), vol=True)}", _color_for_pc(m.get('pc_vol'), vol=True))
        # Divergence detection between OI and Volume ratios
        try:
            oi = float(m.get('pc_oi')) if m.get('pc_oi') is not None else None
            vo = float(m.get('pc_vol')) if m.get('pc_vol') is not None else None
        except Exception:
            oi, vo = None, None
        if oi is not None and vo is not None:
            if oi > 1.10 and vo < 0.90:
                st.caption("Дивергенция: толпа vs умные деньги (бычья)")
            elif oi < 0.90 and vo > 1.10:
                st.caption("Дивергенция: толпа vs умные деньги (медвежья)")


    with c3:
        if m['netgex_sum'] is None:
            ng = '—'
        else:
            if m['netgex_is_millions']:
                ng = f"{_fmt_num(m['netgex_sum'], 0)}M"
            else:
                ng = f"{_fmt_num(m['netgex_sum'], 0)}"
        _colored_line('Net GEX (sum)', f"{ng} {_bracket_netgex(m.get('netgex_sum'), m.get('g_norm'))}", _color_for_netgex(m.get('netgex_sum'), m.get('g_norm')))
        st.caption(caption_suffix)

    with c4:
        _colored_line('ATM IV', f"{_fmt_num(m['atm_iv'], 2)} {_bracket_iv(m.get('atm_iv'), asset_class=asset_class)}", _color_for_iv(m.get('atm_iv'), asset_class=asset_class))
        _colored_line('Skew (Put/Call IV)', f"{_fmt_num(m['skew'], 3)} {_bracket_skew(m.get('skew'))}", _color_for_skew(m.get('skew')))

    return m
