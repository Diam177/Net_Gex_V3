# -*- coding: utf-8 -*-
"""
Advanced Options Market Analysis — calculations & rendering.

Exports:
    - update_ao_summary(df_options, current_price, ...)
    - render_advanced_analysis_block(fallback_ticker=None, vwap_series=None)
"""

from __future__ import annotations

from typing import Optional, Sequence, Mapping, Any
import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # for type checkers

import streamlit as st


# ---------------------------
# Small helpers
# ---------------------------

def _safe_sum(series) -> Optional[float]:
    try:
        if series is None:
            return None
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(arr.sum())
    except Exception:
        return None


def _safe_last(array_like) -> Optional[float]:
    try:
        arr = np.asarray(array_like, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(arr[-1])
    except Exception:
        return None


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0 or not np.isfinite(den) or not np.isfinite(num):
        return None
    try:
        return float(num) / float(den)
    except Exception:
        return None


def _colorize_ratio(val: Optional[float]) -> str:
    """
    Colorize P/C ratio per thresholds:
      < 0.8  -> green
      0.8-1.2 -> yellow
      > 1.2  -> red
    Returns HTML <span>.
    """
    if not isinstance(val, (int, float)) or not np.isfinite(val):
        return "—"
    v = float(val)
    if v < 0.8:
        color = "#22c55e"  # green-500
    elif v > 1.2:
        color = "#ef4444"  # red-500
    else:
        color = "#eab308"  # yellow-500
    return f"<span style='color:{color}'>{v:.2f}</span>"


def _colorize_netgex(val: Optional[float]) -> str:
    if not isinstance(val, (int, float)) or not np.isfinite(val):
        return "—"
    color = "#22c55e" if val >= 0 else "#ef4444"
    return f"<span style='color:{color}'>{val:,.0f}</span>"


# ---------------------------
# Public API
# ---------------------------

def update_ao_summary(
    ticker: str,
    df_options,  # pandas.DataFrame aggregated by strike across selected expirations
    current_price: Optional[float],
    selected_expirations: Optional[Sequence[str]] = None,
    extra_iv: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Compute a compact summary and store it to st.session_state['ao_summary'].
    Expected df_options columns (missing are ok): put_oi, call_oi, put_volume, call_volume, net_gex.
    """
    if df_options is None or pd is None:
        st.session_state["ao_summary"] = {
            "ticker": ticker,
            "S": float(current_price) if isinstance(current_price, (int, float)) and np.isfinite(current_price) else None,
            "pc_oi": None,
            "pc_vol": None,
            "net_gex_total": None,
            "expirations": list(selected_expirations or []),
        }
        if isinstance(extra_iv, Mapping):
            st.session_state["iv_summary"] = dict(extra_iv)
        return

    try:
        put_oi_sum   = _safe_sum(df_options.get("Put OI")  if "Put OI"  in df_options.columns else df_options.get("put_oi"))
        call_oi_sum  = _safe_sum(df_options.get("Call OI") if "Call OI" in df_options.columns else df_options.get("call_oi"))
        put_vol_sum  = _safe_sum(df_options.get("Put Volume")  if "Put Volume"  in df_options.columns else df_options.get("put_volume"))
        call_vol_sum = _safe_sum(df_options.get("Call Volume") if "Call Volume" in df_options.columns else df_options.get("call_volume"))
        net_gex_sum  = _safe_sum(df_options.get("Net GEX") if "Net GEX" in df_options.columns else df_options.get("net_gex"))
    except Exception:
        put_oi_sum = call_oi_sum = put_vol_sum = call_vol_sum = net_gex_sum = None

    summary = {
        "ticker": ticker,
        "S": float(current_price) if isinstance(current_price, (int, float)) and np.isfinite(current_price) else None,
        "pc_oi": _ratio(put_oi_sum, call_oi_sum),
        "pc_vol": _ratio(put_vol_sum, call_vol_sum),
        "net_gex_total": net_gex_sum,
        "expirations": list(selected_expirations or []),
    }

    if isinstance(extra_iv, Mapping):
        st.session_state["iv_summary"] = dict(extra_iv)

    st.session_state["ao_summary"] = summary


def render_advanced_analysis_block(
    fallback_ticker: Optional[str] = None,
    vwap_series=None,
) -> None:
    """
    Render the block using previously computed st.session_state['ao_summary'].
    Place this **below** the Key Levels chart.
    """
    try:
        ao = st.session_state.get("ao_summary", {}) or {}
        ticker = ao.get("ticker") or (fallback_ticker or "")
        # spot/price
        S = ao.get("S")
        if S is None:
            # try common fallbacks that may exist in the app session state
            S = st.session_state.get("spot") or st.session_state.get("last_price") or st.session_state.get("price")
        try:
            S = float(S)
            if not np.isfinite(S):
                S = None
        except Exception:
            S = None

        vwap_last = _safe_last(vwap_series)

        pc_oi_html  = _colorize_ratio(ao.get("pc_oi"))
        pc_vol_html = _colorize_ratio(ao.get("pc_vol"))
        ng_html     = _colorize_netgex(ao.get("net_gex_total"))

        # IV metrics (optional)
        ivs = st.session_state.get("iv_summary", {}) or {}
        atm_iv = ivs.get("atm_iv")
        skew   = ivs.get("skew")
        atm_iv_text = f"{atm_iv:.2f}" if isinstance(atm_iv, (int, float)) and np.isfinite(atm_iv) else "—"
        skew_text   = f"{skew:.2f}"   if isinstance(skew,   (int, float)) and np.isfinite(skew)   else "—"

        # Expected Moves (1d, 1w) from ATM IV
        em1d_text = em1w_text = None
        if (S is not None) and isinstance(atm_iv, (int, float)) and np.isfinite(atm_iv) and atm_iv >= 0:
            vol1d = float(atm_iv) * float(np.sqrt(1.0/252.0))
            vol1w = float(atm_iv) * float(np.sqrt(5.0/252.0))
            em1d = float(S) * vol1d
            em1w = float(S) * vol1w
            rng1d = f"[{S - em1d:.2f}; {S + em1d:.2f}]"
            rng1w = f"[{S - em1w:.2f}; {S + em1w:.2f}]"
            em1d_text = f"±${em1d:.2f} ({vol1d*100:.2f}%) — диапазон {rng1d}"
            em1w_text = f"±${em1w:.2f} ({vol1w*100:.2f}%) — диапазон {rng1w}"

        em_tip = "<span style='cursor:help' title='EM = S × ATM_IV × sqrt(t). t=1/252 (1d), 5/252 (1w)'>ℹ️</span>"

        st.markdown(f"### Advanced Options Market Analysis: {ticker}")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"**Текущая цена:** {S:.2f}" if S is not None else "**Текущая цена:** —")
            if isinstance(vwap_last, (int, float)) and np.isfinite(vwap_last):
                st.markdown(f"**VWAP:** {vwap_last:.2f}")
            else:
                st.markdown("**VWAP:** —")
        with c2:
            st.markdown("**P/C Ratio (OI):** " + pc_oi_html, unsafe_allow_html=True)
            st.markdown("**P/C Ratio (Volume):** " + pc_vol_html, unsafe_allow_html=True)
        with c3:
            st.markdown("**Net GEX (sum):** " + ng_html, unsafe_allow_html=True)
            st.caption("Агрегировано по выбранным экспирациям.")
        with c4:
            st.markdown(f"**ATM IV:** {atm_iv_text}")
            st.markdown(f"**Skew (Put/Call IV):** {skew_text}")
            if em1d_text:
                st.markdown("**Expected Move (1d):** " + em1d_text + " " + em_tip, unsafe_allow_html=True)
            if em1w_text:
                st.markdown("**Expected Move (1w):** " + em1w_text + " " + em_tip, unsafe_allow_html=True)

    except Exception as e:  # never break the page
        st.caption(f"Advanced analysis block error: {e}")
