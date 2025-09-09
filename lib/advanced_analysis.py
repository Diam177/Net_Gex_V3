# -*- coding: utf-8 -*-
"""
Advanced Options Market Analysis — calculations & rendering (minimal, no IV history)

Exposes two functions:
    - update_ao_summary(ticker, df, S, selected_expirations, extra_iv=None)
    - render_advanced_analysis_block(fallback_ticker=None, vwap_series=None)

This version intentionally omits IV Rank / IV Percentile. It shows only current ATM IV and IV Skew.
"""
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

# ---------- helpers ----------

def _safe_last(arr):
    try:
        if arr is None:
            return None
        if hasattr(arr, "__len__") and len(arr) > 0:
            return arr[-1]
    except Exception:
        pass
    return None

def _pc_color_html(val) -> str:
    """Color thresholds: <0.8 green, 0.8–1.2 yellow, >1.2 red."""
    try:
        v = float(val)
        txt = f"{v:.2f}"
        if not np.isfinite(v):
            return txt
        if v < 0.8:
            color = "#16A34A"  # green-600
        elif v > 1.2:
            color = "#DC2626"  # red-600
        else:
            color = "#CA8A04"  # yellow-600
        return f"<span style='color:{color};font-weight:600'>{txt}</span>"
    except Exception:
        return "—"

def _netgex_color_html(val) -> str:
    try:
        v = float(val)
        txt = f"{v:,.0f}"
        if not np.isfinite(v):
            return "—"
        color = "#16A34A" if v > 0 else ("#DC2626" if v < 0 else "#666")
        return f"<span style='color:{color};font-weight:600'>{txt}</span>"
    except Exception:
        return "—"

# ---------- public API ----------

def update_ao_summary(
    ticker: str,
    df: pd.DataFrame,
    S: float,
    selected_expirations,
    extra_iv: Optional[Dict[str, Any]] = None,
) -> None:
    """Compute compact AO summary and stash into st.session_state."""
    # Aggregate inputs (robust to missing columns)
    put_oi = np.asarray(df.get("put_oi", []), dtype=float) if isinstance(df, pd.DataFrame) else np.asarray([])
    call_oi = np.asarray(df.get("call_oi", []), dtype=float) if isinstance(df, pd.DataFrame) else np.asarray([])
    put_vol = np.asarray(df.get("put_volume", []), dtype=float) if isinstance(df, pd.DataFrame) else np.asarray([])
    call_vol = np.asarray(df.get("call_volume", []), dtype=float) if isinstance(df, pd.DataFrame) else np.asarray([])
    net_gex = np.asarray(df.get("net_gex", []), dtype=float) if isinstance(df, pd.DataFrame) else np.asarray([])

    put_oi_sum  = float(np.nansum(put_oi)) if put_oi.size else np.nan
    call_oi_sum = float(np.nansum(call_oi)) if call_oi.size else np.nan
    put_vol_sum = float(np.nansum(put_vol)) if put_vol.size else np.nan
    call_vol_sum= float(np.nansum(call_vol)) if call_vol.size else np.nan
    net_gex_sum = float(np.nansum(net_gex)) if net_gex.size else np.nan

    pc_oi  = (put_oi_sum / call_oi_sum) if (isinstance(call_oi_sum, float) and np.isfinite(call_oi_sum) and call_oi_sum > 0) else np.nan
    pc_vol = (put_vol_sum / call_vol_sum) if (isinstance(call_vol_sum, float) and np.isfinite(call_vol_sum) and call_vol_sum > 0) else np.nan

    # Save AO summary
    st.session_state["ao_summary"] = {
        "ticker": ticker,
        "S": float(S) if isinstance(S, (int, float)) and np.isfinite(S) else None,
        "pc_oi": float(pc_oi) if np.isfinite(pc_oi) else None,
        "pc_vol": float(pc_vol) if np.isfinite(pc_vol) else None,
        "net_gex_total": net_gex_sum if np.isfinite(net_gex_sum) else None,
        "expirations": list(selected_expirations or []),
    }

    # Save IV extras (only current ATM IV and Skew)
    ivs = {}
    if isinstance(extra_iv, dict):
        if extra_iv.get("atm_iv") is not None:
            try:
                ivs["atm_iv"] = float(extra_iv.get("atm_iv"))
            except Exception:
                pass
        if extra_iv.get("skew") is not None:
            try:
                ivs["skew"] = float(extra_iv.get("skew"))
            except Exception:
                pass
    st.session_state["iv_summary"] = ivs

def render_advanced_analysis_block(
    fallback_ticker: Optional[str] = None,
    vwap_series=None,
) -> None:
    """Render block under Key Levels using st.session_state summaries."""
    try:
        st.markdown("")  # spacing
        ao = dict(st.session_state.get("ao_summary") or {})
        ticker = ao.get("ticker") or fallback_ticker or ""

        st.subheader(f"Advanced Options Market Analysis: {ticker}")

        # Current price
        price_val = ao.get("S", None)
        price_text = f"{price_val:.2f}" if isinstance(price_val, (int, float)) and np.isfinite(price_val) else "—"

        # VWAP (last visible)
        vwap_val = _safe_last(vwap_series) if vwap_series is not None else None
        vwap_text = f"{vwap_val:.2f}" if isinstance(vwap_val, (int, float)) and np.isfinite(vwap_val) else "—"

        pc_oi_html  = _pc_color_html(ao.get("pc_oi"))
        pc_vol_html = _pc_color_html(ao.get("pc_vol"))
        ng_html     = _netgex_color_html(ao.get("net_gex_total"))

        # Optional IV metrics (current only)
        ivs = st.session_state.get("iv_summary", {}) or {}
        atm_iv = ivs.get("atm_iv")
        skew   = ivs.get("skew")
        atm_iv_text = f"{atm_iv:.2f}" if isinstance(atm_iv, (int, float)) and np.isfinite(atm_iv) else "—"
        skew_text   = f"{skew:.2f}" if isinstance(skew, (int, float)) and np.isfinite(skew) else "—"

        c1, c2, c3, c4 = st.columns(4)
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
            st.markdown(f"**ATM IV:** {atm_iv_text}")
            st.markdown(f"**Skew (Put/Call IV):** {skew_text}")

    except Exception as e:  # never break the page
        st.caption(f"Advanced analysis block error: {e}")
