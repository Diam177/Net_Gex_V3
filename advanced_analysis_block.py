
# -*- coding: utf-8 -*-
"""
Advanced Options Market Analysis — calculations & rendering.

Exports:
    - update_ao_summary(df_options, current_price, ...)
    - render_advanced_analysis_block(fallback_ticker=None, vwap_series=None)

This module is self-contained and robust to missing fields.
Place it in your project root as `advanced_analysis_block.py`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Mapping, Any
import math
import numpy as np

try:
    import pandas as pd  # optional
except Exception:
    pd = None  # type: ignore

try:
    import streamlit as st
except Exception as e:
    raise RuntimeError("Streamlit is required for this module") from e


# ---------- helpers ----------

def _safe_sum(array_like) -> Optional[float]:
    try:
        arr = np.asarray(array_like, dtype=float)
        if arr.size == 0:
            return None
        s = float(np.nansum(arr))
        return s
    except Exception:
        return None


def _safe_last(series_like) -> Optional[float]:
    try:
        if series_like is None:
            return None
        if hasattr(series_like, "iloc"):
            if len(series_like) == 0:
                return None
            val = series_like.iloc[-1]
        else:
            if len(series_like) == 0:
                return None
            val = series_like[-1]
        v = float(val)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
        return None
    return float(a) / float(b)


def _fmt_price(val: Optional[float]) -> str:
    if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
        return "—"
    v = float(val)
    if v >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    if v >= 10:
        return f"{v:,.2f}"
    return f"{v:,.3f}"


def _fmt_percent(val: Optional[float]) -> str:
    if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
        return "—"
    return f"{100.0*float(val):.2f}%"


def _colorize_ratio(val: Optional[float]) -> str:
    """
    Green if <0.8, Yellow if 0.8..1.2, Red if >1.2
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
    """
    Blue if >=0, Red if <0
    """
    if not isinstance(val, (int, float)) or not np.isfinite(val):
        return "—"
    v = float(val)
    color = "#60A5E7" if v >= 0 else "#D9493A"
    return f"<span style='color:{color}'>{v:,.0f}</span>"


# ---------- public API ----------

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

    cols = {c.lower(): c for c in getattr(df_options, "columns", [])}
    get = lambda name: df_options[cols[name]].to_numpy(dtype=float) if name in cols else None

    put_oi_sum   = _safe_sum(get("put_oi"))
    call_oi_sum  = _safe_sum(get("call_oi"))
    put_vol_sum  = _safe_sum(get("put_volume"))
    call_vol_sum = _safe_sum(get("call_volume"))
    net_gex_sum  = _safe_sum(get("net_gex"))

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

        # iv summary
        iv = st.session_state.get("iv_summary", {}) or {}
        atm_iv = iv.get("atm_iv")  # as decimal, e.g. 0.19
        skew   = iv.get("skew")    # unitless ratio or diff, free-form

        # expected move from ATM IV
        em1d = None
        em1w = None
        if isinstance(atm_iv, (int, float)) and np.isfinite(atm_iv) and isinstance(S, (int, float)) and np.isfinite(S):
            S = float(S)
            vol = float(atm_iv)
            em1d = S * vol * math.sqrt(1.0/252.0)
            em1w = S * vol * math.sqrt(5.0/252.0)

        em_tip = "<span style='opacity:0.7'>(≈ S × ATM_IV × √t)</span>"

        # format parts
        pc_oi_html  = _colorize_ratio(ao.get("pc_oi"))
        pc_vol_html = _colorize_ratio(ao.get("pc_vol"))
        ng_html     = _colorize_netgex(ao.get("net_gex_total"))
        S_text      = _fmt_price(S)
        atm_iv_text = _fmt_percent(atm_iv) if isinstance(atm_iv, (int, float)) and np.isfinite(atm_iv) else "—"
        skew_text   = f"{skew:.2f}" if isinstance(skew, (int, float)) and np.isfinite(skew) else (str(skew) if skew else "—")
        em1d_text   = _fmt_price(em1d) if em1d is not None else None
        em1w_text   = _fmt_price(em1w) if em1w is not None else None
        vwap_last   = _safe_last(vwap_series) if vwap_series is not None else None

        st.markdown(f"### Advanced Analysis — {ticker}" if ticker else "### Advanced Analysis")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"**Spot:** {S_text}")
            if vwap_last is not None:
                st.markdown(f"**VWAP:** {_fmt_price(vwap_last)}")
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
