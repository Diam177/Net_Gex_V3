# -*- coding: utf-8 -*-
"""
Advanced Options Market Analysis — calculations & rendering

This module is intentionally self-contained and side‑effect free.
It exposes two functions you can import and use from the app:

    from lib.advanced_analysis import update_ao_summary, render_advanced_analysis_block

1) update_ao_summary(...):
   - Computes Put/Call ratios (by OI and by Volume), aggregated Net GEX,
     and stores a compact summary in st.session_state["ao_summary"].
   - Call it in app.py right after вы собрали агрегированный DF по выбранным экспирациям
     (тот же DF, по которому строится основной график по страйкам).
   - Ничего не выводит в UI.

2) render_advanced_analysis_block(...):
   - Рисует блок "Advanced Options Market Analysis: <TICKER>"
     и использует st.session_state["ao_summary"] + опционально st.session_state["iv_summary"].
   - Размещайте ВНИЗУ секции Key Levels — сразу после графика/кнопок скачивания.

Обе функции максимально осторожны: если каких‑то данных нет — показывают "—"
и никогда не ломают страницу.
"""

from __future__ import annotations

from typing import Optional, Sequence, Mapping, Any
import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # for type checkers

import streamlit as st  # type: ignore


# ---------------------------
# Calculations / Aggregation
# ---------------------------

def _safe_sum(series) -> Optional[float]:
    """Robust sum that ignores non-finite values. Returns None if empty."""
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
    """Return last finite element if possible."""
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
        return float(num / den)
    except Exception:
        return None


def update_ao_summary(
    ticker: str,
    df_options,  # pandas.DataFrame: aggregated by strike across selected expirations
    current_price: Optional[float],
    selected_expirations: Optional[Sequence[str]] = None,
    extra_iv: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Compute a compact summary and store it to st.session_state["ao_summary"].

    Expected columns in df_options (если каких‑то нет — пропускаем):
      - 'put_oi', 'call_oi'              (open interest per strike)
      - 'put_volume', 'call_volume'      (volume per strike)
      - 'net_gex'                        (per strike; already per your methodology)
    """
    if df_options is None or pd is None:
        st.session_state["ao_summary"] = {
            "ticker": ticker,
            "S": current_price,
            "pc_oi": None,
            "pc_vol": None,
            "net_gex_total": None,
            "expirations": list(selected_expirations or []),
        }
        return

    put_oi_sum   = _safe_sum(df_options["put_oi"])      if "put_oi" in df_options else None
    call_oi_sum  = _safe_sum(df_options["call_oi"])     if "call_oi" in df_options else None
    put_vol_sum  = _safe_sum(df_options["put_volume"])  if "put_volume" in df_options else None
    call_vol_sum = _safe_sum(df_options["call_volume"]) if "call_volume" in df_options else None
    net_gex_sum  = _safe_sum(df_options["net_gex"])     if "net_gex" in df_options else None

    summary = {
        "ticker": ticker,
        "S": float(current_price) if isinstance(current_price, (int, float)) and np.isfinite(current_price) else None,
        "pc_oi": _ratio(put_oi_sum, call_oi_sum),
        "pc_vol": _ratio(put_vol_sum, call_vol_sum),
        "net_gex_total": net_gex_sum,
        "expirations": list(selected_expirations or []),
    }

    # Optional IV metrics can be passed in or set elsewhere into st.session_state["iv_summary"]
    if isinstance(extra_iv, Mapping):
        st.session_state["iv_summary"] = dict(extra_iv)

    st.session_state["ao_summary"] = summary


# ---------------------------
# Rendering
# ---------------------------

def _pc_color_html(val: Optional[float]) -> str:
    """Colorize P/C ratio according to thresholds."""
    if not isinstance(val, (int, float)) or not np.isfinite(val):
        return "—"
    low, high = 0.8, 1.2
    color = "#F1C40F"  # neutral
    if val < low:
        color = "#2ECC71"   # green
    elif val > high:
        color = "#E74C3C"   # red
    return f"<span style='color:{color};font-weight:600'>{val:.2f}</span>"


def _netgex_color_html(val: Optional[float]) -> str:
    if not isinstance(val, (int, float)) or not np.isfinite(val):
        return "—"
    color = "#2ECC71" if val > 0 else ("#E74C3C" if val < 0 else "#F1C40F")
    return f"<span style='color:{color};font-weight:600'>{val:,.0f}</span>"


def render_advanced_analysis_block(
    fallback_ticker: Optional[str] = None,
    vwap_series=None,
) -> None:
    """
    Render the block using previously computed st.session_state["ao_summary"].
    Place this call **below** the Key Levels chart.

    Parameters
    ----------
    fallback_ticker : str, optional
        Если по какой-то причине в ao_summary нет тикера.
    vwap_series : array-like, optional
        Передайте series VWAP (например, из минутных свечей), чтобы мы могли показать последнее значение.
    """
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

        # Optional IV metrics
        ivs = st.session_state.get("iv_summary", {}) or {}
        atm_iv = ivs.get("atm_iv")
        skew    = ivs.get("skew")
        atm_iv_text = f"{atm_iv:.2f}" if isinstance(atm_iv, (int, float)) and np.isfinite(atm_iv) else "—"
        skew_text    = f"{skew:.2f}" if isinstance(skew, (int, float)) and np.isfinite(skew) else "—"

        
        # Expected Moves (1d, 1w)
        em1d_text = em1w_text = None
        if isinstance(S, (int, float)) and np.isfinite(S) and isinstance(atm_iv, (int, float)) and np.isfinite(atm_iv) and atm_iv >= 0:
            vol1d = float(atm_iv) * float(np.sqrt(1.0/252.0))
            vol1w = float(atm_iv) * float(np.sqrt(5.0/252.0))
            em1d = float(S) * vol1d
            em1w = float(S) * vol1w
            rng1d = f"[{S - em1d:.2f}; {S + em1d:.2f}]"
            rng1w = f"[{S - em1w:.2f}; {S + em1w:.2f}]"
            em1d_text = f"±${em1d:.2f} ({vol1d*100:.2f}%) — диапазон {rng1d}"
            em1w_text = f"±${em1w:.2f} ({vol1w*100:.2f}%) — диапазон {rng1w}"

        # tooltip HTML
        em_tip = "<span style='cursor:help' title='EM = S × ATM_IV × sqrt(t). t=1/252 (1d), 5/252 (1w)'>ℹ</span>"
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
            if em1d_text:
                st.markdown("**Expected Move (1d):** " + em1d_text + " " + em_tip, unsafe_allow_html=True)
            if em1w_text:
                st.markdown("**Expected Move (1w):** " + em1w_text + " " + em_tip, unsafe_allow_html=True)

    except Exception as e:  # never break the page
        st.caption(f"Advanced analysis block error: {e}")
