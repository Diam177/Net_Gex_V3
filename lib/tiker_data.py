
from __future__ import annotations
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd

# External project deps (already in your repo):
# - provider_polygon.fetch_option_chain
# - sanitize_window.sanitize_and_window_pipeline, SanitizerConfig

@dataclass
class SanitizedBundle:
    """Per-expiration sanitized outputs (subset of pipeline results)."""
    exp: str
    df_corr: pd.DataFrame
    windows: Dict[str, List[int]]
    # Optional fields you might want later (kept for completeness):
    df_raw: Optional[pd.DataFrame] = None
    df_marked: Optional[pd.DataFrame] = None
    df_weights: Optional[pd.DataFrame] = None
    window_raw: Optional[pd.DataFrame] = None
    window_corr: Optional[pd.DataFrame] = None

@dataclass
class TikerDataResult:
    ticker: str
    spot: float
    expirations: List[str]
    selected: List[str]
    raw_by_exp: Dict[str, List[dict]]
    sanitized_by_exp: Dict[str, SanitizedBundle]

# ---------- Provider helpers ----------

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_chain(ticker: str) -> dict:
    from provider_polygon import fetch_option_chain
    chain = fetch_option_chain(ticker=ticker, host=None, key=None, expiry_unix=None)
    return chain or {}

def _extract_spot_and_exps(chain: dict) -> Tuple[float, List[str]]:
    quote = chain.get("quote") or {}
    # Try common fields for spot:
    spot = quote.get("regularMarketPrice", None)
    if spot is None:
        spot = quote.get("last", None)
    if spot is None:
        # Final fallback
        spot = 0.0
    spot = float(spot)

    exps = chain.get("expirations") or chain.get("expirationDates") or []
    exps = [str(x) for x in exps]
    # Ensure sorted unique
    exps = sorted(set(exps))
    return spot, exps

def _nearest_exp(expirations: List[str], today: Optional[str] = None) -> Optional[str]:
    if not expirations:
        return None
    try:
        today_dt = datetime.strptime(today, "%Y-%m-%d") if today else datetime.utcnow()
        parsed = []
        for e in expirations:
            try:
                parsed.append((e, datetime.strptime(e[:10], "%Y-%m-%d")))
            except Exception:
                continue
        if not parsed:
            return expirations[0]
        # pick the nearest >= today, else earliest
        future = [e for e, d in parsed if d >= today_dt]
        if future:
            return sorted(future, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))[0]
        return sorted([e for e, _ in parsed], key=lambda x: datetime.strptime(x, "%Y-%m-%d"))[0]
    except Exception:
        return expirations[0]

def _record_matches_exp(rec: dict, exp: str) -> bool:
    # Try common keys:
    for k in ("exp", "expiration", "expirationDate", "expiry", "expDate"):
        v = rec.get(k)
        if v is None:
            continue
        s = str(v)
        if s.startswith(exp):
            return True
    # Sometimes date is nested
    d = rec.get("details") or {}
    for k in ("exp", "expiration", "expirationDate"):
        v = d.get(k)
        if v and str(v).startswith(exp):
            return True
    return False

def _split_raw_by_exp(chain: dict, selected_exps: List[str]) -> Dict[str, List[dict]]:
    items = chain.get("options") or chain.get("records") or []
    if not items:
        return {e: [] for e in selected_exps}
    out: Dict[str, List[dict]] = {e: [] for e in selected_exps}
    # If provider already returned only one expiration, just map all to that one
    # Otherwise filter per-record by detected expiration
    for rec in items:
        for e in selected_exps:
            if _record_matches_exp(rec, e):
                out[e].append(rec)
    # Fallback: if filtering yielded empty and provider likely returns single-exp chunks,
    # fill with all items for that chosen expiration.
    for e in selected_exps:
        if not out[e]:
            out[e] = list(items)
    return out

# ---------- Sanitizer bridge ----------

def _sanitize_one_exp(raw_list: List[dict], spot: float) -> SanitizedBundle:
    from sanitize_window import sanitize_and_window_pipeline, SanitizerConfig
    pipes = sanitize_and_window_pipeline(raw_list, S=float(spot), now=None, shares_per_contract=100, cfg=SanitizerConfig())
    # pipes keys: df_raw, df_marked, df_corr, df_weights, windows, window_raw, window_corr
    exp_keys = list(pipes.get("windows", {}).keys())
    exp_name = exp_keys[0] if exp_keys else "UNKNOWN"
    return SanitizedBundle(
        exp=exp_name,
        df_corr=pipes.get("df_corr"),
        windows=pipes.get("windows"),
        df_raw=pipes.get("df_raw"),
        df_marked=pipes.get("df_marked"),
        df_weights=pipes.get("df_weights"),
        window_raw=pipes.get("window_raw"),
        window_corr=pipes.get("window_corr"),
    )

# ---------- Public UI/block ----------

def render_tiker_data_block(title: str = "Блок выбора тикера и даты экспирации (независимый)") -> TikerDataResult:
    st.header(title)

    # Step 1: ticker input (default SPY)
    ticker_default = st.session_state.get("td_ticker", "SPY")
    ticker = st.text_input("Ticker", value=ticker_default, key="td_ticker_input")
    st.session_state["td_ticker"] = ticker

    # Fetch chain & extract expirations
    chain = _fetch_chain(ticker)
    spot, exps = _extract_spot_and_exps(chain)
    if not exps:
        st.warning("Провайдер не вернул список экспираций для данного тикера.")
        return TikerDataResult(ticker=ticker, spot=spot, expirations=[], selected=[], raw_by_exp={}, sanitized_by_exp={})

    # Step 2: default to nearest expiration
    default_exp = _nearest_exp(exps)
    selected = st.multiselect("Дата(ы) экспирации", options=exps, default=[default_exp] if default_exp else [])

    # Step 3: build per-exp raw arrays
    raw_by_exp = _split_raw_by_exp(chain, selected)

    # Step 4: send to sanitize_window.py (per selected exp)
    sanitized_by_exp: Dict[str, SanitizedBundle] = {}
    for e in selected:
        try:
            sanitized_by_exp[e] = _sanitize_one_exp(raw_by_exp[e], spot)
        except Exception as err:
            st.warning(f"Sanitize провалился для {e}: {err}")

    # At this point the block finishes and returns the prepared objects
    return TikerDataResult(
        ticker=ticker,
        spot=spot,
        expirations=exps,
        selected=selected,
        raw_by_exp=raw_by_exp,
        sanitized_by_exp=sanitized_by_exp,
    )
