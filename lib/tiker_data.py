
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import os, inspect

# =====================
# Signature-agnostic provider helpers
# =====================

def _maybe_secret(name: str) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as _st  # type: ignore
        return _st.secrets.get(name)  # type: ignore
    except Exception:
        return None

def _call_with_best_kwargs(func, base_kwargs: dict):
    sig = inspect.signature(func)
    allowed = {}
    for k, v in base_kwargs.items():
        if k in sig.parameters and v is not None:
            allowed[k] = v
    if allowed:
        return func(**allowed)
    # fallback: positional (ticker only)
    params = list(sig.parameters.keys())
    args = []
    if params and params[0] in ("ticker","symbol"):
        args.append(base_kwargs.get("ticker") or base_kwargs.get("symbol"))
    return func(*args)

def _provider_fetch_chain(ticker: str) -> dict:
    func = None
    try:
        from provider_polygon import fetch_option_chain as _f
        func = _f
    except Exception:
        try:
            from provider import fetch_option_chain as _f  # type: ignore
            func = _f
        except Exception:
            raise RuntimeError("fetch_option_chain not found in provider modules")
    key = _maybe_secret("POLYGON_API_KEY") or _maybe_secret("RAPIDAPI_KEY")
    host = _maybe_secret("RAPIDAPI_HOST")
    base = {
        "ticker": ticker,
        "host": host,
        "key": key,
        "apiKey": key,
        "api_key": key,
        "expiry_unix": None,
        "expiry": None,
    }
    try:
        out = _call_with_best_kwargs(func, base)
        return out or {}
    except Exception:
        # last resort try with only ticker
        try:
            return func(ticker=ticker)
        except Exception:
            return {}

def _provider_fetch_ohlc(ticker: str, interval: str, limit: int):
    func = None
    try:
        from provider_polygon import fetch_stock_history as _f
        func = _f
    except Exception:
        try:
            from provider import fetch_stock_history as _f  # type: ignore
            func = _f
        except Exception:
            return None
    key = _maybe_secret("POLYGON_API_KEY") or _maybe_secret("RAPIDAPI_KEY")
    host = _maybe_secret("RAPIDAPI_HOST")
    base = {
        "ticker": ticker,
        "host": host,
        "key": key,
        "apiKey": key,
        "api_key": key,
        "interval": interval,
        "limit": int(limit),
        "dividend": None,
    }
    try:
        return _call_with_best_kwargs(func, base)
    except Exception:
        try:
            return func(ticker=ticker, interval=interval, limit=int(limit))
        except Exception:
            return None

# =====================
# Data structures
# =====================

@dataclass
class TikerRawResult:
    ticker: str
    spot: float
    expirations: List[str]
    selected: List[str]
    raw_by_exp: Dict[str, List[dict]]
    ohlc: Optional[pd.DataFrame] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    ohlc_interval: str = "1m"
    ohlc_limit: int = 500

# =====================
# Helpers
# =====================

def _fetch_chain(ticker: str) -> dict:
    return _provider_fetch_chain(ticker)

def _fetch_ohlc(ticker: str, interval: str = "1m", limit: int = 500):
    return _provider_fetch_ohlc(ticker, interval, limit)

def _extract_spot_and_exps(chain: dict) -> Tuple[float, List[str]]:
    quote = chain.get("quote") or {}
    spot = quote.get("regularMarketPrice", quote.get("last", 0.0))
    spot = float(spot or 0.0)

    exps = chain.get("expirations") or chain.get("expirationDates") or []
    exps = [str(x) for x in exps]
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
        future = [e for e, d in parsed if d >= today_dt]
        if future:
            return sorted(future, key=lambda x: datetime.strptime(x, "%Y-%m-%d"))[0]
        return sorted([e for e, _ in parsed], key=lambda x: datetime.strptime(x, "%Y-%m-%d"))[0]
    except Exception:
        return expirations[0]

def _record_matches_exp(rec: dict, exp: str) -> bool:
    for k in ("exp","expiration","expirationDate","expiry","expDate"):
        v = rec.get(k)
        if v is None:
            continue
        if str(v).startswith(exp):
            return True
    d = rec.get("details") or {}
    for k in ("exp","expiration","expirationDate"):
        v = d.get(k)
        if v and str(v).startswith(exp):
            return True
    return False

def _split_raw_by_exp(chain: dict, selected_exps: List[str]) -> Dict[str, List[dict]]:
    items = chain.get("options") or chain.get("records") or []
    if not items:
        return {e: [] for e in selected_exps}
    out: Dict[str, List[dict]] = {e: [] for e in selected_exps}
    for rec in items:
        for e in selected_exps:
            if _record_matches_exp(rec, e):
                out[e].append(rec)
    for e in selected_exps:
        if not out[e]:
            out[e] = list(items)
    return out

def _normalize_ohlc(payload) -> Optional[pd.DataFrame]:
    if payload is None:
        return None
    try:
        if isinstance(payload, dict) and "results" in payload:
            rows = payload["results"]
        elif isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and "candles" in payload:
            rows = payload["candles"]
        else:
            rows = []
        if not rows:
            return None
        df = pd.DataFrame(rows)
        colmap = {}
        if "time" in df.columns:
            colmap["time"] = "time"
        elif "t" in df.columns:
            df["time"] = pd.to_datetime(df["t"], unit="ms")
            colmap["time"] = "time"
        elif "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            colmap["time"] = "time"
        if "open" in df.columns: colmap["open"] = "open"
        elif "o" in df.columns: colmap["o"] = "open"
        if "high" in df.columns: colmap["high"] = "high"
        elif "h" in df.columns: colmap["h"] = "high"
        if "low" in df.columns: colmap["low"] = "low"
        elif "l" in df.columns: colmap["l"] = "low"
        if "close" in df.columns: colmap["close"] = "close"
        elif "c" in df.columns: colmap["c"] = "close"
        if "volume" in df.columns: colmap["volume"] = "volume"
        elif "v" in df.columns: colmap["v"] = "volume"
        df = df.rename(columns=colmap)
        needed = ["time","open","high","low","close","volume"]
        missing = [x for x in needed if x not in df.columns]
        if missing:
            if all(x in df.columns for x in ["time","open","high","low","close"]):
                df["volume"] = 0
            else:
                return None
        df = df[["time","open","high","low","close","volume"]].copy()
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            pass
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception:
        return None

# =====================
# Public PURE API
# =====================

def get_raw_by_exp(
    ticker: str,
    selected_exps: Optional[List[str]] = None,
    need_ohlc: bool = False,
    ohlc_interval: str = "1m",
    ohlc_limit: int = 500,
) -> TikerRawResult:
    chain = _fetch_chain(ticker)
    spot, exps = _extract_spot_and_exps(chain)
    if not exps:
        return TikerRawResult(ticker=ticker, spot=spot, expirations=[], selected=[], raw_by_exp={}, ohlc=None, day_high=None, day_low=None)
    if not selected_exps:
        default = _nearest_exp(exps)
        selected_exps = [default] if default else []
    raw_by_exp = _split_raw_by_exp(chain, selected_exps)
    ohlc_df = None
    day_hi = day_lo = None
    if need_ohlc:
        payload = _fetch_ohlc(ticker, interval=ohlc_interval, limit=ohlc_limit)
        ohlc_df = _normalize_ohlc(payload)
        if ohlc_df is not None and not ohlc_df.empty:
            try:
                day_hi = float(ohlc_df["high"].max())
                day_lo = float(ohlc_df["low"].min())
            except Exception:
                day_hi = day_lo = None
    return TikerRawResult(
        ticker=ticker,
        spot=spot,
        expirations=exps,
        selected=selected_exps,
        raw_by_exp=raw_by_exp,
        ohlc=ohlc_df,
        day_high=day_hi,
        day_low=day_lo,
        ohlc_interval=ohlc_interval,
        ohlc_limit=ohlc_limit,
    )

# =====================
# OPTIONAL UI helper
# =====================

def render_tiker_data_block(title: str = "Тикер/Экспирации (сырьё + свечи)") -> TikerRawResult:
    import streamlit as st  # local import
    st.header(title)
    ticker_default = st.session_state.get("td2_ticker", "SPY")
    ticker = st.text_input("Ticker", value=ticker_default, key="td2_ticker_input")
    st.session_state["td2_ticker"] = ticker
    with st.expander("Настройки свечей (для чарта Key Levels)", expanded=False):
        interval = st.selectbox("Интервал", options=["1m","2m","5m","15m","30m","1h","1d"], index=0)
        limit = st.number_input("Кол-во баров (limit)", min_value=50, max_value=5000, value=500, step=50)
    base = get_raw_by_exp(ticker, need_ohlc=True, ohlc_interval=interval, ohlc_limit=int(limit))
    if not base.expirations:
        st.warning("Провайдер не вернул список экспираций для данного тикера.")
        return base
    selected = st.multiselect("Дата(ы) экспирации", options=base.expirations, default=base.selected)
    base = get_raw_by_exp(ticker, selected_exps=selected, need_ohlc=True, ohlc_interval=interval, ohlc_limit=int(limit))
    if base.ohlc is not None and not base.ohlc.empty:
        st.caption(f"Свечи: {ticker} • {interval} • {len(base.ohlc)} баров. High={base.day_high}, Low={base.day_low}")
        st.dataframe(base.ohlc.tail(5), use_container_width=True)
    return base
