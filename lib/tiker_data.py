
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

# --- Data structures ---

@dataclass
class TikerRawResult:
    ticker: str
    spot: float
    expirations: List[str]
    selected: List[str]
    raw_by_exp: Dict[str, List[dict]]
    # Added for Key Levels chart:
    ohlc: Optional[pd.DataFrame] = None   # columns: time, open, high, low, close, volume
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    ohlc_interval: str = "1m"
    ohlc_limit: int = 500

# --- Provider helpers (pure, no Streamlit) ---

def _fetch_chain(ticker: str) -> dict:
    # Primary provider abstraction in your repo:
    try:
        from provider_polygon import fetch_option_chain
    except Exception:
        # Fallback to generic provider if available
        from provider import fetch_option_chain  # type: ignore
    chain = fetch_option_chain(ticker=ticker, host=None, key=None, expiry_unix=None)
    return chain or {}

def _fetch_ohlc(ticker: str, interval: str = "1m", limit: int = 500) -> List[dict] | dict | None:
    # Try polygon-first, fallback to generic provider
    try:
        from provider_polygon import fetch_stock_history
    except Exception:
        try:
            from provider import fetch_stock_history  # type: ignore
        except Exception:
            return None
    try:
        data = fetch_stock_history(ticker=ticker, host=None, key=None, interval=interval, limit=limit, dividend=None)
        return data
    except Exception:
        return None

def _extract_spot_and_exps(chain: dict) -> Tuple[float, List[str]]:
    quote = chain.get("quote") or {}
    spot = quote.get("regularMarketPrice", None)
    if spot is None:
        spot = quote.get("last", None)
    if spot is None:
        spot = 0.0
    spot = float(spot)

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

def _normalize_ohlc(payload: List[dict] | dict | None) -> Optional[pd.DataFrame]:
    if payload is None:
        return None
    # Common shapes:
    # 1) {"results":[{"t": epoch_ms, "o":..., "h":..., "l":..., "c":..., "v":...}, ...]}
    # 2) [{"time": "...", "open":..., "high":..., "low":..., "close":..., "volume":...}, ...]
    # 3) Flat list with o/h/l/c/v but different keys
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
        # Try to map columns to a standard schema
        colmap = {}
        # Time
        if "time" in df.columns:
            colmap["time"] = "time"
        elif "t" in df.columns:
            # polygon style epoch ms
            df["time"] = pd.to_datetime(df["t"], unit="ms")
            colmap["time"] = "time"
        elif "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            colmap["time"] = "time"
        # Prices/volume
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

        # Apply mapping
        df = df.rename(columns=colmap)
        needed = ["time", "open", "high", "low", "close", "volume"]
        missing = [x for x in needed if x not in df.columns]
        if missing:
            # If we at least have ohlc, fill volume = 0
            if all(x in df.columns for x in ["time", "open", "high", "low", "close"]):
                df["volume"] = 0
            else:
                return None

        # Keep only necessary columns and sort by time
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            pass
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except Exception:
        return None

# --- Public PURE API ---

def get_raw_by_exp(
    ticker: str,
    selected_exps: Optional[List[str]] = None,
    need_ohlc: bool = False,
    ohlc_interval: str = "1m",
    ohlc_limit: int = 500,
) -> TikerRawResult:
    """Single source of truth for raw option data (per expiration) + optional OHLC for Key Levels chart."""
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

# --- OPTIONAL UI helper (kept separate from pure API) ---

def render_tiker_data_block(title: str = "Блок выбора тикера и даты экспирации (сырьё + свечи)") -> TikerRawResult:
    import streamlit as st  # local import so pure API has no Streamlit dep
    st.header(title)
    ticker_default = st.session_state.get("td2_ticker", "SPY")
    ticker = st.text_input("Ticker", value=ticker_default, key="td2_ticker_input")
    st.session_state["td2_ticker"] = ticker

    # Controls for OHLC retrieval (interval/limit)
    with st.expander("Настройки свечей (для чарта Key Levels)", expanded=False):
        interval = st.selectbox("Интервал", options=["1m","2m","5m","15m","30m","1h","1d"], index=0)
        limit = st.number_input("Кол-во баров (limit)", min_value=50, max_value=5000, value=500, step=50)

    base = get_raw_by_exp(ticker, need_ohlc=True, ohlc_interval=interval, ohlc_limit=int(limit))
    if not base.expirations:
        st.warning("Провайдер не вернул список экспираций для данного тикера.")
        return base

    selected = st.multiselect("Дата(ы) экспирации", options=base.expirations, default=base.selected)
    base = get_raw_by_exp(ticker, selected_exps=selected, need_ohlc=True, ohlc_interval=interval, ohlc_limit=int(limit))

    # Optional: show a small preview of candles
    if base.ohlc is not None and not base.ohlc.empty:
        st.caption(f"Свечи: {ticker} • {interval} • {len(base.ohlc)} баров. High={base.day_high}, Low={base.day_low}")
        st.dataframe(base.ohlc.tail(5), use_container_width=True)

    return base
