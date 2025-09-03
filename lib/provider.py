# -*- coding: utf-8 -*-
"""
Unified Polygon-only provider.
Exposes:
  - fetch_option_chain(ticker, host_unused, api_key, expiry_unix=None)
  - fetch_stock_history(ticker, host_unused, api_key, interval="1m", limit=640, dividend=None)
  - debug_meta()
The functions NEVER fabricate numeric values: if a field is absent from Polygon,
we leave it None or 0 (for counts), and downstream code handles gaps.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List
import time as _time
import datetime as _dt
import requests

POLYGON_BASE_URL = "https://api.polygon.io"

# ---- debug ----
_DEBUG_LAST: Dict[str, Any] = {}

def debug_meta() -> Dict[str, Any]:
    return dict(_DEBUG_LAST)

# ---- helpers ----
def _to_unix(d: str) -> Optional[int]:
    try:
        # Polygon gives expiration_date like "2025-09-20"
        y, m, dd = map(int, d[:10].split("-"))
        return int(_dt.datetime(y, m, dd, 20, 0, 0).timestamp())
    except Exception:
        return None

def _get(d: Any, path: List[str], default=None):
    cur = d
    try:
        for k in path:
            if cur is None:
                return default
            cur = cur.get(k) if isinstance(cur, dict) else None
        return cur if cur is not None else default
    except Exception:
        return default

def _num(*vals):
    for v in vals:
        try:
            if v is None: 
                continue
            if isinstance(v, bool):
                continue
            return float(v)
        except Exception:
            continue
    return None

def _paginate(url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    next_url = url
    next_params = params
    for _ in range(50):  # hard cap
        r = requests.get(next_url, params=next_params, timeout=25)
        if not r.ok:
            r.raise_for_status()
        j = r.json() or {}
        res = j.get("results") or j.get("result") or []
        if isinstance(res, list):
            out.extend(res)
        next_url = j.get("next_url") or j.get("next")
        next_params = None  # already embedded
        if not next_url:
            break
    return out

def _resolve_underlying_price(symbol: str, api_key: str) -> Optional[float]:
    # 1) Stocks snapshot
    try:
        r = requests.get(f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
                         params={"apiKey": api_key}, timeout=15)
        if r.ok:
            j = r.json() or {}
            last = _get(j, ["ticker", "lastTrade", "p"])
            close = _get(j, ["ticker","day","c"])
            px = _num(last, close)
            if px is not None:
                return px
    except Exception:
        pass
    # 2) Indices snapshot
    try:
        r = requests.get(f"{POLYGON_BASE_URL}/v3/snapshot/indices",
                         params={"ticker": symbol, "apiKey": api_key}, timeout=15)
        if r.ok:
            j = r.json() or {}
            res = j.get("results")
            if isinstance(res, list) and res:
                px = _num(res[0].get("price"), _get(res[0], ["value"]), _get(res[0], ["last", "price"]))
                if px is not None:
                    return px
            if isinstance(res, dict):
                px = _num(res.get("price"), _get(res, ["value"]), _get(res, ["last","price"]))
                if px is not None:
                    return px
    except Exception:
        pass
    # 3) Previous close (factual fallback)
    try:
        r = requests.get(f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/prev",
                         params={"adjusted": "true", "apiKey": api_key}, timeout=15)
        if r.ok:
            j = r.json() or {}
            res = j.get("results")
            if isinstance(res, list) and res:
                px = _num(res[0].get("c"), res[0].get("close"))
                if px is not None:
                    return px
    except Exception:
        pass
    return None

# ---- candles (Key Levels) ----
def fetch_stock_history(ticker: str,
                        host_unused: Optional[str],
                        api_key: str,
                        interval: str = "1m",
                        limit: int = 640,
                        dividend: Optional[bool] = None) -> Tuple[Dict[str, Any], bytes]:
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is required for stock history")
    now = _dt.datetime.utcnow()
    start = now - _dt.timedelta(days=6)
    frm = int(_time.mktime(start.timetuple())) * 1000
    to  = int(_time.mktime(now.timetuple())) * 1000
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{frm}/{to}"
    params = {"adjusted": "true", "sort": "asc", "limit": int(limit) if limit else 50000, "apiKey": api_key}
    meta = {"what": "polygon.candles", "url": url, "params": dict(params)}
    try:
        r = requests.get(url, params=params, timeout=20)
        raw = r.content or b""
        meta["status_code"] = r.status_code
        if not r.ok:
            _DEBUG_LAST.update(meta)
            r.raise_for_status()
        j = r.json() if r.content else {}
        _DEBUG_LAST.update({**meta, "ok": True, "count": len(j.get("results") or [])})
        return {"results": j.get("results") or [], "ticker": ticker, "source": "polygon"}, raw
    except Exception as e:
        meta.update({"ok": False, "error": str(e)})
        _DEBUG_LAST.update(meta)
        raise

# ---- option chain ----
def fetch_option_chain(ticker: str,
                       host_unused: Optional[str],
                       api_key: str,
                       expiry_unix: Optional[int] = None) -> Tuple[Dict[str, Any], bytes]:
    if not api_key:
        raise ValueError("POLYGON_API_KEY is empty")

    underlying = ticker.strip().upper()
    if underlying in ("SPX", "^SPX"):
        underlying_symbol = "I:SPX"
    else:
        underlying_symbol = underlying

    # fetch snapshot options, paginated
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{underlying_symbol}"
    params = {"limit": 250, "order": "asc", "sort": "ticker", "apiKey": api_key}
    items = _paginate(url, params=params)

    # try to get underlying price from items quickly
    S = None
    for it in items:
        px = _get(it, ["underlying_asset", "price"])
        if isinstance(px, (int, float)):
            S = float(px)
            break
    if S is None:
        S = _resolve_underlying_price(underlying_symbol, api_key)

    # group by expiration
    by_exp: Dict[int, Dict[str, Any]] = {}
    expirations = set()
    for it in items:
        exp_str = _get(it, ["details","expiration_date"]) or _get(it, ["expiration_date"]) or _get(it, ["exp_date"])
        if not exp_str:
            continue
        e_unix = _to_unix(str(exp_str))
        if not e_unix:
            continue
        expirations.add(e_unix)
        blk = by_exp.get(e_unix)
        if blk is None:
            blk = {"expirationDate": e_unix, "calls": [], "puts": []}
            by_exp[e_unix] = blk
        contract_type = (_get(it, ["details","contract_type"]) or _get(it, ["contract_type"]) or "").lower()
        strike = _num(_get(it, ["details","strike_price"]), it.get("strike_price"), it.get("strike"))
        if strike is None:
            continue
        # fields
        open_interest = _num(it.get("open_interest"), it.get("openInterest"), _get(it, ["last_quote","open_interest"]))
        volume = _num(_get(it, ["day","volume"]), it.get("volume"))
        iv = _num(it.get("implied_volatility"), it.get("impliedVolatility"), _get(it, ["last_quote","implied_volatility"]))
        rec = {
            "contractSymbol": it.get("ticker") or it.get("symbol"),
            "strike": float(strike),
            "openInterest": int(open_interest) if open_interest is not None else 0,
            "volume": int(volume) if volume is not None else 0,
            "impliedVolatility": float(iv) if iv is not None else None,
        }
        if contract_type == "call":
            blk["calls"].append(rec)
        elif contract_type == "put":
            blk["puts"].append(rec)
        else:
            # if unknown, skip
            continue

    expirationDates = sorted(expirations)
    ts_unix = int(_time.time())
    chain_obj = {
        "quote": {
            "regularMarketPrice": S,
            "regularMarketTime": ts_unix,
            "regularMarketDayHigh": None,
            "regularMarketDayLow": None,
        },
        "expirationDates": expirationDates,
        "options": [by_exp[e] for e in expirationDates],
    }
    out = {"optionChain": {"result": [chain_obj], "error": None}}
    raw_bytes = (str(len(items))).encode("utf-8")
    return out, raw_bytes
