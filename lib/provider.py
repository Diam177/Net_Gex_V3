
# lib/provider.py — Polygon-only implementation
from __future__ import annotations
import time
import json
from typing import Any, Dict, List, Optional, Tuple
import requests
import datetime

DEFAULT_TIMEOUT = 20

# Debug state container
DEBUG_STATE: Dict[str, Any] = {'last': {}}

def debug_meta() -> Dict[str, Any]:
    return DEBUG_STATE.get('last', {})

def _request_json(url: str, params: Dict[str, Any], api_key: Optional[str]) -> Tuple[Dict[str, Any], bytes]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json(), r.content
    except Exception:
        return {"raw": r.text}, r.content

def _unix_from_datestr(s: str) -> int:
    # s: 'YYYY-MM-DD' (Polygon expiration)
    dt = datetime.datetime.strptime(s, "%Y-%m-%d")
    # interpret at 00:00:00 UTC of that date
    return int(datetime.datetime(dt.year, dt.month, dt.day, tzinfo=datetime.timezone.utc).timestamp())

def _group_snapshot_results_by_expiration(results: List[Dict[str, Any]]) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    by_exp: Dict[int, Dict[str, Any]] = {}
    exps: set[int] = set()
    for it in (results or []):
        det = it.get("details", {}) or {}
        day = it.get("day", {}) or {}
        k = det.get("strike_price")
        if k is None: 
            continue
        k = float(k)
        exp_str = det.get("expiration_date")
        if not exp_str:
            continue
        exp_unix = _unix_from_datestr(exp_str)
        exps.add(exp_unix)
        typ = (det.get("contract_type") or "").lower()
        # normalize fields
        row = {
            "strike": k,
            "openInterest": int(it.get("open_interest") or 0),
            "volume": int(day.get("volume") or 0),
            "impliedVolatility": float(it.get("implied_volatility")) if it.get("implied_volatility") is not None else None,
        }
        blk = by_exp.setdefault(exp_unix, {"expirationDate": exp_unix, "calls": [], "puts": []})
        if typ == "call":
            blk["calls"].append(row)
        elif typ == "put":
            blk["puts"].append(row)
        else:
            # skip unknown
            pass
    return sorted(exps), by_exp

def _fetch_underlying_quote_polygon(ticker: str, api_key: str) -> Tuple[Dict[str, Any], float, int, List[Dict[str, Any]]]:
    # Try snapshot -> if fails, use previous close
    attempts: List[Dict[str, Any]] = []
    S = None; t0 = None; quote: Dict[str, Any] = {}
    # 1) Try last trade (works for both stocks and indices)
    url = f"https://api.polygon.io/v2/last/trade/{ticker}"
    try:
        js, _ = _request_json(url, {}, api_key)
        attempts.append({"url": url, "ok": True})
        pr = js.get("results", {}).get("p") if isinstance(js.get("results"), dict) else js.get("last", {}).get("price")
        ts = js.get("results", {}).get("t") if isinstance(js.get("results"), dict) else js.get("last", {}).get("sip_timestamp")
        if pr is not None:
            S = float(pr)
        if ts is not None:
            t0 = int(int(ts)/1_000_000_000) if isinstance(ts, int) and ts > 10**12 else int(ts)
    except Exception as e:
        attempts.append({"url": url, "ok": False, "err": str(e)})
    # 2) Fallback: previous close aggregate
    if S is None or t0 is None:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
        try:
            js, _ = _request_json(url, {"adjusted": "true"}, api_key)
            attempts.append({"url": url, "ok": True})
            res = js.get("results")
            if isinstance(res, list) and len(res)>0:
                S = float(res[0].get("c"))
                t0 = int(res[0].get("t")/1000) if res[0].get("t") else int(time.time())
        except Exception as e:
            attempts.append({"url": url, "ok": False, "err": str(e)})
    if S is None:
        S = 0.0
    if t0 is None:
        t0 = int(time.time())
    quote = {"symbol": ticker, "regularMarketPrice": S, "regularMarketTime": t0}
    return quote, S, t0, attempts

def fetch_option_chain(ticker: str,
                       host: Optional[str],
                       key: Optional[str],
                       expiry_unix: Optional[int] = None) -> Tuple[Dict[str, Any], bytes]:
    """Fetch options snapshot from Polygon and normalize to generic structure
    understood by compute.extract_core_from_chain.
    Returns (data_json, raw_bytes)
    """
    api_key = key
    debug = {"mode": "polygon", "attempts": []}
    # 1) Get underlying price/time
    quote, S, t0, q_attempts = _fetch_underlying_quote_polygon(ticker, api_key)
    debug["attempts"].extend(q_attempts)

    # 2) Snapshot options
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    params = {
        "limit": 50000,
        "order": "asc",
        "sort": "ticker",
        "price_source": "v2.prev.close",
    }
    if expiry_unix is not None:
        dt = datetime.datetime.utcfromtimestamp(int(expiry_unix))
        params["expiration_date"] = dt.strftime("%Y-%m-%d")
    js, raw_bytes = _request_json(url, params, api_key)
    debug["attempts"].append({"url": url, "params": params, "ok": True, "count": len(js.get("results", []) or [])})

    results = js.get("results") or js.get("items") or []
    expirations, blocks_by_date = _group_snapshot_results_by_expiration(results)

    # Normalized container
    data = {
        "quote": quote,
        "expirations": expirations,
        "options": [blocks_by_date[e] for e in expirations],
        "result": {"items": results},
    }
    DEBUG_STATE['last'] = debug
    return data, raw_bytes

def fetch_stock_history(ticker: str,
                        host: Optional[str],
                        key: Optional[str],
                        interval: str = "1m",
                        limit: int = 640,
                        dividend: Optional[bool] = None) -> Tuple[Dict[str, Any], bytes]:
    """Fetch recent candles from Polygon aggregates. Returns (json_like, raw_bytes).
    The returned JSON has a top-level 'candles': [{'t','o','h','l','c','v'}...].
    """
    api_key = key
    # Map interval
    m = 1; span = "minute"
    if isinstance(interval, str) and interval.endswith("m"):
        m = int(interval[:-1] or "1"); span = "minute"
    elif isinstance(interval, str) and interval.endswith("h"):
        m = int(interval[:-1] or "1"); span = "hour"
    elif isinstance(interval, str) and interval.endswith("d"):
        m = int(interval[:-1] or "1"); span = "day"
    else:
        try:
            m = int(interval); span = "minute"
        except Exception:
            m = 1; span = "minute"

    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    days_back = 10 if span in ("minute","hour") else 365
    start = now - datetime.timedelta(days=days_back)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{m}/{span}/{start.date()}/{now.date()}"
    params = {"adjusted": "true", "sort": "asc", "limit": int(limit)}
    js, raw = _request_json(url, params, api_key)
    items = js.get("results") or []
    # Normalize
    recs = []
    # take last 'limit' items in ascending order
    items = items[-int(limit):]
    for r in items:
        tms = r.get("t")
        if tms is None: 
            continue
        ts = int(tms/1000)
        recs.append({
            "t": ts,
            "o": r.get("o"),
            "h": r.get("h"),
            "l": r.get("l"),
            "c": r.get("c"),
            "v": r.get("v"),
        })
    data = {"candles": recs}
    DEBUG_STATE['last'] = {"mode": "polygon_candles", "url": url, "params": params, "count": len(recs)}
    return data, raw
