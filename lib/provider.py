# -*- coding: utf-8 -*-
"""
Unified Polygon-only provider (merged provider_polygon.py + helper wrappers).
This module exposes:
  - fetch_option_chain(ticker, host_unused, api_key, expiry_unix=None)
  - fetch_stock_history(ticker, host_unused, api_key, interval="1m", limit=640, dividend=None)
  - debug_meta()
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List
import time as _time
import datetime as _dt
import requests

# Debug state container
_DEBUG_LAST: Dict[str, Any] = {}

def debug_meta() -> Dict[str, Any]:
    """Return meta about the last provider call for UI debugging."""
    return dict(_DEBUG_LAST)

# --- Candles via Polygon v2/aggs (used by Key Levels) ---
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
    if interval not in ("1m","1min","1minute"):
        interval = "1m"
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{frm}/{to}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": int(limit) if limit else 50000,
        "apiKey": api_key,
    }
    meta: Dict[str, Any] = {"what": "polygon.candles", "url": url, "params": dict(params)}
    try:
        r = requests.get(url, params=params, timeout=20)
        raw_bytes = r.content or b""
        meta["status_code"] = r.status_code
        if not r.ok:
            try:
                _DEBUG_LAST.update(meta)
            finally:
                r.raise_for_status()
        j = r.json() if r.content else {}
        results = j.get("results") or []
        out = {"results": results, "ticker": ticker, "source": "polygon"}
        _DEBUG_LAST.update({**meta, "ok": True, "count": len(results)})
        return out, raw_bytes
    except Exception as e:
        meta.update({"ok": False, "error": str(e)})
        _DEBUG_LAST.update(meta)
        raise

# ---- Below is the original provider_polygon.py content (kept as-is) ----
# -*- coding: utf-8 -*-
"""
Polygon provider adapter -> "Yahoo-style" option chain format for this app.
We DO NOT change numeric values; only re-map fields into the schema
expected by compute.extract_core_from_chain().
"""
# (removed duplicate __future__ import)

import datetime as _dt
import time as _time
from typing import Dict, Any, List, Tuple, Optional
import requests

POLYGON_BASE_URL = "https://api.polygon.io"

def _parse_numeric(*vals):
    for v in vals:
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
    return None

def _get_underlying_price(underlying_symbol: str, headers: dict) -> Tuple[Optional[float], str]:
    """
    Try several official Polygon endpoints (read-only) to obtain a factual price for the underlying.
    Returns: (price_or_None, source_string)
    We *never* fabricate values.
    """
    import requests
    # 1) Index snapshot for I:* (e.g., I:SPX)
    if underlying_symbol.startswith("I:"):
        try:
            r = requests.get(f"{POLYGON_BASE_URL}/v3/snapshot/indices",
                             params={"ticker": underlying_symbol},
                             headers=headers, timeout=20)
            if r.ok:
                j = r.json()
                # v3 indices snapshot may return object or list
                res = j.get("results")
                if isinstance(res, list) and res:
                    res = res[0]
                if isinstance(res, dict):
                    price = _parse_numeric(
                        res.get("price"),
                        res.get("value"),
                        (res.get("last") or {}).get("price"),
                        (res.get("last") or {}).get("value"),
                    )
                    if price is not None:
                        return price, "v3.indices.snapshot"
        except Exception:
            pass

    # 2) Stocks snapshot v2 (very stable)
    try:
        r = requests.get(f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{underlying_symbol}",
                         headers=headers, timeout=20)
        if r.ok:
            j = r.json()
            price = _parse_numeric(
                (j.get("lastTrade") or {}).get("p"),
                (j.get("lastQuote") or {}).get("p"),
                ((j.get("day") or {}).get("c")),
            )
            if price is not None:
                return price, "v2.stocks.snapshot"
    except Exception:
        pass

    # 3) Stocks snapshot v3 (some plans return it)
    try:
        r = requests.get(f"{POLYGON_BASE_URL}/v3/snapshot/stocks",
                         params={"ticker": underlying_symbol},
                         headers=headers, timeout=20)
        if r.ok:
            j = r.json()
            res = j.get("results")
            if isinstance(res, list) and res:
                res = res[0]
            if isinstance(res, dict):
                price = _parse_numeric(
                    (res.get("last_trade") or {}).get("price"),
                    (res.get("last_quote") or {}).get("midpoint"),
                    (res.get("day") or {}).get("close"),
                    res.get("price"),
                    res.get("value"),
                )
                if price is not None:
                    return price, "v3.stocks.snapshot"
    except Exception:
        pass

    # 4) Previous close as a final factual fallback
    try:
        r = requests.get(f"{POLYGON_BASE_URL}/v2/aggs/ticker/{underlying_symbol}/prev",
                         params={"adjusted": "true"}, headers=headers, timeout=20)
        if r.ok:
            j = r.json()
            results = j.get("results") or []
            if results and isinstance(results, list):
                price = _parse_numeric(results[0].get("c"))  # previous close
                if price is not None:
                    return price, "v2.prev.close"
    except Exception:
        pass

    return None, "unavailable"

def _to_unix(d: str) -> int:
    try:
        return int(_dt.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc).timestamp())
    except Exception:
        return 0

def _safe_get(d: dict, path: List[str], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _contract_from_item(item: dict) -> Dict[str, Any]:
    strike = (_safe_get(item, ["details", "strike_price"])
              or _safe_get(item, ["strike_price"])
              or _safe_get(item, ["details", "strike"])
              or _safe_get(item, ["strike"]))
    try:
        strike = float(strike)
    except Exception:
        strike = None

    open_interest = (_safe_get(item, ["open_interest"])
                     or _safe_get(item, ["oi"])
                     or _safe_get(item, ["openInterest"]))

    volume = (_safe_get(item, ["day", "volume"])
              or _safe_get(item, ["volume"])
              or _safe_get(item, ["day", "v"]))

    iv = (_safe_get(item, ["greeks", "iv"])
          or _safe_get(item, ["implied_volatility"])
          or _safe_get(item, ["iv"]))

    return {
        "strike": strike,
        "openInterest": open_interest,
        "volume": volume,
        "impliedVolatility": iv,
    }

def _paginate(url: str, headers: dict, params: Optional[dict] = None, cap: int = 60) -> List[dict]:
    out: List[dict] = []
    next_url = url
    next_params = dict(params or {})
    for _ in range(cap):
        r = requests.get(next_url, params=next_params, headers=headers, timeout=30)
        import requests as _requests
        try:
            r.raise_for_status()
        except _requests.HTTPError as e:
            txt = r.text[:1200]
            raise RuntimeError(f"Polygon HTTP {r.status_code}: {txt}") from e
        j = r.json()
        batch = j.get("results") or j.get("data") or []
        if isinstance(batch, list):
            out.extend(batch)
        nxt = j.get("next_url") or j.get("next")
        if not nxt:
            break
        next_url, next_params = nxt, None  # next_url already includes query
    return out

def fetch_option_chain(ticker: str, host_unused: Optional[str], api_key: str, expiry_unix: Optional[int] = None) -> Tuple[Dict[str, Any], bytes]:
    if not api_key:
        raise ValueError("POLYGON_API_KEY is empty")

    underlying = ticker.strip().upper()
    if underlying in ("SPX", "^SPX"):
        underlying_symbol = "I:SPX"
    else:
        underlying_symbol = underlying

    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{underlying_symbol}"
    params = {"limit": 250, "order": "asc", "sort": "ticker"}

    items = _paginate(url, headers=headers, params=params, cap=80)

    # --- Determine underlying price S robustly ---
    S, price_source = None, 'scan'
    for it in items:
        ua = it.get('underlying_asset') or {}
        val = ua.get('price') if isinstance(ua, dict) else None
        if isinstance(val, (int, float)):
            S = float(val); break
    ts_unix = int(_time.time())
    if S is None:
        S, price_source = _get_underlying_price(underlying_symbol, headers)

    # --- Secondary attempt to determine S (only if still None) ---
    if S is None:
        for it in items:
            ua = it.get('underlying_asset') or {}
            val = ua.get('price') if isinstance(ua, dict) else None
        if isinstance(val, (int, float)):
            S = float(val); break
    ts_unix = int(_time.time())
    if S is None:
        # Fallback: query underlying snapshot (stock or index)
        try:
            if underlying_symbol.startswith("I:"):
                uurl = f"{POLYGON_BASE_URL}/v3/snapshot/indices"
                r = requests.get(uurl, params={"ticker": underlying_symbol}, headers=headers, timeout=20)
            else:
                # Try v3 stocks snapshot; if fails, try v2 as a backup
                uurl = f"{POLYGON_BASE_URL}/v3/snapshot/stocks"
                r = requests.get(uurl, params={"ticker": underlying_symbol}, headers=headers, timeout=20)
                if r.status_code == 404 or r.status_code == 400:
                    uurl = f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{underlying_symbol}"
                    r = requests.get(uurl, headers=headers, timeout=20)
            # Parse a numeric price from whatever structure we got
            try:
                j = r.json()
            except Exception:
                j = {}
            # common places
            cand = []
            if isinstance(j.get("results"), dict):
                cand.append(j["results"].get("last").get("price") if isinstance(j["results"].get("last"), dict) else j["results"].get("price"))
                cand.append(j["results"].get("value"))
                cand.append(j["results"].get("p"))
                cand.append(j["results"].get("close"))
            if isinstance(j.get("results"), list) and j["results"]:
                # take first dict and look for common keys
                rr = j["results"][0]
                if isinstance(rr, dict):
                    cand += [rr.get(k) for k in ("price","value","p","close","last","c")]

            # some v2 structures
            if "ticker" in j and isinstance(j.get("lastTrade"), dict):
                cand.append(j["lastTrade"].get("p"))
            if isinstance(j.get("last"), dict):
                cand.append(j["last"].get("price"))

            for v in cand:
                try:
                    if v is None: 
                        continue
                    S = float(v)
                    break
                except Exception:
                    continue
        except Exception:
            S = None


        # (price S determined above)

    # Final fallback: if S is still None, try previous close via aggs
    if S is None:
        try:
            r = requests.get(f"{POLYGON_BASE_URL}/v2/aggs/ticker/{underlying_symbol}/prev",
                             headers=headers, timeout=20)
            if r.ok:
                j = r.json() or {}
                res = j.get('results')
                if isinstance(res, list) and res:
                    v = res[0].get('c') or res[0].get('close')
                    if v is not None:
                        S = float(v)
                        price_source = (price_source + '|aggs_prev') if 'price_source' in locals() else 'aggs_prev'
        except Exception:
            pass
    ts_unix = int(_time.time())

    # Group by expiration
    by_exp: Dict[int, Dict[str, Any]] = {}
    expirations = set()
    for it in items:
        exp_str = (_safe_get(it, ["details", "expiration_date"])
                   or _safe_get(it, ["expiration_date"])
                   or _safe_get(it, ["exp_date"]))
        if not exp_str:
            continue
        exp_unix = _to_unix(str(exp_str)[:10])
        if not exp_unix:
            continue
        if expiry_unix and int(expiry_unix) != exp_unix:
            continue

        ctype = (_safe_get(it, ["details", "contract_type"])
                 or _safe_get(it, ["contract_type"])
                 or _safe_get(it, ["right"]))
        if not ctype:
            continue

        bucket = by_exp.setdefault(exp_unix, {"expirationDate": exp_unix, "calls": [], "puts": []})
        c_l = str(ctype).lower()
        contract = _contract_from_item(it)
        if c_l in ("call", "c"):
            bucket["calls"].append(contract)
        elif c_l in ("put", "p"):
            bucket["puts"].append(contract)
        expirations.add(exp_unix)

    expirationDates = sorted(expirations)
    chain_obj = {
        "quote": {
            "regularMarketPrice": S,
            "regularMarketDayHigh": None,
            "regularMarketDayLow": None,
            "regularMarketTime": ts_unix,
        },
        "expirationDates": expirationDates,
        "options": [by_exp[e] for e in expirationDates],
    }
    out = {"optionChain": {"result": [chain_obj], "error": None}}
    raw_bytes = (f"items={len(items)},price_source={price_source}").encode("utf-8")
    return out, raw_bytes
