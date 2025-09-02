# -*- coding: utf-8 -*-
"""
Provider adapter for Polygon.io to match the internal "Yahoo-style" option chain format.
We DO NOT change numeric values; we only repackage fields to the schema that the rest of the
pipeline already understands.
"""
from __future__ import annotations

import datetime as _dt
import time as _time
from typing import Dict, Any, List, Tuple, Optional
import requests

POLYGON_BASE_URL = "https://api.polygon.io"

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

def _contract_dict_from_polygon(item: dict) -> Dict[str, Any]:
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

def _paginate_get(url: str, headers: dict, params: Optional[dict] = None, page_cap: int = 40) -> List[dict]:
    """Generic v3 pagination helper (respects next_url)."""
    out: List[dict] = []
    next_url = url
    next_params = dict(params or {})
    for _ in range(page_cap):
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
        next_url, next_params = nxt, None  # next_url already contains query string
    return out

def fetch_option_chain(ticker: str, host_unused: Optional[str], api_key: str, expiry_unix: Optional[int] = None) -> Tuple[Dict[str, Any], bytes]:
    """Main entry: returns (chain_like_json, raw_bytes) in Yahoo-style schema."""
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

    all_items = _paginate_get(url, headers=headers, params=params, page_cap=60)

    # Underlying price (if provided in any item)
    S = None
    for it in all_items[:20]:
        maybe = _safe_get(it, ["underlying_asset", "price"])
        if maybe is not None:
            S = maybe
            break
    ts_unix = int(_time.time())

    # Group contracts by expiration
    by_exp: Dict[int, Dict[str, Any]] = {}
    expirations = set()

    for it in all_items:
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
        contract = _contract_dict_from_polygon(it)
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
    # raw bytes not strictly needed; include minimal
    raw_bytes = ("pages=" + str(len(all_items))).encode("utf-8")
    return out, raw_bytes

def _debug_endpoint(ticker: str, api_key: str) -> dict:
    underlying = ticker.strip().upper()
    if underlying in ("SPX", "^SPX"):
        underlying_symbol = "I:SPX"
    else:
        underlying_symbol = underlying

    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{underlying_symbol}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"limit": 5}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    out = {
        "url": r.request.url,
        "status_code": r.status_code,
        "reason": r.reason,
        "used_headers": {"Authorization": headers["Authorization"][:12] + "...(masked)"},
    }
    try:
        j = r.json()
        out["json_keys"] = list(j.keys())
        out["sample"] = {"status": j.get("status"), "results_len": len(j.get("results", []))}
    except Exception:
        out["body_snippet"] = r.text[:800]
    return out
