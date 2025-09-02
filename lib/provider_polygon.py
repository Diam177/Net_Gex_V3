# -*- coding: utf-8 -*-
"""
Provider adapter for Polygon.io to match the internal "Yahoo-style" option chain format
expected by compute.extract_core_from_chain.

We intentionally DO NOT change any numeric values received from Polygon.
We only *repackage* fields into the schema your project already parses.
"""
from __future__ import annotations

import datetime as _dt
import time as _time
from typing import Dict, Any, List, Tuple, Optional
import requests

POLYGON_BASE_URL = "https://api.polygon.io"

def _to_unix(d: str) -> int:
    # d like "2025-09-17"
    try:
        return int(_dt.datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc).timestamp())
    except Exception:
        return 0

def _from_unix(ts: int) -> str:
    return _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

def _safe_get(d: dict, path: List[str], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _contract_dict_from_polygon(item: dict) -> Dict[str, Any]:
    """
    Convert one Polygon snapshot item into a minimal contract dict with
    keys used by your pipeline: strike, openInterest, volume, impliedVolatility.
    """
    strike = (_safe_get(item, ["details", "strike_price"])
              or _safe_get(item, ["strike_price"])
              or _safe_get(item, ["details", "strike"])
              or _safe_get(item, ["strike"]))
    # Normalize to float if possible
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

    # We DO NOT modify values; just pass them through if present
    out = {
        "strike": strike,
        "openInterest": open_interest,
        "volume": volume,
        "impliedVolatility": iv,
    }
    return out

def fetch_option_chain(ticker: str, host_unused: Optional[str], api_key: str, expiry_unix: Optional[int] = None) -> Tuple[Dict[str, Any], bytes]:
    """
    Fetch Polygon index/stock option snapshots for the given underlying.
    Signature mirrors the legacy provider: (ticker, host, key, expiry_unix).
    - host is unused (kept for compatibility).
    - api_key is the Polygon API key.
    Returns: (data_as_python, raw_bytes) where data_as_python matches the internal "Yahoo-style" schema.
    """
    if not api_key:
        raise ValueError("POLYGON_API_KEY is empty")

    # Map underlying for indices: "SPX" -> "I:SPX"; for others leave as-is
    underlying = ticker.strip().upper()
    if underlying == "SPX" or underlying == "^SPX":
        underlying_symbol = "I:SPX"
    else:
        underlying_symbol = underlying

    params = {"limit": 1000, "apiKey": api_key}
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{underlying_symbol}"

    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    results = payload.get("results") or payload.get("data") or []
    if not isinstance(results, list):
        results = []

    # Underlying price/time if present
    # Try to detect from any item; fall back to None
    S = None
    ts_unix = int(_time.time())
    day_high = None
    day_low = None

    for it in results[:10]:  # quick scan
        maybe = _safe_get(it, ["underlying_asset", "price"])
        if maybe is not None:
            S = maybe
            break
    # If still None, leave as None. Downstream code uses .get() and is robust.

    # Group contracts by expiration date
    by_exp: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    expirations_set = set()

    for it in results:
        exp_str = (_safe_get(it, ["details", "expiration_date"])
                   or _safe_get(it, ["expiration_date"])
                   or _safe_get(it, ["exp_date"]))
        if not exp_str:
            continue
        exp_unix = _to_unix(exp_str)
        if not exp_unix:
            continue

        # Filter by expiry_unix if provided
        if expiry_unix and exp_unix != int(expiry_unix):
            continue

        ctype = (_safe_get(it, ["details", "contract_type"])
                 or _safe_get(it, ["contract_type"])
                 or _safe_get(it, ["right"]))  # "call"/"put" or "C"/"P"

        if not ctype:
            continue

        dst = by_exp.setdefault(exp_unix, {"expirationDate": exp_unix, "calls": [], "puts": []})
        contract = _contract_dict_from_polygon(it)
        # Route by type
        ctype_l = str(ctype).lower()
        if ctype_l in ("call", "c"):
            dst["calls"].append(contract)
        elif ctype_l in ("put", "p"):
            dst["puts"].append(contract)

        expirations_set.add(exp_unix)

    expirationDates = sorted(expirations_set)
    # Build final "Yahoo-style" structure
    chain_obj = {
        "quote": {
            "regularMarketPrice": S,
            "regularMarketDayHigh": day_high,
            "regularMarketDayLow": day_low,
            "regularMarketTime": ts_unix,
        },
        "expirationDates": expirationDates,
        "options": [by_exp[e] for e in expirationDates],
    }

    out = {"optionChain": {"result": [chain_obj], "error": None}}
    raw_bytes = resp.content
    return out, raw_bytes
