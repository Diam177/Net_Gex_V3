# -*- coding: utf-8 -*-
"""
Polygon provider adapter → Yahoo-like JSON expected by compute.extract_core_from_chain().
- Uses ONLY query param apiKey (no headers). Avoids duplicate apiKey.
- Robust price fallback incl. indices snapshot for SPX (I:SPX).
"""
from typing import Dict, Any, List, Optional, Tuple
import datetime as _dt
import time as _time
import json
import requests

POLYGON_BASE_URL = "https://api.polygon.io"


# ---------------- helpers ----------------

def _append_key(url: str, api_key: str) -> str:
    return url + (("&" if "?" in url else "?") + "apiKey=" + api_key)


def _to_unix(date_str: str) -> int:
    try:
        return int(_dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc).timestamp())
    except Exception:
        return 0


def _to_iso(ts_unix: int) -> str:
    try:
        return _dt.datetime.utcfromtimestamp(int(ts_unix)).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _paginate(url: str, api_key: str, params: Optional[dict] = None, cap: int = 80) -> List[dict]:
    out: List[dict] = []
    next_url = _append_key(url, api_key)
    # strip apiKey if случайно оказался в params
    next_params = {k: v for k, v in (params or {}).items() if k.lower() != "apikey"}
    for _ in range(cap):
        r = requests.get(next_url, params=next_params, timeout=30)
        r.raise_for_status()
        j = r.json()
        batch = j.get("results") or j.get("data") or []
        if isinstance(batch, list):
            out.extend(batch)
        nxt = j.get("next_url")
        if not nxt:
            break
        next_url = _append_key(nxt, api_key)
        next_params = {}
    return out


def _scan_price_from_items(items: List[dict]) -> Optional[float]:
    for it in items:
        ua = it.get("underlying_asset") or {}
        for k in ("price", "close", "last", "p", "value"):
            v = ua.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
    return None


def _price_fallback(symbol: str, api_key: str) -> Tuple[Optional[float], Optional[int], str]:
    # 1) last trade
    try:
        u = f"{POLYGON_BASE_URL}/v2/last/trade/{symbol}"
        r = requests.get(_append_key(u, api_key), timeout=15)
        if r.ok:
            j = r.json()
            res = j.get("results") or {}
            p = res.get("p") or res.get("price")
            t = res.get("t") or res.get("timestamp")
            if p is not None:
                ts = int(int(t)/1_000_000_000) if isinstance(t, int) and t > 1_000_000_000_000 else int(_time.time())
                return float(p), ts, "v2.last.trade"
    except Exception:
        pass
    # 2) v3 snapshot stocks
    try:
        u = f"{POLYGON_BASE_URL}/v3/snapshot/stocks"
        r = requests.get(u, params={"ticker": symbol, "apiKey": api_key}, timeout=15)
        if r.ok:
            j = r.json()
            res = j.get("results") or {}
            if isinstance(res, dict):
                cand = []
                if isinstance(res.get("last"), dict):
                    cand.append(res["last"].get("price"))
                cand += [res.get(k) for k in ("price", "value", "p", "close")]
                for v in cand:
                    if v is not None:
                        return float(v), int(_time.time()), "v3.snapshot.stocks"
    except Exception:
        pass
    # 2b) v3 snapshot indices (for I:SPX, etc.)
    try:
        u = f"{POLYGON_BASE_URL}/v3/snapshot/indices"
        r = requests.get(u, params={"ticker": symbol, "apiKey": api_key}, timeout=15)
        if r.ok:
            j = r.json()
            res = j.get("results") or {}
            if isinstance(res, dict):
                cand = []
                if isinstance(res.get("last"), dict):
                    cand.append(res["last"].get("price"))
                cand += [res.get(k) for k in ("price", "value", "p", "close")]
                for v in cand:
                    if v is not None:
                        return float(v), int(_time.time()), "v3.snapshot.indices"
    except Exception:
        pass
    # 3) v2 snapshot stock
    try:
        u = f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        r = requests.get(_append_key(u, api_key), timeout=15)
        if r.ok:
            j = r.json()
            last_trade = j.get("lastTrade") or {}
            p = last_trade.get("p") or last_trade.get("price")
            if p is None and isinstance(j.get("day"), dict):
                p = j["day"].get("close") or j["day"].get("c")
            if p is not None:
                return float(p), int(_time.time()), "v2.snapshot.stocks"
    except Exception:
        pass
    # 4) previous close
    try:
        u = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/prev"
        r = requests.get(u, params={"apiKey": api_key, "adjusted": "true"}, timeout=15)
        if r.ok:
            j = r.json()
            res = j.get("results") or []
            if res:
                p = res[0].get("close") or res[0].get("c")
                t = res[0].get("t")
                ts = int(int(t)/1000) if isinstance(t, int) and t > 1_000_000_000_000 else int(_time.time())
                if p is not None:
                    return float(p), ts, "v2.prev.close"
    except Exception:
        pass
    return None, None, "none"


def _remap_chain(items: List[dict], S: Optional[float], ts_unix: int) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    exp_map: Dict[str, int] = {}

    def push(exp_str: str, side: str, rec: Dict[str, Any]):
        blk = grouped.setdefault(exp_str, {"expirationDate": exp_str, "calls": [], "puts": []})
        blk["calls" if side == "call" else "puts"].append(rec)

    for it in items:
        details = it.get("details") or {}
        ctype = (details.get("contract_type") or "").lower()
        if ctype not in ("call", "put"):
            continue
        exp = details.get("expiration_date")
        strike = details.get("strike_price")
        if not exp or strike is None:
            continue

        if exp not in exp_map:
            exp_map[exp] = _to_unix(exp)

        day = it.get("day") or {}
        last_trade = it.get("last_trade") or {}

        rec = {
            "contractSymbol": details.get("ticker"),
            "strike": float(strike),
            "openInterest": int((it.get("open_interest") or 0) or 0),
            "volume": int((day.get("volume") or 0) or 0),
            "impliedVolatility": float(it.get("implied_volatility")) if it.get("implied_volatility") is not None else None,
            "lastPrice": float(last_trade.get("price") or last_trade.get("p")) if (last_trade.get("price") or last_trade.get("p")) else None,
            "expiration": exp,
        }
        push(exp, ctype, rec)

    expirationDates = sorted(exp_map.values())
    options_blocks = []
    for exp in sorted(grouped.keys(), key=lambda s: exp_map.get(s, 0)):
        blk = dict(grouped[exp])
        blk["expirationDate"] = exp_map[exp]  # unix
        options_blocks.append(blk)

    return {
        "optionChain": {
            "result": [{
                "regularMarketPrice": float(S) if S is not None else None,
                    "regularMarketPreviousClose": float(S) if S is not None else None,
                "quote": {
                    "regularMarketPrice": float(S) if S is not None else None,
                    "regularMarketPreviousClose": float(S) if S is not None else None,
                    "regularMarketDayHigh": None,
                    "regularMarketDayLow": None,
                    "regularMarketTime": int(ts_unix),
                },
                "expirationDates": expirationDates,
                "options": options_blocks,
            }],
            "error": None,
        }
    }


# ---------------- public API ----------------

def fetch_option_chain(ticker: str, host_unused: Optional[str], api_key: str, expiry_unix: Optional[int] = None) -> Tuple[Dict[str, Any], bytes]:
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is missing")

    symbol = (ticker or "").strip().upper()
    poly_symbol = "I:SPX" if symbol in ("SPX", "^SPX") else symbol

    # Load options chain
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{poly_symbol}"
    params = {"limit": 250, "order": "asc", "sort": "ticker"}
    exp_iso = None
    if expiry_unix:
        exp_iso = _to_iso(int(expiry_unix))
        if exp_iso:
            params["expiration_date"] = exp_iso

    items = _paginate(url, api_key=api_key, params=params, cap=80)

    # Underlying price
    S = _scan_price_from_items(items)
    ts = int(_time.time())
    price_source = "items.underlying_asset"
    if S is None:
        S, ts, price_source = _price_fallback(poly_symbol, api_key)

    # Build normalized json
    out_json = _remap_chain(items, S, ts)

    # Raw bytes
    raw_dump = {
        "endpoint": "/v3/snapshot/options",
        "ticker": poly_symbol,
        "expiration_date": exp_iso,
        "price_source": price_source,
        "items_count": len(items),
        "items": items,
    }
    raw_bytes = json.dumps(raw_dump, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return out_json, raw_bytes
