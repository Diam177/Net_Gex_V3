# -*- coding: utf-8 -*-
"""
Polygon provider adapter → 'Yahoo-like' JSON для приложения.
Передаёт ключ ТОЛЬКО через query (apiKey=...), без headers.
"""

from typing import Dict, Any, List, Optional, Tuple
import datetime as _dt
import time as _time
import json
import requests

POLYGON_BASE_URL = "https://api.polygon.io"

# ----------------- helpers -----------------

def _append_api_key(url: str, api_key: str) -> str:
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
    """Проходит все страницы Polygon (results + next_url). На каждой добавляем apiKey."""
    out: List[dict] = []
    next_url = _append_api_key(url, api_key)
    next_params = dict(params or {})
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
        next_url = _append_api_key(nxt, api_key)
        next_params = {}  # дальше всё в next_url
    return out

def _scan_underlying_price_from_items(items: List[dict]) -> Optional[float]:
    """Если провайдер вернул цену подложки в элементах цепочки."""
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

def _get_underlying_price_http(symbol: str, api_key: str) -> Tuple[Optional[float], Optional[int], str]:
    # 1) last trade
    try:
        u = f"{POLYGON_BASE_URL}/v2/last/trade/{symbol}"
        r = requests.get(_append_api_key(u, api_key), timeout=15)
        if r.ok:
            j = r.json()
            res = j.get("results") or {}
            p = res.get("p") or res.get("price")
            t = res.get("t") or res.get("timestamp")
            if p is not None:
                ts_unix = int(int(t)/1_000_000_000) if isinstance(t, int) and t > 1_000_000_000_000 else int(_time.time())
                return float(p), ts_unix, "v2.last.trade"
    except Exception:
        pass
    # 2) v3 stocks snapshot
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
                        try:
                            return float(v), int(_time.time()), "v3.snapshot.stocks"
                        except Exception:
                            pass
    except Exception:
        pass
    # 3) v2 stocks snapshot
    try:
        u = f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        r = requests.get(_append_api_key(u, api_key), timeout=15)
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
                ts_unix = int(int(t)/1_000) if isinstance(t, int) and t > 1_000_000_000_000 else int(_time.time())
                if p is not None:
                    return float(p), ts_unix, "v2.prev.close"
    except Exception:
        pass
    return None, None, "none"

def _remap_to_yahoo_like(items: List[dict], S: Optional[float], ts_unix: Optional[int]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    exp_unix_map: Dict[str, int] = {}

    def _push(exp_str: str, kind: str, rec: Dict[str, Any]):
        blk = grouped.setdefault(exp_str, {"expirationDate": exp_str, "calls": [], "puts": []})
        blk["calls" if kind == "call" else "puts"].append(rec)

    for it in items:
        details = it.get("details") or {}
        ctype = (details.get("contract_type") or "").lower()
        if ctype not in ("call", "put"):
            continue
        exp_str = details.get("expiration_date")
        strike = details.get("strike_price")
        if not exp_str or strike is None:
            continue

        if exp_str not in exp_unix_map:
            exp_unix_map[exp_str] = _to_unix(exp_str)

        day = it.get("day") or {}
        last_trade = it.get("last_trade") or {}

        rec = {
            "contractSymbol": details.get("ticker"),
            "strike": float(strike),
            "openInterest": int((it.get("open_interest") or 0) or 0),
            "volume": int((day.get("volume") or 0) or 0),
            "impliedVolatility": float(it.get("implied_volatility")) if it.get("implied_volatility") is not None else None,
            "lastPrice": float(last_trade.get("price") or last_trade.get("p")) if (last_trade.get("price") or last_trade.get("p")) else None,
            "expiration": exp_str,
        }
        _push(exp_str, ctype, rec)

    expirationDates = sorted(exp_unix_map.values())

    options_blocks: List[Dict[str, Any]] = []
    for exp_str in sorted(grouped.keys(), key=lambda s: exp_unix_map.get(s, 0)):
        blk = dict(grouped[exp_str])
        blk["expirationDate"] = exp_unix_map[exp_str]  # unix!
        options_blocks.append(blk)

    chain_obj = {
        "quote": {
            "regularMarketPrice": float(S) if S is not None else None,
            "postMarketPrice": float(S) if S is not None else None,
            "regularMarketDayHigh": None,
            "regularMarketDayLow": None,
            "regularMarketTime": int(ts_unix or _time.time()),
        },
        "expirationDates": expirationDates,
        "options": options_blocks,
    }
    return {"optionChain": {"result": [chain_obj], "error": None}}

# ----------------- public API -----------------

def fetch_option_chain(ticker: str,
                       host_unused: Optional[str],
                       api_key: str,
                       expiry_unix: Optional[int] = None) -> Tuple[Dict[str, Any], bytes]:
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is missing")

    underlying = (ticker or "").strip().upper()
    underlying_symbol = "I:SPX" if underlying in ("SPX", "^SPX") else underlying

    # 1) основная выборка (с фильтром expiration_date при необходимости)
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{underlying_symbol}"
    params = {"limit": 250, "order": "asc", "sort": "ticker", "apiKey": api_key}
    expiration_date_iso = None
    if expiry_unix:
        expiration_date_iso = _to_iso(int(expiry_unix))
        if expiration_date_iso:
            params["expiration_date"] = expiration_date_iso

    items = _paginate(url, api_key=api_key, params=params, cap=80)

    # 2) цена подложки — СНАЧАЛА пытаемся получить из stocks snapshot / last trade (самая свежая)
    S = None
    ts_unix = int(_time.time())
    price_source = None
    try:
        S, ts_unix, price_source = _get_underlying_price_http(underlying_symbol, api_key)
    except Exception:
        S = None
        price_source = None

    # Если не удалось — берём из items.underlying_asset (может быть устаревшим)
    if S is None:
        S = _scan_underlying_price_from_items(items)
        ts_unix = int(_time.time())
        price_source = "items.underlying_asset"
    out_json = _remap_to_yahoo_like(items, S, ts_unix)
    
    # 4) сырые байты провайдера
    raw_dump = {
        "endpoint": "/v3/snapshot/options",
        "ticker": underlying_symbol,
        "expiration_date": expiration_date_iso,
        "price_source": price_source,
        "items_count": len(items),
        "items": items,
    }
    raw_bytes = json.dumps(raw_dump, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return out_json, raw_bytes


    # ------------- stock candles (intraday) -------------

def fetch_stock_history(ticker: str,
                        host_unused: Optional[str],
                        api_key: str,
                        interval: str = "1m",
                        limit: int = 640,
                        dividend: Optional[bool] = None,
                        timeout: int = 20) -> Tuple[Dict[str, Any], bytes]:
    """
    Получить исторические свечи у Polygon (v2 aggregates).
    Возвращает (json_dict, raw_bytes) где json_dict имеет ключ "candles": [ ... ]
    Поля каждой свечи: timestamp_unix, open, high, low, close, volume.
    """
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is missing")

    symbol = (ticker or "").strip().upper()
    # Map interval "1m","5m","15m","1h" -> (mult, timespan)
    span = "minute"
    mult = 1
    try:
        if isinstance(interval, str) and interval.endswith("m"):
            mult = max(1, int(float(interval[:-1])))
            span = "minute"
        elif isinstance(interval, str) and interval.endswith("h"):
            mult = max(1, int(float(interval[:-1])) * 60)
            span = "minute"
        elif isinstance(interval, str) and interval.endswith("d"):
            mult = max(1, int(float(interval[:-1])))
            span = "day"
        else:
            mult = 1
            span = "minute"
    except Exception:
        mult = 1
        span = "minute"

    # Compute time window
    now_utc = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
    # take a slightly larger window to ensure we get 'limit' sorted asc
    minutes = mult * max(int(limit), 1)
    from_dt = now_utc - _dt.timedelta(minutes=minutes * 2)
    to_dt = now_utc
    # Polygon expects ISO timestamps or dates
    from_ms = int(from_dt.timestamp() * 1000)
    to_ms = int(to_dt.timestamp() * 1000)

    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{from_ms}/{to_ms}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": int(limit),
        "apiKey": api_key,
    }
    r = requests.get(url, params=params, timeout=timeout)
    raw_bytes = r.content
    r.raise_for_status()
    j = r.json()

    # Fallback: if no results, try last regular trading session window in ET (09:30-16:00)
    def _retry_last_session():
        try:
            import pytz
            tz = pytz.timezone("America/New_York")
            now_et = _dt.datetime.now(tz)
            # move to previous weekday if weekend
            while now_et.weekday() >= 5:
                now_et = now_et - _dt.timedelta(days=1)
            session_start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            session_end   = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            # if we're before 10:00 ET, use previous weekday
            if now_et < session_start + _dt.timedelta(minutes=30):
                prev = now_et - _dt.timedelta(days=1)
                while prev.weekday() >= 5:
                    prev = prev - _dt.timedelta(days=1)
                session_start = prev.replace(hour=9, minute=30, second=0, microsecond=0)
                session_end   = prev.replace(hour=16, minute=0, second=0, microsecond=0)
            from_ms = int(session_start.astimezone(_dt.timezone.utc).timestamp()*1000)
            to_ms   = int(session_end.astimezone(_dt.timezone.utc).timestamp()*1000)
            url2 = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/{mult}/{span}/{from_ms}/{to_ms}"
            r2 = requests.get(url2, params=params, timeout=timeout)
            rb2 = r2.content
            r2.raise_for_status()
            j2 = r2.json()
            return j2, rb2
        except Exception:
            return None, None

    if not j.get("results"):
        j2, rb2 = _retry_last_session()
        if j2 is not None:
            j = j2
            raw_bytes = rb2 or raw_bytes

    # Expected j: { "results": [ { "t": 1717600800000, "o":..., "h":..., "l":..., "c":..., "v":... }, ... ] }
    records: List[Dict[str, Any]] = []
    try:
        for rec in j.get("results", []) or []:
            t = rec.get("t")
            ts_unix = int(int(t) / 1000) if isinstance(t, (int, float)) else None
            records.append({
                "timestamp_unix": ts_unix,
                "open": rec.get("o"),
                "high": rec.get("h"),
                "low":  rec.get("l"),
                "close":rec.get("c"),
                "volume": rec.get("v"),
            })
    except Exception:
        # leave empty on failure
        records = []

    out = {"candles": records}
    return out, raw_bytes
