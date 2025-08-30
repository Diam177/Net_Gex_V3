# lib/provider.py
import json
import requests
from typing import Tuple, Dict, Any, Optional

DEFAULT_TIMEOUT = 20

# Храним последний отладочный мета-лог, чтобы показать его в Streamlit
_LAST_DEBUG_META: Dict[str, Any] = {}


def debug_meta() -> Dict[str, Any]:
    """Вернуть отладочную мета-информацию о последнем запросе к провайдеру."""
    return _LAST_DEBUG_META.copy()


def _request_json(url: str, headers: dict, params: dict, timeout: int):
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {url} {params} -> {r.text[:200]}")
    try:
        return r.json(), r.content
    except Exception:
        return json.loads(r.content.decode("utf-8", errors="ignore")), r.content


def _extract_price_from_quote_obj(quote: dict):
    price = None
    for key in (
        "regularMarketPrice", "postMarketPrice", "last", "lastPrice",
        "price", "close", "regularMarketPreviousClose"
    ):
        v = quote.get(key)
        if v is None:
            continue
        try:
            price = float(v)
            break
        except Exception:
            continue

    t0 = 0
    for key in ("regularMarketTime", "postMarketTime", "time", "timestamp", "lastTradeDate"):
        v = quote.get(key)
        if v is None:
            continue
        try:
            t0 = int(v)
            break
        except Exception:
            try:
                t0 = int(float(v))
                break
            except Exception:
                continue
    return price, t0


def _locate_root_mutable(chain_json: dict):
    # 1) Yahoo canonical
    try:
        res = chain_json.get("optionChain", {}).get("result", [])
        if isinstance(res, list) and res:
            return res[0], "optionChain.result[0]"
    except Exception:
        pass

    # 2) body
    body = chain_json.get("body")
    if isinstance(body, list) and body:
        return body[0], "body[0]"
    if isinstance(body, dict):
        return body, "body"

    # 3) data
    data = chain_json.get("data")
    if isinstance(data, list) and data:
        return data[0], "data[0]"
    if isinstance(data, dict):
        return data, "data"

    # 4) result
    res2 = chain_json.get("result")
    if isinstance(res2, list) and res2:
        return res2[0], "result[0]"
    if isinstance(res2, dict):
        return res2, "result"

    # 5) fallback
    if isinstance(chain_json, dict):
        return chain_json, "<root>"

    raise RuntimeError("Cannot locate root in chain JSON")


def _ensure_quote_with_price(root: dict, meta: dict, host: str, api_key: str, ticker: str, timeout: int):
    """Если в root['quote'] нет цены — подтягиваем её через quote-эндпоинты того же хоста (без выдумывания)."""
    quote = root.get("quote")
    if not isinstance(quote, dict):
        quote = {}
        root["quote"] = quote

    before_price, before_t0 = _extract_price_from_quote_obj(quote)
    meta["price_before"] = before_price
    meta["t0_before"] = before_t0
    if before_price is not None and before_price > 0:
        meta["price_source"] = "quote"
        return

    # иногда цена лежит в 'underlying'
    u = root.get("underlying", {})
    if isinstance(u, dict):
        p2, t02 = _extract_price_from_quote_obj(u)
        if p2 is not None and p2 > 0:
            quote.setdefault("regularMarketPrice", p2)
            if t02:
                quote.setdefault("regularMarketTime", t02)
            meta["price_source"] = "underlying"
            meta["price_after"] = p2
            meta["t0_after"] = t02
            return

    # пробуем несколько quote-эндпоинтов
    base = f"https://{host}"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }
    q_candidates = [
        (f"{base}/api/v1/market/quotes", {"tickers": ticker}),
        (f"{base}/api/v1/market/quotes", {"ticker": ticker}),
        (f"{base}/api/v1/markets/quotes", {"tickers": ticker}),
        (f"{base}/api/v1/markets/quotes", {"ticker": ticker}),
        (f"{base}/api/yahoo/quotes/{ticker}", {}),
    ]
    meta["quote_attempts"] = []
    for url, params in q_candidates:
        try:
            qjson, _ = _request_json(url, headers, params, timeout)
            meta["quote_attempts"].append({"url": url, "params": params, "ok": True})
            # a) quoteResponse.result[0]
            try:
                res = qjson.get("quoteResponse", {}).get("result", [])
                if isinstance(res, list) and res:
                    p, t = _extract_price_from_quote_obj(res[0])
                    if p is not None and p > 0:
                        quote["regularMarketPrice"] = p
                        if t:
                            quote["regularMarketTime"] = t
                        meta["price_source"] = "quoteResponse.result[0]"
                        meta["price_after"] = p
                        meta["t0_after"] = t
                        return
            except Exception:
                pass
            # b) body[..] или body{}
            body = qjson.get("body")
            if isinstance(body, list) and body:
                p, t = _extract_price_from_quote_obj(body[0])
                if p is not None and p > 0:
                    quote["regularMarketPrice"] = p
                    if t:
                        quote["regularMarketTime"] = t
                    meta["price_source"] = "body[0]"
                    meta["price_after"] = p
                    meta["t0_after"] = t
                    return
            elif isinstance(body, dict):
                p, t = _extract_price_from_quote_obj(body)
                if p is not None and p > 0:
                    quote["regularMarketPrice"] = p
                    if t:
                        quote["regularMarketTime"] = t
                    meta["price_source"] = "body"
                    meta["price_after"] = p
                    meta["t0_after"] = t
                    return
            # c) плоский объект
            p, t = _extract_price_from_quote_obj(qjson)
            if p is not None and p > 0:
                quote["regularMarketPrice"] = p
                if t:
                    quote["regularMarketTime"] = t
                meta["price_source"] = "flat"
                meta["price_after"] = p
                meta["t0_after"] = t
                return
        except Exception as e:
            meta["quote_attempts"].append({"url": url, "params": params, "ok": False, "err": str(e)})
            continue
    meta["price_source"] = "not_found"


def fetch_option_chain(
    ticker: str,
    host: str,
    api_key: str,
    expiry_unix: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[dict, bytes]:
    """
    Основной запрос опционной цепочки:
        первично:  GET /api/v1/markets/options?ticker=...&display=chain&expiration=...
        запасные:  /api/yahoo/... (с ?date=)
    Плюс — автодобавление цены в root['quote'] при отсутствии (без выдумывания).
    Возвращает: (json_dict, raw_bytes)
    """
    global _LAST_DEBUG_META
    meta = {
        "ticker": ticker,
        "host": host,
        "expiry_unix": int(expiry_unix) if expiry_unix is not None else None,
        "attempts": [],
        "used": None,
        "root_path": None,
    }

    base_url = f"https://{host}"
    candidates = [
        {"url": f"{base_url}/api/v1/markets/options", "mode": "v1"},
        {"url": f"{base_url}/api/yahoo/options/{ticker}", "mode": "path"},
        {"url": f"{base_url}/api/yahoo/finance/options/{ticker}", "mode": "path"},
        {"url": f"{base_url}/api/yahoo/finance/option/{ticker}", "mode": "path"},
        {"url": f"{base_url}/api/yahoo/finance/stock/options/{ticker}", "mode": "path"},
        {"url": f"{base_url}/api/yahoo/finance/v2/options/{ticker}", "mode": "path"},
        {"url": f"{base_url}/api/yahoo/options", "mode": "query_symbol"},
        {"url": f"{base_url}/api/yahoo/finance/options", "mode": "query_symbol"},
    ]
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }

    errors = []
    raw_bytes = b""

    for c in candidates:
        params = {}
        if c["mode"] == "v1":
            params["ticker"] = ticker
            params["display"] = "chain"
            if expiry_unix is not None:
                params["expiration"] = int(expiry_unix)
        elif c["mode"] == "query_symbol":
            params["symbol"] = ticker
            if expiry_unix is not None:
                params["date"] = int(expiry_unix)
        else:  # path
            if expiry_unix is not None:
                params["date"] = int(expiry_unix)

        try:
            data, content = _request_json(c["url"], headers, params, timeout)
            raw_bytes = content
            meta["attempts"].append({"url": c["url"], "params": params, "ok": True})
            meta["used"] = {"url": c["url"], "params": params}

            # найдём корень + гарантируем цену
            try:
                root, where = _locate_root_mutable(data)
                meta["root_path"] = where
                _ensure_quote_with_price(root, meta, host, api_key, ticker, timeout)
            except Exception as e:
                meta["root_path"] = f"locate_failed: {e}"

            _LAST_DEBUG_META = meta
            return data, raw_bytes

        except Exception as e:
            err = {"url": c["url"], "params": params, "ok": False, "err": str(e)}
            meta["attempts"].append(err)
            errors.append(str(e))
            continue

    _LAST_DEBUG_META = meta
    raise RuntimeError("Option chain fetch failed. Tried:\n" + "\n".join(errors))
