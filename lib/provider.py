# lib/provider.py
import json
import requests

DEFAULT_TIMEOUT = 20


def _request_json(url: str, headers: dict, params: dict, timeout: int):
    """HTTP GET с разбором JSON и компактной ошибкой."""
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {url} {params} -> {r.text[:200]}")
    try:
        return r.json(), r.content
    except Exception:
        return json.loads(r.content.decode("utf-8", errors="ignore")), r.content


def _extract_price_from_quote_obj(quote: dict):
    """Пытаемся извлечь цену/время из quote-подобного словаря (без выдумывания значений)."""
    price = None
    for key in (
        "regularMarketPrice",
        "postMarketPrice",
        "last",
        "lastPrice",
        "price",
        "close",
        "regularMarketPreviousClose",
    ):
        v = quote.get(key, None)
        if v is None:
            continue
        try:
            price = float(v)
            break
        except Exception:
            continue

    t0 = 0
    for key in ("regularMarketTime", "postMarketTime", "time", "timestamp", "lastTradeDate"):
        v = quote.get(key, None)
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
    """Находим корень, содержащий quote/expirationDates/options (в разных форматах провайдера)."""
    # 1) Классический Yahoo
    try:
        res = chain_json.get("optionChain", {}).get("result", [])
        if isinstance(res, list) and res:
            return res[0], "optionChain.result[0]"
    except Exception:
        pass

    # 2) body
    body = chain_json.get("body", None)
    if isinstance(body, list) and body:
        return body[0], "body[0]"
    if isinstance(body, dict):
        return body, "body"

    # 3) data
    data = chain_json.get("data", None)
    if isinstance(data, list) and data:
        return data[0], "data[0]"
    if isinstance(data, dict):
        return data, "data"

    # 4) result
    res2 = chain_json.get("result", None)
    if isinstance(res2, list) and res2:
        return res2[0], "result[0]"
    if isinstance(res2, dict):
        return res2, "result"

    # 5) запасной вариант — сам корень
    if isinstance(chain_json, dict):
        return chain_json, "<root>"

    raise RuntimeError("Cannot locate root in chain JSON")


def _ensure_quote_with_price(root: dict, host: str, api_key: str, ticker: str, timeout: int):
    """Гарантируем наличие цены в root['quote']; если её нет, подтягиваем с quote-эндпоинтов."""
    quote = root.get("quote", None)
    if quote is None or not isinstance(quote, dict):
        quote = {}
        root["quote"] = quote

    price, t0 = _extract_price_from_quote_obj(quote)
    if price is not None and price > 0:
        return  # всё есть

    # иногда цена лежит в 'underlying'
    u = root.get("underlying", {})
    if isinstance(u, dict):
        price2, t02 = _extract_price_from_quote_obj(u)
        if price2 is not None and price2 > 0:
            quote.setdefault("regularMarketPrice", price2)
            if t02:
                quote.setdefault("regularMarketTime", t02)
            return

    # пробуем несколько quote-эндпоинтов на том же хосте
    base = f"https://{host}"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }
    q_candidates = [
        (f"{base}/api/v1/market/quotes", {"tickers": ticker}),  # snapshots
        (f"{base}/api/v1/market/quotes", {"ticker": ticker}),
        (f"{base}/api/v1/markets/quotes", {"tickers": ticker}),
        (f"{base}/api/v1/markets/quotes", {"ticker": ticker}),
        (f"{base}/api/yahoo/quotes/{ticker}", {}),
    ]
    for url, params in q_candidates:
        try:
            qjson, _ = _request_json(url, headers, params, timeout)
            # a) { quoteResponse: { result: [ {...} ] } }
            try:
                res = qjson.get("quoteResponse", {}).get("result", [])
                if isinstance(res, list) and res:
                    p, t = _extract_price_from_quote_obj(res[0])
                    if p is not None and p > 0:
                        quote["regularMarketPrice"] = p
                        if t:
                            quote["regularMarketTime"] = t
                        return
            except Exception:
                pass
            # b) { body: [...] } or body:{...}
            body = qjson.get("body", None)
            if isinstance(body, list) and body:
                p, t = _extract_price_from_quote_obj(body[0])
                if p is not None and p > 0:
                    quote["regularMarketPrice"] = p
                    if t:
                        quote["regularMarketTime"] = t
                    return
            elif isinstance(body, dict):
                p, t = _extract_price_from_quote_obj(body)
                if p is not None and p > 0:
                    quote["regularMarketPrice"] = p
                    if t:
                        quote["regularMarketTime"] = t
                    return
            # c) плоский объект
            p, t = _extract_price_from_quote_obj(qjson)
            if p is not None and p > 0:
                quote["regularMarketPrice"] = p
                if t:
                    quote["regularMarketTime"] = t
                return
        except Exception:
            continue
    # если так и не нашли — оставляем как есть (ничего не придумываем)


def fetch_option_chain(ticker: str, host: str, api_key: str, expiry_unix: int | None = None, timeout: int = DEFAULT_TIMEOUT):
    """
    Основной запрос опционного цепочки:
      первично:  GET /api/v1/markets/options?ticker=...&display=chain&expiration=...
      запасные:  /api/yahoo/... варианты (с ?date=)
    Плюс автодобавление цены в root['quote'] при отсутствии.
    Возвращает: (json_dict, raw_bytes)
    """
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
    errors: list[str] = []
    raw_bytes = None

    for c in candidates:
        params: dict = {}
        if c["mode"] == "v1":
            params["ticker"] = ticker
            params["display"] = "chain"  # хотим calls/puts, а не straddles
            if expiry_unix is not None:
                params["expiration"] = int(expiry_unix)
        elif c["mode"] == "query_symbol":
            params["symbol"] = ticker
            if expiry_unix is not None:
                params["date"] = int(expiry_unix)
        else:  # "path"
            if expiry_unix is not None:
                params["date"] = int(expiry_unix)

        try:
            data, content = _request_json(c["url"], headers, params, timeout)
            raw_bytes = content  # для кнопки «скачать сырой JSON»
            # гарантируем наличие цены в quote (raw_bytes не трогаем)
            try:
                root, _ = _locate_root_mutable(data)
                _ensure_quote_with_price(root, host, api_key, ticker, timeout)
            except Exception:
                pass
            return data, raw_bytes
        except Exception as e:
            errors.append(str(e))
            continue

    raise RuntimeError("Option chain fetch failed. Tried:\n" + "\n".join(errors))
