import requests, time, json

DEFAULT_TIMEOUT = 20

def fetch_option_chain(ticker, host, api_key, expiry_unix=None, timeout=20):
    import requests, json
    base_url = f"https://{host}"
    # 1) канонический путь провайдера из твоего скрина
    candidates = [
        {"url": f"{base_url}/api/v1/markets/options",                 "mode": "query_ticker"},
        # 2) запасные варианты (встречаются у того же хоста в разных версиях пакета)
        {"url": f"{base_url}/api/yahoo/options/{ticker}",             "mode": "path"},
        {"url": f"{base_url}/api/yahoo/finance/options/{ticker}",     "mode": "path"},
        {"url": f"{base_url}/api/yahoo/finance/option/{ticker}",      "mode": "path"},
        {"url": f"{base_url}/api/yahoo/finance/stock/options/{ticker}","mode": "path"},
        {"url": f"{base_url}/api/yahoo/finance/v2/options/{ticker}",  "mode": "path"},
        {"url": f"{base_url}/api/yahoo/options",                      "mode": "query_symbol"},
        {"url": f"{base_url}/api/yahoo/finance/options",              "mode": "query_symbol"},
    ]
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }
    errors = []
    for c in candidates:
        params = {}
        if c["mode"] == "query_ticker":
            params["ticker"] = ticker
            if expiry_unix is not None:
                params["expiration"] = int(expiry_unix)  # ВАЖНО: у этого эндпоинта параметр называется expiration
        elif c["mode"] == "query_symbol":
            params["symbol"] = ticker
            if expiry_unix is not None:
                # некоторые альтернативные пути принимают date
                params["date"] = int(expiry_unix)
        else:
            if expiry_unix is not None:
                params["date"] = int(expiry_unix)

        try:
            r = requests.get(c["url"], headers=headers, params=params, timeout=timeout)
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    data = json.loads(r.content.decode("utf-8", errors="ignore"))
                return data, r.content
            errors.append(f"{r.status_code} {c['url']} {params} -> {r.text[:200]}")
        except Exception as e:
            errors.append(f"EXC {c['url']} {params} -> {e}")
    raise RuntimeError("Option chain fetch failed. Tried:\n" + "\n".join(errors))
