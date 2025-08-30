import requests, json

DEFAULT_TIMEOUT = 20

def fetch_option_chain(ticker, host, api_key, expiry_unix=None, timeout=DEFAULT_TIMEOUT):
    """
    Fetch options chain JSON from yahoo-finance15 RapidAPI host.

    Primary (per provider docs / your screenshots):
      GET https://{host}/api/v1/markets/options?ticker=SPY&display=chain&expiration=UNIX

    Fallbacks: several legacy yahoo-like paths that some deployments expose.
    Returns (json_dict, raw_bytes). Raises RuntimeError with a compact trace on failure.
    """
    base_url = f"https://{host}"
    candidates = [
        {"url": f"{base_url}/api/v1/markets/options",                 "mode": "v1"},            # ticker, display=chain, expiration
        {"url": f"{base_url}/api/yahoo/options/{ticker}",             "mode": "path"},          # ?date=
        {"url": f"{base_url}/api/yahoo/finance/options/{ticker}",     "mode": "path"},          # ?date=
        {"url": f"{base_url}/api/yahoo/finance/option/{ticker}",      "mode": "path"},          # ?date=
        {"url": f"{base_url}/api/yahoo/finance/stock/options/{ticker}","mode": "path"},         # ?date=
        {"url": f"{base_url}/api/yahoo/finance/v2/options/{ticker}",  "mode": "path"},          # ?date=
        {"url": f"{base_url}/api/yahoo/options",                      "mode": "query_symbol"},  # ?symbol=&date=
        {"url": f"{base_url}/api/yahoo/finance/options",              "mode": "query_symbol"},  # ?symbol=&date=
    ]
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }
    errors = []
    for c in candidates:
        params = {}
        if c["mode"] == "v1":
            params["ticker"] = ticker
            params["display"] = "chain"  # CRITICAL: we need calls/puts rather than straddles
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
