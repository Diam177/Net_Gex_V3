import requests, time, json

DEFAULT_TIMEOUT = 20

def fetch_option_chain(ticker, host, api_key, expiry_unix=None, timeout=DEFAULT_TIMEOUT):
    base_url = f"https://{host}"
    candidates = [
        f"{base_url}/api/yahoo/options/{ticker}",
        f"{base_url}/api/yahoo/finance/options/{ticker}",
        f"{base_url}/api/yahoo/option/chain/{ticker}",
    ]
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }
    params = {}
    if expiry_unix is not None:
        params["date"] = int(expiry_unix)

    last_err = None
    for url in candidates:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    data = json.loads(r.content.decode("utf-8", errors="ignore"))
                return data, r.content
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"Option chain fetch failed: {last_err}")
