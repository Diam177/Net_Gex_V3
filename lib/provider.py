# lib/provider.py
# Polygon-only provider. RapidAPI fully removed.
from __future__ import annotations

import os
import time
import requests
from typing import Any, Dict, List, Optional

POLYGON_API_URL = "https://api.polygon.io"


class PolygonConfigError(RuntimeError):
    pass


# --- helpers ---------------------------------------------------------------
def _polygon_api_key() -> str:
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        raise PolygonConfigError(
            "POLYGON_API_KEY не найден. Установите переменную окружения или secrets.toml"
        )
    return key


INDEX_SYMBOLS = {
    "SPX", "NDX", "DJX", "RUT", "VIX", "OEX", "XSP"
}


def _as_underlying(symbol: str) -> str:
    s = symbol.strip().upper()
    # drop caret if passed like ^SPX
    if s.startswith("^"):
        s = s[1:]
    # Polygon expects I: prefix for indices
    return f"I:{s}" if s in INDEX_SYMBOLS else s


def _http_get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(params or {})
    # next_url from Polygon may already include apiKey; only add if missing
    if "apiKey=" not in url:
        params["apiKey"] = _polygon_api_key()

    # cautious retries for 429/503
    for attempt in range(3):
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code in (429, 503) and attempt < 2:
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Не удалось получить данные от Polygon после повторов")


# --- public API ------------------------------------------------------------
def fetch_option_chain(
    symbol: str,
    expiration: Optional[str] = None,
    limit: int = 250,
    **_ignored
) -> Dict[str, Any]:
    """Fetch option chain via Polygon Option Chain Snapshot.

    Args:
        symbol: underlying ticker (e.g., 'AAPL', 'SPY', 'SPX').
        expiration: optional 'YYYY-MM-DD' filter.
        limit: page size (1..250).

    Returns:
        dict with key 'results' — list of contracts compatible with
        downstream compute.extract_core_from_chain logic. We keep Polygon's
        native fields like 'details', 'day', 'open_interest', 'greeks',
        'implied_volatility', 'last_quote', etc.
    """
    underlying = _as_underlying(symbol)
    base_url = f"{POLYGON_API_URL}/v3/snapshot/options/{underlying}"

    params: Dict[str, Any] = {"limit": max(1, min(int(limit), 250))}
    if expiration:
        params["expiration_date"] = expiration  # YYYY-MM-DD

    results: List[Dict[str, Any]] = []

    data = _http_get(base_url, params)
    results.extend(data.get("results", []) or [])

    next_url = data.get("next_url")
    # safety limit: up to ~5000 contracts (20 pages * 250)
    safety = 0
    while next_url and safety < 20:
        data = _http_get(next_url)
        results.extend(data.get("results", []) or [])
        next_url = data.get("next_url")
        safety += 1

    return {"results": results}
