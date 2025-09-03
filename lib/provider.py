# lib/provider.py (Polygon-only wrapper)
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List
import time as _time
import datetime as _dt
import requests

from . import provider_polygon as _poly

# Debug state container
_DEBUG_LAST: Dict[str, Any] = {}

def debug_meta() -> Dict[str, Any]:
    """Return meta about the last provider call for UI debugging."""
    return dict(_DEBUG_LAST)

# Re-export option chain fetcher directly from Polygon adapter
fetch_option_chain = _poly.fetch_option_chain

# ---- Stock history (intraday candles) via Polygon ----
# We keep signature compatible with previous code: (ticker, host, key, interval="1m", limit=640, dividend=None)
# - 'host' is ignored (RapidAPI removed)
# - interval supports "1m" only for now (app uses 1m)
# Returns: (json_obj, raw_bytes)
def fetch_stock_history(ticker: str,
                        host: Optional[str],
                        api_key: str,
                        interval: str = "1m",
                        limit: int = 640,
                        dividend: Optional[bool] = None) -> Tuple[Dict[str, Any], bytes]:
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is required for stock history")
    # Determine range: fetch enough recent minutes to cover last and current session
    # We take ~3 trading days back as a safe window.
    now = _dt.datetime.utcnow()
    start = now - _dt.timedelta(days=6)  # buffer; app will pick "last session" internally
    # Polygon expects UNIX ms for from/to in v2 aggs (when using date strings it's YYYY-MM-DD)
    frm = int(_time.mktime(start.timetuple())) * 1000
    to  = int(_time.mktime(now.timetuple())) * 1000

    if interval not in ("1m", "1min", "1minute"):
        # Fallback to 1 minute; the intraday chart expects minute bars.
        interval = "1m"

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{frm}/{to}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": int(limit) if limit else 50000,
        "apiKey": api_key,
    }

    meta: Dict[str, Any] = {"what": "polygon.candles", "url": url, "params": dict(params)}
    try:
        r = requests.get(url, params=params, timeout=20)
        raw_bytes = r.content or b""
        meta["status_code"] = r.status_code
        if not r.ok:
            try:
                _DEBUG_LAST.update(meta)
            finally:
                r.raise_for_status()

        j = r.json() if r.content else {}
        results = j.get("results") or []
        # Normalize to a flat list of dicts with keys similar to Yahoo-style expected by _normalize_candles_json:
        # Polygon uses: t (ms), o, h, l, c, v
        # We'll return {"results": [...]} as-is to let downstream normalizer handle.
        out = {"results": results, "ticker": ticker, "source": "polygon"}
        _DEBUG_LAST.update({**meta, "ok": True, "count": len(results)})
        return out, raw_bytes
    except Exception as e:
        meta.update({"ok": False, "error": str(e)})
        _DEBUG_LAST.update(meta)
        raise
