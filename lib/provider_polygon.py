# -*- coding: utf-8 -*-
"""
Polygon provider adapter → "Yahoo-like" JSON для приложения.
Использует только apiKey в query (никаких headers).
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import datetime as _dt
import time as _time
import json, requests

POLYGON_BASE_URL = "https://api.polygon.io"

# ---------- helpers ----------
def _append_api_key(url: str, api_key: str) -> str:
    return url + (("&" if "?" in url else "?") + f"apiKey={api_key}")

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

def _paginate(url: str, api_key: str, params: Optional[dict] = None, cap: int = 80)
