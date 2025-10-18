from __future__ import annotations
import io
import json
from datetime import date, datetime
from typing import Dict, Iterable, List, Tuple

import requests


POLYGON_BASE = "https://api.polygon.io"
HEADERS_TEMPLATE = {"Authorization": "Bearer {api_key}"}


class PolygonError(RuntimeError):
    pass


def _headers(api_key: str) -> Dict[str, str]:
    if not api_key or not isinstance(api_key, str):
        raise PolygonError("POLYGON_API_KEY отсутствует или некорректен")
    h = HEADERS_TEMPLATE.copy()
    h["Authorization"] = h["Authorization"].format(api_key=api_key)
    return h


def _get_with_cursor(url: str, headers: Dict[str, str], timeout: int, max_pages: int) -> Iterable[dict]:
    """
    Универсальный пагинатор по cursor/next_url (Polygon v3).
    Возвращает генератор страниц (dict).
    """
    pages = 0
    next_url = url
    sess = requests.Session()
    while next_url and pages < max_pages:
        resp = sess.get(next_url, headers=headers, timeout=timeout)
        if resp.status_code == 429:
            raise PolygonError("Polygon вернул 429 (rate limit). Попробуйте позже или уменьшите частоту запросов.")
        if not resp.ok:
            raise PolygonError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        yield data
        next_url = data.get("next_url")
        pages += 1


def list_future_expirations(ticker: str, api_key: str, *, max_pages: int = 8, timeout: int = 20) -> List[str]:
    """
    Возвращает отсортированный список будущих дат экспираций (YYYY-MM-DD) для базового актива.
    Источник: /v3/reference/options/contracts?underlying_ticker=...
    """
    t = (ticker or "").strip().upper()
    if not t:
        raise ValueError("ticker не задан")
    url = (
        f"{POLYGON_BASE}/v3/reference/options/contracts"
        f"?underlying_ticker={t}"
        f"&expired=false"
        f"&limit=1000"
        f"&order=asc"
        f"&sort=expiration_date"
    )
    headers = _headers(api_key)
    today = date.today()
    uniq = set()

    for page in _get_with_cursor(url, headers, timeout, max_pages):
        results = page.get("results") or []
        for r in results:
            d = (r.get("expiration_date") or "").strip()
            if not d:
                continue
            # берём только даты в формате YYYY-MM-DD и не в прошлом
            try:
                dt = datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            if dt >= today:
                uniq.add(d)

    out = sorted(uniq)
    return out


def download_snapshot_json(ticker: str, expiration_date: str, api_key: str, *, timeout: int = 30, max_pages: int = 40) -> dict:
    """
    Возвращает объединённый JSON-снэпшот всех опционов по данной дате экспирации.
    Источник: /v3/snapshot/options/{UNDERLYING}?expiration_date=YYYY-MM-DD
    Обходит страницы cursor до max_pages.
    """
    t_raw = (ticker or '').strip()
    t = t_raw.upper()
    # Normalize common indices to Polygon index tickers
    _IDX = {'SPX','NDX','VIX','RUT','DJX'}
    if t in _IDX and not t_raw.startswith('I:'):
        t = f'I:{t}'
    if not t:
        raise ValueError("ticker не задан")
    if not expiration_date:
        raise ValueError("expiration_date не задана")

    base = f"{POLYGON_BASE}/v3/snapshot/options/{t}?expiration_date={expiration_date}&limit=250"
    headers = _headers(api_key)

    all_results: List[dict] = []
    # Первая попытка — без limit (чтобы избежать известной ошибки 'Limit ... max')
    for page in _get_with_cursor(base, headers, timeout, 10000):
        results = page.get("results") or []
        all_results.extend(results)

    return {
        "ticker": t,
        "expiration_date": expiration_date,
        "results_count": len(all_results),
        "results": all_results,
    }


def snapshots_zip_bytes(ticker: str, dates: Iterable[str], api_key: str, *, timeout: int = 30, max_pages: int = 40) -> Tuple[bytes, str]:
    """
    Для нескольких дат экспираций — собирает JSON по каждой и упаковывает в ZIP (в памяти).
    Возвращает (zip_bytes, filename).
    """
    t = (ticker or "").strip().upper()
    if not t:
        raise ValueError("ticker не задан")

    # сохраним список дат для имени и итерации
    dates_list = list(dates)

    buf = io.BytesIO()
    import zipfile

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d in dates_list:
            js = download_snapshot_json(t, d, api_key, timeout=timeout, max_pages=max_pages)
            zf.writestr(f"{t}_{d}.json", json.dumps(js, ensure_ascii=False))

    buf.seek(0)
    return buf.read(), f"{t}_snapshots_{len(dates_list)}.zip"

# --- BEGIN: spot price helper (safe addition) ---------------------------------
from datetime import datetime, timedelta, timezone

def _poly_get_json(url: str, api_key: str, params: dict | None = None, timeout: float = 10.0) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code == 429:
        raise PolygonError(f"Rate limit (429) while GET {url}")
    if not r.ok:
        # NOT_AUTHORIZED и прочие ошибки пробрасываем вверх — вызывающий сам решит фолбэк
        raise PolygonError(f"Polygon GET {url} -> HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json() or {}
    except Exception:
        return {}

def get_spot_price(ticker: str, api_key: str, *_, **__) -> tuple[float, int, str]:
    """snapshot-only wrapper for backward compatibility"""
    s = get_spot_snapshot(ticker, api_key)
    return s, 0, "snapshot"
# --- END: spot price helper ---------------------------------------------------


# --- Snapshot-based spot (no fallbacks) --------------------------------------
def get_spot_snapshot(ticker: str, api_key: str, *, timeout: float = 15.0) -> float:
    """
    Единый источник S из Polygon Snapshot.
    - Индексы: /v3/snapshot/indices?ticker=I:SPX -> results[0].value
    - Акции/ETF: /v2/snapshot/locale/us/markets/stocks/tickers/SPY -> ticker.min.c
    """
    if not ticker:
        raise PolygonError("get_spot_snapshot: empty ticker")
    t = ticker.strip().upper()
    import requests
    if t.startswith("I:"):
        url = f"{POLYGON_BASE}/v3/snapshot/indices"
        params = {"ticker": t, "apiKey": api_key}
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        results = js.get("results") or []
        if not results:
            raise PolygonError(f"get_spot_snapshot: no results for {t}")
        val = results[0].get("value")
        if val is None:
            raise PolygonError(f"get_spot_snapshot: missing value for {t}")
        return float(val)
    # stocks/ETF
    url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{t}"
    params = {"apiKey": api_key}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    try:
        return float(js["ticker"]["min"]["c"])
    except Exception as e:
        raise PolygonError(f"get_spot_snapshot: missing min.c for {t}") from e
# --- END: snapshot spot helper ------------------------------------------------
