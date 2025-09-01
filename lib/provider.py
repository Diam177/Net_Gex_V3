# lib/provider.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 20

def _normalize_candles_payload(raw_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Нормализация форматов ответа по историческим свечам (intraday).
    Возвращает {"records": List[dict], "root": <path-string>} без модификации значений.
    Поддерживает:
      - {"meta": {...}, "body": [ {timestamp/timestamp_unix, open, high, low, close, volume}, ... ]}
      - {"data": {..., "items": [...]}} / {"result": {...}} / {"candles": [...]}
    """
    # 1) Прямой формат: meta/body
    body = raw_json.get("body")
    if isinstance(body, list) and body and isinstance(body[0], dict):
        return {"records": body, "root": "body"}

    # 2) data.items
    data = raw_json.get("data")
    if isinstance(data, dict):
        items = data.get("items") or data.get("body") or data.get("candles")
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return {"records": items, "root": "data.items|body|candles"}

    # 3) result.candles
    result = raw_json.get("result")
    if isinstance(result, dict):
        items = result.get("candles") or result.get("items") or result.get("body")
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return {"records": items, "root": "result.candles|items|body"}

    # 4) candles на верхнем уровне
    items = raw_json.get("candles")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return {"records": items, "root": "candles"}

    # Fallback: если пришёл массив верхнего уровня
    if isinstance(raw_json, list):
        return {"records": raw_json, "root": "<list>"}

    return {"records": [], "root": "locate_failed"}


def fetch_stock_history(ticker: str,
                        host: str,
                        api_key: str,
                        interval: str = "1m",
                        limit: int = 640,
                        dividend: Optional[bool] = None,
                        timeout: int = DEFAULT_TIMEOUT) -> Tuple[Dict[str, Any], bytes]:
    """
    Получить исторические свечи у провайдера RapidAPI YH Finance:
    GET https://{host}/api/v2/markets/stock/history?symbol=...&interval=...&limit=...
    Возвращаем (json_dict, raw_bytes) без преобразований.
    """
    base_url = f"https://{host}"
    url = f"{base_url}/api/v2/markets/stock/history"
    params = {"symbol": ticker, "interval": interval, "limit": int(limit)}
    if dividend is not None:
        params["dividend"] = "true" if dividend else "false"

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }

    meta = {
        "when": datetime.datetime.utcnow().isoformat() + "Z",
        "endpoint": url,
        "params": params.copy(),
        "headers": {"X-RapidAPI-Host": host},
        "ok": False,
        "error": None,
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        raw_bytes = r.content
        r.raise_for_status()
        data = r.json()
        meta["ok"] = True

        # Для отладки сохраним короткую сводку
        payload = _normalize_candles_payload(data)
        meta["records"] = len(payload.get("records", []))
        meta["root_path"] = payload.get("root", "?")
        global _LAST_DEBUG_META
        _LAST_DEBUG_META = meta

        return data, raw_bytes
    except Exception as e:
        meta["error"] = str(e)
        global _LAST_DEBUG_META
        _LAST_DEBUG_META = meta
        raise

# Глобальная «память» для отладочной панели в UI
_LAST_DEBUG_META: Dict[str, Any] = {}


def debug_meta() -> Dict[str, Any]:
    """Вернуть отладочную мета-информацию о последнем запросе к провайдеру."""
    return _LAST_DEBUG_META.copy()


# ---------- утилиты http/json ----------

def _request_json(url: str, headers: dict, params: dict, timeout: int) -> Tuple[dict, bytes]:
    """GET → JSON (+ исходные байты). Ошибку сервера отдаём компактно."""
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {url} {params} -> {r.text[:200]}")
    try:
        return r.json(), r.content
    except Exception:
        # на всякий случай раскодируем вручную (не трогаем исходные байты)
        return json.loads(r.content.decode("utf-8", errors="ignore")), r.content


def _locate_root_mutable(chain_json: dict) -> Tuple[dict, str]:
    """
    Находим «корень» с полями вида quote / expirationDates / options[..].
    Поддерживаем несколько реальных форматов провайдера.
    """
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

    # 5) запасной — сам корень, если это dict
    if isinstance(chain_json, dict):
        return chain_json, "<root>"

    raise RuntimeError("Cannot locate root in chain JSON")


def _price_in_quote(root: dict) -> Tuple[Optional[float], Optional[int], str]:
    """
    Проверяем, есть ли числовая цена в root['quote'] или root['underlying'].
    Ничего не скачиваем дополнительно и не придумываем.
    """
    def _extract(q: dict) -> Tuple[Optional[float], Optional[int]]:
        price = None
        for k in ("regularMarketPrice", "postMarketPrice", "last", "lastPrice",
                  "price", "close", "regularMarketPreviousClose"):
            v = q.get(k)
            if v is None:
                continue
            try:
                price = float(v)
                break
            except Exception:
                continue
        t0 = None
        for k in ("regularMarketTime", "postMarketTime", "time", "timestamp", "lastTradeDate"):
            v = q.get(k)
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

    q = root.get("quote", {})
    if isinstance(q, dict):
        p, t = _extract(q)
        if p is not None:
            return p, t, "quote"

    u = root.get("underlying", {})
    if isinstance(u, dict):
        p, t = _extract(u)
        if p is not None:
            return p, t, "underlying"

    return None, None, "not_found"


# ---------- основной вызов провайдера ----------

def fetch_option_chain(
    ticker: str,
    host: str,
    api_key: str,
    expiry_unix: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[dict, bytes]:
    """
    Запрашиваем опционную цепочку у yahoo-finance15 (через RapidAPI-host).
    Порядок попыток подобран так, чтобы избежать ошибки «The selected display is invalid.»:
      1) v1/markets/options — без параметра display
      2) v1/markets/options — display=straddle
      3) v1/markets/options — display=chain (как самый последний вариант)
      4) набор yahoo-fallback’ов (path / query_symbol)

    Никаких дополнительных запросов за ценой не делаем — мы её не "выдумываем".
    Возвращаем (json_dict, raw_bytes). Исходные байты не модифицируем.
    """
    global _LAST_DEBUG_META
    base_url = f"https://{host}"

    # Список попыток
    candidates: List[Dict[str, Any]] = [
        {"url": f"{base_url}/api/v1/markets/options", "mode": "v1_nodisplay"},
        {"url": f"{base_url}/api/v1/markets/options", "mode": "v1_display_straddle"},
        {"url": f"{base_url}/api/v1/markets/options", "mode": "v1_display_chain"},
        # fallback’и
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

    meta: Dict[str, Any] = {
        "ticker": ticker,
        "host": host,
        "expiry_unix": int(expiry_unix) if expiry_unix is not None else None,
        "attempts": [],
        "used": None,
        "root_path": None,
        "price_probe": None,
    }

    errors: List[str] = []
    raw_bytes: bytes = b""

    for c in candidates:
        params: Dict[str, Any] = {}
        mode = c["mode"]

        if mode.startswith("v1_"):
            params["ticker"] = ticker
            if expiry_unix is not None:
                params["expiration"] = int(expiry_unix)
            if mode == "v1_display_straddle":
                params["display"] = "straddle"
            elif mode == "v1_display_chain":
                params["display"] = "chain"
            # v1_nodisplay — без display
        elif mode == "query_symbol":
            params["symbol"] = ticker
            if expiry_unix is not None:
                params["date"] = int(expiry_unix)
        else:  # mode == "path"
            if expiry_unix is not None:
                params["date"] = int(expiry_unix)

        try:
            data, content = _request_json(c["url"], headers, params, timeout)
            raw_bytes = content
            meta["attempts"].append({"url": c["url"], "params": params, "ok": True})
            meta["used"] = {"url": c["url"], "params": params}

            # Найдём корень и проверим наличие цены (но ничего не дополняем снаружи)
            try:
                root, where = _locate_root_mutable(data)
                meta["root_path"] = where
                p, t, src = _price_in_quote(root)
                meta["price_probe"] = {
                    "price": p,
                    "t0": t,
                    "source": src,
                }
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
