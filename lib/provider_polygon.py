# -*- coding: utf-8 -*-
"""
Polygon provider adapter → "Yahoo-style" option chain format for this app.
Мы не меняем числовые значения, а только приводим структуру к формату,
который ожидает compute.extract_core_from_chain().

Главная функция: fetch_option_chain(ticker, host, key, expiry_unix=None) -> (json_like, raw_bytes)
- ticker: базовый актив, напр. "SPY"
- host: не используется (для совместимости с существующим кодом)
- key: POLYGON_API_KEY
- expiry_unix: если задан, то фильтруем по expiration_date (UTC, YYYY-MM-DD)
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import datetime as _dt
import time as _time
import requests

POLYGON_BASE_URL = "https://api.polygon.io"
DEFAULT_TIMEOUT = 25

def _to_iso_date_from_unix(ts_unix: int) -> str:
    return _dt.datetime.utcfromtimestamp(int(ts_unix)).strftime("%Y-%m-%d")

def _to_unix_from_iso(s: str) -> int:
    # Polygon даёт expiration_date в формате YYYY-MM-DD (UTC)
    dt = _dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    return int(dt.timestamp())

def _append_api_key(url: str, api_key: str) -> str:
    return (url + ("&" if ("?" in url) else "?") + f"apiKey={api_key}")

def _get_underlying_price(ticker: str, api_key: str) -> Tuple[Optional[float], Optional[int], str]:
    """
    Пытаемся аккуратно получить цену базового актива S и timestamp t0.
    Делаем несколько попыток (с наименьшими правками кода):
      1) /v2/last/trade/{ticker}
      2) /v2/snapshot/locale/us/markets/stocks/tickers/{ticker}
      3) /v2/aggs/ticker/{ticker}/prev (yesterday close)
    Возвращаем: (S, ts_unix, source_tag)
    """
    # 1) last trade
    try:
        u = f"{POLYGON_BASE_URL}/v2/last/trade/{ticker}"
        resp = requests.get(u, params={"apiKey": api_key}, timeout=DEFAULT_TIMEOUT)
        if resp.ok:
            j = resp.json()
            # Новые ответы: {"results":{"p": 123.45, "t": 169...}, "status":"success"}
            res = j.get("results") or {}
            p = res.get("p") or res.get("price")
            t = res.get("t") or res.get("timestamp")
            if p is not None:
                ts_unix = int(int(t)/1_000_000_000) if isinstance(t, int) and t>1e12 else (int(t) if t is not None else int(_time.time()))
                return float(p), ts_unix, "last_trade"
    except Exception:
        pass

    # 2) stocks snapshot
    try:
        u = f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        resp = requests.get(u, params={"apiKey": api_key}, timeout=DEFAULT_TIMEOUT)
        if resp.ok:
            j = resp.json()
            t = j.get("ticker") or j.get("results") or {}
            # Пытаемся взять последнюю цену / время
            last = (t.get("lastTrade") or t.get("last_quote") or t.get("last")) or {}
            p = last.get("p") or last.get("price") or last.get("P")
            ts = last.get("t") or last.get("timestamp")
            if p is None:
                # иногда дают close в day
                day = t.get("day") or {}
                p = day.get("c") or day.get("close")
                ts = day.get("t") or day.get("timestamp") or ts
            if p is not None:
                ts_unix = int(int(ts)/1_000_000_000) if isinstance(ts, int) and ts>1e12 else (int(ts) if ts is not None else int(_time.time()))
                return float(p), ts_unix, "stock_snapshot"
    except Exception:
        pass

    # 3) previous close (agg)
    try:
        u = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        resp = requests.get(u, params={"apiKey": api_key, "adjusted": "true"}, timeout=DEFAULT_TIMEOUT)
        if resp.ok:
            j = resp.json()
            results = j.get("results") or []
            if results:
                r0 = results[0]
                p = r0.get("c") or r0.get("close")
                ts = r0.get("t")
                ts_unix = int(int(ts)/1_000) if isinstance(ts, int) and ts>1e12 else (int(ts) if ts is not None else int(_time.time()))
                if p is not None:
                    return float(p), ts_unix, "prev_close"
    except Exception:
        pass

    return None, None, "none"

def _list_snapshot_options_chain(ticker: str, api_key: str, expiration_date: Optional[str]) -> Tuple[List[dict], dict]:
    """
    Стягивает все страницы /v3/snapshot/options/{ticker} c пагинацией.
    Возвращает: (items, debug_meta)
    """
    items: List[dict] = []
    debug = {"attempt": "v3.snapshot.options", "pages": []}

    params = {
        "order": "asc",
        "limit": 250,      # максимум у Polygon
        "sort": "ticker",
    }
    if expiration_date:
        params["expiration_date"] = expiration_date

    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{ticker}"
    while True:
        resp = requests.get(url, params={**params, "apiKey": api_key}, timeout=DEFAULT_TIMEOUT)
        page_meta = {"url": resp.url, "ok": resp.ok, "status_code": resp.status_code}
        debug["pages"].append(page_meta)

        if not resp.ok:
            raise RuntimeError(f"Polygon snapshot request failed: {resp.status_code} {resp.text[:200]}")

        j = resp.json()
        page_items = j.get("results") or []
        if not isinstance(page_items, list):
            break
        items.extend(page_items)

        next_url = j.get("next_url")
        if not next_url:
            break
        # В next_url обычно нет apiKey
        url = _append_api_key(next_url, api_key)
        params = {}  # дальнейшие параметры уже "встроены" в next_url

    return items, debug

def _remap_to_yahoo_like(items: List[dict],
                         S: Optional[float],
                         ts_unix: Optional[int]) -> Dict[str, Any]:
    """
    Преобразуем список конрактов Polygon в структуру "как у Yahoo":
    {
      "optionChain": {
        "result": [{
            "quote": {...},
            "expirationDates": [...],
            "options": [{
               "expirationDate": "YYYY-MM-DD",
               "calls": [...],
               "puts":  [...]
            }, ...]
        }],
        "error": null
      }
    }
    """
    # Группируем по дате экспирации
    by_exp: Dict[str, Dict[str, Any]] = {}
    expirations_unix: List[int] = []

    def _push(exp: str, kind: str, rec: Dict[str, Any]):
        blk = by_exp.setdefault(exp, {"expirationDate": exp, "calls": [], "puts": []})
        blk["calls" if kind == "call" else "puts"].append(rec)

    for it in items:
        details = it.get("details") or {}
        day     = it.get("day") or {}
        ctype   = (details.get("contract_type") or "").lower()
        if ctype not in ("call", "put"):
            continue
        strike = details.get("strike_price")
        exp    = details.get("expiration_date")  # "YYYY-MM-DD"
        if strike is None or not exp:
            continue

        # Веса/объёмы/IV
        oi  = it.get("open_interest")
        vol = day.get("volume")
        iv  = it.get("implied_volatility")

        # price/last не обязателен для нашего расчёта, добавим если есть
        last_trade = it.get("last_trade") or {}
        last_price = last_trade.get("price") or last_trade.get("p")

        rec = {
            "contractSymbol": details.get("ticker"),
            "strike": float(strike),
            "openInterest": int(oi or 0),
            "volume": int(vol or 0),
            "impliedVolatility": float(iv) if (iv is not None) else None,
            "lastPrice": float(last_price) if (last_price is not None) else None,
            # Yahoo-подобное поле (строка ок)
            "expiration": exp,
        }
        _push(exp, ctype, rec)

    # Список экспираций: в юникс-секундах (так ожидает наш UI)
    expirations_sorted = sorted(by_exp.keys())
    for e in expirations_sorted:
        try:
            expirations_unix.append(_to_unix_from_iso(e))
        except Exception:
            # если вдруг формат не как YYYY-MM-DD, пропустим
            pass

    # quote + время
    now_unix = int(_time.time())
    q_time = int(ts_unix or now_unix)
    q_price = float(S) if (S is not None) else None

    chain_obj = {
        "quote": {
            "regularMarketPrice": q_price,
            "regularMarketDayHigh": None,
            "regularMarketDayLow": None,
            "regularMarketTime": q_time,
        },
        "expirationDates": expirations_unix,
        "options": [by_exp[e] for e in expirations_sorted],
    }
    return {"optionChain": {"result": [chain_obj], "error": None}}

def fetch_option_chain(ticker: str,
                       host: Optional[str],
                       key: str,
                       expiry_unix: Optional[int] = None) -> Tuple[Dict[str, Any], bytes]:
    """
    Главная точка входа для приложения.
    1) тянем option chain snapshot (возможно с фильтром по expiration_date)
    2) тянем цену базового актива S (несколько стратегий)
    3) собираем Yahoo-подобный JSON
    4) raw_bytes — это "сырые" данные провайдера Polygon: полный список items в исходном виде
    """
    if not key:
        raise RuntimeError("POLYGON_API_KEY is missing")

    expiration_date = None
    if expiry_unix:
        expiration_date = _to_iso_date_from_unix(int(expiry_unix))

    # 1) вся цепочка (с пагинацией)
    items, debug_meta = _list_snapshot_options_chain(ticker, key, expiration_date)

    # 2) цена подложки
    S, ts_unix, price_source = _get_underlying_price(ticker, key)

    # 3) нормализуем в "Yahoo-формат"
    out_json = _remap_to_yahoo_like(items, S, ts_unix)

    # 4) сырые байты провайдера для кнопки скачивания
    #    НЕ режем содержимое, чтобы пользователь получил оригинальные данные
    try:
        raw_dump = {
            "endpoint": "v3/snapshot/options",
            "ticker": ticker,
            "expiration_date": expiration_date,
            "price_source": price_source,
            "items": items,
            "debug": debug_meta,
        }
        raw_bytes = json.dumps(raw_dump, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    except Exception as e:
        raw_bytes = f"[polygon_raw_build_error] {e}".encode("utf-8")

    return out_json, raw_bytes
