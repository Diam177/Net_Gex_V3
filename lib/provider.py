# lib/provider.py
from __future__ import annotations

import json
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 20

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


# ---------- OHLCV: последняя полностью закрытая сессия ----------
def fetch_previous_session_ohlcv(symbol: str, host: str, api_key: str,
                                 interval: str = "1m", range_: str = "5d",
                                 timeout: int = DEFAULT_TIMEOUT,
                                 regular_hours_only: bool = True):
    """
    Возвращает минутные свечи за последнюю полностью закрытую сессию.
    Колонки: ['datetime','open','high','low','close','volume'] (tz=America/New_York).
    Никаких домыслов — только пришедшие данные; фильтруем по датам.
    """
    base_url = f"https://{host}"
    url = f"{base_url}/api/v2/stock/history"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Accept": "application/json",
    }
    params = {"symbol": symbol, "interval": interval, "range": range_}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {url} {params} -> {r.text[:200]}")
    js = r.json()
    items = js.get("body", [])
    if not items:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])

    df = pd.DataFrame(items)
    # защищённо берем нужные столбцы
    needed_cols = ["timestamp_unix","open","high","low","close","volume"]
    for c in needed_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in stock/history response")


    # Конвертация времени → America/New_York
    dt = pd.to_datetime(df["timestamp_unix"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df = df.assign(datetime=dt).sort_values("datetime")
    df["date"] = df["datetime"].dt.date

    today_ny = pd.Timestamp.now(tz="America/New_York").date()
    past_dates = [d for d in df["date"].unique() if d != today_ny]
    if not past_dates:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    last_full = max(past_dates)

    day_df = df[df["date"] == last_full].copy()
    if regular_hours_only:
        # Срез только регулярных торгов 09:30–16:00 NY
        t_open = pd.to_datetime("09:30").time()
        t_close = pd.to_datetime("16:00").time()
        day_df = day_df[(day_df["datetime"].dt.time >= t_open) & (day_df["datetime"].dt.time <= t_close)]

    day_df = day_df.drop(columns=["date", "timestamp_unix"]).reset_index(drop=True)
    return day_df[["datetime","open","high","low","close","volume"]]


# ===== Added for intraday previous session fetch (fallback v2→v1) =====
import requests, pandas as pd

def _parse_history_json(js):
    items = js.get("body", [])
    if not isinstance(items, list):
        # иногда приходит под ключом "items"
        items = js.get("items", [])
    return pd.DataFrame(items)

def fetch_previous_session_ohlcv(symbol: str, host: str, key: str,
                                 interval: str = "1m", range_: str = "5d",
                                 timeout: int = 15, regular_hours_only: bool = True) -> pd.DataFrame:
    """
    Возвращает минутные свечи за последнюю полностью закрытую сессию.
    Колонки: ['datetime','open','high','low','close','volume'] (tz=America/New_York).
    Никаких домыслов: берём только то, что пришло от провайдера.
    """
    headers = {"x-rapidapi-host": host, "x-rapidapi-key": key}
    params_symbol = {"symbol": symbol, "interval": interval, "range": range_}
    params_ticker = {"ticker": symbol, "interval": interval, "range": range_}

    def _try(url, params):
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 404 or "does not exist" in r.text:
            raise FileNotFoundError("endpoint")
        r.raise_for_status()
        return _parse_history_json(r.json())

    df_raw = None
    # 1) v2 + symbol
    try:
        df_raw = _try("https://{host}/api/v2/stock/history", params_symbol)
    except FileNotFoundError:
        pass
    # 2) v1 + symbol
    if df_raw is None or df_raw.empty:
        try:
            df_raw = _try("https://{host}/api/v1/stock/history", params_symbol)
        except FileNotFoundError:
            pass
    # 3) v1 + ticker (на всякий случай для старых бэкендов)
    if df_raw is None or df_raw.empty:
        df_raw = _try("https://{host}/api/v1/stock/history", params_ticker)

    # Ожидаемые поля: timestamp_unix, open, high, low, close, volume
    cols = {"timestamp_unix": "timestamp_unix", "timestamp": "timestamp_unix"}
    if "timestamp" in df_raw.columns and "timestamp_unix" not in df_raw.columns:
        df_raw = df_raw.rename(columns=cols)

    required = ["timestamp_unix", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Unexpected payload, missing: {missing}")

    df_raw = df_raw[required].copy()
    dt = pd.to_datetime(df_raw["timestamp_unix"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df = (df_raw.assign(datetime=dt)
                 .sort_values("datetime")
                 .reset_index(drop=True))

    # выделяем последнюю полностью закрытую дату в NY
    df["date"] = df["datetime"].dt.date
    today_ny = pd.Timestamp.now(tz="America/New_York").date()
    past_dates = [d for d in df["date"].unique() if d != today_ny]
    if not past_dates:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    last_full = max(past_dates)
    day_df = df[df["date"] == last_full].copy()

    if regular_hours_only:
        day_df = day_df[(day_df["datetime"].dt.time >= pd.to_datetime("09:30").time()) &
                        (day_df["datetime"].dt.time <= pd.to_datetime("16:00").time())]

    return day_df[["datetime","open","high","low","close","volume"]].reset_index(drop=True)


import requests, pandas as pd

def _parse_history_json(js):
    # Accept multiple possible shapes
    for key in ("body", "items", "result", "results", "data"):
        if isinstance(js.get(key), list):
            return pd.DataFrame(js[key])
    # Sometimes the payload is a dict of arrays
    if isinstance(js, dict):
        # Try to coerce to DataFrame if keys look like columns
        try:
            return pd.DataFrame(js)
        except Exception:
            pass
    return pd.DataFrame()

def fetch_previous_session_ohlcv(symbol: str, host: str, key: str,
                                 interval: str = "1m", range_: str = "5d",
                                 timeout: int = 15, regular_hours_only: bool = True) -> pd.DataFrame:
    headers = {"x-rapidapi-host": host, "x-rapidapi-key": key}
    variants = [("https://{host}/api/v2/markets/stock/history", {"symbol": symbol, "interval": interval, "limit": 640}),
        ("https://{host}/api/v2/markets/stock/history", {"ticker": symbol, "interval": interval, "limit": 640}),
        
        ("https://{host}/api/v2/stock/history", {"symbol": symbol, "interval": interval, "range": range_}),
        ("https://{host}/api/v2/stock/history", {"ticker": symbol, "interval": interval, "range": range_}),
        ("https://{host}/api/v1/stock/history", {"symbol": symbol, "interval": interval, "range": range_}),
        ("https://{host}/api/v1/stock/history", {"ticker": symbol, "interval": interval, "range": range_}),
        ("https://{host}/api/v1/market/history", {"symbol": symbol, "interval": interval, "range": range_}),
        ("https://{host}/api/v1/market/history", {"ticker": symbol, "interval": interval, "range": range_}),
    ]

    last_err = None
    df_raw = pd.DataFrame()
    for url, params in variants:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 404 or "does not exist" in r.text:
                last_err = f"404 at {url} with params keys {list(params.keys())}"
                continue
            r.raise_for_status()
            df_raw = _parse_history_json(r.json())
            if not df_raw.empty:
                break
            last_err = f"Empty payload at {url}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e} at {url}"
            continue

    if df_raw.empty:
        raise RuntimeError(f"All history endpoints failed. Last error: {last_err}")

    # Normalize columns
    if "timestamp_unix" not in df_raw.columns:
        if "timestamp" in df_raw.columns and pd.api.types.is_integer_dtype(df_raw["timestamp"]):
            df_raw = df_raw.rename(columns={"timestamp": "timestamp_unix"})
        elif "date" in df_raw.columns and pd.api.types.is_integer_dtype(df_raw["date"]):
            df_raw = df_raw.rename(columns={"date": "timestamp_unix"})

    required = ["timestamp_unix", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Unexpected payload (missing columns: {missing}); head: {df_raw.head(3).to_dict()}")

    df_raw = df_raw[required].copy()
    dt = pd.to_datetime(df_raw["timestamp_unix"], unit="s", utc=True).dt.tz_convert("America/New_York")
    df = (df_raw.assign(datetime=dt)
                 .sort_values("datetime")
                 .reset_index(drop=True))

    # Select last fully closed NY session
    df["date"] = df["datetime"].dt.date
    today_ny = pd.Timestamp.now(tz="America/New_York").date()
    past_dates = [d for d in df["date"].unique() if d != today_ny]
    if not past_dates:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    last_full = max(past_dates)
    day_df = df[df["date"] == last_full].copy()

    if regular_hours_only:
        start_t = pd.to_datetime("09:30").time()
        end_t   = pd.to_datetime("16:00").time()
        day_df = day_df[(day_df["datetime"].dt.time >= start_t) & (day_df["datetime"].dt.time <= end_t)]

    return day_df[["datetime","open","high","low","close","volume"]].reset_index(drop=True)
