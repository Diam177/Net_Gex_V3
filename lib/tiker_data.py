
# -*- coding: utf-8 -*-
"""
tiker_data.py — автономный блок выбора тикера и экспирации.
Жёсткое требование: НЕ зависеть от внутренних модулей проекта, кроме lib/provider_polygon.py.
Вся логика извлечения spot/expirations и сборки сырых записей — внутри этого файла.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import datetime as _dt
import os

# единственная внешняя зависимость — адаптер Polygon
from lib import provider_polygon as _prov  # type: ignore
import inspect


def _safe_fetch_option_chain(ticker: str) -> dict:
    """
    Вызывает provider_polygon.fetch_option_chain, независимо от того,
    как у него названы параметры (ticker/symbol) и принимает ли он позиционный аргумент.
    """
    func = getattr(_prov, "fetch_option_chain", None)
    if func is None:
        raise RuntimeError("provider_polygon.fetch_option_chain не найден")

    try:
        sig = inspect.signature(func)
    except Exception:
        sig = None

    # Попробуем по имени параметров
    if sig is not None:
        params = list(sig.parameters.keys())
        try:
            if "ticker" in params:
                return func(ticker=ticker)
            if "symbol" in params:
                return func(symbol=ticker)
        except TypeError:
            pass  # попробуем позиционно

    # Позиционный вызов на всякий случай
    try:
        return func(ticker)
    except TypeError:
        # Последний шанс: без аргументов (если провайдер сам читает st.session_state)
        return func()


# ---------------------------
# Вспомогательные локальные утилиты
# ---------------------------

def _parse_chain_extract_spot_and_exps(chain: dict) -> Tuple[Optional[float], List[str]]:
    """
    Достаёт spot и список экспираций прямо из chain['records'].
    Ожидается формат, совместимый с provider_polygon.fetch_option_chain.
    """
    if not isinstance(chain, dict):
        return None, []
    recs = chain.get("records") or []

    # spot: пробуем в заголовке, затем в первом контракте
    spot = None
    for key in ("spot", "underlyingPrice", "S"):
        try:
            if key in chain and chain[key] is not None:
                spot = float(chain[key])
                break
        except Exception:
            pass
    if spot is None:
        try:
            if recs and recs[0].get("S") is not None:
                spot = float(recs[0]["S"])
        except Exception:
            spot = None

    # expirations: из самих контрактов
    exps = sorted({r.get("expiration") for r in recs if r.get("expiration")})
    return spot, exps


def _choose_default_expiration(exps: List[str]) -> Optional[str]:
    """
    Ближайшая дата >= сегодня. Если таких нет — последняя доступная.
    """
    if not exps:
        return None
    today = _dt.date.today()
    parsed = []
    for e in exps:
        try:
            parsed.append((_dt.date.fromisoformat(e), e))
        except Exception:
            continue
    if not parsed:
        return None
    parsed.sort()
    for d, raw in parsed:
        if d >= today:
            return raw
    return parsed[-1][1]  # самая дальняя


def _split_raw_by_exp(chain: dict, selected_exps: List[str]) -> Dict[str, List[dict]]:
    """
    Группирует контракты по выбранным датам экспирации.
    """
    ans: Dict[str, List[dict]] = {e: [] for e in (selected_exps or [])}
    recs = (chain or {}).get("records") or []
    sels = set(selected_exps or [])
    for r in recs:
        e = r.get("expiration")
        if e in sels:
            ans[e].append(r)
    return ans


def _fetch_chain_polygon(ticker: str) -> dict:
    """
    Единственная точка загрузки опционной цепочки из Polygon.
    """
    api_key = os.environ.get('POLYGON_API_KEY') or ''
    return _prov.fetch_option_chain(ticker=ticker, host_unused=None, api_key=api_key, expiry_unix=None)


def _fetch_ohlc_polygon(ticker: str, interval: str = "1m", limit: int = 500) -> dict:
    """
    История свечей для Key Levels. Возвращает payload адаптера Polygon.
    """
    api_key = os.environ.get('POLYGON_API_KEY') or ''
    return _prov.fetch_stock_history(ticker=ticker, host_unused=None, api_key=api_key, interval=interval, limit=limit, dividend=False, timeout=30)


# ---------------------------
# Публичный API
# ---------------------------

@dataclass
class TikerRawResult:
    ticker: str
    spot: Optional[float]
    expirations: List[str]
    selected: List[str]
    raw_by_exp: Dict[str, List[dict]]
    ohlc: Optional[dict] = None  # сырой payload от провайдера
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    ohlc_interval: str = "1m"
    ohlc_limit: int = 500


def get_raw_by_exp(
    ticker: str,
    selected_exps: Optional[List[str]] = None,
    need_ohlc: bool = False,
    ohlc_interval: str = "1m",
    ohlc_limit: int = 500,
) -> TikerRawResult:
    """
    Тянет цепочку по тикеру, восстанавливает список экспираций из цепочки,
    выбирает дефолт, формирует raw_by_exp по выбранным датам.
    НЕТ зависимостей на другие модули (кроме provider_polygon).
    """
    chain = _fetch_chain_polygon(ticker)
    spot, exps = _parse_chain_extract_spot_and_exps(chain)

    if not exps:
        return TikerRawResult(
            ticker=ticker, spot=spot, expirations=[], selected=[],
            raw_by_exp={}, ohlc=None, day_high=None, day_low=None,
            ohlc_interval=ohlc_interval, ohlc_limit=ohlc_limit
        )

    if not selected_exps:
        default = _choose_default_expiration(exps)
        selected_exps = [default] if default else []

    raw_by_exp = _split_raw_by_exp(chain, selected_exps or [])

    ohlc_payload: Optional[dict] = None
    day_hi = day_lo = None
    if need_ohlc:
        try:
            ohlc_payload = _fetch_ohlc_polygon(ticker, interval=ohlc_interval, limit=int(ohlc_limit))
            # Попробуем вычислить High/Low без pandas
            bars = (ohlc_payload or {}).get("results") or (ohlc_payload or {}).get("bars") or []
            highs = []
            lows = []
            for b in bars:
                h = b.get("high", b.get("h"))
                l = b.get("low",  b.get("l"))
                if h is not None:
                    try: highs.append(float(h))
                    except Exception: pass
                if l is not None:
                    try: lows.append(float(l))
                    except Exception: pass
            if highs and lows:
                day_hi = max(highs)
                day_lo = min(lows)
        except Exception:
            pass

    return TikerRawResult(
        ticker=ticker,
        spot=spot,
        expirations=exps,
        selected=selected_exps or [],
        raw_by_exp=raw_by_exp,
        ohlc=ohlc_payload,
        day_high=day_hi,
        day_low=day_lo,
        ohlc_interval=ohlc_interval,
        ohlc_limit=ohlc_limit,
    )


def render_tiker_data_block(st) -> None:
    """
    Рисует блок: Ticker + Expiration (одна) и кладёт выбор в session_state.
    Не возвращает объект st и ничего не печатает, чтобы не было артефактов в UI.
    """
    st.markdown("**Ticker**")
    ticker_default = st.session_state.get("_last_ticker", "SPY")
    ticker = st.text_input("Ticker", value=ticker_default, key="td2_ticker_input")
    st.session_state["td2_ticker"] = ticker

    # Один запрос цепочки для построения списка экспираций
    base = get_raw_by_exp(ticker, need_ohlc=True, ohlc_interval="1m", ohlc_limit=500)

    if not base.expirations:
        st.warning("Не удалось получить список экспираций из цепочки (provider_polygon).")
        return

    # Дефолтный выбор
    default = _choose_default_expiration(base.expirations)
    try:
        default_idx = base.expirations.index(default) if default in base.expirations else 0
    except Exception:
        default_idx = 0

    exp = st.selectbox("Экспирация", options=base.expirations, index=default_idx, key="td2_exp_select")

    # Обновим выбор уже целевым вызовом (raw_by_exp только по exp)
    final = get_raw_by_exp(ticker, selected_exps=[exp], need_ohlc=True, ohlc_interval=base.ohlc_interval, ohlc_limit=base.ohlc_limit)

    # Кладём в session_state — финальная таблица и другие блоки смогут работать без доп. импортов
    st.session_state["raw_records"] = final.raw_by_exp.get(exp, [])
    st.session_state["spot"] = final.spot
    st.session_state["_last_ticker"] = ticker
    st.session_state["_last_exp_sig"] = exp

    # «Замок» для финальной таблицы (чтобы она не рисовала дублирующий селектор)
    st.session_state["tiker_selected_exps"] = [exp]
    st.session_state["exp_locked_by_tiker_data"] = True

    # Небольшая справка
    with st.expander("Инфо (данные для таблицы и Key Levels)", expanded=False):
        st.caption(f"{ticker} @ spot={final.spot or '—'}; выбрана экспирация: {exp}")
        if final.day_high is not None and final.day_low is not None:
            st.caption(f"Свечи: интервал={final.ohlc_interval}, баров≈{final.ohlc_limit}, High={final.day_high}, Low={final.day_low}")
        st.caption(f"raw_records[{exp}]: {len(st.session_state.get('raw_records', []))} контрактов")

    # Ничего не возвращаем
    return None
