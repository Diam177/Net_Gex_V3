
import streamlit as st
import requests
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional

st.set_page_config(page_title="Ticker & Expiration Selector", layout="wide")

# --- Utils --------------------------------------------------------------------

POLY_BASE = "https://api.polygon.io"

def _poly_headers(api_key: str) -> Dict[str, str]:
    # Polygon v3 supports Bearer auth; we also append the query param as fallback for some proxies.
    return {"Authorization": f"Bearer {api_key}"}

def fetch_option_contracts_page(
    api_key: str,
    underlying: str,
    cursor: Optional[str] = None,
    limit: int = 1000,
    expired: bool = False,
) -> Dict[str, Any]:
    """Fetch a single page of option contracts for an underlying from Polygon v3."""
    url = f"{POLY_BASE}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": underlying.upper(),
        "limit": limit,
        "order": "asc",
        "sort": "expiration_date",
        "expired": str(expired).lower(),
    }
    if cursor:
        params["cursor"] = cursor
    resp = requests.get(url, headers=_poly_headers(api_key), params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

def fetch_expirations_from_contracts(api_key: str, underlying: str, max_pages: int = 5) -> List[str]:
    """
    Retrieve a de-duplicated, sorted list of expiration dates (YYYY-MM-DD) for the underlying.
    We gather from /v3/reference/options/contracts to avoid endpoint drift; paginates a few pages.
    """
    cursor = None
    expirations = set()
    pages = 0
    while pages < max_pages:
        data = fetch_option_contracts_page(api_key, underlying, cursor=cursor, expired=False)
        results = data.get("results") or []
        for item in results:
            exp = item.get("expiration_date")
            if exp:
                expirations.add(exp[:10])
        cursor = data.get("next_url") or data.get("next_cursor") or data.get("next")
        # Polygon returns 'next_url' in some SDKs and 'next_url'/'next'/'next_cursor' variants; stop if none
        if not cursor:
            break
        # normalize cursor string if it's a URL; Polygon accepts passing the 'cursor' token only
        if isinstance(cursor, str) and "cursor=" in cursor:
            # keep token after last 'cursor='
            cursor = cursor.split("cursor=")[-1]
        pages += 1
    # Keep only future/ today expirations
    today_str = date.today().isoformat()
    future = [e for e in expirations if e >= today_str]
    future.sort()
    return future

def choose_nearest(expirations: List[str]) -> Optional[str]:
    if not expirations:
        return None
    today = date.today().isoformat()
    for e in expirations:
        if e >= today:
            return e
    return expirations[0]  # fallback

# --- UI -----------------------------------------------------------------------
# Read API key from Streamlit Secrets
api_key = st.secrets.get("POLYGON_API_KEY", None)
if not api_key:
    st.error("POLYGON_API_KEY отсутствует в Secrets. Откройте Settings → Secrets и добавьте ключ.")
    st.stop()

# Поле ввода тикера (по умолчанию SPY)
ticker = st.text_input("Тикер", value="SPY", placeholder="Например: SPX, QQQ, AMD...", key="ticker_input").strip().upper()
if not ticker:
    ticker = "SPY"

# Fetch expirations when ticker is chosen
with st.spinner(f"Запрашиваю даты экспираций для {ticker}..."):
    try:
        expirations = fetch_expirations_from_contracts(api_key, ticker, max_pages=30)
        st.session_state['expirations_list'] = list(expirations)
    except requests.HTTPError as e:
        st.error(f"Ошибка Polygon API: {e} — проверьте права ключа и тикер.")
        st.stop()
    except Exception as e:
        st.error(f"Не удалось получить экспирации: {e}")
        st.stop()

if not expirations:
    st.warning("Не найдено доступных будущих экспираций для выбранного тикера.")
    st.stop()

nearest = choose_nearest(expirations)
# Build dropdown with nearest pre-selected
# single-select removed; using multiselect below
nearest = choose_nearest(expirations)
# Multi-select для выбора одной или нескольких дат
selected_expirations = st.multiselect("Выберите даты экспирации", options=expirations, default=[nearest])
st.session_state["selected_expirations"] = selected_expirations

st.caption("Источник данных: Polygon v3. Авторизация — Bearer token из Streamlit Secrets (POLYGON_API_KEY).")
