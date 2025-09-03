import streamlit as st
import pandas as pd
import numpy as np
import time, json, math, io, datetime

from lib.provider import fetch_option_chain
from lib.compute import (
    extract_core_from_chain,
    compute_series_metrics_for_expiry,
    aggregate_series,
)
from lib.utils import choose_default_expiration, env_or_secret
from lib.plotting import make_figure

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Net GEX / AG / PZ / PZ_FP", layout="wide")
st.title("Net GEX / AG / PZ / PZ_FP — Streamlit (Polygon-only)")

# -----------------------------------------------------------------------------
# Secrets & env
# -----------------------------------------------------------------------------
POLYGON_API_KEY = env_or_secret(st, "POLYGON_API_KEY", None)
if not POLYGON_API_KEY:
    st.warning("Установите POLYGON_API_KEY в окружении или .streamlit/secrets.toml")

# -----------------------------------------------------------------------------
# Sidebar controls (минимально и безопасно)
# -----------------------------------------------------------------------------
with st.sidebar:
    ticker = st.text_input("Ticker", value="SPX").strip().upper()
    st.caption("Для индексов (SPX/NDX/RUT/…) префикс I: добавится автоматически")

# -----------------------------------------------------------------------------
# Data fetch — аккуратно, без RapidAPI, не меняя вычислительную логику
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60)
def _load_all_options(_ticker: str):
    """Загружает цепочку без фильтра по экспирации (для выбора default)."""
    chain = fetch_option_chain(_ticker)
    core = extract_core_from_chain(chain)
    return chain, core

@st.cache_data(show_spinner=False, ttl=60)
def _load_by_expiry(_ticker: str, _expiration: str):
    chain = fetch_option_chain(_ticker, expiration=_expiration)
    core = extract_core_from_chain(chain)
    return chain, core

# 1) загрузка общей цепочки — только для списка дат и выбора дефолтной
chain_all, core_all = _load_all_options(ticker)

# построим список доступных экспираций из core_all
expirations = sorted({row.get("expiration_date") for row in core_all if row.get("expiration_date")})
default_expiration = choose_default_expiration(core_all) if core_all else None
if default_expiration and default_expiration not in expirations and default_expiration is not None:
    expirations.append(default_expiration)
    expirations = sorted(expirations)

# UI для выбора даты экспирации
if expirations:
    default_index = expirations.index(default_expiration) if default_expiration in expirations else 0
    expiration = st.sidebar.selectbox("Expiration", expirations, index=default_index)
else:
    expiration = st.sidebar.text_input("Expiration (YYYY-MM-DD)", value="")

# 2) загрузка по выбранной экспирации
if expiration:
    chain_sel, core_sel = _load_by_expiry(ticker, expiration)
    # compute series/aggregates без изменения вашей логики (используем те же функции)
    series = compute_series_metrics_for_expiry(core_sel, ticker, expiration)
    aggregates = aggregate_series(series)

    # рисуем фигуру через ваш plotting helper
    fig = make_figure(aggregates, series, title=f"{ticker} — {expiration}")
    # попытка отрисовать: если это Plotly — используем st.plotly_chart; если Matplotlib — st.pyplot
    # Пытаемся детектировать тип объекта по атрибутам
    if hasattr(fig, "to_plotly_json"):
        st.plotly_chart(fig, use_container_width=True)
    else:
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            st.pyplot(fig, clear_figure=False, use_container_width=True)
        except Exception:
            st.write(fig)
else:
    st.info("Нет доступных дат экспирации. Уточните тикер или введите дату вручную.")
