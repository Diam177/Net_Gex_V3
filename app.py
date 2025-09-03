import streamlit as st
import pandas as pd
import numpy as np
import time, json, math, io, datetime

from lib.provider import fetch_option_chain
from lib.compute import extract_core_from_chain, compute_series_metrics_for_expiry
from lib.compute import aggregate_series
from lib.utils import choose_default_expiration, env_or_secret
from lib.plotting import make_figure

st.set_page_config(page_title="Net GEX / AG / PZ / PZ_FP", layout="wide")

st.title("Net GEX / AG / PZ / PZ_FP — Streamlit")

RAPIDAPI_HOST = env_or_secret(st, "RAPIDAPI_HOST", None)
RAPIDAPI_KEY  = env_or_secret(st, "RAPIDAPI_KEY",  None)
POLYGON_API_KEY = env_or_secret(st, "POLYGON_API_KEY", None)

_PROVIDER = "polygon" if POLYGON_API_KEY else "rapid"

ticker = st.sidebar.text_input("Ticker", "AAPL")

expiry_placeholder = st.sidebar.empty()
download_placeholder = st.empty()
download_polygon_placeholder = st.empty()
table_download_placeholder = st.empty()

# Информация о провайдере
st.caption(f"Data provider: **{_PROVIDER}**")

# Явное отображение тикера (SPX → I:SPX)
_display_ticker = ticker.strip().upper()
if _display_ticker in ("SPX", "^SPX"):
    st.caption("Ticker: **SPX**  (Polygon symbol: `I:SPX`)")
else:
    st.caption(f"Ticker: **{_display_ticker}**")

# === Данные от провайдера ===
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_chain_cached(ticker, host, key, expiry=None):
    return fetch_option_chain(ticker, host, key, expiry)

try:
    raw_data, raw_bytes = _fetch_chain_cached(
        ticker,
        None if _PROVIDER == "polygon" else RAPIDAPI_HOST,
        POLYGON_API_KEY if _PROVIDER == "polygon" else RAPIDAPI_KEY,
        None
    )
except Exception as e:
    st.error(f"Ошибка при запросе данных: {e}")
    st.stop()

try:
    quote, underlying_price, t0, expirations, blocks_by_date = extract_core_from_chain(raw_data)
except Exception as e:
    st.error(f"Invalid chain: {e}")
    st.stop()

exp_labels = [datetime.datetime.utcfromtimestamp(e).strftime("%Y-%m-%d") for e in expirations]
default_index = 0

sel_label = expiry_placeholder.selectbox("Expiration", options=exp_labels, index=default_index)
selected_exp = expirations[exp_labels.index(sel_label)]

# Если выбранной даты нет в блоках — дотягиваем конкретный expiry
if selected_exp not in blocks_by_date and ((_PROVIDER=="polygon" and POLYGON_API_KEY) or (_PROVIDER=="rapid" and RAPIDAPI_HOST and RAPIDAPI_KEY)):
    try:
        by_date_json, by_date_bytes = _fetch_chain_cached(
            ticker,
            None if _PROVIDER=="polygon" else RAPIDAPI_HOST,
            POLYGON_API_KEY if _PROVIDER=="polygon" else RAPIDAPI_KEY,
            selected_exp
        )
        _, _, _, expirations2, blocks_by_date2 = extract_core_from_chain(by_date_json)
        blocks_by_date.update(blocks_by_date2)
        raw_bytes = by_date_bytes
    except Exception as e:
        st.warning(f"Не удалось получить блок для выбранной даты: {e}")

# Кнопка "Скачать сырой JSON"
download_placeholder.download_button(
    "Download JSON",
    data=raw_bytes if raw_bytes is not None else json.dumps(raw_data, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name=f"{ticker}_{selected_exp}_raw.json",
    mime="application/json"
)

# Доп. кнопка: сырые данные от Polygon
if _PROVIDER == "polygon":
    download_polygon_placeholder.download_button(
        "Download Polygon JSON",
        data=raw_bytes if raw_bytes is not None else json.dumps(raw_data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{ticker}_{selected_exp}_polygon_raw.json",
        mime="application/json"
    )

# === Контекст для PZ/PZ_FP (all_series_ctx) ===
all_series_ctx = []
for e, block in blocks_by_date.items():
    try:
        metrics = compute_series_metrics_for_expiry(
            S=underlying_price, t0=t0, expiry_unix=e,
            block=block,
            day_high=quote.get("regularMarketDayHigh"),
            day_low=quote.get("regularMarketDayLow"),
            all_series=all_series_ctx
        )
    except Exception as ex:
        st.warning(f"Не удалось вычислить метрики для {e}: {ex}")

# === Метрики для выбранной экспирации ===
if selected_exp not in blocks_by_date:
    st.error("Не найден блок выбранной экспирации у провайдера.")
    st.stop()

day_high = quote.get("regularMarketDayHigh", None)
day_low  = quote.get("regularMarketDayLow", None)

metrics = compute_series_metrics_for_expiry(
    S=underlying_price, t0=t0, expiry_unix=selected_exp,
    block=blocks_by_date[selected_exp],
    day_high=day_high, day_low=day_low,
    all_series=all_series_ctx
)

# === Таблица по страйкам ===
df = pd.DataFrame({
    "strike": metrics["strikes"],
    "net_gex": metrics["net_gex"],
    "abs_gamma": metrics["abs_gamma"],
    "put_oi": metrics["put_oi"],
    "call_oi": metrics["call_oi"],
    "put_vol": metrics["put_vol"],
    "call_vol": metrics["call_vol"],
    "pz": metrics["pz"],
    "pz_fp": metrics["pz_fp"],
})

st.dataframe(df)

table_download_placeholder.download_button(
    "Download table (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker}_{selected_exp}_metrics.csv",
    mime="text/csv"
)

# === График ===
fig = make_figure(metrics, ticker, selected_exp, quote)
st.plotly_chart(fig, use_container_width=True)
