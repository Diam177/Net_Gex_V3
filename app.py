import streamlit as st
import pandas as pd
import numpy as np
import time, json, math, io, datetime

from lib.provider import fetch_option_chain, debug_meta
from lib.compute import extract_core_from_chain, compute_series_metrics_for_expiry, aggregate_series
from lib.utils import choose_default_expiration, env_or_secret
from lib.plotting import make_figure

st.set_page_config(page_title="Net GEX / AG / PZ / PZ_FP", layout="wide")
st.title(ticker)

# === Secrets / env ===
RAPIDAPI_HOST = env_or_secret(st, "RAPIDAPI_HOST", None)
RAPIDAPI_KEY  = env_or_secret(st, "RAPIDAPI_KEY",  None)

with st.sidebar:
    st.markdown("### RapidAPI")
    st.write("Ключи читаются из `.streamlit/secrets.toml` или переменных окружения.")
    st.text(f"HOST: {RAPIDAPI_HOST or '—'}")
    st.text(f"KEY:  {'set' if RAPIDAPI_KEY else '—'}")

# === Inputs ===
col_t, col_e = st.columns([1,1])
with col_t:
    ticker = st.text_input("Ticker", value="SPY").strip().upper()
with col_e:
    st.caption("Экспирация по умолчанию — ближайшая будущая")
    expiry_placeholder = st.empty()

st.divider()
st.subheader("Источник данных")
col_f, col_u = st.columns([1,1])

raw_data = None
raw_bytes = None

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_chain_cached(ticker, host, key, expiry_unix=None):
    data, content = fetch_option_chain(ticker, host, key, expiry_unix=expiry_unix)
    return data, content

# === Fetch from RapidAPI ===
with col_f:
    if RAPIDAPI_HOST and RAPIDAPI_KEY:
        try:
            base_json, base_bytes = _fetch_chain_cached(ticker, RAPIDAPI_HOST, RAPIDAPI_KEY, None)
            raw_data, raw_bytes = base_json, base_bytes
            st.success("Данные получены из RapidAPI.")
        except Exception as e:
            st.error(f"Ошибка запроса RapidAPI: {e}")
    else:
        st.warning("Не заданы RAPIDAPI_HOST/RAPIDAPI_KEY. Можно загрузить JSON вручную.")

# === Upload ===
with col_u:
    up = st.file_uploader("Или загрузите файл JSON от провайдера", type=["json"], accept_multiple_files=False)
    if up is not None:
        try:
            raw_bytes = up.read()
            raw_data  = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
            st.success("Загружен JSON из файла.")
        except Exception as e:
            st.error(f"Ошибка чтения JSON: {e}")

if raw_data is None:
    st.stop()

# === DEBUG BLOCK (виден всегда, ещё до парсинга) ===
def _sniff_roots(d: dict):
    out = {}
    try:
        oc = d.get("optionChain", {}).get("result", [])
        out["optionChain.result[0]"] = list(oc[0].keys())[:50] if isinstance(oc, list) and oc else None
    except Exception as e:
        out["optionChain.result[0]"] = f"err: {e}"
    for name in ["body", "data", "result"]:
        try:
            node = d.get(name)
            if isinstance(node, list) and node:
                out[f"{name}[0]"] = list(node[0].keys())[:50]
            elif isinstance(node, dict):
                out[name] = list(node.keys())[:50]
            else:
                out[name] = None
        except Exception as e:
            out[name] = f"err: {e}"
    return out

with st.expander("🔎 Debug / Скачать сырые данные", expanded=False):
    meta = debug_meta()
    st.markdown("**Endpoint, который сработал:**")
    st.code(json.dumps(meta.get("used", {}), ensure_ascii=False, indent=2), language="json")

    st.markdown("**Все попытки:**")
    st.code(json.dumps(meta.get("attempts", []), ensure_ascii=False, indent=2), language="json")

    st.markdown("**Где найден корень данных (root):**")
    st.write(meta.get("root_path"))

    st.markdown("**Цена до/после поиска источника цены:**")
    st.code(json.dumps({
        "price_before": meta.get("price_before"),
        "t0_before": meta.get("t0_before"),
        "price_after": meta.get("price_after"),
        "t0_after": meta.get("t0_after"),
        "price_source": meta.get("price_source"),
    }, ensure_ascii=False, indent=2), language="json")

    if "quote_attempts" in meta:
        st.markdown("**Попытки получить цену (quotes):**")
        st.code(json.dumps(meta["quote_attempts"], ensure_ascii=False, indent=2), language="json")

    st.markdown("**Обнаруженная структура корней:**")
    try:
        st.code(json.dumps(_sniff_roots(raw_data), ensure_ascii=False, indent=2), language="json")
    except Exception as e:
        st.write(f"sniff error: {e}")

    # Скачать ровно то, что пришло от провайдера (RAW)
    st.download_button(
        "⬇️ Скачать RAW JSON провайдера",
        data=raw_bytes if raw_bytes is not None else json.dumps(raw_data).encode("utf-8"),
        file_name=f"{ticker}_raw_provider.json",
        mime="application/json",
    )
    # Скачивание «красивой» версии распарсенного JSON
    st.download_button(
        "⬇️ Скачать распарсенный JSON (pretty)",
        data=json.dumps(raw_data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{ticker}_parsed_pretty.json",
        mime="application/json",
    )
    # Скачивание meta-лога
    st.download_button(
        "⬇️ Скачать debug-meta",
        data=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{ticker}_debug_meta.json",
        mime="application/json",
    )

# === Parse core ===
try:
    quote, t0, S, expirations, blocks_by_date = extract_core_from_chain(raw_data)
except Exception as e:
    st.error(f"Неверная структура JSON: {e}")
    st.info("Скачайте RAW/pretty JSON и debug-meta в блоке Debug выше и пришлите мне — подгоню адаптер.")
    st.stop()

now_unix = int(time.time())
if not expirations:
    st.error("Список дат экспирации пуст. Проверьте источник.")
    st.stop()

default_exp = choose_default_expiration(expirations, now_unix)

def fmt_ts(ts):
    return datetime.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")

exp_labels = [f"{e}  ({fmt_ts(e)})" for e in expirations]
try:
    default_index = expirations.index(default_exp)
except ValueError:
    default_index = 0

sel_label = expiry_placeholder.selectbox("Expiration (UTC)", options=exp_labels, index=default_index)
selected_exp = int(sel_label.split()[0])

# Если выбранной даты нет в уже пришедшем блоке — дотягиваем конкретный expiry
if selected_exp not in blocks_by_date and RAPIDAPI_HOST and RAPIDAPI_KEY:
    try:
        by_date_json, by_date_bytes = _fetch_chain_cached(ticker, RAPIDAPI_HOST, RAPIDAPI_KEY, selected_exp)
        quote2, t02, S2, expirations2, blocks_by_date2 = extract_core_from_chain(by_date_json)
        blocks_by_date.update(blocks_by_date2)
        raw_bytes = by_date_bytes  # для кнопки "скачать"
    except Exception as e:
        st.warning(f"Не удалось получить блок для выбранной даты: {e}")

# Кнопка "Скачать сырой JSON"
st.download_button(
    "Скачать сырой JSON от провайдера",
    data=raw_bytes if raw_bytes is not None else json.dumps(raw_data, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name=f"{ticker}_{selected_exp}_raw.json",
    mime="application/json"
)

# Подготовка всех серий (для PZ/PZ_FP нужен контекст всех экспираций)
all_series = []
for e in expirations:
    blk = blocks_by_date.get(int(e))
    if blk is None:
        continue
    strikes, call_oi, put_oi, call_vol, put_vol, iv_call, iv_put = aggregate_series(blk)
    T = max((int(e) - int(quote.get('regularMarketTime', now_unix))) / (365*24*3600), 1e-6)
    all_series.append({
        "E": int(e), "T": float(T), "strikes": strikes,
        "call_oi": call_oi, "put_oi": put_oi,
        "call_vol": call_vol, "put_vol": put_vol,
        "iv_call": iv_call, "iv_put": iv_put
    })

selected_block = blocks_by_date.get(selected_exp)
if selected_block is None:
    st.error("Не найден блок по выбранной экспирации.")
    st.stop()

day_high = quote.get("regularMarketDayHigh", None)
day_low  = quote.get("regularMarketDayLow", None)

S_used = float(quote.get('regularMarketPrice', S))
metrics = compute_series_metrics_for_expiry(
    S=S_used,
    t0=int(quote.get("regularMarketTime", t0)),
    expiry_unix=selected_exp,
    block=selected_block,
    day_high=day_high,
    day_low=day_low,
    all_series=all_series
)

# === Table ===
st.divider()
st.subheader("Таблица")
df = pd.DataFrame({
    "Strike": metrics["strikes"],
    "Put OI": metrics["put_oi"],
    "Call OI": metrics["call_oi"],
    "Put Volume": metrics["put_vol"],
    "Call Volume": metrics["call_vol"],
    "Net Gex": metrics["net_gex"],
    "AG": metrics["ag"],
    "PZ": np.round(metrics["pz"], 6),
    "PZ_FP": np.round(metrics["pz_fp"], 6),
})
st.dataframe(df, use_container_width=True)

# === Plot ===
st.divider()
st.subheader("График")
cols = st.columns(8)
toggles = {}
names = ["Net Gex","Put OI","Call OI","Put Volume","Call Volume","AG","PZ","PZ_FP"]
defaults = {"Net Gex": True, "Put OI": False, "Call OI": False, "Put Volume": False, "Call Volume": False, "AG": False, "PZ": False, "PZ_FP": False}
for i, name in enumerate(names):
    with cols[i]:
        toggles[name] = st.toggle(name, value=defaults.get(name, False), key=f"tgl_{name}")

series_dict = {
    "Net Gex": df["Net Gex"].values,
    "Put OI": df["Put OI"].values,
    "Call OI": df["Call OI"].values,
    "Put Volume": df["Put Volume"].values,
    "Call Volume": df["Call Volume"].values,
    "AG": df["AG"].values,
    "PZ": df["PZ"].values,
    "PZ_FP": df["PZ_FP"].values,
}

fig = make_figure(df["Strike"].values, df["Net Gex"].values, toggles, series_dict, price=S_used, ticker=ticker)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
