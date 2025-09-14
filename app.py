
import json
import streamlit as st

from lib.tiker_data import (
    list_future_expirations,
    download_snapshot_json,
    snapshots_zip_bytes,
    PolygonError,
)

st.set_page_config(page_title="Main", layout="wide")

api_key = st.secrets.get("POLYGON_API_KEY", "")


def _load_expirations():
    ticker = st.session_state.get("ticker", "SPY").strip().upper()
    if not api_key:
        st.session_state["expirations"] = []
        st.session_state["last_loaded_ticker"] = None
        st.error("В .streamlit/secrets.toml должен быть задан POLYGON_API_KEY")
        return
    try:
        with st.spinner("Загружаем даты экспираций…"):
            dates = list_future_expirations(ticker, api_key)
        st.session_state["expirations"] = dates
        st.session_state["last_loaded_ticker"] = ticker

    except Exception as e:
        st.session_state["expirations"] = []
        st.session_state["last_loaded_ticker"] = None
        st.error(f"Ошибка загрузки дат: {e}")


# --- Тикер в основной области ---
col_ticker, _ = st.columns([2, 3])
with col_ticker:
    st.text_input(
        "Тикер",
        value=st.session_state.get("ticker", "SPY"),
        key="ticker",
        max_chars=15,
        
        on_change=_load_expirations,
    )

# Первичная загрузка при первом открытии страницы
if "expirations" not in st.session_state:
    _load_expirations()

expirations = st.session_state.get("expirations", [])
# Локально вычисляем default для мультиселекта, НИЧЕГО не пишем в session_state["selected"]
selected_default = []
if expirations:
    prev = st.session_state.get("selected", [])
    if prev and prev[0] in expirations:
        selected_default = prev
    else:
        selected_default = [expirations[0]]

selected = st.session_state.get("selected", [])

col1, col2 = st.columns([3, 2])
with col1:
    if expirations:
        st.multiselect(
            "Даты экспирации",
            options=expirations,
            default=selected_default,
            key="selected",
            label_visibility="collapsed",
        )
        selected = st.session_state.get("selected", [])
    else:
        st.info("Нет доступных дат — проверьте тикер или ключ API.")

with col2:
    if expirations:
        csv_lines = ["expiration_date"] + expirations
        csv_bytes = ("\n".join(csv_lines) + "\n").encode("utf-8")
        st.download_button(
            label="Скачать CSV с датами",
            data=csv_bytes,
            file_name=f"{st.session_state.get('ticker', 'TICKER')}_expirations.csv",
            mime="text/csv",
        )

        if selected:
            if len(selected) == 1:
                d = selected[0]
                try:
                    js = download_snapshot_json(st.session_state.get('ticker', 'TICKER'), d, api_key)
                    st.download_button(
                        label=f"Скачать JSON snapshot ({d})",
                        data=(json.dumps(js, ensure_ascii=False)).encode("utf-8"),
                        file_name=f"{st.session_state.get('ticker', 'TICKER')}_{d}.json",
                        mime="application/json",
                    )

                except PolygonError as e:
                    st.error(f"Ошибка Polygon: {e}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")
            else:
                try:
                    zbytes, fname = snapshots_zip_bytes(st.session_state.get('ticker', 'TICKER'), selected, api_key)
                    st.download_button(
                        label=f"Скачать ZIP ({len(selected)} дат)",
                        data=zbytes,
                        file_name=fname,
                        mime="application/zip",
                    )
                except PolygonError as e:
                    st.error(f"Ошибка Polygon: {e}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")
    else:
        pass
