import json
import streamlit as st

from lib.tiker_data import (
    list_future_expirations,
    download_snapshot_json,
    snapshots_zip_bytes,
    PolygonError,
)

st.set_page_config(page_title=\"Main\", layout=\"wide\")
st.title(\"Main — Выбор тикера и дат экспирации\")

api_key = st.secrets.get(\"POLYGON_API_KEY\", \"\")

# --- helpers ---
def _load_expirations():
    ticker = st.session_state.get(\"ticker\", \"SPY\").strip().upper()
    if not api_key:
        st.session_state[\"expirations\"] = []
        st.session_state[\"last_loaded_ticker\"] = None
        st.error(\"В .streamlit/secrets.toml должен быть задан POLYGON_API_KEY\")
        return
    try:
        with st.spinner(\"Загружаем даты экспираций…\"):
            dates = list_future_expirations(ticker, api_key)
        st.session_state[\"expirations\"] = dates
        st.session_state[\"last_loaded_ticker\"] = ticker
        st.toast(f\"Найдено дат: {len(dates)}\", icon=\"✅\")
    except Exception as e:
        st.session_state[\"expirations\"] = []
        st.session_state[\"last_loaded_ticker\"] = None
        st.error(f\"Ошибка загрузки дат: {e}\")

# --- UI: ticker input in main (not in sidebar) ---
col_ticker, _ = st.columns([2, 3])
with col_ticker:
    st.text_input(
        \"Тикер базового актива\",
        value=st.session_state.get(\"ticker\", \"SPY\"),
        key=\"ticker\",
        max_chars=15,
        help=\"Например: SPY, AAPL, MSFT\",
        on_change=_load_expirations,  # авто‑подгрузка при изменении
    )

# Первичная загрузка при первом открытии страницы
if \"expirations\" not in st.session_state:
    _load_expirations()

expirations = st.session_state.get(\"expirations\", [])
selected = st.session_state.get(\"selected\", [])

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader(\"Выбор дат\")
    if expirations:
        selected = st.multiselect(\"Даты экспирации\", options=expirations, default=selected, key=\"selected\")
    else:
        st.info(\"Нет доступных дат — проверьте тикер или ключ API.\")

with col2:
    st.subheader(\"Скачивание\")
    if expirations:
        # CSV с датами (без pandas)
        csv_lines = [\"expiration_date\"] + expirations
        csv_bytes = (\"\\n\".join(csv_lines) + \"\\n\").encode(\"utf-8\")
        st.download_button(
            label=\"Скачать CSV с датами\",
            data=csv_bytes,
            file_name=f\"{st.session_state.get('ticker', 'TICKER')}_expirations.csv\",
            mime=\"text/csv\",
        )

        if selected:
            if len(selected) == 1:
                d = selected[0]
                try:
                    js = download_snapshot_json(st.session_state.get('ticker', 'TICKER'), d, api_key)
                    st.download_button(
                        label=f\"Скачать JSON snapshot ({d})\",
                        data=(json.dumps(js, ensure_ascii=False)).encode(\"utf-8\"),
                        file_name=f\"{st.session_state.get('ticker', 'TICKER')}_{d}.json\",
                        mime=\"application/json\",
                    )
                    st.caption(f\"Всего опционов: {js.get('results_count', 0)}\")
                except PolygonError as e:
                    st.error(f\"Ошибка Polygon: {e}\")
                except Exception as e:
                    st.error(f\"Ошибка: {e}\")
            else:
                try:
                    zbytes, fname = snapshots_zip_bytes(st.session_state.get('ticker', 'TICKER'), selected, api_key)
                    st.download_button(
                        label=f\"Скачать ZIP ({len(selected)} дат)\",
                        data=zbytes,
                        file_name=fname,
                        mime=\"application/zip\",
                    )
                except PolygonError as e:
                    st.error(f\"Ошибка Polygon: {e}\")
                except Exception as e:
                    st.error(f\"Ошибка: {e}\")
    else:
        st.caption(\"Нет дат для скачивания.\")

st.divider()
st.markdown(
    \"\"\"
    **Памятка:**  
    1) Укажите `POLYGON_API_KEY` в `.streamlit/secrets.toml`.  
    2) Введите тикер — даты загрузятся автоматически.  
    3) Выберите дату(ы) и скачайте JSON/ZIP.
    \"\"\"
)
