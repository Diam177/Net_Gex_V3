import json
import streamlit as st

from lib.tiker_data import (
    list_future_expirations,
    download_snapshot_json,
    snapshots_zip_bytes,
    PolygonError,
)

st.set_page_config(page_title="Main", layout="wide")
st.title("Main — Выбор тикера и дат экспирации")

api_key = st.secrets.get("POLYGON_API_KEY", "")

with st.sidebar:
    st.markdown("### Параметры")
    ticker = st.text_input("Тикер базового актива", value=st.session_state.get("ticker", "SPY")).strip().upper()
    st.session_state["ticker"] = ticker

    if not api_key:
        st.error("В .streamlit/secrets.toml должен быть задан POLYGON_API_KEY")
    else:
        if st.button("Загрузить даты экспираций"):
            try:
                dates = list_future_expirations(ticker, api_key)
                st.session_state["expirations"] = dates
                st.success(f"Найдено дат: {len(dates)}")
            except Exception as e:
                st.session_state["expirations"] = []
                st.error(f"Ошибка загрузки дат: {e}")

expirations = st.session_state.get("expirations", [])

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Выбор дат")
    if expirations:
        selected = st.multiselect("Даты экспирации", options=expirations, default=st.session_state.get("selected", []))
        st.session_state["selected"] = selected
    else:
        st.info("Сначала загрузите даты экспираций (кнопка в сайдбаре).")

with col2:
    st.subheader("Скачивание")
    if expirations:
        # CSV с датами (без pandas)
        csv_lines = ["expiration_date"] + expirations
        csv_bytes = ("\n".join(csv_lines) + "\n").encode("utf-8")
        st.download_button(
            label="Скачать CSV с датами",
            data=csv_bytes,
            file_name=f"{st.session_state.get('ticker', 'TICKER')}_expirations.csv",
            mime="text/csv",
        )

        selected = st.session_state.get("selected", [])
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
                    st.caption(f"Всего опционов: {js.get('results_count', 0)}")
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
        st.caption("Нет дат для скачивания.")

st.divider()
st.markdown(
    """
    **Памятка:**  
    1) Укажите `POLYGON_API_KEY` в `.streamlit/secrets.toml`.  
    2) Введите тикер и нажмите «Загрузить даты экспираций».  
    3) Выберите дату(ы) и скачайте JSON/ZIP.
    """
)
