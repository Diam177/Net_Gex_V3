
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


# --- DEBUG SPOT DISPLAY ---
try:
    # аккуратно пробуем получить цену через helper
    from lib.tiker_data import get_spot_price  # не трогаем остальной импорт
    _ticker = st.session_state.get("ticker")
    if _ticker and api_key:
        try:
            _s, _ts_ms, _src = get_spot_price(_ticker.strip().upper(), api_key)
            st.session_state["spot_price"] = _s
            st.session_state["spot_ts_ms"] = _ts_ms
            st.session_state["spot_source"] = _src
        except Exception as _e:
            # очищаем старые значения, чтобы не показывать цену от предыдущего тикера
            st.session_state["spot_price"] = None
            st.session_state["spot_ts_ms"] = None
            st.session_state["spot_source"] = None
            st.session_state["spot_error"] = str(_e)
    else:
        st.session_state["spot_price"] = None
        st.session_state["spot_ts_ms"] = None
        st.session_state["spot_source"] = None

    # читаем query params: поддерживаем и новые, и старые API Streamlit
    def _qp_true(name: str) -> bool:
        val = None
        try:
            qp = getattr(st, "query_params", None)
            if qp is not None:
                try:
                    val = qp.get(name)
                except Exception:
                    try:
                        val = dict(qp).get(name)
                    except Exception:
                        pass
                if val is None and hasattr(qp, "to_dict"):
                    val = qp.to_dict().get(name)
        except Exception:
            pass
        if val is None:
            try:
                val = st.experimental_get_query_params().get(name)
            except Exception:
                val = None
        if isinstance(val, list):
            val = val[0] if val else ""
        return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

    if _qp_true("debug_spot"):
        _s = st.session_state.get("spot_price")
        _src = st.session_state.get("spot_source")
        _ts = st.session_state.get("spot_ts_ms")
        _ts_str = "—"
        try:
            if _ts:
                from datetime import datetime, timezone
                _ts_str = datetime.fromtimestamp(int(_ts)/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            pass
        st.info(f"DEBUG SPOT — {_ticker}: {_s} ({_src}, {_ts_str})")
except Exception:
    # никакой UI не ломаем, любые ошибки в дебаге игнорируются
    pass
# --- /DEBUG SPOT DISPLAY ---


# --- ALWAYS SHOW SPOT ---
try:
    _ticker = st.session_state.get("ticker")
    _s = st.session_state.get("spot_price")
    _src = st.session_state.get("spot_source")
    if _ticker and (_s is not None):
        st.markdown(f"**Цена {_ticker}:** {_s:.2f}  \u2009*(источник: {_src})*")
except Exception:
    pass
# --- /ALWAYS SHOW SPOT ---


# --- RAW TABLE DISPLAY ---
# Здесь реализуем отображение и скачивание первой таблицы df_raw из санитайзера.
# Пользователь выбирает единственную дату экспирации и нажимает кнопку "Получить df_raw".
# Приложение загружает JSON‑snapshot, корректно импортирует sanitize_window (из lib или из файла),
# строит сырую таблицу и отображает её. Также доступна загрузка CSV.
try:
    # Показываем секцию только если выбран ровно один срок
    if selected and len(selected) == 1:
        expander = st.expander("Просмотр df_raw (сырые данные)")
        with expander:
            if st.button("Получить df_raw"):
                try:
                    # Получаем snapshot JSON с рынка
                    ticker = st.session_state.get("ticker", "SPY").strip().upper()
                    expiry = selected[0]
                    snap = download_snapshot_json(ticker, expiry, api_key)
                    # Получаем спот‑цену
                    from lib.tiker_data import get_spot_price
                    s_price, _, _ = get_spot_price(ticker, api_key)
                    # Динамически импортируем sanitize_window
                    import importlib
                    import importlib.util
                    import os
                    from pathlib import Path
                    module = None
                    # Сначала пытаемся импортировать как пакет lib.sanitize_window
                    try:
                        module = importlib.import_module("lib.sanitize_window")
                    except Exception:
                        try:
                            # затем пытаемся импортировать как sanitize_window на верхнем уровне
                            module = importlib.import_module("sanitize_window")
                        except Exception:
                            # в противном случае ищем файл sanitize_window.py в директории приложения и подкаталоге lib
                            current_dir = Path(__file__).resolve().parent
                            candidates = [current_dir / "lib" / "sanitize_window.py", current_dir / "sanitize_window.py"]
                            for cand in candidates:
                                if cand.exists():
                                    spec = importlib.util.spec_from_file_location("sanitize_window_dynamic", cand)
                                    module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(module)  # type: ignore
                                    break
                    if module is None:
                        raise ModuleNotFoundError("sanitize_window module not found")
                    build_raw_table = getattr(module, "build_raw_table")
                    SanitizerConfig = getattr(module, "SanitizerConfig")
                    # Формируем сырую таблицу
                    df_raw = build_raw_table(snap, s_price)
                    st.dataframe(df_raw)
                    # Кнопка для скачивания CSV
                    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Скачать df_raw CSV",
                        data=csv_bytes,
                        file_name=f"{ticker}_{expiry}_df_raw.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Не удалось сформировать df_raw: {e}")
except Exception:
    # При любых ошибках в секции df_raw UI сохраняем стабильность
    pass
# --- /RAW TABLE DISPLAY ---
