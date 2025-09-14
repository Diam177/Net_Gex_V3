
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
# Позволяет сформировать и просмотреть таблицу df_raw по выбранной экспирации
# и скачать её в CSV. Динамически загружает модуль sanitize_window по пути,
# чтобы избежать ошибки ModuleNotFoundError при запуске Streamlit.
try:
    _ticker_raw = st.session_state.get("ticker", "").strip().upper()
    _selected_raw = st.session_state.get("selected", [])
    # Показываем раздел, если выбран ровно один срок и есть API‑ключ
    if api_key and _ticker_raw and _selected_raw and len(_selected_raw) == 1:
        exp_date_raw = _selected_raw[0]
        with st.expander("Просмотр df_raw (сырые данные)", expanded=False):
            if st.button("Получить df_raw", key="btn_df_raw"):
                try:
                    # загрузить snapshot по выбранной дате
                    snapshot_js = download_snapshot_json(_ticker_raw, exp_date_raw, api_key)
                    raw_records = snapshot_js.get("results") or []
                    # расчёт спота из дневных close; fallback на spot_price
                    import numpy as _np
                    closes = []
                    for rec in raw_records:
                        day_info = rec.get("day") or {}
                        c = day_info.get("close")
                        if isinstance(c, (int, float)):
                            closes.append(float(c))
                    if closes:
                        S_val = float(_np.nanmedian(closes))
                    else:
                        S_val = float(st.session_state.get("spot_price") or 0.0)
                    # динамический импорт sanitize_window
                    import importlib, os, sys, importlib.util
                    try:
                        sw = importlib.import_module("sanitize_window")
                    except ModuleNotFoundError:
                        try:
                            sys.path.append(os.path.dirname(__file__))
                            sw = importlib.import_module("sanitize_window")
                        except ModuleNotFoundError:
                            module_path = os.path.join(os.path.dirname(__file__), "sanitize_window.py")
                            spec = importlib.util.spec_from_file_location("sanitize_window", module_path)
                            sw = importlib.util.module_from_spec(spec)
                            if spec.loader is not None:
                                spec.loader.exec_module(sw)
                    build_raw_table = getattr(sw, "build_raw_table")
                    SanitizerConfig = getattr(sw, "SanitizerConfig")
                    cfg_raw = SanitizerConfig()
                    df_raw_val = build_raw_table(raw_records, S=S_val, cfg=cfg_raw)
                    st.session_state["df_raw"] = df_raw_val
                except Exception as raw_err:
                    st.session_state["df_raw"] = None
                    st.error(f"Не удалось сформировать df_raw: {raw_err}")
            # отображение и скачивание df_raw
            _df_raw = st.session_state.get("df_raw")
            if _df_raw is not None:
                st.dataframe(_df_raw, use_container_width=True, hide_index=True)
                try:
                    csv_bytes = _df_raw.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Скачать CSV df_raw",
                        data=csv_bytes,
                        file_name=f"df_raw_{_ticker_raw}_{exp_date_raw}.csv",
                        mime="text/csv",
                    )
                except Exception:
                    pass
except Exception:
    pass
# --- /RAW TABLE DISPLAY ---
