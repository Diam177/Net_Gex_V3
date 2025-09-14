
import json
import streamlit as st

from lib.tiker_data import (
    list_future_expirations,
    download_snapshot_json,
    snapshots_zip_bytes,
    PolygonError,
)

# для обработки raw-данных используем санитайзер
from sanitize_window import build_raw_table, SanitizerConfig

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
# Позволяет пользователю сформировать и посмотреть таблицу df_raw (сырой входной набор)
# для выбранной экспирации и скачать её в CSV. Использует sanitize_window.build_raw_table.
try:
    # Берём текущий тикер и выбранные даты
    _ticker_raw = st.session_state.get("ticker", "").strip().upper()
    _selected_raw = st.session_state.get("selected", [])
    # Показ df_raw только при наличии ключа API, тикера и одной выбранной даты
    if api_key and _ticker_raw and _selected_raw and len(_selected_raw) == 1:
        exp_date_raw = _selected_raw[0]
        with st.expander("Просмотр df_raw (сырой набор)", expanded=False):
            if st.button("Получить df_raw", key="btn_df_raw"):
                try:
                    # загружаем снапшот через Polygon
                    snapshot_js = download_snapshot_json(_ticker_raw, exp_date_raw, api_key)
                    raw_records = snapshot_js.get("results") or []
                    # оцениваем спот для расчёта времени до экспирации и смещения по страйкам
                    import numpy as _np
                    closes = []
                    for r in raw_records:
                        day_info = r.get("day") or {}
                        c = day_info.get("close")
                        if isinstance(c, (int, float)):
                            closes.append(float(c))
                    if closes:
                        S_val = float(_np.nanmedian(closes))
                    else:
                        # fallback: используем spot_price из session_state или 0.0
                        S_val = float(st.session_state.get("spot_price") or 0.0)
                    cfg_raw = SanitizerConfig()
                    df_raw = build_raw_table(raw_records, S=S_val, cfg=cfg_raw)
                    st.session_state["df_raw"] = df_raw
                except Exception as _err:
                    st.session_state["df_raw"] = None
                    st.error(f"Не удалось сформировать df_raw: {_err}")
            df_raw_show = st.session_state.get("df_raw")
            if df_raw_show is not None:
                # отображаем таблицу
                st.dataframe(df_raw_show, use_container_width=True, hide_index=True)
                # кнопка для скачивания CSV
                try:
                    csv_bytes = df_raw_show.to_csv(index=False).encode("utf-8")
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
