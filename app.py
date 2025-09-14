
import json
import streamlit as st

from lib.tiker_data import (
    list_future_expirations,
    download_snapshot_json,
    snapshots_zip_bytes,
    PolygonError,
)

st.set_page_config(page_title="Main", layout="wide")

# Пайплайн подготовки данных
from lib.sanitize_window import sanitize_and_window_pipeline, build_window_panels
import pandas as pd
import io

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


# --- PIPELINE TABLES (intermediate) ---
try:
    _raw_records = st.session_state.get("raw_records")  # ожидаем список dict (Polygon snapshot["results"])
    _spot = st.session_state.get("spot_price")          # float
    if _raw_records and (_spot is not None):
        @st.cache_data(show_spinner=False)
        def _run_pipeline(records, S):
            res = sanitize_and_window_pipeline(records, S)
            # дополнительно соберём панели
            try:
                panels = build_window_panels(res["df_weights"], res["df_corr"], res["windows"])
            except Exception:
                panels = {}
            res["panels"] = panels
            return res
        _res = _run_pipeline(_raw_records, float(_spot))

        def _dl(df, name):
            st.caption(f"Скачать {name}")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(f"CSV — {name}", data=csv_bytes, file_name=f"{name}.csv", mime="text/csv", key=f"csv_{name}")
            try:
                import pyarrow as pa, pyarrow.parquet as pq
                buf = io.BytesIO()
                pq.write_table(pa.Table.from_pandas(df), buf)
                st.download_button(f"Parquet — {name}", data=buf.getvalue(), file_name=f"{name}.parquet", mime="application/octet-stream", key=f"pq_{name}")
            except Exception:
                pass

        st.subheader("1) df_raw")
        st.dataframe(_res["df_raw"], use_container_width=True)
        _dl(_res["df_raw"], "df_raw")

        st.subheader("2) df_marked")
        st.dataframe(_res["df_marked"], use_container_width=True)
        _dl(_res["df_marked"], "df_marked")

        st.subheader("3) df_corr")
        st.dataframe(_res["df_corr"], use_container_width=True)
        _dl(_res["df_corr"], "df_corr")

        st.subheader("4) df_weights")
        st.dataframe(_res["df_weights"], use_container_width=True)
        _dl(_res["df_weights"], "df_weights")

        # windows: dict exp -> idx; развернём в таблицу с K
        try:
            rows = []
            g = _res["df_weights"].sort_values(["exp","K"]).reset_index(drop=True)
            for exp, idxs in _res["windows"].items():
                if hasattr(idxs, "tolist"):
                    idxs = list(idxs)
                for i in idxs:
                    if 0 <= i < len(g):
                        row = g.iloc[i]
                        rows.append({"exp": exp, "idx": int(i), "K": float(row["K"])})
            df_windows = pd.DataFrame(rows)
        except Exception:
            df_windows = pd.DataFrame()

        st.subheader("5) windows (развёрнуто)")
        if not df_windows.empty:
            st.dataframe(df_windows, use_container_width=True)
            _dl(df_windows, "windows")
        else:
            st.info("windows: нет данных (или не удалось построить таблицу из индексов).")

        st.subheader("6) window_raw")
        st.dataframe(_res["window_raw"], use_container_width=True)
        _dl(_res["window_raw"], "window_raw")

        st.subheader("7) window_corr")
        st.dataframe(_res["window_corr"], use_container_width=True)
        _dl(_res["window_corr"], "window_corr")

        # панели по окну (если удалось)
        if isinstance(_res.get("panels"), dict) and _res["panels"]:
            for exp, dfp in _res["panels"].items():
                st.subheader(f"8) panel — {exp}")
                st.dataframe(dfp, use_container_width=True)
                _dl(dfp, f"panel_{exp}")
    else:
        st.info("Чтобы вывести промежуточные таблицы: в session_state должны быть raw_records (список результатов снапшота) и spot_price (float).")
except Exception as _e:
    st.warning(f"PIPELINE TABLES: {type(_e).__name__}: {_e}")
# --- /PIPELINE TABLES ---
