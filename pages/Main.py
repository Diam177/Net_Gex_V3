# Main page: Ticker/Expiration selector + Final Table
# Location: pages/Main.py
# Usage: streamlit run app.py  (Streamlit will auto-detect pages/Main.py)

import streamlit as st

st.set_page_config(page_title="Main — Ticker & Final Table", layout="wide")

# Lazy imports to avoid any side effects on app startup
try:
    from lib import tiker_data  # expected to exist: render_tiker_data_block, get_raw_by_exp, sanitize_from_tiker_data
except Exception as e:
    tiker_data = None
    _tiker_err = e

try:
    from lib.ui_final_table import render_final_table
except Exception as e:
    render_final_table = None
    _table_err = e

st.title("Main")

# 1) Block: ticker + expiration (from lib/tiker_data.py)
with st.container(border=True):
    st.subheader("Выбор тикера и даты экспирации")
    if tiker_data is None:
        st.error("lib/tiker_data.py не найден или не импортируется. Проверь файл. "
                 f"Детали: {_tiker_err!s}")
    else:
        # This call is expected to populate st.session_state with raw_records / spot
        # and, after processing, df_corr / windows for downstream components.
        tiker_data.render_tiker_data_block(st)

# 2) Final table (lib/ui_final_table.py -> lib/final_table.py)
with st.container(border=True):
    st.subheader("Финальная таблица")
    if render_final_table is None:
        st.error("Не удалось импортировать lib/ui_final_table.render_final_table. "
                 f"Детали: {_table_err!s}")
    else:
        # render_final_table internally reads st.session_state and decides the data source:
        # - preferred: df_corr + windows from sanitize_window pipeline
        # - fallback: raw_records + spot
        render_final_table(st)

# Small helper: show minimal session debug toggle (collapsed by default)
with st.expander("Отладочная информация (сеанс)"):
    keys = [
        "raw_records", "spot", "df_corr", "windows",
        "_last_exp_sig", "_last_ticker"
    ]
    for k in keys:
        st.write(k, "→", "есть" if k in st.session_state else "нет")
