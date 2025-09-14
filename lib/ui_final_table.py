
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple
from final_table import FinalTableConfig, process_from_raw

# This UI block is COMPLETELY INDEPENDENT from the sidebar.
# It loads expirations and raw option chain DIRECTLY from the provider (provider_polygon),
# builds the final table for the chosen expiration, and exposes CSV/Parquet + "raw provider" CSV.

def _fetch_provider_chain_and_expirations(ticker: str) -> Tuple[list, float, List[str]]:
    """Return (raw_records, spot, expirations) directly from provider."""
    from provider_polygon import fetch_option_chain  # provider abstraction already in project
    chain = fetch_option_chain(ticker=ticker, host=None, key=None, expiry_unix=None)
    raw_records = chain.get("options") or chain.get("records") or []
    # spot price
    quote = chain.get("quote") or {}
    spot = float(quote.get("regularMarketPrice") or quote.get("last") or 0.0)
    # expirations list (strings like 'YYYY-MM-DD')
    expirations = chain.get("expirations") or chain.get("expirationDates") or []
    expirations = [str(x) for x in expirations]
    expirations = sorted(set(expirations))
    return raw_records, spot, expirations

def render_final_table(section_title: str = "Финальная таблица (окно, NetGEX/AG, PZ/ER)"):
    st.header(section_title)

    with st.expander("Настройки финальной таблицы (полностью независимы от сайдбара)", expanded=True):
        ticker = st.text_input("Ticker", value=st.session_state.get("_table_ticker", "SPY"))
        if ticker != st.session_state.get("_table_ticker"):
            st.session_state["_table_ticker"] = ticker
            st.session_state["_table_exp"] = None  # reset exp on ticker change

        # Pull expirations directly from provider
        raw_records, spot, expirations = _fetch_provider_chain_and_expirations(ticker)
        if not expirations:
            st.warning("Провайдер не вернул список экспираций.")
            return

        default_exp = st.session_state.get("_table_exp") or expirations[0]
        exp_selected = st.selectbox("Экспирация", options=expirations, index=max(0, expirations.index(default_exp) if default_exp in expirations else 0))
        st.session_state["_table_exp"] = exp_selected

    if not raw_records or not spot:
        st.info("Нет сырых данных от провайдера или не определена текущая цена.")
        return

    # Build final tables using ONLY provider data (independent of sidebar/session df_corr)
    cfg = FinalTableConfig(scale_millions=True)
    tables = process_from_raw(raw_records, spot, final_cfg=cfg)
    if not tables:
        st.info("Не удалось собрать таблицу по данным провайдера.")
        return

    # Use selected expiration (fallback to first available if needed)
    table = tables.get(exp_selected) or tables[list(tables.keys())[0]]
    st.dataframe(table, use_container_width=True)

    st.download_button("Скачать CSV", data=table.to_csv(index=False).encode("utf-8"),
                       file_name=f"final_table_{ticker}_{exp_selected}.csv", mime="text/csv")

    try:
        parquet_bytes = table.to_parquet(index=False)
        st.download_button("Скачать Parquet", data=parquet_bytes,
                           file_name=f"final_table_{ticker}_{exp_selected}.parquet", mime="application/octet-stream")
    except Exception:
        pass

    # Raw provider CSV (for the same expiration) using sanitize_window.build_raw_table
    try:
        from sanitize_window import build_raw_table, SanitizerConfig
        df_raw = build_raw_table(raw_records, S=float(spot), cfg=SanitizerConfig())
        if "exp" in df_raw.columns:
            df_raw = df_raw[df_raw["exp"] == exp_selected]
        st.download_button("Скачать исходную таблицу провайдера (CSV)",
                           data=df_raw.to_csv(index=False).encode("utf-8"),
                           file_name=f"provider_raw_{ticker}_{exp_selected}.csv", mime="text/csv")
    except Exception as e:
        st.warning(f"Не удалось подготовить исходную таблицу провайдера: {e}")
