
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
from final_table import FinalTableConfig, build_final_tables_from_corr, process_from_raw

# Changes in this version:
# 1) Independent controls for Final Table: you can pick a ticker & expiration inside this section,
#    independent from the sidebar selection.
# 2) A safe "Download raw provider table" button (CSV).
# 3) Robust rendering even if new Greek columns are partially missing.

def _get_provider_tables_by_exp(raw_records, spot, sanitizer_cfg=None, final_cfg=None) -> Dict[str, pd.DataFrame]:
    try:
        return process_from_raw(raw_records, float(spot), sanitizer_cfg=sanitizer_cfg, final_cfg=final_cfg)
    except Exception as e:
        st.warning(f"Не удалось собрать из raw_records: {e}")
        return {}

def _get_tables_from_corr(df_corr, windows, final_cfg=None) -> Dict[str, pd.DataFrame]:
    try:
        scale = 1_000_000.0 if (final_cfg and final_cfg.scale_millions) else 1.0
        return build_final_tables_from_corr(df_corr, windows, scale, final_cfg or FinalTableConfig())
    except Exception as e:
        st.warning(f"Не удалось собрать из df_corr/windows: {e}")
        return {}

def _render_raw_download(selected_exp: Optional[str]):
    try:
        from sanitize_window import build_raw_table, SanitizerConfig
    except Exception:
        return
    if ("raw_records" not in st.session_state) or ("spot" not in st.session_state):
        return
    try:
        df_raw = build_raw_table(st.session_state["raw_records"], S=float(st.session_state["spot"]), cfg=SanitizerConfig())
        if selected_exp and ("exp" in df_raw.columns):
            df_raw = df_raw[df_raw["exp"] == selected_exp]
        st.download_button(
            "Скачать исходную таблицу провайдера (CSV)",
            data=df_raw.to_csv(index=False).encode("utf-8"),
            file_name=f"provider_raw_{selected_exp or 'all'}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"Не удалось подготовить исходную таблицу провайдера: {e}")

def render_final_table(section_title: str = "Финальная таблица (окно, NetGEX/AG, PZ/ER)"):
    st.header(section_title)

    # Independent controls (ticker & expiration) for the table section
    with st.expander("Настройки финальной таблицы (независимо от сайдбара)", expanded=False):
        ticker_default = st.session_state.get("_table_ticker", st.session_state.get("_last_ticker", "SPY"))
        ticker = st.text_input("Ticker (для финальной таблицы)", value=ticker_default, key="table_ticker_input")
        st.session_state["_table_ticker"] = ticker

        # Determine expirations list if available from df_corr or raw tables; fallback to a text input
        expirations = []
        if "df_corr" in st.session_state and "exp" in getattr(st.session_state["df_corr"], "columns", []):
            expirations = sorted(st.session_state["df_corr"]["exp"].dropna().astype(str).unique().tolist())
        elif "raw_records" in st.session_state:
            try:
                from sanitize_window import build_raw_table, SanitizerConfig
                df_raw_all = build_raw_table(st.session_state["raw_records"], S=float(st.session_state.get("spot", 0.0)), cfg=SanitizerConfig())
                if "exp" in df_raw_all.columns:
                    expirations = sorted(df_raw_all["exp"].dropna().astype(str).unique().tolist())
            except Exception:
                pass
        exp_selected = st.selectbox("Экспирация (для финальной таблицы)", options=expirations or ["—"], index=0)

    final_cfg = FinalTableConfig(scale_millions=True)

    # Prefer df_corr/windows pipeline if present
    tables: Dict[str, pd.DataFrame] = {}
    if ("df_corr" in st.session_state) and ("windows" in st.session_state):
        tables = _get_tables_from_corr(st.session_state["df_corr"], st.session_state["windows"], final_cfg)
    elif ("raw_records" in st.session_state) and ("spot" in st.session_state):
        tables = _get_provider_tables_by_exp(st.session_state["raw_records"], st.session_state["spot"], final_cfg=final_cfg)

    if not tables:
        st.info("Нет данных для финальной таблицы. Выберите экспирацию/получите сырые данные.")
        return

    # choose selected table
    tbl = None
    if exp_selected in tables:
        tbl = tables[exp_selected]
    else:
        # pick first available as fallback
        exp_selected = list(tables.keys())[0]
        tbl = tables[exp_selected]

    st.dataframe(tbl, use_container_width=True)

    # download buttons
    st.download_button("Скачать CSV", data=tbl.to_csv(index=False).encode("utf-8"), file_name=f"final_table_{exp_selected}.csv", mime="text/csv")
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        parquet_bytes = tbl.to_parquet(index=False)
        st.download_button("Скачать Parquet", data=parquet_bytes, file_name=f"final_table_{exp_selected}.parquet", mime="application/octet-stream")
    except Exception:
        pass

    _render_raw_download(exp_selected)
