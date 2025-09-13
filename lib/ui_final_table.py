import re
import streamlit as st
import pandas as pd

from lib.final_table import get_final_table_cached

# ----- helpers ---------------------------------------------------------------

def _get_current_expirations_from_state() -> list[str]:
    # Common keys for multiselect of expirations
    for k in ("expirations_multiselect", "expirations", "Expiration", "Expirations"):
        exps = st.session_state.get(k)
        if isinstance(exps, (list, tuple)) and len(exps) > 0:
            return [str(x) for x in exps]
    return []

def _get_current_ticker_from_state() -> str | None:
    # Try a list of common keys used across different app versions
    candidate_keys = (
        "ticker", "Ticker", "selected_ticker", "ticker_input",
        "symbol", "Symbol", "asset", "Asset"
    )
    for k in candidate_keys:
        v = st.session_state.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()

    # Safe fallback: scan session_state values that look like a ticker (A-Z, dot/hyphen, 1..12 chars)
    # but skip known non-ticker strings.
    skip = {"POLYGON", "RAPIDAPI", "DATA RECEIVED", "KEY LEVELS", "DOWNLOAD JSON"}
    for v in st.session_state.values():
        if isinstance(v, str):
            s = v.strip().upper()
            if s in skip:
                continue
            if re.fullmatch(r"[A-Z][A-Z0-9\.\-]{0,11}", s):
                return s
    return None

def _get_current_provider_from_state(default: str = "polygon") -> str:
    for k in ("data_provider", "provider", "selected_provider", "Data provider"):
        v = st.session_state.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return default

# ----- main ------------------------------------------------------------------

def render_final_table(
    ticker: str | None = None,
    expirations: list[str] | None = None,
    scale_musd: float = 1_000_000.0,
    provider: str | None = None,
    title: str = "Финальная таблица (окно, NetGEX/AG, PZ/ER)",
) -> None:
    """
    Renders the final per-strike table for a SINGLE selected expiration.

    Robust to different app wiring:
    - Ticker/expirations/provider can be omitted and will be inferred from session_state.
    - Cache is keyed by (ticker, expiration, scale_musd, provider).
    - No internal defaulting/overrides of expiration.
    """
    st.subheader(title)

    if ticker is None:
        ticker = _get_current_ticker_from_state()
    if not ticker:
        st.warning("Не передан ticker и он не найден в session_state.")
        return

    if provider is None:
        provider = _get_current_provider_from_state()

    if not expirations:
        expirations = _get_current_expirations_from_state()

    if not expirations:
        st.info("Выберите хотя бы одну экспирацию слева.")
        return

    # Initialize selection in session state if missing or stale
    if "exp_for_table" not in st.session_state or st.session_state.exp_for_table not in expirations:
        st.session_state.exp_for_table = expirations[0]

    exp_for_table = st.selectbox(
        "Экспирация",
        options=expirations,
        index=expirations.index(st.session_state.exp_for_table),
        key="exp_for_table_select",
    )
    st.session_state.exp_for_table = exp_for_table

    df: pd.DataFrame = get_final_table_cached(
        ticker=ticker,
        expiration=exp_for_table,
        scale_musd=float(scale_musd),
        provider=provider,
    )

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Скачать CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"final_table_{ticker}_{exp_for_table}.csv",
        mime="text/csv",
    )
