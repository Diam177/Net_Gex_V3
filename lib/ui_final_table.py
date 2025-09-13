import streamlit as st
import pandas as pd

# We import the cached table builder from final_table
from lib.final_table import get_final_table_cached

def _get_current_expirations_from_state() -> list[str]:
    exps = st.session_state.get("expirations_multiselect")
    if isinstance(exps, list) and exps:
        return [str(x) for x in exps]
    return []

def _get_current_ticker_from_state() -> str | None:
    for k in ("ticker", "selected_ticker", "ticker_input"):
        v = st.session_state.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    return None

def _get_current_provider_from_state(default: str = "polygon") -> str:
    for k in ("data_provider", "provider", "selected_provider"):
        v = st.session_state.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return default

def render_final_table(
    ticker: str | None = None,
    expirations: list[str] | None = None,
    scale_musd: float = 1_000_000.0,
    provider: str | None = None,
    title: str = "Финальная таблица (окно, NetGEX/AG, PZ/ER)",
) -> None:
    """Renders the final per-strike table for a SINGLE selected expiration.

    Key points:
    - Accepts a *list* of expirations (from the sidebar multi-select).
    - Ensures a deterministic single 'exp_for_table' via a selectbox bound to session state.
    - Uses cache keyed by (ticker, expiration, scale_musd, provider).
    - No hidden defaulting inside compute/aggregation layers.
    - To minimize integration friction, 'ticker' and 'provider' are optional and can be read from session_state.
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

    # Deterministic single selection visible above the table
    exp_for_table = st.selectbox(
        "Экспирация",
        options=expirations,
        index=expirations.index(st.session_state.exp_for_table),
        key="exp_for_table_select",
    )
    st.session_state.exp_for_table = exp_for_table

    # Build table strictly for the chosen expiration
    df: pd.DataFrame = get_final_table_cached(
        ticker=ticker,
        expiration=exp_for_table,
        scale_musd=float(scale_musd),
        provider=provider,
    )

    # Display
    st.dataframe(df, use_container_width=True)

    # CSV download for the current selection
    st.download_button(
        "Скачать CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"final_table_{ticker}_{exp_for_table}.csv",
        mime="text/csv",
    )
