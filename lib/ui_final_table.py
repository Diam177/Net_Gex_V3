import streamlit as st
import pandas as pd

# We import the cached table builder from final_table
from lib.final_table import get_final_table_cached

def _get_current_expirations_from_state() -> list[str]:
    # fallback if the parent app didn't pass expirations explicitly
    exps = st.session_state.get("expirations_multiselect")
    if isinstance(exps, list) and exps:
        # normalize to strings
        return [str(x) for x in exps]
    return []

def render_final_table(
    ticker: str,
    expirations: list[str] | None = None,
    scale_musd: float = 1_000_000.0,
    provider: str = "polygon",
    title: str = "Финальная таблица (окно, NetGEX/AG, PZ/ER)",
) -> None:
    """Renders the final per-strike table for a SINGLE selected expiration.

    Key points:
    - Accepts a *list* of expirations (from the sidebar multi-select).
    - Ensures a deterministic single 'exp_for_table' via a selectbox bound to session state.
    - Uses cache keyed by (ticker, expiration, scale_musd, provider).
    - No hidden defaulting inside compute/aggregation layers.
    """
    st.subheader(title)

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
