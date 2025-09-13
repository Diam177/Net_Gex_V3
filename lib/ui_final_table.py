import re
import streamlit as st
import pandas as pd
from typing import Iterable

from lib.final_table import get_final_table_cached

ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}$")

# ----- helpers ---------------------------------------------------------------

def _as_list(obj) -> list:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return [obj]

def _looks_like_iso_date(s: str) -> bool:
    return bool(ISO_DATE_RE.fullmatch(s.strip()))

def _scan_session_for_expirations() -> list[str]:
    # 1) Named keys we expect
    for k in ("expirations_multiselect", "expirations", "Expiration", "Expirations"):
        v = st.session_state.get(k)
        if isinstance(v, (list, tuple)) and v:
            vals = [str(x) for x in v if isinstance(x, (str, int))]
            dates = [x for x in vals if _looks_like_iso_date(str(x))]
            if dates:
                return dates

    # 2) Fallback: scan *all* session_state values for any iterable of ISO dates
    for v in st.session_state.values():
        if isinstance(v, (list, tuple, set)) and v:
            vals = [str(x) for x in v if isinstance(x, (str, int))]
            dates = [x for x in vals if _looks_like_iso_date(str(x))]
            if len(dates) >= 1:
                return dates

    # 3) As a last resort, look for a single ISO date string value
    for v in st.session_state.values():
        if isinstance(v, str) and _looks_like_iso_date(v):
            return [v]

    return []

def _get_current_expirations_from_state() -> list[str]:
    return _scan_session_for_expirations()

def _get_current_ticker_from_state() -> str | None:
    candidate_keys = (
        "ticker", "Ticker", "selected_ticker", "ticker_input",
        "symbol", "Symbol", "asset", "Asset"
    )
    for k in candidate_keys:
        v = st.session_state.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()

    # Heuristic scan (short uppercase-like token)
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
    """Render final table for a SINGLE selected expiration.

    Robust discovery:
    - Ticker/expirations/provider can be omitted and will be inferred from session_state (including heuristic scan).
    - Cache key includes all params to avoid stale tables.
    - No internal defaulting of expiration beyond user choice.
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

    # Initialize or correct selection
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
