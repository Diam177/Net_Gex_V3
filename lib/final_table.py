import streamlit as st
import pandas as pd
from typing import Optional

# We will try to import robust window/chain builder from compute.
# Different projects may expose different entry points; we guard with fallbacks.
try:
    from lib.compute import build_window_dataframe  # preferred explicit entry
except Exception:
    build_window_dataframe = None  # type: ignore

try:
    from lib.compute import get_window_dataframe  # alt naming
except Exception:
    get_window_dataframe = None  # type: ignore

# As a last resort, we import sanitize/window pipeline and minimal fetch
try:
    from lib.sanitize_window import sanitize_and_window_pipeline  # type: ignore
except Exception:
    sanitize_and_window_pipeline = None  # type: ignore

try:
    from lib.provider import fetch_chain  # generic provider facade expected by the project
except Exception:
    fetch_chain = None  # type: ignore


def _materialize_window_df(ticker: str, expiration: str, provider: str) -> pd.DataFrame:
    """Obtain the per-strike base window dataframe for a *specific* expiration.

    Absolutely no internal defaulting: 'expiration' must be used as passed.
    """
    # Preferred API
    if build_window_dataframe is not None:
        return build_window_dataframe(ticker=ticker, expiration=expiration, provider=provider)  # type: ignore

    if get_window_dataframe is not None:
        return get_window_dataframe(ticker=ticker, expiration=expiration, provider=provider)  # type: ignore

    # Fallback path: fetch + sanitize pipeline if exposed
    if fetch_chain is not None and sanitize_and_window_pipeline is not None:
        raw = fetch_chain(ticker=ticker, expiration=expiration, provider=provider)  # type: ignore
        return sanitize_and_window_pipeline(raw)  # type: ignore

    raise RuntimeError(
        "Не найдено подходящего entry-point для построения окна (ожидались: "
        "compute.build_window_dataframe | compute.get_window_dataframe | fetch_chain+sanitize_and_window_pipeline)."
    )


@st.cache_data(show_spinner=False)
def get_final_table_cached(ticker: str, expiration: str, scale_musd: float, provider: str) -> pd.DataFrame:
    """Return the final table for the specific expiration.

    Cache key includes ALL params to avoid stale tables when switching expirations.
    The function assumes downstream logic (NetGEX/AG, PZ/ER, call_vol/put_vol) is applied
    in the window builder or in subsequent steps of your pipeline.
    """
    df = _materialize_window_df(ticker=ticker, expiration=expiration, provider=provider)

    # If scaling columns exist, compute their _M variants deterministically
    # without changing your original logic (idempotent guard).
    if "NetGEX_1pct" in df.columns and "NetGEX_1pct_M" not in df.columns:
        try:
            # scale_musd is "million $ per 1% move". Convert to million scale factor carefully.
            # Your original UI treats it as a divisor; keep that behavior.
            denom = (scale_musd / 1_000_000.0) if scale_musd else 1.0
            df["NetGEX_1pct_M"] = df["NetGEX_1pct"] / denom
        except Exception:
            pass

    if "AG_1pct" in df.columns and "AG_1pct_M" not in df.columns:
        try:
            denom = (scale_musd / 1_000_000.0) if scale_musd else 1.0
            df["AG_1pct_M"] = df["AG_1pct"] / denom
        except Exception:
            pass

    # Ensure expected volume columns exist (no-ops if already present)
    for col in ("call_vol", "put_vol"):
        if col not in df.columns and all(c in df.columns for c in ("side", "volume")):
            # pivot if raw side/volume present
            try:
                piv = df.pivot_table(values="volume", index=df.index, columns="side", aggfunc="sum").fillna(0)
                if "call" in piv:
                    df["call_vol"] = piv["call"].values
                if "put" in piv:
                    df["put_vol"] = piv["put"].values
            except Exception:
                pass

    # Normalize column order (only if columns exist; we won't drop anything)
    preferred_order = [
        "exp", "K", "S", "F",
        "call_oi", "put_oi", "call_vol", "put_vol",
        "dg1pct_call", "dg1pct_put",
        "AG_1pct", "NetGEX_1pct",
        "AG_1pct_M", "NetGEX_1pct_M",
        "PZ", "ER_Up", "ER_Down"
    ]
    existing = [c for c in preferred_order if c in df.columns]
    others = [c for c in df.columns if c not in existing]
    df = df[existing + others]

    return df
