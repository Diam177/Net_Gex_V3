# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _to_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def build_provider_strike_df(block: dict) -> pd.DataFrame:
    """
    block: {"calls":[...], "puts":[...]}, элементы с полями:
    contractSymbol, strike, lastPrice, openInterest, volume, impliedVolatility
    """
    calls = block.get("calls", []) or []
    puts  = block.get("puts", []) or []

    def collect(lst, side: str):
        data = {}
        for r in lst:
            k = _to_float(r.get("strike"))
            if k is None:
                continue
            data[k] = {
                f"{side}_symbol": r.get("contractSymbol"),
                f"{side}_last": _to_float(r.get("lastPrice")),
                f"{side}_oi": _to_int(r.get("openInterest")),
                f"{side}_vol": _to_int(r.get("volume")),
                f"{side}_iv": _to_float(r.get("impliedVolatility")),
            }
        return data

    c = collect(calls, "call")
    p = collect(puts, "put")

    strikes = sorted(set(c.keys()) | set(p.keys()))
    rows = []
    for k in strikes:
        row = {"strike": float(k)}
        row.update(c.get(k, {"call_symbol": None, "call_last": None, "call_oi": 0, "call_vol": 0, "call_iv": None}))
        row.update(p.get(k, {"put_symbol": None, "put_last": None, "put_oi": 0, "put_vol": 0, "put_iv": None}))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

def render_provider_strike_table(blocks_by_date: dict, selected_exp: int):
    block = blocks_by_date.get(selected_exp)
    if not block:
        st.info("Нет данных провайдера для выбранной экспирации.")
        return None

    st.subheader("Provider data — per strike")
    df = build_provider_strike_df(block)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download provider table (CSV)",
        data=csv,
        file_name=f"provider_strikes_{selected_exp}.csv",
        mime="text/csv"
    )
    return df
