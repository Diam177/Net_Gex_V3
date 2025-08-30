# lib/plotting.py (robust + legacy API + ΔOI window + debug helpers)
from __future__ import annotations

import math
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Visual palette & specs (match your reference screenshots) ---
PALETTE = {
    "bg": "#0b0b0e",
    "fg": "#e6e6eb",
    "grid": "rgba(255,255,255,0.12)",
    "net_pos": "#39c5ff",
    "net_neg": "#ff4b4b",
    "price":  "#ffae00",
    "put_vol": "#f79b14",
    "call_vol": "#1f6de0",
    "put_oi": "#6b0f1a",
    "call_oi": "#00b15b",
    "ag": "#a47cff",
    "pz": "#d1ff00",
    "pz_fp": "#00ffd1",
}

LINE_SPECS = {
    "width_main": 2.2,
    "width_aux": 1.8,
    "smoothing": 0.35,
    "marker_size": 5,
    "area_opacity": 0.28,
    "bar_opacity": 0.95,
    "bar_gap": 0.15,
}

# --- Normalization helpers ---
_NORM_MAP = {
    # strike
    "strike": "strike", "Strike": "strike", "K": "strike",
    # net gex
    "net gex": "net_gex", "Net Gex": "net_gex", "Net GEX": "net_gex",
    "net_gex": "net_gex", "NET_GEX": "net_gex",
    # OI
    "Put OI": "put_oi", "put_oi": "put_oi", "PUT_OI": "put_oi",
    "Call OI": "call_oi", "call_oi": "call_oi", "CALL_OI": "call_oi",
    # Volume
    "Put Volume": "put_volume", "put_volume": "put_volume", "PUT_VOLUME": "put_volume",
    "Call Volume": "call_volume", "call_volume": "call_volume", "CALL_VOLUME": "call_volume",
    # AG / PZ
    "AG": "ag", "ag": "ag",
    "PZ": "pz", "pz": "pz",
    "PZ_FP": "pz_fp", "PZ FP": "pz_fp", "pz_fp": "pz_fp",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        key = _NORM_MAP.get(str(c), None) or _NORM_MAP.get(str(c).strip(), None)
        if key is not None:
            ren[c] = key
    if ren:
        df = df.rename(columns=ren)
    return df

def _as_array(x):
    try:
        return np.asarray(x)
    except Exception:
        return None

def _pick_series(df: pd.DataFrame, candidates):
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None

def _infer_step(strikes: np.ndarray) -> float:
    if strikes.size < 2:
        return 1.0
    diffs = np.diff(np.unique(np.sort(strikes)))
    return float(np.median(diffs)) if diffs.size else 1.0

def _subset_strikes(df: pd.DataFrame, spot: Optional[float], max_per_side: int) -> pd.DataFrame:
    if spot is None or not np.isfinite(spot):
        return df.copy()
    strikes = np.asarray(df["strike"].values, dtype=float)
    step = _infer_step(strikes)
    left = spot - max_per_side * step
    right = spot + max_per_side * step
    out = df[(df["strike"] >= left) & (df["strike"] <= right)].copy()
    return out if not out.empty else df.copy()

def _bar_widths(strikes: np.ndarray) -> np.ndarray:
    step = _infer_step(strikes)
    width = step * (1.0 - LINE_SPECS["bar_gap"])
    return np.full_like(strikes, width, dtype=float)

# --- Core: deterministic contiguous ATM-window by |ΔOI| ---
def _select_window_by_delta_oi(
    df: pd.DataFrame,
    spot: Optional[float],
    p: float = 0.95,
    q: float = 0.05,
    nmin: int = 15,
    nmax: int = 45,
) -> pd.DataFrame:
    call = _pick_series(df, ["call_oi", "Call OI", "callOI", "call_oi_total"])
    put  = _pick_series(df, ["put_oi", "Put OI", "putOI", "put_oi_total"])
    if spot is None or call is None or put is None:
        return df.copy()

    work = df[["strike"]].copy()
    w = (call.fillna(0.0).astype(float) - put.fillna(0.0).astype(float)).abs()
    work["w"] = w.values

    work = work.sort_values("strike").reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    weights = work["w"].values.astype(float)
    strikes = work["strike"].values.astype(float)
    n = len(work)
    if n == 0:
        return df.copy()

    atm_idx = int(np.argmin(np.abs(strikes - float(spot))))

    w_total = float(np.nansum(np.abs(weights))) or 1.0
    w_max = float(np.nanmax(np.abs(weights))) if n > 0 else 1.0
    tail_thr = q * w_max

    left = right = atm_idx

    def coverage_ok(lo: int, hi: int) -> bool:
        return float(np.nansum(np.abs(weights[lo:hi+1]))) >= p * w_total

    def tails_attenuated(lo: int, hi: int) -> bool:
        k = 3
        # left edge weights: lo .. lo+k-1
        left_slice = weights[lo:min(lo+k, hi+1)]
        # right edge weights: hi-k+1 .. hi
        right_slice = weights[max(lo, hi-k+1):hi+1]
        left_ok = (np.nanmean(np.abs(left_slice)) <= tail_thr) if left_slice.size else True
        right_ok = (np.nanmean(np.abs(right_slice)) <= tail_thr) if right_slice.size else True
        return left_ok and right_ok

    # Expand symmetrically
    while True:
        length = right - left + 1
        if length >= nmax or coverage_ok(left, right) or tails_attenuated(left, right):
            break
        expand_left = left > 0
        expand_right = right < n - 1
        if not expand_left and not expand_right:
            break
        if expand_left and expand_right:
            left -= 1; right += 1
        elif expand_left:
            left -= 1
        else:
            right += 1

    while (right - left + 1) < nmin and (left > 0 or right < n - 1):
        if left > 0:
            left -= 1
        if (right - left + 1) < nmin and right < n - 1:
            right += 1

    rows = work.iloc[left:right+1]["_orig_idx"].tolist()
    out = df.iloc[rows].copy().sort_values("strike").reset_index(drop=True)
    return out

# --- Public debug helper for selection window ---
def selection_window_debug(df: pd.DataFrame | None = None, *, spot: Optional[float] = None,
                           call_oi=None, put_oi=None,
                           p: float = 0.95, q: float = 0.05, nmin: int = 15, nmax: int = 45) -> Dict[str, Any]:
    """
    Return diagnostics of the ΔOI window selection.
    Accepts either a DataFrame with columns ('strike','call_oi','put_oi') or arrays via call_oi/put_oi.
    """
    if df is not None:
        df = _normalize_columns(df.copy())
        if "strike" not in df.columns:
            raise ValueError("selection_window_debug: df must contain 'strike'")
        co = _pick_series(df, ["call_oi", "Call OI", "callOI"])
        po = _pick_series(df, ["put_oi", "Put OI", "putOI"])
        strikes = df["strike"].astype(float).values
        call = co.values if co is not None else None
        put = po.values if po is not None else None
    else:
        strikes = None

    if strikes is None:
        raise ValueError("selection_window_debug: provide df with 'strike' and OI columns")

    # compute weights
    call = np.asarray(call, dtype=float)
    put = np.asarray(put, dtype=float)
    w = np.abs(call - put)
    order = np.argsort(strikes)
    strikes_s = strikes[order]
    w_s = w[order]

    n = len(strikes_s)
    atm_idx = int(np.argmin(np.abs(strikes_s - float(spot)))) if spot is not None else None
    w_total = float(np.nansum(w_s)) or 1.0
    w_max = float(np.nanmax(w_s)) if n > 0 else 1.0
    tail_thr = q * w_max

    # simulate expansion to capture exact bounds and metrics
    if atm_idx is None:
        return {"error": "spot is None", "columns": list(df.columns)}

    lo = hi = atm_idx

    def coverage(lo, hi): return float(np.nansum(w_s[lo:hi+1])) / w_total
    def edge_means(lo, hi):
        k=3
        left_slice = w_s[lo:min(lo+k, hi+1)]
        right_slice = w_s[max(lo, hi-k+1):hi+1]
        return float(np.nanmean(left_slice)) if left_slice.size else None, float(np.nanmean(right_slice)) if right_slice.size else None

    stop_reason = None
    while True:
        length = hi - lo + 1
        cov = coverage(lo, hi)
        lm, rm = edge_means(lo, hi)
        tails_ok = ((lm is None or lm <= tail_thr) and (rm is None or rm <= tail_thr))

        if length >= nmax:
            stop_reason = "nmax"; break
        if cov >= p:
            stop_reason = "coverage"; break
        if tails_ok:
            stop_reason = "tails"; break

        expand_left = lo > 0
        expand_right = hi < n - 1
        if not expand_left and not expand_right:
            stop_reason = "exhausted"; break
        if expand_left and expand_right:
            lo -= 1; hi += 1
        elif expand_left:
            lo -= 1
        else:
            hi += 1

    while (hi - lo + 1) < nmin and (lo > 0 or hi < n - 1):
        if lo > 0:
            lo -= 1
        if (hi - lo + 1) < nmin and hi < n - 1:
            hi += 1

    return {
        "spot": float(spot) if spot is not None else None,
        "atm_strike": float(strikes_s[atm_idx]) if atm_idx is not None else None,
        "bounds_idx": [int(lo), int(hi)],
        "bounds_strikes": [float(strikes_s[lo]), float(strikes_s[hi])],
        "window_len": int(hi - lo + 1),
        "coverage_fraction": coverage(lo, hi),
        "edge_means": {"left": float(np.nanmean(w_s[lo:min(lo+3, hi+1)])),
                       "right": float(np.nanmean(w_s[max(lo, hi-2):hi+1]))},
        "tail_threshold": tail_thr,
        "total_weight": w_total,
        "max_weight": w_max,
    }

# --- Core figure builder (expects normalized df) ---
def _make_figure_core(
    df: pd.DataFrame,
    spot: Optional[float] = None,
    title: Optional[str] = None,
    show_netgex: bool = True,
    show_put_oi: bool = False,
    show_call_oi: bool = False,
    show_put_vol: bool = False,
    show_call_vol: bool = False,
    show_ag: bool = False,
    show_pz: bool = False,
    show_pz_fp: bool = False,
    max_bars_per_side: int = 30,
) -> "go.Figure":
    if "strike" not in df.columns:
        raise ValueError("df must contain 'strike'")
    if "net_gex" not in df.columns and show_netgex:
        raise ValueError("df must contain 'net_gex' when show_netgex=True")

    dfv = _normalize_columns(df.copy()).sort_values("strike").reset_index(drop=True)
    dfv = _select_window_by_delta_oi(dfv, spot) if spot is not None else _subset_strikes(dfv, spot, max_bars_per_side)

    x = dfv["strike"].astype(float).values
    widths = _bar_widths(x)

    fig = go.Figure()

    # Net GEX bars
    if show_netgex and "net_gex" in dfv.columns:
        y = dfv["net_gex"].astype(float).values
        pos_mask = y >= 0
        neg_mask = ~pos_mask

        if np.any(pos_mask):
            fig.add_bar(x=x[pos_mask], y=y[pos_mask], width=widths[pos_mask],
                        name="Net GEX", marker_color=PALETTE["net_pos"],
                        opacity=LINE_SPECS["bar_opacity"], hovertemplate="Strike=%{x}<br>Net GEX=%{y}<extra></extra>",
                        offset=0, showlegend=True)
        if np.any(neg_mask):
            fig.add_bar(x=x[neg_mask], y=y[neg_mask], width=widths[neg_mask],
                        name="Net GEX", marker_color=PALETTE["net_neg"],
                        opacity=LINE_SPECS["bar_opacity"], hovertemplate="Strike=%{x}<br>Net GEX=%{y}<extra></extra>",
                        offset=0, showlegend=False)

    # Helper to add secondary lines/areas
    def add_aux(col: str, label: str, color_key: str, area=True, smoothing=LINE_SPECS["smoothing"]):
        if col not in dfv.columns:
            return
        y = dfv[col].astype(float).values
        if not np.any(np.isfinite(y)):  # all NaN or zeros
            return
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label, mode="lines+markers",
            line=dict(width=LINE_SPECS["width_aux"],
                      shape="spline", smoothing=smoothing,
                      color=PALETTE[color_key]),
            marker=dict(size=LINE_SPECS["marker_size"], color=PALETTE[color_key]),
            fill="tozeroy" if area else None,
            opacity=LINE_SPECS["area_opacity"] if area else 1.0,
            yaxis="y2",
            hovertemplate=f"Strike=%{{x}}<br>{label}=%{{y}}<extra></extra>",
        ))

    if show_put_vol:  add_aux("put_volume", "Put Volume", "put_vol", area=True)
    if show_call_vol: add_aux("call_volume", "Call Volume", "call_vol", area=True)
    if show_put_oi:   add_aux("put_oi", "Put OI", "put_oi", area=True)
    if show_call_oi:  add_aux("call_oi", "Call OI", "call_oi", area=True)
    if show_ag:       add_aux("ag", "AG", "ag", area=True)
    if show_pz:       add_aux("pz", "PZ", "pz", area=False, smoothing=0.0)
    if show_pz_fp and "pz_fp" in dfv.columns:
        y = dfv["pz_fp"].astype(float).values
        fig.add_trace(go.Scatter(
            x=x, y=y, name="PZ_FP", mode="lines+markers",
            line=dict(width=LINE_SPECS["width_aux"], color=PALETTE["pz_fp"], dash="dash"),
            marker=dict(size=LINE_SPECS["marker_size"], color=PALETTE["pz_fp"]),
            yaxis="y2",
            hovertemplate="Strike=%{x}<br>PZ_FP=%{y}<extra></extra>",
        ))

    if spot is not None and math.isfinite(spot):
        fig.add_vline(x=spot, line=dict(color=PALETTE["price"], width=2.0),
                      annotation_text=f"Price: {spot:.2f}",
                      annotation_position="top",
                      annotation_font_color=PALETTE["price"])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["fg"], size=14,
                  family="Inter, Segoe UI, system-ui, -apple-system, sans-serif"),
        barmode="overlay", bargap=0.0,
        margin=dict(l=70, r=70, t=60, b=90),
        title=dict(text=title or "", x=0.01, xanchor="left", y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Net GEX", gridcolor=PALETTE["grid"],
                     zeroline=True, zerolinewidth=1, zerolinecolor=PALETTE["grid"],
                     rangemode="tozero", showline=False)
    fig.update_layout(yaxis2=dict(title="Other series", overlaying="y", side="right",
                                  showgrid=False, zeroline=False))

    strikes_unique = np.unique(x)
    if strikes_unique.size > 30:
        every = max(1, int(round(strikes_unique.size / 30)))
        tickvals = strikes_unique[::every]
    else:
        tickvals = strikes_unique
    fig.update_xaxes(title_text="Strike", tickmode="array", tickvals=tickvals,
                     ticktext=[f"{v:g}" for v in tickvals], tickangle=60,
                     showgrid=False, mirror=True, linecolor=PALETTE["grid"])

    return fig

# --- Public API wrapper (DataFrame or legacy arrays) ---
def make_figure(*args, **kwargs):
    # New API: df=DataFrame or positional first arg DataFrame
    spot = kwargs.pop("spot", None)
    title = kwargs.pop("title", None)

    if len(args) == 1 and isinstance(args[0], pd.DataFrame):
        df = _normalize_columns(args[0].copy())
        return _make_figure_core(df, spot=spot, title=title, **kwargs)

    if "df" in kwargs and isinstance(kwargs["df"], pd.DataFrame):
        df = _normalize_columns(kwargs.pop("df").copy())
        return _make_figure_core(df, spot=spot, title=title, **kwargs)

    # Legacy: (strike_array, net_gex_array, toggles_dict, series_dict, **opts)
    if len(args) >= 4:
        strike_arr = _as_array(args[0])
        netgex_arr = _as_array(args[1])
        toggles = args[2] if isinstance(args[2], dict) else {}
        series = args[3] if isinstance(args[3], dict) else {}

        data = {"strike": strike_arr, "net_gex": netgex_arr}
        # optional series
        for k, v in series.items():
            key = _NORM_MAP.get(str(k), None) or _NORM_MAP.get(str(k).strip(), None)
            if key in {"put_oi","call_oi","put_volume","call_volume","ag","pz","pz_fp"}:
                data[key] = _as_array(v)

        if spot is None:
            spot = series.get("spot")
        if title is None:
            title = series.get("title")

        df_leg = pd.DataFrame({k: v for k, v in data.items() if v is not None})
        df_leg = _normalize_columns(df_leg)

        def t(name, default=False): return bool(toggles.get(name, default))
        return _make_figure_core(
            df_leg, spot=spot, title=title,
            show_netgex=t("Net Gex", True),
            show_put_oi=t("Put OI"),
            show_call_oi=t("Call OI"),
            show_put_vol=t("Put Volume"),
            show_call_vol=t("Call Volume"),
            show_ag=t("AG"),
            show_pz=t("PZ"),
            show_pz_fp=t("PZ_FP"),
            max_bars_per_side=int(kwargs.pop("max_bars_per_side", 30)),
        )

    raise ValueError("make_figure: unexpected arguments. Pass a DataFrame (new API) or (strike, net_gex, toggles, series_dict).")
