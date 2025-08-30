# lib/plotting.py
from __future__ import annotations

import math
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# --- Visual style (colors & sizes) tuned to match your reference screenshots ---
PALETTE = {
    "bg": "#0b0b0e",
    "fg": "#e6e6eb",
    "grid": "rgba(255,255,255,0.12)",
    "net_pos": "#39c5ff",   # light sky blue
    "net_neg": "#ff4b4b",   # soft red
    "price":  "#ffae00",    # orange for price line
    "put_vol": "#f79b14",   # amber/orange
    "call_vol": "#1f6de0",  # azure/blue
    "put_oi": "#6b0f1a",    # burgundy
    "call_oi": "#00b15b",   # green
    "ag": "#a47cff",        # purple
    "pz": "#d1ff00",        # lime for PZ
    "pz_fp": "#00ffd1",     # aqua for PZ_FP
}

LINE_SPECS = {
    "width_main": 2.2,
    "width_aux": 1.8,
    "smoothing": 0.35,   # purely visual; we keep markers so raw points are visible
    "marker_size": 5,
    "area_opacity": 0.28,
    "bar_opacity": 0.95,
    "bar_gap": 0.15,     # 0..1
}


def _infer_step(strikes: np.ndarray) -> float:
    """Infer typical increment between strikes to size bar widths and x-window."""
    if strikes.size < 2:
        return 1.0
    diffs = np.diff(np.unique(np.sort(strikes)))
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))



def _pick_series(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None


def _select_window_by_delta_oi(
    df: pd.DataFrame,
    spot: Optional[float],
    p: float = 0.95,     # coverage threshold
    q: float = 0.05,     # tail attenuation fraction of max |ΔOI|
    nmin: int = 15,      # UI lower bound
    nmax: int = 45,      # UI upper bound
) -> pd.DataFrame:
    """
    Deterministic selection of a *contiguous* window around ATM by the ΔOI methodology:
      ΔOI(K) = Call OI - Put OI. Importance weight = |ΔOI|.
    Expand symmetrically around K_atm (closest strike to spot) until either:
      - coverage >= p of total |ΔOI|, OR
      - both tails are attenuated: mean of 3 edge weights <= q * max_all_weight,
      - or we reach nmax.
    If final length < nmin, pad symmetrically (when possible).
    Falls back to no subsetting if Call/Put OI are unavailable or spot is missing.
    """
    # Need call_oi and put_oi (support several common aliases)
    call = _pick_series(df, ["call_oi", "Call OI", "callOI", "call_oi_total"])
    put  = _pick_series(df, ["put_oi", "Put OI", "putOI", "put_oi_total"])

    if spot is None or call is None or put is None:
        return df.copy()

    work = df[["strike"]].copy()
    w = (call.fillna(0.0).astype(float) - put.fillna(0.0).astype(float)).abs()
    work["w"] = w.values

    # Sort by strike
    work = work.sort_values("strike").reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    weights = work["w"].values.astype(float)
    strikes = work["strike"].values.astype(float)
    n = len(work)
    if n == 0:
        return df.copy()

    # Find ATM index (closest strike to spot)
    atm_idx = int(np.argmin(np.abs(strikes - float(spot))))

    # Precompute totals
    w_total = float(np.nansum(np.abs(weights)))
    w_total = w_total if np.isfinite(w_total) and w_total > 0 else 1.0
    w_max = float(np.nanmax(np.abs(weights))) if n > 0 else 1.0
    tail_thr = q * w_max

    # Start window at ATM
    left = right = atm_idx

    def coverage_ok(lo: int, hi: int) -> bool:
        return float(np.nansum(np.abs(weights[lo:hi+1]))) >= p * w_total

    def tails_attenuated(lo: int, hi: int) -> bool:
        # average of the 3 edge weights on each side (if fewer than 3, take what's available)
        k = 3
        left_slice = weights[max(lo, hi-k+1):lo+1]  # if window is tiny, this handles gracefully
        # Correct slices: left edge is [lo : lo+k), right edge is (hi-k+1 : hi+1)
        left_slice = weights[lo:min(lo+k, hi+1)]
        right_slice = weights[max(lo, hi-k+1):hi+1]
        left_ok = float(np.nanmean(np.abs(left_slice))) <= tail_thr if left_slice.size else True
        right_ok = float(np.nanmean(np.abs(right_slice))) <= tail_thr if right_slice.size else True
        return left_ok and right_ok

    # Expand symmetrically
    while True:
        # Check stop conditions
        length = right - left + 1
        if length >= nmax or coverage_ok(left, right) or tails_attenuated(left, right):
            break

        # expand one step (prefer symmetrical if both sides available)
        expand_left = left > 0
        expand_right = right < n - 1
        if not expand_left and not expand_right:
            break  # can't expand further

        if expand_left and expand_right:
            left -= 1
            right += 1
        elif expand_left:
            left -= 1
        else:
            right += 1

    # Enforce nmin by padding if possible
    while (right - left + 1) < nmin and (left > 0 or right < n - 1):
        if left > 0:
            left -= 1
        if (right - left + 1) < nmin and right < n - 1:
            right += 1

    # Map back to original df rows
    rows = work.iloc[left:right+1]["_orig_idx"].tolist()
    out = df.iloc[rows].copy().sort_values("strike").reset_index(drop=True)
    return out


def _bar_widths(strikes: np.ndarray) -> np.ndarray:
    step = _infer_step(strikes)
    width = step * (1.0 - LINE_SPECS["bar_gap"])
    return np.full_like(strikes, width, dtype=float)


def _maybe(series: pd.Series | None) -> pd.Series:
    return series if series is not None else pd.Series(dtype=float)


def make_figure(
    df: pd.DataFrame,
    spot: Optional[float] = None,
    title: Optional[str] = None,
    # toggles
    show_netgex: bool = True,
    show_put_oi: bool = False,
    show_call_oi: bool = False,
    show_put_vol: bool = False,
    show_call_vol: bool = False,
    show_ag: bool = False,
    show_pz: bool = False,
    show_pz_fp: bool = False,
    # view params
    max_bars_per_side: int = 30,
) -> "go.Figure":
    """
    Build Plotly figure matching the reference visuals.
    Required df columns:
        'strike', 'net_gex' (others are optional: 'put_oi','call_oi','put_volume','call_volume','ag','pz','pz_fp')
    All series are plotted *as is* without any transformations.
    """
    if "strike" not in df.columns:
        raise ValueError("df must contain 'strike'")
    if "net_gex" not in df.columns and show_netgex:
        raise ValueError("df must contain 'net_gex' when show_netgex=True")

    dfv = df.copy()
    dfv = dfv.sort_values("strike").reset_index(drop=True)
    dfv = _select_window_by_delta_oi(dfv, spot) if spot is not None else _subset_strikes(dfv, spot, max_bars_per_side)

    x = dfv["strike"].astype(float).values
    widths = _bar_widths(x)

    fig = go.Figure()

    # --- Net GEX bars (positive cyan, negative red) on left Y axis ---
    if show_netgex and "net_gex" in dfv.columns:
        y = dfv["net_gex"].astype(float).values
        pos_mask = y >= 0
        neg_mask = ~pos_mask

        if np.any(pos_mask):
            fig.add_bar(
                x=x[pos_mask],
                y=y[pos_mask],
                width=widths[pos_mask],
                name="Net GEX",
                marker_color=PALETTE["net_pos"],
                opacity=LINE_SPECS["bar_opacity"],
                hovertemplate="Strike=%{x}<br>Net GEX=%{y}<extra></extra>",
                offset=0,
                showlegend=True,
            )
        if np.any(neg_mask):
            fig.add_bar(
                x=x[neg_mask],
                y=y[neg_mask],
                width=widths[neg_mask],
                name="Net GEX",
                marker_color=PALETTE["net_neg"],
                opacity=LINE_SPECS["bar_opacity"],
                hovertemplate="Strike=%{x}<br>Net GEX=%{y}<extra></extra>",
                offset=0,
                showlegend=False,  # keep single legend entry
            )

    # --- Helper to add right-axis line/area traces ---
    def add_aux_line(
        col: str,
        label: str,
        color_key: str,
        as_area: bool = True,
        smoothing: float = LINE_SPECS["smoothing"],
    ) -> None:
        if col not in dfv.columns:
            return
        y = dfv[col].astype(float).values
        if not np.any(np.isfinite(y)):
            return
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=label,
                mode="lines+markers",
                line=dict(width=LINE_SPECS["width_aux"], shape="spline", smoothing=smoothing, color=PALETTE[color_key]),
                marker=dict(size=LINE_SPECS["marker_size"], color=PALETTE[color_key]),
                fill="tozeroy" if as_area else None,
                opacity=LINE_SPECS["area_opacity"] if as_area else 1.0,
                yaxis="y2",
                hovertemplate="Strike=%{x}<br>%s=%{y}<extra></extra>" % label,
            )
        )

    # Order: volumes, OI, AG, PZ
    if show_put_vol:
        add_aux_line("put_volume", "Put Volume", "put_vol", as_area=True)
    if show_call_vol:
        add_aux_line("call_volume", "Call Volume", "call_vol", as_area=True)
    if show_put_oi:
        add_aux_line("put_oi", "Put OI", "put_oi", as_area=True)
    if show_call_oi:
        add_aux_line("call_oi", "Call OI", "call_oi", as_area=True)
    if show_ag:
        add_aux_line("ag", "AG", "ag", as_area=True)
    if show_pz:
        add_aux_line("pz", "PZ", "pz", as_area=False, smoothing=0.0)  # PZ typically shown as crisp line
    if show_pz_fp:
        # PZ_FP as dashed
        if "pz_fp" in dfv.columns:
            y = dfv["pz_fp"].astype(float).values
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, name="PZ_FP", mode="lines+markers",
                    line=dict(width=LINE_SPECS["width_aux"], color=PALETTE["pz_fp"], dash="dash"),
                    marker=dict(size=LINE_SPECS["marker_size"], color=PALETTE["pz_fp"]),
                    yaxis="y2",
                    hovertemplate="Strike=%{x}<br>PZ_FP=%{y}<extra></extra>",
                )
            )

    # --- Spot price vertical line ---
    if spot is not None and math.isfinite(spot):
        fig.add_vline(
            x=spot,
            line=dict(color=PALETTE["price"], width=2.0),
            annotation_text=f"Price: {spot:.2f}",
            annotation_position="top",
            annotation_font_color=PALETTE["price"],
        )

    # --- Layout to match screenshots ---
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["fg"], size=14, family="Inter, Segoe UI, system-ui, -apple-system, sans-serif"),
        barmode="overlay",
        bargap=0.0,
        margin=dict(l=70, r=70, t=60, b=90),
        title=dict(text=title or "", x=0.01, xanchor="left", y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )

    # Left axis: Net GEX
    fig.update_yaxes(
        title_text="Net GEX",
        gridcolor=PALETTE["grid"],
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=PALETTE["grid"],
        rangemode="tozero",
        showline=False,
    )

    # Right axis: others
    fig.update_layout(
        yaxis2=dict(
            title="Other series",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        )
    )

    # X axis (strikes): show many ticks and angle them
    strikes_unique = np.unique(x)
    # dynamic dtick to reduce overlap
    if strikes_unique.size > 30:
        # aim for ~30 labels
        every = max(1, int(round(strikes_unique.size / 30)))
        tickvals = strikes_unique[::every]
    else:
        tickvals = strikes_unique

    fig.update_xaxes(
        title_text="Strike",
        tickmode="array",
        tickvals=tickvals,
        ticktext=[f"{v:g}" for v in tickvals],
        tickangle=60,
        showgrid=False,
        mirror=True,
        linecolor=PALETTE["grid"],
    )

    return fig
