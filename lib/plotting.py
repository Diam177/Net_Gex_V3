
from __future__ import annotations

from typing import Dict, Iterable, Optional
import numpy as np
import plotly.graph_objects as go


# ---- Palette (kept identical usage-wise; tune only colors) ----
COLORS = {
    "net_pos": "#25B8FF",   # positive Net GEX bars
    "net_neg": "#E5534B",   # negative Net GEX bars
    "put_oi":  "#7B1030",   # maroon
    "call_oi": "#1FC87B",   # green
    "put_vol": "#8A5A00",   # brown
    "call_vol":"#1C7ED6",   # blue
    "ag":      "#A47BFF",   # violet
    "pz":      "#FFD21A",   # yellow
    "pz_fp":   "#C8CDD3",   # light gray
}

def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _mk_hover_template(series_tail: str) -> str:
    # Common hover "table" for line series; we keep content intact and only color via hoverlabel.
    return (
        "<b>Strike:</b> %{x}<br>"
        "<b>Call OI:</b> %{customdata[0]}<br>"
        "<b>Put OI:</b> %{customdata[1]}<br>"
        "<b>Call Volume:</b> %{customdata[2]}<br>"
        "<b>Put Volume:</b> %{customdata[3]}<br>"
        f"<b>{series_tail}:</b> %{y}<extra></extra>"
    )


def make_figure(
    strikes: Iterable,
    net_gex: Iterable,
    toggles: Dict[str, bool],
    series_dict: Dict[str, np.ndarray],
    price: Optional[float] = None,
    ticker: Optional[str] = None,
) -> go.Figure:
    """Build main chart. Only hover label colors have been modified per request."""
    x = np.asarray(strikes)
    gex = np.asarray(net_gex, dtype=float)

    fig = go.Figure()

    # ---- Net GEX bars (split into negative/positive) ----
    neg_mask = gex < 0
    pos_mask = gex > 0

    if np.any(neg_mask):
        fig.add_bar(
            x=x[neg_mask],
            y=gex[neg_mask],
            name="Net Gex -",
            marker_color=COLORS["net_neg"],
            hovertemplate="<b>Strike:</b> %{x}<br><b>Net GEX:</b> %{y}<extra></extra>",
            hoverlabel=dict(
                bgcolor=COLORS["net_neg"],   # (4) match bar color
                bordercolor="white",         # (7) white border
                font=dict(color="white"),    # keep white text for negative
            ),
        )

    if np.any(pos_mask):
        fig.add_bar(
            x=x[pos_mask],
            y=gex[pos_mask],
            name="Net Gex +",
            marker_color=COLORS["net_pos"],
            hovertemplate="<b>Strike:</b> %{x}<br><b>Net GEX:</b> %{y}<extra></extra>",
            hoverlabel=dict(
                bgcolor=COLORS["net_pos"],   # (5) match bar color
                bordercolor="white",         # (7)
                font=dict(color="black"),    # (6) black text for positive
            ),
        )

    # Convenience: build customdata matrix used by all line traces
    # order: [Call OI, Put OI, Call Volume, Put Volume]
    cd_cols = [
        np.asarray(series_dict.get("call_oi", np.zeros_like(gex)), dtype=float),
        np.asarray(series_dict.get("put_oi", np.zeros_like(gex)), dtype=float),
        np.asarray(series_dict.get("call_vol", np.zeros_like(gex)), dtype=float),
        np.asarray(series_dict.get("put_vol", np.zeros_like(gex)), dtype=float),
    ]
    customdata = np.vstack(cd_cols).T

    def add_line(key: str, name: str, color_key: str, show: bool, tail_field: str, text_black: bool = False):
        if not show:
            return
        y = np.asarray(series_dict.get(key, np.zeros_like(gex)), dtype=float)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode="lines+markers",
                line=dict(color=COLORS[color_key], width=1.2, shape="spline", smoothing=1.3),
                marker=dict(color=COLORS[color_key], size=4, line=dict(width=0)),
                fill="tozeroy",
                fillcolor=_rgba(COLORS[color_key], 0.18),
                customdata=customdata,
                hovertemplate=_mk_hover_template(tail_field),
                hoverlabel=dict(
                    bgcolor=COLORS[color_key],         # same as line color
                    bordercolor="white",                # (7) white border
                    font=dict(color="black" if text_black else "white"),  # (1)(2)(3)
                ),
                yaxis="y2",  # these line series typically mapped to secondary axis; retain
            )
        )

    # Lines for each toggle. Only text color exceptions: Call OI, PZ, PZ_FP.
    add_line("put_oi",   "Put OI",        "put_oi",   toggles.get("put_oi", False),   "Put OI")
    add_line("call_oi",  "Call OI",       "call_oi",  toggles.get("call_oi", False),  "Call OI",  text_black=True)  # (1)
    add_line("put_vol",  "Put Volume",    "put_vol",  toggles.get("put_vol", False),  "Put Volume")
    add_line("call_vol", "Call Volume",   "call_vol", toggles.get("call_vol", False), "Call Volume")
    add_line("ag",       "AG",            "ag",       toggles.get("ag", False),       "AG")
    add_line("pz",       "PZ",            "pz",       toggles.get("pz", False),       "PZ", text_black=True)        # (2)
    add_line("pz_fp",    "PZ_FP",         "pz_fp",    toggles.get("pz_fp", False),    "PZ_FP", text_black=True)     # (3)

    # Price line (unchanged except we keep previously chosen width/format)
    if price is not None:
        fig.add_vline(
            x=price,
            line_color="#FFAE00",
            line_width=1.2,
            annotation_text=f"Price: {price:.2f}",
            annotation_position="top",
            annotation_font_color="#FFAE00",
        )

    # Axes & layout (left intact)
    fig.update_layout(
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=70, r=70, t=40, b=60),
        template="plotly_dark",
        xaxis=dict(title="Strike"),
        yaxis=dict(title="Net GEX"),
        yaxis2=dict(overlaying="y", side="right", title="Other series"),
    )

    # Make hover label border white globally as a fallback
    fig.update_traces(hoverlabel=dict(bordercolor="white"), selector=dict())

    # Chart title (if provided)
    if ticker:
        fig.update_layout(title=dict(text=ticker, x=0.02, xanchor="left"))

    return fig
