
import numpy as np
import plotly.graph_objects as go

POS_COLOR = "#48B4FF"   # голубые положительные
NEG_COLOR = "#FF3B30"   # красные отрицательные

def _select_atm_window(strikes, call_oi, put_oi, S, p=0.95, q=0.05, Nmin=15, Nmax=49):
    strikes = np.asarray(strikes, dtype=float)
    call_oi = np.asarray(call_oi, dtype=float)
    put_oi  = np.asarray(put_oi,  dtype=float)
    d_oi = call_oi - put_oi
    abs_d = np.abs(d_oi)
    n = len(strikes)
    if n == 0:
        return np.array([], dtype=int)

    i_atm = int(np.argmin(np.abs(strikes - float(S))))
    total_abs = abs_d.sum()
    max_abs = abs_d.max() if n>0 else 0.0

    L = R = i_atm
    def coverage_ok(L, R):
        if total_abs <= 0: return True
        return abs_d[L:R+1].sum() >= p * total_abs

    def tails_ok(L, R):
        k = 3
        left_seg  = abs_d[L:min(L+k, R+1)]
        right_seg = abs_d[max(R-k+1, L):R+1]
        left_mean  = left_seg.mean()  if left_seg.size  else 0.0
        right_mean = right_seg.mean() if right_seg.size else 0.0
        return (left_mean <= q * max_abs) and (right_mean <= q * max_abs)

    while (R - L + 1) < Nmax and not (coverage_ok(L, R) or tails_ok(L, R)):
        if L > 0: L -= 1
        if R < n - 1: R += 1
        if L == 0 and R == n-1:
            break

    while (R - L + 1) < Nmin:
        if L > 0: L -= 1
        if R < n - 1: R += 1
        if L == 0 and R == n-1:
            break

    return np.arange(L, R+1, dtype=int)

def _format_labels(vals):
    labels = []
    for v in vals:
        try:
            fv = float(v)
            if abs(fv - int(round(fv))) < 1e-9:
                labels.append(str(int(round(fv))))
            else:
                labels.append(f"{fv:g}")
        except Exception:
            labels.append(str(v))
    return labels

def make_figure(strikes, net_gex, series_enabled, series_dict, price=None):
    strikes = np.asarray(strikes, dtype=float)

    # Determine indices to keep for bars
    idx_keep = np.arange(len(strikes), dtype=int)
    if (price is not None) and ("Call OI" in series_dict) and ("Put OI" in series_dict):
        try:
            idx_keep = _select_atm_window(strikes, series_dict["Call OI"], series_dict["Put OI"], float(price))
        except Exception:
            idx_keep = np.arange(len(strikes), dtype=int)

    strikes_keep = strikes[idx_keep]
    x_labels = _format_labels(strikes_keep)

    # Build figure
    fig = go.Figure()

    # Adaptive gaps for small samples
    n = len(strikes_keep)
    if n <= 5:
        bargap = 0.55
    elif n <= 10:
        bargap = 0.40
    elif n <= 20:
        bargap = 0.28
    else:
        bargap = 0.15

    # Net Gex bars with per-bar color by sign
    if series_enabled.get("Net Gex", True):
        y = np.asarray(series_dict["Net Gex"], dtype=float)[idx_keep]
        colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in y]
        fig.add_trace(go.Bar(x=x_labels, y=y, name="Net Gex", marker_color=colors, opacity=0.92))

    # Optional lines (use filtered x for alignment)
    for name in ["Put OI","Call OI","Put Volume","Call Volume","AG","PZ","PZ_FP"]:
        if series_enabled.get(name, False) and name in series_dict:
            y_full = np.asarray(series_dict[name], dtype=float)[idx_keep]
            fig.add_trace(go.Scatter(
                x=x_labels, y=y_full, name=name, mode="lines+markers", yaxis="y2"
            ))

    # Vertical price line anchored to the NEAREST kept strike label
    if price is not None and n > 0:
        i_near = int(np.argmin(np.abs(strikes_keep - float(price))))
        x_line = x_labels[i_near]
        fig.add_shape(
            type="line",
            x0=x_line, x1=x_line, xref="x",
            y0=0, y1=1, yref="paper",
            line=dict(width=2, color="#f0a000")
        )
        fig.add_annotation(
            x=x_line, y=1.02, xref="x", yref="paper",
            text=f"Price: {float(price):.2f}",
            showarrow=False, font=dict(size=12, color="#f0a000")
        )

    fig.update_layout(
        barmode="overlay",
        bargap=bargap,
        bargroupgap=0.0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="Strike",
            type="category",
            categoryorder="array",
            categoryarray=x_labels,
        tickmode='array',
        tickvals=x_labels,
        ticktext=x_labels,
        range=[-0.5, len(x_labels)-0.5]
        ),
        yaxis=dict(title="Net Gex (thousands)"),
        yaxis2=dict(title="Other series", overlaying="y", side="right"),
        hovermode="x unified",
        height=560
    )
    return fig
