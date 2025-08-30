
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
    max_abs = abs_d.max() if n > 0 else 0.0

    L = R = i_atm

    def coverage_ok(L, R):
        if total_abs <= 0:
            return True
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
        if L == 0 and R == n - 1:
            break

    while (R - L + 1) < Nmin:
        if L > 0: L -= 1
        if R < n - 1: R += 1
        if L == 0 and R == n - 1:
            break

    return np.arange(L, R + 1, dtype=int)

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

def make_figure(strikes, net_gex, series_enabled, series_dict, price=None, ticker=None):
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

    
    # Net Gex bars split by sign with custom hover (table-like) and colored hover box
    if series_enabled.get("Net Gex", True):
        y_all = np.asarray(series_dict["Net Gex"], dtype=float)[idx_keep]
        call_oi_f = np.asarray(series_dict.get("Call OI", np.zeros_like(y_all)), dtype=float)[idx_keep]
        put_oi_f  = np.asarray(series_dict.get("Put OI",  np.zeros_like(y_all)), dtype=float)[idx_keep]
        call_v_f  = np.asarray(series_dict.get("Call Volume", np.zeros_like(y_all)), dtype=float)[idx_keep]
        put_v_f   = np.asarray(series_dict.get("Put Volume", np.zeros_like(y_all)), dtype=float)[idx_keep]

        mask_pos = y_all >= 0
        mask_neg = ~mask_pos

        # Positive bars
        x_pos = [lbl for lbl, m in zip(x_labels, mask_pos) if m]
        y_pos = y_all[mask_pos]
        cd_pos = np.stack([
            np.array([lbl for lbl, m in zip(x_labels, mask_pos) if m], dtype=object),
            call_oi_f[mask_pos],
            put_oi_f[mask_pos],
            call_v_f[mask_pos],
            put_v_f[mask_pos],
            y_pos
        ], axis=-1)
        fig.add_trace(go.Bar(
            x=x_pos, y=y_pos, name="Net Gex +",
            marker_color=POS_COLOR, opacity=0.92,
            customdata=cd_pos,
            hovertemplate=(
                "Strike: %{customdata[0]}<br>"
                "Call OI: %{customdata[1]:,.0f}<br>"
                "Put OI: %{customdata[2]:,.0f}<br>"
                "Call Volume: %{customdata[3]:,.0f}<br>"
                "Put Volume: %{customdata[4]:,.0f}<br>"
                "Net GEX: %{customdata[5]:,.1f}"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor=POS_COLOR)
        ))

        # Negative bars
        x_neg = [lbl for lbl, m in zip(x_labels, mask_neg) if m]
        y_neg = y_all[mask_neg]
        cd_neg = np.stack([
            np.array([lbl for lbl, m in zip(x_labels, mask_neg) if m], dtype=object),
            call_oi_f[mask_neg],
            put_oi_f[mask_neg],
            call_v_f[mask_neg],
            put_v_f[mask_neg],
            y_neg
        ], axis=-1)
        fig.add_trace(go.Bar(
            x=x_neg, y=y_neg, name="Net Gex -",
            marker_color=NEG_COLOR, opacity=0.92,
            customdata=cd_neg,
            hovertemplate=(
                "Strike: %{customdata[0]}<br>"
                "Call OI: %{customdata[1]:,.0f}<br>"
                "Put OI: %{customdata[2]:,.0f}<br>"
                "Call Volume: %{customdata[3]:,.0f}<br>"
                "Put Volume: %{customdata[4]:,.0f}<br>"
                "Net GEX: %{customdata[5]:,.1f}"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor=NEG_COLOR)
        ))
    # Optional lines
 (aligned to filtered x)
    for name in ["Put OI","Call OI","Put Volume","Call Volume","AG","PZ","PZ_FP"]:
        if series_enabled.get(name, False) and name in series_dict:
            y_full = np.asarray(series_dict[name], dtype=float)[idx_keep]
            fig.add_trace(go.Scatter(
                x=x_labels, y=y_full, name=name, mode="lines+markers", yaxis="y2"
            ))

    # Vertical price line anchored to the nearest kept strike
        if price is not None and n > 0:
            i_near = int(np.argmin(np.abs(strikes_keep - float(price))))
            x_idx = i_near
            # full-height vertical line
            fig.add_shape(
                type="line",
                x0=x_idx, x1=x_idx, xref="x",
                y0=0, y1=1, yref="paper",
                line=dict(width=2, color="#f0a000"),
                layer="above"
            )
            # centered label exactly above the line
            fig.add_annotation(
                x=x_idx, y=1.0, xref="x", yref="paper",
                text=f"Price: {float(price):.2f}",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                align="center",
                yshift=8,
                font=dict(size=12, color="#f0a000")
            )

        # Layout with full-width category axis
        fig.update_layout(
            barmode="overlay",
            bargap=bargap,
            bargroupgap=0.0,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis=dict(
                title="Strike",
                type="category",
                categoryorder="array",
                categoryarray=x_labels,
                tickmode="array",
                tickvals=x_labels,
                ticktext=x_labels,
                range=[-0.5, len(x_labels)-0.5],
                showgrid=False
            ),
            yaxis=dict(title="Net Gex", showgrid=False),
            yaxis2=dict(title="Other series", overlaying="y", side="right", showgrid=False),
            hovermode="x unified",
            height=560
        )
# Add ticker label at top-left (inside chart, under 'График')
    if ticker:
        fig.add_annotation(
            x=0.0, y=1.08, xref="paper", yref="paper",
            text=str(ticker),
            showarrow=False,
            xanchor="left",
            font=dict(size=18)
        )

    return fig
