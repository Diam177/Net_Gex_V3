
import numpy as np
import plotly.graph_objects as go

def _select_atm_window(strikes, call_oi, put_oi, S, p=0.95, q=0.05, Nmin=15, Nmax=49):
    strikes = np.asarray(strikes, dtype=float)
    call_oi = np.asarray(call_oi, dtype=float)
    put_oi  = np.asarray(put_oi,  dtype=float)
    d_oi = call_oi - put_oi
    abs_d = np.abs(d_oi)
    n = len(strikes)
    if n == 0:
        return np.array([], dtype=int)

    # ATM index by nearest strike to S
    i_atm = int(np.argmin(np.abs(strikes - float(S))))

    # coverage threshold and max for tail rule
    total_abs = abs_d.sum()
    max_abs = abs_d.max() if n>0 else 0.0

    L = R = i_atm
    def coverage_ok(L, R):
        if total_abs <= 0: return True
        return abs_d[L:R+1].sum() >= p * total_abs

    def tails_ok(L, R):
        # mean over last 3 on each tail <= q * max_abs
        k = 3
        left_seg  = abs_d[L:min(L+k, R+1)]
        right_seg = abs_d[max(R-k+1, L):R+1]
        left_mean  = left_seg.mean()  if left_seg.size  else 0.0
        right_mean = right_seg.mean() if right_seg.size else 0.0
        return (left_mean <= q * max_abs) and (right_mean <= q * max_abs)

    # Expand symmetrically while criteria not met and size < Nmax
    while (R - L + 1) < Nmax and not (coverage_ok(L, R) or tails_ok(L, R)):
        if L > 0: L -= 1
        if R < n - 1: R += 1
        if L == 0 and R == n-1:
            break

    # Ensure minimum size
    while (R - L + 1) < Nmin:
        if L > 0: L -= 1
        if R < n - 1: R += 1
        if L == 0 and R == n-1:
            break

    idx = np.arange(L, R+1, dtype=int)
    return idx

def make_figure(strikes, net_gex, series_enabled, series_dict, price=None):
    strikes = np.asarray(strikes, dtype=float)

    # Build figure
    fig = go.Figure()

    # Compute filtered indices for Net Gex bars, if possible
    idx_keep = None
    if (price is not None) and ("Call OI" in series_dict) and ("Put OI" in series_dict):
        try:
            idx_keep = _select_atm_window(strikes, series_dict["Call OI"], series_dict["Put OI"], float(price))
        except Exception:
            idx_keep = None

    if series_enabled.get("Net Gex", True):
        if idx_keep is not None and idx_keep.size > 0:
            x = strikes[idx_keep]
            y = np.asarray(series_dict["Net Gex"], dtype=float)[idx_keep]
        else:
            x = strikes
            y = np.asarray(series_dict["Net Gex"], dtype=float)
        fig.add_trace(go.Bar(x=x, y=y, name="Net Gex", opacity=0.85))

    # Other series are plotted in full
    for name in ["Put OI","Call OI","Put Volume","Call Volume","AG","PZ","PZ_FP"]:
        if series_enabled.get(name, False):
            fig.add_trace(go.Scatter(
                x=strikes, y=series_dict[name], name=name, mode="lines+markers", yaxis="y2"
            ))

    fig.update_layout(
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Strike",
        yaxis=dict(title="Net Gex (thousands)"),
        yaxis2=dict(title="Other series", overlaying="y", side="right"),
        hovermode="x unified",
        height=560
    )
    return fig
