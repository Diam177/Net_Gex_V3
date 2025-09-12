
import numpy as np
import plotly.graph_objects as go

from datetime import datetime, time as _dt_time
from dateutil import tz as _tz

def _price_label_prefix_now_et() -> str:
    """Return 'Price', 'Price(Pre)', or 'Price(Post)' based on ET time.
    - Pre:   before 09:30 ET on a weekday
    - RTH:   09:30–16:00 ET (no suffix)
    - Post:  after 16:00 ET on a weekday
    Weekends/unknown tz fallback -> no suffix (just 'Price').
    """
    try:
        tz_et = _tz.gettz("America/New_York")
        now_et = datetime.now(tz=tz_et)
        # Only Mon–Fri are trading days. On weekends just show 'Price'.
        if now_et.weekday() >= 5:
            return "Price"
        t = now_et.time()
        if t < _dt_time(9, 30):
            return "Price(Pre)"
        if t >= _dt_time(16, 0):
            return "Price(Post)"
        return "Price"
    except Exception:
        # Conservative fallback
        return "Price"

POS_COLOR = "#48B4FF"   # positive Net Gex bars (blue)
NEG_COLOR = "#FF3B30"   # negative Net Gex bars (red)

# Line colors & fills for each optional series
LINE_STYLE = {
    "Put OI":     {"line": "#7F0020", "fill": "rgba(127,0,32,0.25)"},
    "Call OI":    {"line": "#2FD06F", "fill": "rgba(47,208,111,0.25)"},
    "Put Volume": {"line": "#8C5A0A", "fill": "rgba(140,90,10,0.25)"},
    "Call Volume":{"line": "#2D83FF", "fill": "rgba(45,131,255,0.25)"},
    "AG":         {"line": "#8A63F6", "fill": "rgba(138,99,246,0.25)"},
    "PZ":         {"line": "#FFC400", "fill": "rgba(255,196,0,0.30)"},
    "PZ_FP":      {"line": "#B0B8C5", "fill": "rgba(176,184,197,0.30)"},
}

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
    total_abs = float(abs_d.sum())
    max_abs = float(abs_d.max()) if n > 0 else 0.0
    L = R = i_atm

    def coverage_ok(L, R):
        if total_abs <= 0:
            return True
        return float(abs_d[L:R+1].sum()) >= p * total_abs

    def tails_ok(L, R):
        k = 3
        left_seg  = abs_d[L:min(L+k, R+1)]
        right_seg = abs_d[max(R-k+1, L):R+1]
        left_mean  = float(left_seg.mean())  if left_seg.size  else 0.0
        right_mean = float(right_seg.mean()) if right_seg.size else 0.0
        return (left_mean <= q * max_abs) and (right_mean <= q * max_abs)

    while (R - L + 1) < Nmax and not (coverage_ok(L, R) or tails_ok(L, R)):
        if L > 0: L -= 1
        if R < n - 1: R += 1
        if L == 0 and R == n - 1: break

    while (R - L + 1) < Nmin:
        if L > 0: L -= 1
        if R < n - 1: R += 1
        if L == 0 and R == n - 1: break

    return np.arange(L, R + 1, dtype=int)



def _gaussian_weight(dist_in_strikes: np.ndarray, bandwidth: int = 15) -> np.ndarray:
    """Smoothly decaying weight as we move away from ATM in strike-index space."""
    dist_in_strikes = np.asarray(dist_in_strikes, dtype=float)
    bw = max(1, int(bandwidth))
    return np.exp(- (dist_in_strikes / bw) ** 2)


def _select_window_score(
    strikes: np.ndarray,
    call_oi: np.ndarray,
    put_oi: np.ndarray,
    atm_idx: int,
    call_vol: np.ndarray | None = None,
    put_vol: np.ndarray | None = None,
    *,
    Nmin: int = 15,
    Nmax: int = 99,
    weights: dict | None = None,
    bandwidth: int = 15,
) -> np.ndarray:
    """Hybrid score window using multiple signals and distance to price.

    score_k = max( |C-P|/max|C-P|,
                   wC * C/maxC,
                   wP * P/maxP,
                   wCV * CV/maxCV,
                   wPV * PV/maxPV ) * gaussian(|k - atm_idx|)

    Then choose a contiguous segment length in [Nmin, Nmax] that maximizes score sum
    and contains ATM.
    """
    strikes = np.asarray(strikes)
    C = np.asarray(call_oi, dtype=float)
    P = np.asarray(put_oi, dtype=float)
    n = len(strikes)
    if n == 0:
        return np.arange(0, dtype=int)

    abs_diff = np.abs(C - P)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    maxC = float(C.max()) if C.size else 0.0
    maxP = float(P.max()) if P.size else 0.0

    CV = np.asarray(call_vol, dtype=float) if call_vol is not None else np.zeros_like(C)
    PV = np.asarray(put_vol, dtype=float) if put_vol is not None else np.zeros_like(P)
    maxCV = float(CV.max()) if CV.size else 0.0
    maxPV = float(PV.max()) if PV.size else 0.0

    # default weights
    w = dict(absdiff=1.0, call_oi=0.7, put_oi=0.7, call_vol=0.4, put_vol=0.4)
    if isinstance(weights, dict):
        w.update({k: float(v) for k, v in weights.items() if k in w})

    comp_abs = (abs_diff / max_abs) if max_abs > 0 else np.zeros(n)
    comp_c   = (C / maxC)         if maxC    > 0 else np.zeros(n)
    comp_p   = (P / maxP)         if maxP    > 0 else np.zeros(n)
    comp_cv  = (CV / maxCV)       if maxCV   > 0 else np.zeros(n)
    comp_pv  = (PV / maxPV)       if maxPV   > 0 else np.zeros(n)

    raw_score = np.maximum.reduce([
        comp_abs,
        w["call_oi"] * comp_c,
        w["put_oi"] * comp_p,
        w["call_vol"] * comp_cv,
        w["put_vol"] * comp_pv,
    ])

    idx = np.arange(n)
    dist = np.abs(idx - int(atm_idx))
    weight = _gaussian_weight(dist, bandwidth=bandwidth)
    score = raw_score * weight

    Nmin = max(1, int(Nmin))
    Nmax = max(Nmin, int(min(Nmax, n)))

    best_sum = -1.0
    best_L = 0
    best_R = max(Nmin - 1, 0)

    # Iterate possible widths ensuring ATM is inside segment
    for width in range(Nmin, Nmax + 1):
        L_min = max(0, int(atm_idx) - width + 1)
        L_max = min(int(atm_idx), n - width)
        for L in range(L_min, L_max + 1):
            R = L + width - 1
            s = float(score[L:R+1].sum())
            if s > best_sum:
                best_sum = s
                best_L, best_R = L, R

    return np.arange(best_L, best_R + 1, dtype=int)

def _format_labels(vals):
    out = []
    for v in vals:
        try:
            fv = float(v)
            if abs(fv - int(round(fv))) < 1e-9:
                out.append(str(int(round(fv))))
            else:
                out.append(f"{fv:g}")
        except Exception:
            out.append(str(v))
    return out

def make_figure(strikes, net_gex, series_enabled, series_dict, price=None, ticker=None, g_flip=None, call_volume=None, put_volume=None, full_chain: bool = False, window_mode: str = "atm", score_weights: dict | None = None, score_bandwidth: int = 15, score_nmax: int | None = None, idx_keep_override=None):
    strikes = np.asarray(strikes, dtype=float)
    net_gex = np.asarray(net_gex, dtype=float)

    
    idx_keep = np.arange(len(strikes), dtype=int)
    try:
        if idx_keep_override is not None:
            idx_keep = np.asarray(idx_keep_override, dtype=int)
        elif (price is not None) and ("Call OI" in series_dict) and ("Put OI" in series_dict):
            if full_chain:
                idx_keep = np.arange(len(strikes), dtype=int)
            elif window_mode == "score":
                # compute atm index
                try:
                    atm_idx = int(np.argmin(np.abs(np.asarray(strikes, dtype=float) - float(price))))
                except Exception:
                    atm_idx = 0
                idx_keep = _select_window_score(
                    np.asarray(strikes, dtype=float),
                    np.asarray(series_dict.get("Call OI"), dtype=float),
                    np.asarray(series_dict.get("Put OI"), dtype=float),
                    atm_idx=atm_idx,
                    call_vol=np.asarray(series_dict.get("Call Volume"), dtype=float) if "Call Volume" in series_dict else None,
                    put_vol=np.asarray(series_dict.get("Put Volume"), dtype=float) if "Put Volume" in series_dict else None,
                    Nmin=15,
                    Nmax=int(score_nmax or 99),
                    weights=score_weights,
                    bandwidth=int(score_bandwidth),
                )
            else:
                idx_keep = _select_atm_window(
                    np.asarray(strikes, dtype=float),
                    np.asarray(series_dict.get("Call OI"), dtype=float),
                    np.asarray(series_dict.get("Put OI"), dtype=float),
                    float(price)
                )
    except Exception:
        idx_keep = np.arange(len(strikes), dtype=int)

    strikes_keep = strikes[idx_keep]
    x_labels = _format_labels(strikes_keep)
    n = len(strikes_keep)

    fig = go.Figure()

    # Adaptive bar spacing
    if n <= 5: bargap = 0.55
    elif n <= 10: bargap = 0.40
    elif n <= 20: bargap = 0.28
    else: bargap = 0.15

    # Shared arrays for hover customdata
    call_oi_f = np.asarray(series_dict.get("Call OI", np.zeros_like(net_gex)), dtype=float)[idx_keep]
    put_oi_f  = np.asarray(series_dict.get("Put OI",  np.zeros_like(net_gex)), dtype=float)[idx_keep]
    call_v_f  = np.asarray(series_dict.get("Call Volume", np.zeros_like(net_gex)), dtype=float)[idx_keep]
    put_v_f   = np.asarray(series_dict.get("Put Volume", np.zeros_like(net_gex)), dtype=float)[idx_keep]

    # Net Gex bars
    if series_enabled.get("Net Gex", True):
        y_all = np.asarray(series_dict.get("Net Gex", net_gex), dtype=float)[idx_keep]
        mask_pos = y_all >= 0
        mask_neg = ~mask_pos

        def build_cd(mask, yvals):
            x_use = [lbl for lbl, m in zip(x_labels, mask) if m]
            cd = np.stack([
                np.array(x_use, dtype=object),
                call_oi_f[mask],
                put_oi_f[mask],
                call_v_f[mask],
                put_v_f[mask],
                yvals[mask]
            ], axis=-1)
            return x_use, cd

        x_pos, cd_pos = build_cd(mask_pos, y_all)
        x_neg, cd_neg = build_cd(mask_neg, y_all)

        fig.add_trace(go.Bar(
            x=x_pos, y=y_all[mask_pos], name="Net GEX",
            marker_color=POS_COLOR, opacity=0.92,
            customdata=cd_pos,
            hovertemplate=(
                "Strike: %{customdata[0]}<br>"
                "Call OI: %{customdata[1]:,.0f}<br>"
                "Put OI: %{customdata[2]:,.0f}<br>"
                "Call Volume: %{customdata[3]:,.0f}<br>"
                "Put Volume: %{customdata[4]:,.0f}<br>"
                "Net Gex: %{customdata[5]:,.1f}"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor=POS_COLOR, bordercolor=POS_COLOR, font=dict(color="#000000"))
        ))
        fig.add_trace(go.Bar(
            x=x_neg, y=y_all[mask_neg], name="Net GEX", showlegend=False,
            marker_color=NEG_COLOR, opacity=0.92,
            customdata=cd_neg,
            hovertemplate=(
                "Strike: %{customdata[0]}<br>"
                "Call OI: %{customdata[1]:,.0f}<br>"
                "Put OI: %{customdata[2]:,.0f}<br>"
                "Call Volume: %{customdata[3]:,.0f}<br>"
                "Put Volume: %{customdata[4]:,.0f}<br>"
                "Net Gex: %{customdata[5]:,.1f}"
                "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor=NEG_COLOR)
        ))

    # Optional series as smooth spline lines with fill
    SERIES_ORDER = [
        ("Put OI", "Put OI"),
        ("Call OI", "Call OI"),
        ("Put Volume", "Put Volume"),
        ("Call Volume", "Call Volume"),
        ("AG", "AG"),
        ("PZ", "PZ"),
        ("PZ_FP", "PZ_FP"),
    ]

    for ser_key, ser_label in SERIES_ORDER:
        if series_enabled.get(ser_key, False) and (ser_key in series_dict):
            y_full = np.asarray(series_dict[ser_key], dtype=float)[idx_keep]
            cd = np.stack([
                np.array(x_labels, dtype=object),
                call_oi_f, put_oi_f, call_v_f, put_v_f, y_full
            ], axis=-1)

            colors = LINE_STYLE.get(ser_key, {"line": "#BBBBBB", "fill": "rgba(187,187,187,0.2)"})
            hovertemplate = (
                "Strike: %{customdata[0]}<br>"
                "Call OI: %{customdata[1]:,.0f}<br>"
                "Put OI: %{customdata[2]:,.0f}<br>"
                "Call Volume: %{customdata[3]:,.0f}<br>"
                "Put Volume: %{customdata[4]:,.0f}<br>"
                f"{ser_label}: " + "%{customdata[5]:,.1f}"
                "<extra></extra>"
            )

            fig.add_trace(go.Scatter(
                x=x_labels, y=y_full, name=ser_key,
                mode="lines+markers", yaxis="y2",
                line=dict(color=colors["line"], width=1.1, shape="spline"),
                marker=dict(color=colors["line"], size=5),
                fill="tozeroy", fillcolor=colors["fill"], opacity=0.95,
                customdata=cd, hovertemplate=hovertemplate,
                hoverlabel=dict(bgcolor=colors["line"], font=dict(color="#000000"))
            ))

    # Price marker
    if (price is not None) and (n > 0):
        try: price_val = float(price)
        except Exception: price_val = None
        if price_val is not None:
            i_near = int(np.argmin(np.abs(strikes_keep - price_val)))
            x_idx = i_near
            fig.add_shape(type="line", x0=x_idx, x1=x_idx, xref="x",
                          y0=0, y1=1, yref="paper",
                          line=dict(width=1, color="#f0a000"), layer="above")
            fig.add_annotation(x=x_idx, y=1.07, xref="x", yref="paper",
                               text=f"{_price_label_prefix_now_et()}: {price_val:.2f}", showarrow=False,
                               xanchor="center", yanchor="bottom",
                               font=dict(size=14, color="#f0a000"))
        # G-Flip marker (optional, dashed)
        try:
            if series_enabled.get("G-Flip", False) and (g_flip is not None) and (n > 0):
                k_val = float(g_flip)
                i_near_g = int(np.argmin(np.abs(strikes_keep - k_val)))
                x_idx_g = i_near_g
                k_snap = float(strikes_keep[i_near_g])
                fig.add_shape(type="line", x0=x_idx_g, x1=x_idx_g, xref="x",
                              y0=0, y1=1, yref="paper",
                              line=dict(width=1, color="#AAAAAA", dash="dash"), layer="above")
                fig.add_annotation(x=x_idx_g, y=1.08, xref="x", yref="paper",
                                   text=f"G-Flip: {int(round(k_snap))}", showarrow=False,
                                   xanchor="center", yanchor="top",
                                   font=dict(size=14, color="#AAAAAA"))
                # add legend entry (non-clickable via layout at end)
                try:
                    yvals = np.asarray(series_dict.get("Net Gex", net_gex), dtype=float)[idx_keep]
                    y_min = float(np.nanmin(yvals)) if np.any(np.isfinite(yvals)) else 0.0
                    y_max = float(np.nanmax(yvals)) if np.any(np.isfinite(yvals)) else 1.0
                    fig.add_trace(go.Scatter(
                        x=[x_idx_g, x_idx_g], y=[y_min, y_max], name="G-Flip",
                        mode="lines", line=dict(color="#AAAAAA", width=1, dash="dash"),
                        hoverinfo="skip", showlegend=True, yaxis="y"))
                except Exception:
                    pass
        except Exception:
            pass
        # Build dynamic right-axis title (exclude Net Gex; show only specific series)
        whitelist = ["Put OI","Call OI","Put Volume","Call Volume","AG","PZ","PZ_FP"]
        picked = [k for k in whitelist if series_enabled.get(k, False)]
        right_title = "Other parameters" + ((" (" + ", ".join(picked) + ")") if picked else "")
    
    fig.update_layout(
        barmode="overlay", bargap=bargap, bargroupgap=0.0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="Strike", type="category", categoryorder="array",
                   categoryarray=x_labels, tickmode="array", tickvals=x_labels,
                   ticktext=x_labels, range=[-0.5, len(x_labels)-0.5],
                   showgrid=False, fixedrange=True),
        yaxis=dict(title="Net GEX", showgrid=False, fixedrange=True),
        yaxis2=dict(title=right_title, overlaying="y", side="right",
                    showgrid=False, fixedrange=True),
        hovermode="closest", height=560,
    )

    if ticker:
        fig.add_annotation(x=0.0, y=1.08, xref="paper", yref="paper",
                           text=str(ticker), showarrow=False,
                           xanchor="left", font=dict(size=18))


    try:

        for _tr in fig.data:

            _n = (getattr(_tr, "name", None) or "").strip().lower()

            if _n in ("put oi", "put volume"):

                _tr.update(hoverlabel=dict(font=dict(color="white")))

    except Exception:

        pass

    # === end enforced tooltip text colors ===


    

    # === force area fill alpha = 0.6 for specific series ===

    try:

        def _parse_rgba(c):

            s = (c or "").strip().lower()

            if s.startswith("rgba(") and s.endswith(")"):

                p = [x.strip() for x in s[5:-1].split(",")]

                if len(p) == 4:

                    try:

                        return int(float(p[0])), int(float(p[1])), int(float(p[2])), float(p[3])

                    except Exception:

                        return None

            if s.startswith("rgb(") and s.endswith(")"):

                p = [x.strip() for x in s[4:-1].split(",")]

                if len(p) == 3:

                    try:

                        return int(float(p[0])), int(float(p[1])), int(float(p[2])), 1.0

                    except Exception:

                        return None

            if s.startswith("#") and (len(s) in (7,9)):

                try:

                    r = int(s[1:3],16); g=int(s[3:5],16); b=int(s[5:7],16)

                    a = int(s[7:9],16)/255.0 if len(s)==9 else 1.0

                    return r,g,b,a

                except Exception:

                    return None

            return None

    

        targets = {"put oi","call oi","put volume","call volume","ag","pz","pz_fp"}

        for tr in fig.data:

            name = (getattr(tr,"name","") or "").strip().lower()

            if name in targets and getattr(tr,"fill",None) not in (None,"none"):

                rgb = None

                lc = getattr(getattr(tr,"line",None),"color",None)

                fc = getattr(tr,"fillcolor",None)

                rgba = _parse_rgba(lc) or _parse_rgba(fc)

                if rgba:

                    r,g,b,_ = rgba

                    tr.update(fillcolor=f"rgba({int(r)},{int(g)},{int(b)},0.6)")

    except Exception:

        pass

    # === end force area fill alpha ===

    


    # --- final legend override: right-aligned & non-clickable ---

    try:

        fig.update_layout(legend=dict(orientation="h", x=1, y=1.1, xanchor="right",

                                      itemclick=False, itemdoubleclick=False))

    except Exception:

        pass

    # --- end legend override ---

    return fig
