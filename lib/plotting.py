
import numpy as np
import plotly.graph_objects as go

POS_COLOR = "#48B4FF"   # positive Net GEX bars (blue)
NEG_COLOR = "#FF3B30"   # negative Net GEX bars (red)

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

def make_figure(strikes, net_gex, series_enabled, series_dict, price=None, ticker=None, g_flip=None):
    strikes = np.asarray(strikes, dtype=float)
    net_gex = np.asarray(net_gex, dtype=float)

    idx_keep = np.arange(len(strikes), dtype=int)
    if (price is not None) and ("Call OI" in series_dict) and ("Put OI" in series_dict):
        try:
            idx_keep = _select_atm_window(strikes, series_dict["Call OI"], series_dict["Put OI"], float(price))
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

    # Net GEX bars
    if series_enabled.get("Net GEX", True):
        y_all = np.asarray(series_dict.get("Net GEX", net_gex), dtype=float)[idx_keep]
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
                "Net GEX: %{customdata[5]:,.1f}"
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
                "Net GEX: %{customdata[5]:,.1f}"
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
                               text=f"Price: {price_val:.2f}", showarrow=False,
                               xanchor="center", yanchor="bottom",
                               font=dict(size=14, color="#f0a000"))
        # G-Flip marker (optional, dashed)
        try:
            if series_enabled.get("G-Flip", False) and (g_flip is not None) and (n > 0):
                k_val = float(g_flip)
                i_near_g = int(np.argmin(np.abs(strikes_keep - k_val)))
                x_idx_g = i_near_g
                fig.add_shape(type="line", x0=x_idx_g, x1=x_idx_g, xref="x",
                              y0=0, y1=1, yref="paper",
                              line=dict(width=1, color="#AAAAAA", dash="dash"), layer="above")
                fig.add_annotation(x=x_idx_g, y=1.08, xref="x", yref="paper",
                                   text=f"G-Flip: {k_val:.2f}", showarrow=False,
                                   xanchor="center", yanchor="top",
                                   font=dict(size=14, color="#AAAAAA"))
                # add legend entry (non-clickable via layout at end)
                try:
                    yvals = np.asarray(series_dict.get("Net GEX", net_gex), dtype=float)[idx_keep]
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
        # Build dynamic right-axis title (exclude Net GEX; show only specific series)
        whitelist = ["Put OI","Call OI","Put Volume","Call Volume","AG","PZ","PZ_FP"]
        picked = [k for k in whitelist if series_enabled.get(k, False)]
        right_title = "Other parameters" + ((" (" + ", ".join(picked) + ")") if picked else "")
    
    fig.update_layout(
        barmode="overlay", bargap=bargap, bargroupgap=0.0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=64, r=40, t=40, b=40),
        xaxis=dict(title="Strike", type="category", categoryorder="array",
                   categoryarray=x_labels, tickmode="array", tickvals=x_labels,
                   ticktext=x_labels, range=[-0.5, len(x_labels)-0.5],
                   showgrid=False, fixedrange=True),
        yaxis=dict(title="Net GEX", showgrid=False, fixedrange=True, automargin=True, tickformat=",.0f"),
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
