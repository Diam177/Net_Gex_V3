import numpy as np
import plotly.graph_objects as go

# Цвета под референс-скриншот
POS = "#40C4FF"  # голубой (положительный Net GEX)
NEG = "#FF5252"  # красный (отрицательный Net GEX)

def _signed_colors(values):
    y = np.asarray(values, dtype=float)
    return np.where(y >= 0, POS, NEG)

def make_figure(strikes, net_gex, series_enabled, series_dict, price=None):
    fig = go.Figure()

    # Net GEX как столбцы с окраской по знаку
    if series_enabled.get("Net Gex", True):
        colors = _signed_colors(series_dict["Net Gex"])
        fig.add_trace(go.Bar(
            x=strikes,
            y=series_dict["Net Gex"],
            name="Net Gex",
            marker=dict(color=colors, line=dict(width=0)),
            opacity=0.95
        ))

    # Остальные серии — линии (на правой оси), как и раньше
    for name in ["Put OI","Call OI","Put Volume","Call Volume","AG","PZ","PZ_FP"]:
        if series_enabled.get(name, False):
            fig.add_trace(go.Scatter(
                x=strikes,
                y=series_dict[name],
                name=name,
                mode="lines+markers",
                yaxis="y2"
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
