import plotly.graph_objects as go

def make_figure(strikes, net_gex, series_enabled, series_dict):
    fig = go.Figure()
    if series_enabled.get("Net Gex", True):
        fig.add_trace(go.Bar(
            x=strikes, y=series_dict["Net Gex"], name="Net Gex", opacity=0.85
        ))
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
