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

def _autodetect_price(series_dict):
    for k in ("price", "Price", "_price", "spot", "Spot"):
        if isinstance(series_dict, dict) and k in series_dict:
            try:
                return float(series_dict[k])
            except Exception:
                pass
    return None

def _finite_min_max(arr, default=(0.0, 1.0)):
    import numpy as _np
    a = _np.asarray(arr, dtype=float)
    if a.size == 0:
        return default
    a = a[_np.isfinite(a)]
    if a.size == 0:
        return default
    return float(a.min()), float(a.max())

def _draw_price_line(fig, price_val, y_values):
    # Надёжная отрисовка вертикальной линии через Scatter (поверх баров)
    y_min, y_max = _finite_min_max(y_values, default=(0.0, 1.0))
    # небольшие поля сверху/снизу
    pad_top = 0.12 * (y_max - y_min if y_max != y_min else (abs(y_max) + 1.0))
    pad_bot = 0.08  * (y_max - y_min if y_max != y_min else (abs(y_min) + 1.0))

    y0 = y_min - pad_bot
    y1 = y_max + pad_top

    # Линия как trace (чтобы точно была видна)
    fig.add_trace(go.Scatter(
        x=[price_val, price_val],
        y=[y0, y1],
        mode="lines",
        line=dict(color="#FFA500", width=3),
        name="Price",
        hoverinfo="skip",
        showlegend=False,
        yaxis="y"
    ))

    # Подпись над линией
    fig.add_annotation(
        x=price_val, y=y1,
        text=f"Price: {price_val:.2f}",
        showarrow=False,
        yanchor="bottom",
        xanchor="center",
        font=dict(color="#FFA500", size=16)
    )

    # Зафиксируем диапазон Y, чтобы подпись не обрезалась
    try:
        fig.update_yaxes(range=[y0, y1], matches=None)
    except Exception:
        pass
