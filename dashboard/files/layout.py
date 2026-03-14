# ─────────────────────────────────────────────
#  layout.py
# ─────────────────────────────────────────────

from dash import dcc, html
from config import TASK_NAME, MODEL_NAME, COLORS

def _card(children, extra_style: dict = None):
    style = {
        "background": COLORS["surface"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "8px",
        "padding": "6px",
        "boxShadow": "0 4px 24px rgba(0,0,0,0.4)",
    }
    if extra_style:
        style.update(extra_style)
    return html.Div(children, style=style)

def _graph_card(graph_id: str):
    return _card(
        dcc.Graph(
            id=graph_id,
            config={"displayModeBar": False},
            style={"height": "280px"},
        )
    )

def _status_dot():
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "8px",
               "fontFamily": "'JetBrains Mono', monospace",
               "fontSize": "11px", "color": "#64748b"},
        children=[
            html.Div(id="status-dot", style={
                "width": "8px", "height": "8px", "borderRadius": "50%",
                "background": "#22c55e", "boxShadow": "0 0 6px #22c55e",
                "animation": "pulse 2s infinite",
            }),
            html.Span(id="status-text", children="LIVE"),
        ],
    )

def _sample_card(idx: int):
    return html.Div(
        id=f"sample-{idx}",
        style={"display": "flex", "flexDirection": "column", "alignItems": "center",
               "background": "#0d1520", "border": f"1px solid {COLORS['border']}",
               "borderRadius": "8px", "padding": "8px", "gap": "6px", "minWidth": "0"},
        children=[
            html.Img(id=f"sample-img-{idx}",
                     style={"width": "100%", "aspectRatio": "1/1", "objectFit": "cover",
                            "borderRadius": "4px", "display": "none"}),
            html.Div(id=f"sample-label-{idx}",
                     style={"fontFamily": "'JetBrains Mono', monospace", "fontSize": "10px",
                            "color": "#64748b", "textAlign": "center", "lineHeight": "1.4"}),
        ],
    )

def _summary_metric(metric_id: str, label: str, color: str):
    return _card(html.Div(
        className="metric-card",
        style={"padding": "8px 10px"},
        children=[
            html.Div(label, style={"fontSize": "9px", "letterSpacing": "2px",
                                   "color": "#475569", "marginBottom": "6px"}),
            html.Div(id=metric_id, children="—",
                     style={"fontSize": "22px", "fontWeight": "700",
                            "color": color, "letterSpacing": "-1px"}),
        ],
    ))

def _session_metric(label: str, value_id: str, color: str = "#94a3b8"):
    """A single labeled value cell for the session info bar."""
    return html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
        children=[
            html.Div(label, style={"fontSize": "9px", "letterSpacing": "2px",
                                   "color": "#475569", "fontWeight": "600"}),
            html.Div(id=value_id, children="—",
                     style={"fontSize": "13px", "fontWeight": "600",
                            "color": color, "fontFamily": "'JetBrains Mono', monospace"}),
        ],
    )

def build_layout():
    return html.Div(
        style={"minHeight": "100vh", "background": COLORS["bg"],
               "fontFamily": "'JetBrains Mono', monospace",
               "padding": "24px 28px", "boxSizing": "border-box"},
        children=[
            # ── Header ────────────────────────────────────────────────────────
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "flex-end", "marginBottom": "20px",
                       "borderBottom": f"1px solid {COLORS['border']}",
                       "paddingBottom": "16px"},
                children=[
                    html.Div([
                        html.Div("TRAINING MONITOR",
                                 style={"fontSize": "10px", "letterSpacing": "4px",
                                        "color": COLORS["train"], "marginBottom": "4px",
                                        "fontWeight": "600"}),
                        html.H1(TASK_NAME,
                                style={"margin": "0", "fontSize": "22px", "fontWeight": "700",
                                       "color": "#e2e8f0", "letterSpacing": "-0.5px"}),
                        html.Div(MODEL_NAME,
                                 style={"fontSize": "11px", "color": "#475569", "marginTop": "2px"}),
                    ]),
                    html.Div(
                        style={"display": "flex", "flexDirection": "column",
                               "alignItems": "flex-end", "gap": "6px"},
                        children=[
                            _status_dot(),
                            html.Div(id="epoch-counter",
                                     style={"fontSize": "11px", "color": "#475569"}),
                        ],
                    ),
                ],
            ),

            # ── Session info bar (start time + elapsed) ───────────────────────
            _card(
                html.Div(
                    style={"display": "flex", "gap": "48px",
                           "alignItems": "center", "padding": "10px 12px"},
                    children=[
                        html.Div("SESSION",
                                 style={"fontSize": "9px", "letterSpacing": "3px",
                                        "color": "#334155", "fontWeight": "600",
                                        "alignSelf": "center"}),
                        _session_metric("STARTED AT",  "session-start-time", COLORS["train"]),
                        _session_metric("ELAPSED",     "session-elapsed",    COLORS["test"]),
                    ],
                ),
                extra_style={"marginBottom": "16px"},
            ),

            # ── Summary strip ─────────────────────────────────────────────────
            html.Div(
                id="summary-strip",
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                       "gap": "12px", "marginBottom": "24px"},
                children=[
                    _summary_metric("latest-train-loss", "TRAIN LOSS",  COLORS["train"]),
                    _summary_metric("latest-test-loss",  "TEST LOSS",   COLORS["test"]),
                    _summary_metric("latest-train-acc",  "TRAIN ACC",   COLORS["train"]),
                    _summary_metric("latest-test-acc",   "TEST ACC",    COLORS["test"]),
                ],
            ),

            # ── 6 Graphs ──────────────────────────────────────────────────────
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                       "gap": "16px", "marginBottom": "24px"},
                children=[
                    _graph_card("graph-step-train-loss"),
                    _graph_card("graph-test-loss"),
                    _graph_card("graph-train-acc"),
                    _graph_card("graph-test-acc"),
                    _graph_card("graph-combined-loss"),
                    _graph_card("graph-combined-acc"),
                ],
            ),

            # ── Bottom row: samples + configs side by side ────────────────────
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 340px",
                       "gap": "16px", "marginBottom": "24px", "alignItems": "start"},
                children=[
                    # Sample predictions
                    _card([
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "12px",
                                   "padding": "4px 6px"},
                            children=[
                                html.Div("SAMPLE PREDICTIONS",
                                         style={"fontSize": "10px", "letterSpacing": "3px",
                                                "color": "#475569", "fontWeight": "600"}),
                                html.Div(id="samples-epoch-tag",
                                         style={"fontSize": "10px", "color": "#334155"}),
                            ],
                        ),
                        html.Div(
                            id="samples-grid",
                            style={"display": "grid",
                                   "gridTemplateColumns": "repeat(10, 1fr)", "gap": "10px"},
                            children=[_sample_card(i) for i in range(10)],
                        ),
                    ]),

                    # Configs panel
                    _card(
                        html.Div([
                            html.Div("RUN CONFIGURATION",
                                     style={"fontSize": "9px", "letterSpacing": "3px",
                                            "color": "#475569", "fontWeight": "600",
                                            "marginBottom": "14px", "padding": "4px 6px"}),
                            html.Div(id="configs-table",
                                     style={"padding": "0 6px"}),
                        ]),
                        extra_style={"overflowY": "auto", "maxHeight": "320px"},
                    ),
                ],
            ),

            # ── Interval ──────────────────────────────────────────────────────
            dcc.Interval(id="interval", interval=1500, n_intervals=0),
        ],
    )
