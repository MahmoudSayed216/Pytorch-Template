# ─────────────────────────────────────────────
#  callbacks.py
# ─────────────────────────────────────────────

from dash import Input, Output, html
from datetime import datetime, timezone
import time
import data_store
import graphs


def register_callbacks(app):

    # ── Graphs + summary strip ─────────────────────────────────────────────────
    @app.callback(
        Output("graph-step-train-loss", "figure"),
        Output("graph-test-loss",        "figure"),
        Output("graph-train-acc",        "figure"),
        Output("graph-test-acc",         "figure"),
        Output("graph-combined-loss",    "figure"),
        Output("graph-combined-acc",     "figure"),
        Output("latest-train-loss",  "children"),
        Output("latest-test-loss",   "children"),
        Output("latest-train-acc",   "children"),
        Output("latest-test-acc",    "children"),
        Output("epoch-counter",      "children"),
        Input("interval", "n_intervals"),
    )
    def update_graphs(_n):
        snap = data_store.get_snapshot()
        step_loss   = snap["step_train_loss"]
        epochs      = snap["epochs"]
        avg_tr_loss = snap["avg_train_loss"]
        te_loss     = snap["test_loss"]
        tr_acc      = snap["train_acc"]
        te_acc      = snap["test_acc"]

        fig_step  = graphs.make_step_train_loss_fig(step_loss)
        fig_tl    = graphs.make_test_loss_fig(epochs, te_loss)
        fig_tra   = graphs.make_train_acc_fig(epochs, tr_acc)
        fig_tea   = graphs.make_test_acc_fig(epochs, te_acc)
        fig_closs = graphs.make_combined_loss_fig(epochs, avg_tr_loss, te_loss)
        fig_cacc  = graphs.make_combined_acc_fig(epochs, tr_acc, te_acc)

        fmt_loss = lambda v: f"{v:.5f}" if v else "—"
        fmt_acc  = lambda v: f"{v*100:.2f}%" if v else "—"

        return (
            fig_step, fig_tl, fig_tra, fig_tea, fig_closs, fig_cacc,
            fmt_loss(step_loss[-1] if step_loss else None),
            fmt_loss(te_loss[-1]   if te_loss   else None),
            fmt_acc(tr_acc[-1]     if tr_acc    else None),
            fmt_acc(te_acc[-1]     if te_acc    else None),
            f"Epoch {epochs[-1]}" if epochs else "Waiting…",
        )

    # ── Session info (start time + elapsed) ────────────────────────────────────
    @app.callback(
        Output("session-start-time", "children"),
        Output("session-elapsed",    "children"),
        Input("interval", "n_intervals"),
    )
    def update_session_info(_n):
        snap = data_store.get_snapshot()
        t0   = snap["training_start_time"]

        if t0 is None:
            return "—", "—"

        start_str = datetime.fromtimestamp(t0).strftime("%Y-%m-%d  %H:%M:%S")

        elapsed = time.time() - t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        elapsed_str = f"{h:02d}h {m:02d}m {s:02d}s"

        return start_str, elapsed_str

    # ── Configs table ──────────────────────────────────────────────────────────
    @app.callback(
        Output("configs-table", "children"),
        Input("interval", "n_intervals"),
    )
    def update_configs(_n):
        snap    = data_store.get_snapshot()
        configs = snap["run_configs"]

        if not configs:
            return html.Div("Waiting for configs…",
                            style={"fontSize": "11px", "color": "#334155",
                                   "padding": "8px"})

        rows = []
        def flatten(d, prefix=""):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, full_key)
                else:
                    rows.append((full_key, str(v)))

        flatten(configs)

        return html.Div([
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "baseline",
                       "padding": "5px 0",
                       "borderBottom": f"1px solid {data_store._lock and '#1e2a3a'}"},
                children=[
                    html.Span(key,
                              style={"fontSize": "10px", "color": "#475569",
                                     "marginRight": "12px", "flexShrink": "0"}),
                    html.Span(val,
                              style={"fontSize": "10px", "color": "#94a3b8",
                                     "textAlign": "right", "wordBreak": "break-all"}),
                ],
            )
            for key, val in rows
        ])

    # ── Sample images ──────────────────────────────────────────────────────────
    img_outputs  = [Output(f"sample-img-{i}",   "src")     for i in range(10)]
    img_styles   = [Output(f"sample-img-{i}",   "style")   for i in range(10)]
    label_outs   = [Output(f"sample-label-{i}", "children") for i in range(10)]

    @app.callback(
        *img_outputs, *img_styles, *label_outs,
        Output("samples-epoch-tag", "children"),
        Input("interval", "n_intervals"),
    )
    def update_samples(_n):
        snap    = data_store.get_snapshot()
        samples = snap["samples"][:10]
        epochs  = snap["epochs"]

        srcs, styles, labels = [], [], []
        base = {"width": "100%", "aspectRatio": "1/1", "objectFit": "cover", "borderRadius": "4px"}

        for i in range(10):
            if i < len(samples):
                s   = samples[i]
                b64 = s.get("image_b64", "")
                srcs.append(f"data:image/jpeg;base64,{b64}" if b64 else "")
                correct = s.get("true_label", "?") == s.get("pred_label", "!")
                styles.append({**base, "display": "block",
                                "border": f"2px solid {'#22c55e' if correct else '#ef4444'}"})
                conf = s.get("confidence", 0)
                labels.append(f"True: {s.get('true_label','?')}\nPred: {s.get('pred_label','?')}\nConf: {conf*100:.1f}%")
            else:
                srcs.append("")
                styles.append({**base, "display": "none"})
                labels.append("")

        return (*srcs, *styles, *labels, f"@ epoch {epochs[-1]}" if epochs else "")
