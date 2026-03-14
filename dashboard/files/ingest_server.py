# ─────────────────────────────────────────────
#  ingest_server.py
# ─────────────────────────────────────────────

from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading
import data_store
from config import FLASK_PORT

ingest_app = Flask(__name__)

@ingest_app.route("/log/step", methods=["POST"])
def log_step():
    data = request.get_json(force=True)
    data_store.append_step_loss(float(data["loss"]))
    return jsonify({"status": "ok"}), 200

@ingest_app.route("/log/epoch", methods=["POST"])
def log_epoch():
    data = request.get_json(force=True)
    data_store.append_epoch_metrics(
        epoch=int(data["epoch"]),
        avg_train_loss=float(data["avg_train_loss"]),
        test_loss=float(data["test_loss"]),
        train_acc=float(data["train_acc"]),
        test_acc=float(data["test_acc"]),
    )
    return jsonify({"status": "ok"}), 200

@ingest_app.route("/log/samples", methods=["POST"])
def log_samples():
    data = request.get_json(force=True)
    data_store.set_samples(data.get("samples", []))
    return jsonify({"status": "ok"}), 200

@ingest_app.route("/log/configs", methods=["POST"])
def log_configs():
    """
    Receive filtered training configs from Kaggle.
    Expected JSON: {"configs": {"model": {...}, "training": {...}, ...}}
    """
    data = request.get_json(force=True)
    data_store.set_run_configs(data.get("configs", {}))
    return jsonify({"status": "ok"}), 200

@ingest_app.route("/reset", methods=["POST"])
def reset():
    data_store.clear()
    return jsonify({"status": "cleared"}), 200

@ingest_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "alive"}), 200

def start_ingest_server(use_ngrok: bool = True, ngrok_token: str = "") -> str:
    public_url = f"http://localhost:{FLASK_PORT}"
    if use_ngrok and ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        tunnel = ngrok.connect(FLASK_PORT)
        public_url = tunnel.public_url
        print(f"[Ingest] ngrok public URL: {public_url}")
    else:
        print(f"[Ingest] Running locally on port {FLASK_PORT}")
    thread = threading.Thread(
        target=lambda: ingest_app.run(port=FLASK_PORT, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return public_url
