# ─────────────────────────────────────────────────────────────────────────────
#  dashboard_reporter.py  –  runs on Kaggle
#
#  Drop this file next to your train.py on Kaggle.
#  It provides one class:  DashboardReporter
#
#  Usage inside your training loop:
#
#      reporter = DashboardReporter(server_url="https://xxxx.ngrok-free.app")
#      reporter.reset()                            # clear old data at run start
#
#      # inside mini-batch loop:
#      reporter.log_step(loss=loss.item())
#
#      # end of epoch:
#      reporter.log_epoch(epoch, avg_train_loss, test_loss, train_acc, test_acc)
#
#      # send sample predictions (call once after computing them):
#      reporter.log_samples(images, true_labels, pred_labels, confidences)
#
# ─────────────────────────────────────────────────────────────────────────────

import base64
import io
import requests
import urllib3
import numpy as np
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DashboardReporter:
    """
    Sends training metrics to the remote Dash dashboard via HTTP POST.
    All methods are fire-and-forget: failures are printed but never raise.
    """

    def __init__(self, server_url: str, timeout: float = 3.0):
        """
        Args:
            server_url: Base URL of the ingest server, e.g.
                        "https://xxxx.ngrok-free.app"  (no trailing slash)
            timeout:    Seconds to wait for each request.
        """
        self.base = server_url.rstrip("/")
        self.timeout = timeout

    # ── Internal ──────────────────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: dict) -> None:
        try:
            requests.post(
                f"{self.base}{endpoint}",
                json=payload,
                timeout=self.timeout,
                headers={
                    "Content-Type": "application/json",
                    "ngrok-skip-browser-warning": "true",   # bypass ngrok interstitial page
                },
                verify=False,   # tolerate self-signed / unexpected EOF on free ngrok tunnels
            )
        except Exception as e:
            print(f"[DashboardReporter] WARNING – could not reach {endpoint}: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all stored data on the server (call at the start of a run)."""
        self._post("/reset", {})

    def log_step(self, loss: float) -> None:
        """Send per-mini-batch train loss."""
        self._post("/log/step", {"loss": loss})

    def log_epoch(
        self,
        epoch: int,
        avg_train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float,
    ) -> None:
        """Send all per-epoch metrics."""
        self._post("/log/epoch", {
            "epoch":          epoch,
            "avg_train_loss": avg_train_loss,
            "test_loss":      test_loss,
            "train_acc":      train_acc,
            "test_acc":       test_acc,
        })

    def log_samples(
        self,
        images,          # list/array of images – numpy [H,W,C] uint8  or PIL Images
        true_labels,     # list of str  e.g. ["Dog", "Cat", ...]
        pred_labels,     # list of str
        confidences,     # list of float (0-1)
    ) -> None:
        """
        Encode images as base64 JPEG and send them with their labels.
        Silently skips images that can't be encoded.
        """
        samples = []
        for img, tl, pl, conf in zip(images, true_labels, pred_labels, confidences):
            try:
                if isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img.astype(np.uint8))
                elif isinstance(img, Image.Image):
                    pil_img = img
                else:
                    continue

                # Resize to 128×128 to keep payloads small
                pil_img = pil_img.resize((128, 128), Image.BILINEAR)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                samples.append({
                    "image_b64":  b64,
                    "true_label": str(tl),
                    "pred_label": str(pl),
                    "confidence": float(conf),
                })
            except Exception as e:
                print(f"[DashboardReporter] WARNING – could not encode sample: {e}")

        if samples:
            self._post("/log/samples", {"samples": samples})