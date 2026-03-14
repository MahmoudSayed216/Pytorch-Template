# ─────────────────────────────────────────────
#  data_store.py
#  Single source of truth for all live metrics.
#  Both the Flask ingest routes and Dash callbacks read/write here.
# ─────────────────────────────────────────────

import threading
import time

_lock = threading.Lock()

# ── Per-step data (updated every mini-batch) ──────────────────────────────────
_step_train_loss: list[float] = []

# ── Per-epoch data ─────────────────────────────────────────────────────────────
_epochs:          list[int]   = []
_avg_train_loss:  list[float] = []
_test_loss:       list[float] = []
_train_acc:       list[float] = []
_test_acc:        list[float] = []

# ── Sample images ──────────────────────────────────────────────────────────────
_samples: list[dict] = []

# ── Session metadata ───────────────────────────────────────────────────────────
_training_start_time: float | None = None   # unix timestamp, set on reset()
_run_configs: dict = {}                     # filtered configs sent from Kaggle


# ── Thread-safe accessors ──────────────────────────────────────────────────────

def append_step_loss(loss: float) -> None:
    with _lock:
        _step_train_loss.append(loss)

def append_epoch_metrics(epoch: int, avg_train_loss: float,
                         test_loss: float, train_acc: float, test_acc: float) -> None:
    with _lock:
        _epochs.append(epoch)
        _avg_train_loss.append(avg_train_loss)
        _test_loss.append(test_loss)
        _train_acc.append(train_acc)
        _test_acc.append(test_acc)

def set_samples(samples: list[dict]) -> None:
    with _lock:
        global _samples
        _samples = samples

def set_run_configs(configs: dict) -> None:
    with _lock:
        global _run_configs
        _run_configs = configs

def get_snapshot() -> dict:
    """Return a consistent copy of all data – safe to call from any thread."""
    with _lock:
        return {
            "step_train_loss":     list(_step_train_loss),
            "epochs":              list(_epochs),
            "avg_train_loss":      list(_avg_train_loss),
            "test_loss":           list(_test_loss),
            "train_acc":           list(_train_acc),
            "test_acc":            list(_test_acc),
            "samples":             list(_samples),
            "training_start_time": _training_start_time,
            "run_configs":         dict(_run_configs),
        }

def clear() -> None:
    """Reset everything and record training start time."""
    global _training_start_time
    with _lock:
        _step_train_loss.clear()
        _epochs.clear()
        _avg_train_loss.clear()
        _test_loss.clear()
        _train_acc.clear()
        _test_acc.clear()
        _samples.clear()
        _run_configs.clear()
        _training_start_time = time.time()
