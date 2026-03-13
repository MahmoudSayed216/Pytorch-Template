# ─────────────────────────────────────────────
#  Dashboard Configuration
#  Edit this file to adapt the dashboard to a new task
# ─────────────────────────────────────────────

TASK_NAME = "Dogs vs Cats Classification"
MODEL_NAME = "SimpleCNN"

# Server settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8050
FLASK_PORT  = 5000

# Metric display names  (used in graph titles / tooltips)
METRIC_LABELS = {
    "train_loss":  "Train Loss",
    "test_loss":   "Test Loss",
    "train_acc":   "Train Accuracy",
    "test_acc":    "Test Accuracy",
    "avg_train_loss": "Avg Train Loss",
}

# Color palette for the graphs
COLORS = {
    "train":   "#00e5ff",   # cyan
    "test":    "#ff6b35",   # orange
    "accent":  "#a855f7",   # purple
    "bg":      "#0a0e1a",
    "surface": "#111827",
    "border":  "#1f2937",
}

# How many random sample images to show
NUM_SAMPLES = 10

# Class labels for the task  (index → display name)
CLASS_NAMES = {0: "Cat", 1: "Dog"}
