"""
evaluate.py
VoiceIntent — standalone evaluation script.
Loads the saved model and produces accuracy, per-class report, and confusion matrix.

Usage
-----
    python evaluate.py --data data/fluent_speech_commands
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from utils.dataset import load_split


MODEL_PATH   = Path("models/voiceintent_cnn.keras")
ENCODER_PATH = Path("models/label_encoder.pkl")
OUTPUT_DIR   = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate(data_root: str, split: str = "test") -> None:
    # ── Load model & encoder ──────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train.py first.")

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    # ── Load data ─────────────────────────────────────────────────────────────
    csv_path = Path(data_root) / "data" / f"{split}_data.csv"
    print(f"Loading {split} split from {csv_path}...")
    X, y_true_str = load_split(str(csv_path), data_root)

    y_true = le.transform(y_true_str)
    n_classes = len(le.classes_)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"Running inference on {len(X):,} samples...")
    probs  = model.predict(X, batch_size=64, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = (y_pred == y_true).mean() * 100
    print(f"\nTop-1 Accuracy: {acc:.2f}%\n")
    print("Per-class report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#0a0a0f")

    im = ax.imshow(cm, cmap="magma", aspect="auto")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right", fontsize=7, color="#e8e8f0")
    ax.set_yticklabels(le.classes_, fontsize=7, color="#e8e8f0")
    ax.set_xlabel("Predicted", color="#5a5a7a", fontsize=10)
    ax.set_ylabel("True",      color="#5a5a7a", fontsize=10)
    ax.set_title("Confusion Matrix — VoiceIntent CNN", color="#e8e8f0", fontsize=12, pad=16)

    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e2e")

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="#5a5a7a", labelsize=7)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nConfusion matrix saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VoiceIntent CNN")
    parser.add_argument("--data",  type=str, default="data/fluent_speech_commands")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    evaluate(args.data, args.split)