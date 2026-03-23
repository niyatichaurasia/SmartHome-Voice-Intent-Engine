"""
train.py
VoiceIntent — model training script.

Usage
-----
    python train.py --data data/fluent_speech_commands --epochs 30

Or import train_model() directly (called from the Streamlit sidebar).
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from models.model  import build_cnn, compile_model
from utils.dataset import load_all_splits


# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_model(
    data_root: str,
    epochs: int = 30,
    batch_size: int = 64,
    callback=None,          # optional progress callback(epoch, total, loss, acc)
    max_per_split: int = None,
) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    data_root     : path to the fluent_speech_commands directory
    epochs        : training epochs
    batch_size    : mini-batch size
    callback      : optional fn(epoch, total, loss, acc) for UI progress
    max_per_split : cap samples per split (set ~500 for fast debug runs)

    Returns
    -------
    dict with keys: val_acc, val_loss, history
    """
    print("=" * 60)
    print("VoiceIntent — Training")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    t0 = time.time()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_all_splits(
        data_root, max_per_split=max_per_split
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Feature shape: {X_train.shape[1:]}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 2. Encode labels ──────────────────────────────────────────────────────
    print("\n[2/4] Encoding labels...")
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_val, y_test]))
    n_classes = len(le.classes_)
    print(f"  {n_classes} intent classes: {list(le.classes_[:5])} ...")

    def to_onehot(y_str):
        idx = le.transform(y_str)
        return tf.keras.utils.to_categorical(idx, num_classes=n_classes)

    Y_train = to_onehot(y_train)
    Y_val   = to_onehot(y_val)
    Y_test  = to_onehot(y_test)

    # Save label encoder
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)
    print(f"  Label encoder saved → {encoder_path}")

    # ── 3. Build & compile model ──────────────────────────────────────────────
    print("\n[3/4] Building model...")
    model = build_cnn(n_classes=n_classes, input_shape=X_train.shape[1:])
    model = compile_model(model, n_classes)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_path = MODEL_DIR / "voiceintent_cnn.keras"

    keras_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Inject UI progress callback if provided
    if callback:
        class UICallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                callback(
                    epoch + 1,
                    epochs,
                    float(logs.get("loss", 0)),
                    float(logs.get("accuracy", 0)),
                )
        keras_callbacks.append(UICallback())

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Training for up to {epochs} epochs...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=keras_callbacks,
        verbose=1,
    )

    # ── 5. Evaluate on test set ───────────────────────────────────────────────
    print("\n[Eval] Test set evaluation...")
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"  Test accuracy : {test_acc*100:.2f}%")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"\n✓ Best model saved → {checkpoint_path}")

    return {
        "val_acc":  max(history.history.get("val_accuracy", [0])),
        "val_loss": min(history.history.get("val_loss", [0])),
        "test_acc": test_acc,
        "history":  history.history,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VoiceIntent CNN")
    parser.add_argument("--data",       type=str, default="data/fluent_speech_commands",
                        help="Path to Fluent Speech Commands dataset root")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Mini-batch size")
    parser.add_argument("--max-samples",type=int, default=None,
                        help="Cap samples per split (for quick debug runs)")
    args = parser.parse_args()

    metrics = train_model(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_per_split=args.max_samples,
    )

    print("\n" + "=" * 60)
    print(f"Training complete.")
    print(f"  Best val accuracy : {metrics['val_acc']*100:.2f}%")
    print(f"  Test accuracy     : {metrics['test_acc']*100:.2f}%")
    print("=" * 60)