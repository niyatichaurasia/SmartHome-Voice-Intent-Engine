"""
utils/dataset.py
Fluent Speech Commands dataset loader.

Expected dataset layout (after downloading from https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/):

    data/fluent_speech_commands/
    ├── data/
    │   ├── train_data.csv
    │   ├── valid_data.csv
    │   └── test_data.csv
    └── wavs/
        └── speakers/
            └── .../*.wav

CSV columns: path, speakerId, transcription, action, object, location
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from utils.audio    import load_file, pad_or_trim
from utils.features import extract_mfcc, MAX_FRAMES, N_MFCC
from utils.intents  import build_intent_label


def load_split(
    csv_path: str,
    data_root: str,
    max_samples: int = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one CSV split of the Fluent Speech Commands dataset.

    Parameters
    ----------
    csv_path    : path to train_data.csv / valid_data.csv / test_data.csv
    data_root   : root of the dataset (parent of the 'wavs' folder)
    max_samples : cap for quick debugging runs

    Returns
    -------
    X : np.ndarray  shape (N, N_MFCC, MAX_FRAMES, 1)
    y : np.ndarray  shape (N,)  string intent labels
    """
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    X_list, y_list = [], []
    skipped = 0

    for i, row in df.iterrows():
        wav_path = Path(data_root) / row["path"]
        if not wav_path.exists():
            skipped += 1
            continue

        try:
            waveform, sr = load_file(str(wav_path))
            waveform     = pad_or_trim(waveform, sr)
            mfcc         = extract_mfcc(waveform, sr)          # (40, 128)
            intent       = build_intent_label(
                row["action"], row["object"], row["location"]
            )
            X_list.append(mfcc)
            y_list.append(intent)
        except Exception as e:
            skipped += 1
            continue

        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(df)} — skipped {skipped}")

    X = np.stack(X_list, axis=0)[..., np.newaxis]   # (N, 40, 128, 1)
    y = np.array(y_list)

    if verbose:
        print(f"Loaded {len(X)} samples (skipped {skipped})")

    return X.astype(np.float32), y


def load_all_splits(data_root: str, max_per_split: int = None):
    """
    Convenience loader for all three splits.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    root = Path(data_root)
    splits = {}
    for split in ("train", "valid", "test"):
        csv = root / "data" / f"{split}_data.csv"
        print(f"\nLoading {split} split...")
        X, y = load_split(str(csv), str(root), max_samples=max_per_split)
        splits[split] = (X, y)

    return splits["train"], splits["valid"], splits["test"]