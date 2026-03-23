"""
utils/features.py
MFCC feature extraction — must match training-time parameters exactly.
"""

import numpy as np

# ── Feature hyperparameters (must match train.py) ───────────────────────────
N_MFCC      = 40    # number of MFCC coefficients
N_FFT       = 512   # FFT window size
HOP_LENGTH  = 160   # hop between frames (10 ms at 16 kHz)
WIN_LENGTH  = 400   # analysis window  (25 ms at 16 kHz)
N_MELS      = 64    # mel filterbanks used internally by librosa
MAX_FRAMES  = 128   # fixed time axis (pad/trim to this)
SR          = 16_000


def extract_mfcc(
    waveform: np.ndarray,
    sr: int = SR,
    n_mfcc: int = N_MFCC,
    max_frames: int = MAX_FRAMES,
) -> np.ndarray:
    """
    Extract MFCC features from a waveform.

    Parameters
    ----------
    waveform   : 1-D float32 array
    sr         : sample rate (resampled to SR if different)
    n_mfcc     : number of MFCC coefficients
    max_frames : fixed time axis length (pad or trim)

    Returns
    -------
    mfcc : np.ndarray  shape (n_mfcc, max_frames)  float32
    """
    import librosa

    if sr != SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SR)
        sr = SR

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000,
    )  # shape: (n_mfcc, T)

    # ── Fixed-length normalisation ───────────────────────────────────────────
    T = mfcc.shape[1]
    if T >= max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        pad = np.zeros((n_mfcc, max_frames - T), dtype=np.float32)
        mfcc = np.concatenate([mfcc, pad], axis=1)

    # ── Per-feature normalisation (mean 0, std 1) ────────────────────────────
    mean = mfcc.mean(axis=1, keepdims=True)
    std  = mfcc.std(axis=1, keepdims=True) + 1e-8
    mfcc = (mfcc - mean) / std

    return mfcc.astype(np.float32)


def batch_extract(waveforms: list, sr: int = SR) -> np.ndarray:
    """
    Extract MFCCs for a batch of waveforms.

    Returns
    -------
    np.ndarray  shape (N, N_MFCC, MAX_FRAMES, 1)  — ready for CNN input
    """
    features = [extract_mfcc(w, sr) for w in waveforms]
    arr = np.stack(features, axis=0)         # (N, 40, 128)
    arr = arr[..., np.newaxis]               # (N, 40, 128, 1)
    return arr.astype(np.float32)