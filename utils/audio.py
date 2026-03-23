"""
utils/audio.py
Audio I/O helpers — handles both dataset WAV files AND browser mic recordings.

Browser mic audio differs from dataset audio in several ways:
  - Recorded at 48kHz (not 16kHz)
  - Often stereo
  - Has automatic gain control / noise suppression applied by browser
  - Encoded as OGG/WebM or WAV depending on browser
  - May have leading/trailing silence
  - Different amplitude characteristics

This module normalises all of these differences before feature extraction.
"""

import io
import numpy as np

TARGET_SR     = 16_000
CLIP_DURATION = 3.0


def bytes_to_array(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert raw audio bytes to a clean mono float32 array at TARGET_SR.
    Handles both clean dataset WAV files and noisy browser mic recordings.
    """
    import librosa

    buf = io.BytesIO(audio_bytes)

    # librosa handles WAV, FLAC, OGG, MP3, WebM reliably
    # sr=None preserves original sample rate so we can resample properly
    try:
        waveform, sr = librosa.load(buf, sr=None, mono=True)
    except Exception:
        # Fallback: try soundfile then force mono manually
        import soundfile as sf
        buf.seek(0)
        waveform, sr = sf.read(buf, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)

    # ── Step 1: Resample to 16kHz ─────────────────────────────────────────
    # Browser mic is typically 48kHz. Model expects 16kHz.
    if sr != TARGET_SR:
        waveform = librosa.resample(
            waveform,
            orig_sr=sr,
            target_sr=TARGET_SR,
            res_type="kaiser_best",   # high quality resample
        )
        sr = TARGET_SR

    # ── Step 2: Trim leading/trailing silence ─────────────────────────────
    # Browser recordings often have 0.2-0.5s of silence before speech starts.
    # top_db=20 is gentle — only removes true silence, not quiet speech.
    waveform, _ = librosa.effects.trim(waveform, top_db=20, frame_length=512, hop_length=128)

    # ── Step 3: Robust amplitude normalisation ────────────────────────────
    # Browser AGC means amplitude varies wildly between recordings.
    # RMS normalisation matches what the training data looks like better
    # than simple peak normalisation.
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms > 1e-6:
        # Target RMS of 0.1 — matches typical FSC dataset level
        waveform = waveform * (0.1 / rms)
    
    # Hard clip to prevent any values > 1.0 after RMS normalisation
    waveform = np.clip(waveform, -1.0, 1.0)

    return waveform.astype(np.float32), sr


def load_file(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file from disk."""
    with open(path, "rb") as f:
        return bytes_to_array(f.read())


def pad_or_trim(waveform: np.ndarray, sr: int,
                duration: float = CLIP_DURATION) -> np.ndarray:
    """Pad with zeros or trim to a fixed length."""
    target_len = int(sr * duration)
    if len(waveform) >= target_len:
        return waveform[:target_len]
    pad = np.zeros(target_len - len(waveform), dtype=np.float32)
    return np.concatenate([waveform, pad])