"""Audio loading and fixed-length segmentation.

AASIST and most ASVspoof systems use 16 kHz mono audio cut or padded to
64,600 samples (~4 s). Short clips are padded by repeating the signal (the
convention used by the official AASIST code), not by zeros, so silence padding
does not become a shortcut the model can learn.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
NUM_SAMPLES = 64_600


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target_sr:
        return x.astype(np.float32)
    g = gcd(sr, target_sr)
    return resample_poly(x, target_sr // g, sr // g).astype(np.float32)


def pad_or_crop(x: np.ndarray, n: int = NUM_SAMPLES, random_crop: bool = False,
                rng: np.random.Generator | None = None) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) >= n:
        start = 0
        if random_crop:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, len(x) - n + 1))
        return x[start:start + n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def segments(x: np.ndarray, n: int = NUM_SAMPLES, hop: int | None = None) -> np.ndarray:
    """Split a long clip into overlapping fixed-length windows (for scoring whole files
    and for the per-segment timeline in the report). Returns shape (num_windows, n)."""
    hop = hop or n // 2
    if len(x) <= n:
        return pad_or_crop(x, n)[None, :]
    starts = list(range(0, len(x) - n + 1, hop))
    if starts[-1] + n < len(x):
        starts.append(len(x) - n)
    return np.stack([x[s:s + n] for s in starts])


def _load_with_ffmpeg(path: str | Path, target_sr: int) -> np.ndarray:
    """Decode with ffmpeg to mono float32 at target_sr. ffmpeg's FLAC decoder tolerates
    stream quirks that libFLAC/libsndfile reject: about 42% of the official ASVspoof 2021
    DF eval files (MD5-verified against Zenodo) fail in soundfile but decode here."""
    import subprocess

    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as mono float32 at target_sr: soundfile first, ffmpeg if
    soundfile cannot decode it."""
    import shutil

    import soundfile as sf  # imported lazily: only needed when reading real files

    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except RuntimeError as e:  # sf.LibsndfileError subclasses RuntimeError
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"soundfile cannot read {path} ({e!r}) and ffmpeg is not installed") from e
        try:
            return _load_with_ffmpeg(path, target_sr)
        except Exception as e2:
            raise RuntimeError(f"neither soundfile ({e!r}) nor ffmpeg can read {path}") from e2
    return resample(to_mono(x), sr, target_sr)
