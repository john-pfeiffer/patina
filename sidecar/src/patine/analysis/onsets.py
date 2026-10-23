"""§4.4 Onsets: transient detection over the whole file, and the per-region
`first_onset_s` = first onset at or after the region's harmonic start.

Onsets are backtracked to the local minimum of the onset envelope that
precedes each peak, which is where the transient actually starts, so playback
from `first_onset_s` never clips an attack (§7.2)."""

from __future__ import annotations

import librosa
import numpy as np

from .tempo import ONSET_HOP, onset_envelope


def detect_onsets(
    env: np.ndarray | None,
    sr: int,
    hop_length: int = ONSET_HOP,
    *,
    y: np.ndarray | None = None,
) -> list[float]:
    """Onset times in seconds from an onset envelope (or from `y`)."""
    if env is None:
        if y is None:
            raise ValueError("detect_onsets needs an onset envelope or audio")
        env = onset_envelope(y, sr, hop_length)
    if env.size < 4 or float(env.max()) <= 0.0:
        return []
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop_length, units="frames", backtrack=True
    )
    frames = np.unique(np.asarray(frames, dtype=int))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    return [float(t) for t in np.atleast_1d(times)]


def first_onset_in(
    onsets: list[float] | np.ndarray,
    start_s: float,
    end_s: float,
    *,
    tolerance_s: float = 0.03,
) -> float:
    """First onset at or after `start_s` (an onset up to `tolerance_s` earlier
    counts as "at", clamped to the region). Falls back to `start_s`."""
    arr = np.asarray(onsets, dtype=np.float64)
    if arr.size == 0:
        return float(start_s)
    sel = arr[(arr >= start_s - tolerance_s) & (arr < end_s)]
    if sel.size == 0:
        return float(start_s)
    return float(max(start_s, sel.min()))


def snap_to_onset(t: float, onsets: list[float] | np.ndarray, tolerance_s: float) -> float:
    """Nearest onset within ±tolerance of `t`, else `t` itself."""
    arr = np.asarray(onsets, dtype=np.float64)
    if arr.size == 0:
        return float(t)
    i = int(np.argmin(np.abs(arr - t)))
    return float(arr[i]) if abs(arr[i] - t) <= tolerance_s else float(t)
