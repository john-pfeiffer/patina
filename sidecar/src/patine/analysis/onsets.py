"""§4.4 Onsets: transient detection over the whole file, and the per-region
`first_onset_s` = first onset at or after the region's harmonic start.

Onsets are backtracked to the local minimum of the onset envelope that
precedes each peak, which is where the transient actually starts, so playback
from `first_onset_s` never clips an attack (§7.2). A file that starts
abruptly (a one-shot trimmed to its attack) gets an onset at 0.0, which the
envelope alone cannot see because nothing precedes the first frame."""

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
    """Onset times in seconds from an onset envelope (or from `y`). When `y`
    is given, an abrupt start adds an onset at 0.0."""
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
    times = [float(t) for t in np.atleast_1d(librosa.frames_to_time(frames, sr=sr, hop_length=hop_length))]
    if y is not None and starts_abruptly(y, sr) and not any(t < 0.05 for t in times):
        times.insert(0, 0.0)
    return times


def starts_abruptly(y: np.ndarray, sr: int, *, head_s: float = 0.02, ratio: float = 0.25) -> bool:
    """True when the first `head_s` already carry a sizeable share of the
    file's loud-frame RMS, i.e. the audio begins mid-transient."""
    y = np.asarray(y, dtype=np.float32)
    head = y[: max(1, int(head_s * sr))]
    if head.size == 0:
        return False
    head_rms = float(np.sqrt(np.mean(head.astype(np.float64) ** 2)))
    frame = max(1, int(0.05 * sr))
    n = y.size // frame
    if n == 0:
        return False
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    loud = float(np.percentile(rms, 95))
    return loud > 1e-6 and head_rms >= ratio * loud


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
