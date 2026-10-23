"""§6 Loudness: match integrated LUFS (ITU-R BS.1770, pyloudnorm) to the
anchor. Gain only — no limiting, no dynamics."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pyloudnorm as pyln

from .filters import as_2d, rms_db

log = logging.getLogger(__name__)

MAX_GAIN_DB = 24.0
SILENCE_LUFS = -70.0


def measure_lufs(y: np.ndarray, sr: int) -> float:
    """Integrated loudness. Short clips use a shorter gating block; silence
    returns -inf."""
    y = as_2d(y)
    n = y.shape[1]
    if n == 0 or not np.any(y):
        return float("-inf")
    duration = n / sr
    block = 0.4 if duration > 0.5 else max(0.05, duration * 0.8)
    if duration < 0.1:
        # too short for K-weighted gating; RMS in dBFS is the honest fallback
        return rms_db(y) - 0.691
    meter = pyln.Meter(sr, block_size=block)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = y.T.astype(np.float64)
        return float(meter.integrated_loudness(data))


def loudness_gain_db(measured_lufs: float, target_lufs: float | None) -> float:
    if target_lufs is None or not np.isfinite(measured_lufs) or measured_lufs < SILENCE_LUFS:
        return 0.0
    return float(np.clip(target_lufs - measured_lufs, -MAX_GAIN_DB, MAX_GAIN_DB))


def match_loudness(
    y: np.ndarray, sr: int, target_lufs: float | None, *, measured_lufs: float | None = None
) -> tuple[np.ndarray, float]:
    """Gain `y` so its integrated loudness equals `target_lufs`. Returns
    (audio, gain_db). `measured_lufs` skips the measurement when known."""
    y = as_2d(y)
    if target_lufs is None:
        return y, 0.0
    measured = measure_lufs(y, sr) if measured_lufs is None else measured_lufs
    gain_db = loudness_gain_db(measured, target_lufs)
    if gain_db == 0.0:
        return y, 0.0
    return (y * np.float32(10.0 ** (gain_db / 20.0))).astype(np.float32), gain_db
