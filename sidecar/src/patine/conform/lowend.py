"""§6 Low-end floor. If the region has more sub than the anchor (`sub_ratio`),
high-pass at the anchor's `low_rolloff_hz`. Mono below the anchor's
`bass_mono_hz` when the region's bass is stereo lower than the anchor's.
Never adds low end — every step only removes energy."""

from __future__ import annotations

import numpy as np

from ..models import TonalMetrics
from .filters import as_2d, zero_phase_highpass, zero_phase_lowpass

MIN_HPF_HZ = 15.0
MAX_HPF_HZ = 400.0
MIN_MONO_HZ = 20.0
MAX_MONO_HZ = 1000.0


def lowend_decision(
    region: TonalMetrics | None, anchor: TonalMetrics | None, channels: int
) -> tuple[float | None, float | None]:
    """(high-pass Hz or None, mono-below Hz or None) from the metrics."""
    if region is None or anchor is None:
        return None, None
    hpf: float | None = None
    if (
        region.sub_ratio is not None
        and anchor.sub_ratio is not None
        and region.sub_ratio > anchor.sub_ratio * 1.05 + 1e-6
        and anchor.low_rolloff_hz is not None
        and anchor.low_rolloff_hz >= MIN_HPF_HZ
    ):
        hpf = float(min(anchor.low_rolloff_hz, MAX_HPF_HZ))
    mono: float | None = None
    if (
        channels >= 2
        and anchor.bass_mono_hz is not None
        and anchor.bass_mono_hz >= MIN_MONO_HZ
        and (region.bass_mono_hz is None or region.bass_mono_hz < anchor.bass_mono_hz * 0.95)
    ):
        mono = float(min(anchor.bass_mono_hz, MAX_MONO_HZ))
    return hpf, mono


def apply_lowend(y: np.ndarray, sr: int, hpf_hz: float | None, mono_hz: float | None) -> np.ndarray:
    y = as_2d(y)
    if y.shape[1] < 64:
        return y
    out = y
    if hpf_hz is not None and hpf_hz > 0:
        out = zero_phase_highpass(out, sr, hpf_hz)
    if mono_hz is not None and mono_hz > 0 and out.shape[0] >= 2:
        low = zero_phase_lowpass(out, sr, mono_hz)
        high = out - low
        low_mono = low.mean(axis=0, keepdims=True)
        out = (high + low_mono).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def align_lowend(
    y: np.ndarray, sr: int, region: TonalMetrics | None, anchor: TonalMetrics | None
) -> tuple[np.ndarray, float | None, float | None]:
    """Returns (audio, hpf_hz applied or None, mono_hz applied or None)."""
    y = as_2d(y)
    hpf, mono = lowend_decision(region, anchor, y.shape[0])
    return apply_lowend(y, sr, hpf, mono), hpf, mono
