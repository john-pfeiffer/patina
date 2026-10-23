"""§6 Tuning: correct a sample's tuning offset by resampling. Runs first,
before the temporal layer; the length change it causes is absorbed by the
stretch stage (`warp.scale_plan_source`).

Target: the anchor's tuning when there is one — the anchor is identity, so a
member is conformed *to the anchor's* offset rather than to A440 (otherwise a
sharp anchor and an A440-corrected member would disagree); with no anchor the
target is A440."""

from __future__ import annotations

import numpy as np

from ..temporal.stretch import resample_ratio


def tuning_shift_cents(sample_cents: float | None, anchor_cents: float | None) -> float:
    """Cents to shift the sample by (positive = up) so it matches the anchor's
    tuning, or A440 when there is no anchor."""
    if sample_cents is None:
        return 0.0
    target = float(anchor_cents) if anchor_cents is not None else 0.0
    return target - float(sample_cents)


def speed_for_cents(shift_cents: float) -> float:
    """Playback speed that shifts pitch by `shift_cents` (2 = +1200 cents)."""
    return float(2.0 ** (shift_cents / 1200.0))


def correct_tuning(y: np.ndarray, sr: int, shift_cents: float) -> tuple[np.ndarray, float]:
    """Resample so pitch moves by `shift_cents`. Returns (audio, speed): the
    audio's time axis is scaled by 1 / speed, which the caller folds into the
    warp plan."""
    speed = speed_for_cents(shift_cents)
    if abs(shift_cents) < 0.05:
        return np.asarray(y, dtype=np.float32), 1.0
    return resample_ratio(y, sr, speed), speed
