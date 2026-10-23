"""§6 Tilt: octave-band difference curve (region → anchor) from `tilt_bands`,
capped ±6 dB per band, scaled by the amount, applied as a linear-phase FIR
with its group delay removed. Coarse on purpose — match the tilt and the
room, not the fingerprint."""

from __future__ import annotations

import numpy as np

from ..models import TILT_BAND_CENTRES_HZ
from .filters import apply_fir_zero_delay, as_2d, fir_from_band_curve

CAP_DB = 6.0


def tilt_correction_db(
    region_bands: list[float] | None,
    anchor_bands: list[float] | None,
    amount_pct: float | None,
    cap_db: float = CAP_DB,
) -> list[float]:
    """Per-band gain to apply, after cap and amount. Bands measured at -120 dB
    (above Nyquist / empty) on either side are left at 0."""
    n = len(TILT_BAND_CENTRES_HZ)
    if not region_bands or not anchor_bands:
        return [0.0] * n
    amount = (100.0 if amount_pct is None else float(amount_pct)) / 100.0
    amount = min(max(amount, 0.0), 1.0)
    out: list[float] = []
    for i in range(n):
        r = region_bands[i] if i < len(region_bands) else None
        a = anchor_bands[i] if i < len(anchor_bands) else None
        if r is None or a is None or r <= -100.0 or a <= -100.0:
            out.append(0.0)
            continue
        diff = float(np.clip(a - r, -cap_db, cap_db))
        out.append(diff * amount)
    return out


def apply_tilt(y: np.ndarray, sr: int, correction_db: list[float]) -> np.ndarray:
    y = as_2d(y)
    if not any(abs(c) > 1e-3 for c in correction_db) or y.shape[1] < 64:
        return y
    h = fir_from_band_curve(sr, list(TILT_BAND_CENTRES_HZ), list(correction_db))
    return apply_fir_zero_delay(y, h)


def match_tilt(
    y: np.ndarray,
    sr: int,
    region_bands: list[float] | None,
    anchor_bands: list[float] | None,
    amount_pct: float | None,
) -> tuple[np.ndarray, list[float]]:
    """Returns (audio, per-band dB applied)."""
    corr = tilt_correction_db(region_bands, anchor_bands, amount_pct)
    return apply_tilt(y, sr, corr), corr
