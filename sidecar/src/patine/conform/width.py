"""§6 Width: scale Side relative to Mid per band (<200 Hz, 200 Hz–3 kHz,
>3 kHz) to match the anchor's `width_bands`, scaled by the amount, capped at
1.5× widening. Mid is untouched; a mono input stays mono."""

from __future__ import annotations

import numpy as np

from .filters import as_2d, split_bands

WIDEN_CAP = 1.5
BAND_EDGES_HZ = [200.0, 3000.0]


def width_scales(
    region_bands: list[float] | None, anchor_bands: list[float] | None, amount_pct: float | None
) -> list[float]:
    """Side amplitude scale per band after amount and cap. A band the region
    has no side energy in (mono) stays at 1 — there is nothing to scale."""
    if not region_bands or not anchor_bands:
        return [1.0] * 3
    amount = (100.0 if amount_pct is None else float(amount_pct)) / 100.0
    amount = min(max(amount, 0.0), 1.0)
    out: list[float] = []
    for i in range(3):
        r = region_bands[i] if i < len(region_bands) else None
        a = anchor_bands[i] if i < len(anchor_bands) else None
        if r is None or a is None or r <= 1e-6:
            out.append(1.0)
            continue
        target = float(np.sqrt(max(a, 0.0) / r))     # energy ratio → amplitude
        s = 1.0 + (target - 1.0) * amount
        out.append(float(min(max(s, 0.0), WIDEN_CAP)))
    return out


def apply_width(y: np.ndarray, sr: int, scales: list[float]) -> np.ndarray:
    y = as_2d(y)
    if y.shape[0] < 2 or y.shape[1] < 64 or all(abs(s - 1.0) < 1e-4 for s in scales):
        return y
    mid = (y[0] + y[1]) * 0.5
    side = (y[0] - y[1]) * 0.5
    bands = split_bands(side[None, :], sr, BAND_EDGES_HZ)
    side2 = sum(b[0] * np.float32(s) for b, s in zip(bands, scales, strict=False))
    out = y.copy()
    out[0] = mid + side2
    out[1] = mid - side2
    return out


def match_width(
    y: np.ndarray,
    sr: int,
    region_bands: list[float] | None,
    anchor_bands: list[float] | None,
    amount_pct: float | None,
) -> tuple[np.ndarray, list[float]]:
    """Returns (audio, side scale per band)."""
    y = as_2d(y)
    if y.shape[0] < 2:
        return y, [1.0, 1.0, 1.0]
    scales = width_scales(region_bands, anchor_bands, amount_pct)
    return apply_width(y, sr, scales), scales
