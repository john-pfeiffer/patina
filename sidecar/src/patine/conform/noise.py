"""§6 Noise floor. If the region's floor is below the anchor's, bed a floor
under it. The floor is harvested from the anchor — its quietest contiguous
stretch of at least 200 ms, looped with a crossfade and levelled so the
region's floor comes up to the anchor's `noise_floor_db`. Never subtracts or
cleans. When the anchor has no quiet stretch (its quietest 200 ms is program
material, not floor), a floor is synthesised from white noise shaped by the
anchor's `noise_spectrum` (open question 5)."""

from __future__ import annotations

import logging

import numpy as np

from ..models import TILT_BAND_CENTRES_HZ
from .filters import apply_fir_zero_delay, as_2d, fir_from_band_curve, frame_rms_db, mono, rms_db

log = logging.getLogger(__name__)

MIN_FLOOR_S = 0.2
FRAME_S = 0.01
QUIET_MARGIN_DB = 6.0       # frames within this of the quietest window still count as floor
HARVEST_MAX_ABOVE_DB = 12.0  # harvested stretch louder than floor + this is not a floor
LOOP_XFADE_S = 0.02


def bed_level_db(region_floor_db: float | None, anchor_floor_db: float | None) -> float | None:
    """Level of the floor to add so that region ⊕ bed reaches the anchor's
    floor (power sum), or None when nothing should be added."""
    if region_floor_db is None or anchor_floor_db is None:
        return None
    if region_floor_db >= anchor_floor_db - 0.5:
        return None
    pa = 10.0 ** (anchor_floor_db / 10.0)
    pr = 10.0 ** (region_floor_db / 10.0)
    return float(10.0 * np.log10(max(pa - pr, 1e-12)))


def harvest_floor(anchor_y: np.ndarray, sr: int, min_len_s: float = MIN_FLOOR_S) -> np.ndarray | None:
    """The anchor's quietest contiguous ≥ `min_len_s` stretch (channels, frames),
    or None when there is none."""
    y = as_2d(anchor_y)
    n = y.shape[1]
    frame = max(1, int(round(FRAME_S * sr)))
    need = int(np.ceil(min_len_s * sr / frame))
    levels = frame_rms_db(y, sr, FRAME_S)
    if levels.size < need:
        return None
    # quietest window of `need` frames (mean power)
    power = 10.0 ** (levels / 10.0)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    window_power = (csum[need:] - csum[:-need]) / need
    start = int(np.argmin(window_power))
    floor_db = 10.0 * np.log10(max(window_power[start], 1e-12))
    # grow while neighbours stay within the margin
    lo, hi = start, start + need
    while lo > 0 and levels[lo - 1] <= floor_db + QUIET_MARGIN_DB:
        lo -= 1
    while hi < levels.size and levels[hi] <= floor_db + QUIET_MARGIN_DB:
        hi += 1
    a, b = lo * frame, min(n, hi * frame)
    if b - a < min_len_s * sr:
        return None
    return np.ascontiguousarray(y[:, a:b])


def loop_to_length(piece: np.ndarray, n: int, sr: int, xfade_s: float = LOOP_XFADE_S) -> np.ndarray:
    """Loop `piece` to `n` frames with a raised-cosine crossfade at each seam."""
    piece = as_2d(piece)
    ch, m = piece.shape
    if m == 0:
        return np.zeros((ch, n), dtype=np.float32)
    xf = min(int(round(xfade_s * sr)), m // 3)
    out = np.zeros((ch, n + m), dtype=np.float32)
    hop = m - xf
    fade_in = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf, endpoint=False))).astype(np.float32)
    pos = 0
    first = True
    while pos < n:
        seg = piece.copy()
        if xf:
            if not first:
                seg[:, :xf] *= fade_in
            seg[:, m - xf:] *= 1.0 - fade_in
        out[:, pos:pos + m] += seg
        pos += hop if hop > 0 else m
        first = False
    return out[:, :n]


def synth_floor(sr: int, n: int, channels: int, spectrum_db: list[float] | None, seed: int = 0) -> np.ndarray:
    """White noise shaped by the anchor's octave-band noise spectrum."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((channels, n)).astype(np.float32) * 0.1
    if spectrum_db and any(abs(v) > 1e-6 for v in spectrum_db) and n > 64:
        ref = max(spectrum_db)
        rel = [v - ref if v > -100.0 else -60.0 for v in spectrum_db]
        h = fir_from_band_curve(sr, list(TILT_BAND_CENTRES_HZ), rel, numtaps=int(round(0.03 * sr)) | 1)
        noise = apply_fir_zero_delay(noise, h)
    return noise


def level_to_db(piece: np.ndarray, target_db: float) -> np.ndarray:
    """Scale so the mono mix's RMS is `target_db` — the same measure as the
    floor metric, so a decorrelated stereo bed lands where it should."""
    cur = rms_db(mono(piece))
    if cur <= -119.0:
        return piece
    return (piece * np.float32(10.0 ** ((target_db - cur) / 20.0))).astype(np.float32)


def bed_noise_floor(
    y: np.ndarray,
    sr: int,
    region_floor_db: float | None,
    anchor_floor_db: float | None,
    anchor_y: np.ndarray | None,
    anchor_spectrum: list[float] | None = None,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, float | None]:
    """Returns (audio, level of the bedded floor in dBFS or None)."""
    y = as_2d(y)
    level = bed_level_db(region_floor_db, anchor_floor_db)
    if level is None or y.shape[1] == 0:
        return y, None
    ch, n = y.shape
    piece = harvest_floor(anchor_y, sr) if anchor_y is not None else None
    if piece is not None and anchor_floor_db is not None and rms_db(mono(piece)) > anchor_floor_db + HARVEST_MAX_ABOVE_DB:
        log.info("noise: anchor's quietest stretch is %.1f dB, not a floor; synthesising", rms_db(mono(piece)))
        piece = None
    if piece is None:
        bed = synth_floor(sr, n, ch, anchor_spectrum, seed)
    else:
        if piece.shape[0] != ch:
            piece = np.broadcast_to(piece.mean(axis=0, keepdims=True), (ch, piece.shape[1])).copy()
        bed = loop_to_length(piece, n, sr)
    bed = level_to_db(bed, level)
    return (y + bed).astype(np.float32), level
