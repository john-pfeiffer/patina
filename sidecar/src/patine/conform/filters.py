"""Shared DSP for the conform modules (handoff §6): octave-band measurement,
zero-phase crossovers, linear-phase FIR design from a band curve.

Everything works on float32 (channels, frames) arrays and never changes the
length: crossovers are forward-backward Butterworth (zero phase), FIRs are
linear-phase with the group delay removed."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..models import TILT_BAND_CENTRES_HZ

SQRT2 = float(np.sqrt(2.0))


def as_2d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    return np.ascontiguousarray(y)


def mono(y: np.ndarray) -> np.ndarray:
    y = as_2d(y)
    return y.mean(axis=0) if y.shape[0] > 1 else y[0]


def band_edges(centre_hz: float) -> tuple[float, float]:
    return centre_hz / SQRT2, centre_hz * SQRT2


def octave_band_energy_db(
    y: np.ndarray, sr: int, centres: list[float] | None = None, nperseg: int = 4096
) -> list[float]:
    """Long-term average spectrum (Welch) collapsed to octave bands, dB (§4.8
    `tilt_bands`). Bands above Nyquist report -120 dB."""
    centres = centres or TILT_BAND_CENTRES_HZ
    x = mono(y).astype(np.float64)
    if x.size < 64:
        return [-120.0] * len(centres)
    nperseg = min(nperseg, x.size)
    f, p = signal.welch(x, fs=sr, nperseg=nperseg)
    out: list[float] = []
    for c in centres:
        lo, hi = band_edges(c)
        if lo >= sr / 2:
            out.append(-120.0)
            continue
        m = (f >= lo) & (f < min(hi, sr / 2))
        e = float(np.sum(p[m])) if np.any(m) else 0.0
        out.append(10.0 * np.log10(max(e, 1e-12)))
    return out


def band_energy(y: np.ndarray, sr: int, lo_hz: float, hi_hz: float, nperseg: int = 4096) -> float:
    """Linear energy of the mono mix between lo_hz and hi_hz (Welch)."""
    x = mono(y).astype(np.float64)
    if x.size < 64:
        return 0.0
    f, p = signal.welch(x, fs=sr, nperseg=min(nperseg, x.size))
    m = (f >= lo_hz) & (f < hi_hz)
    return float(np.sum(p[m])) if np.any(m) else 0.0


def _butter_sos(sr: int, fc: float, kind: str, order: int = 2) -> np.ndarray:
    fc = float(min(max(fc, 1.0), sr / 2 * 0.98))
    return signal.butter(order, fc, btype=kind, fs=sr, output="sos")


def zero_phase_lowpass(y: np.ndarray, sr: int, fc: float, order: int = 2) -> np.ndarray:
    """Forward-backward Butterworth low-pass; complementary to
    `zero_phase_highpass` at the same fc and order (|L|² + |H|² = 1)."""
    y = as_2d(y)
    if y.shape[1] < 32:
        return y.copy()
    sos = _butter_sos(sr, fc, "low", order)
    return signal.sosfiltfilt(sos, y, axis=1).astype(np.float32)


def zero_phase_highpass(y: np.ndarray, sr: int, fc: float, order: int = 2) -> np.ndarray:
    y = as_2d(y)
    if y.shape[1] < 32:
        return y.copy()
    sos = _butter_sos(sr, fc, "high", order)
    return signal.sosfiltfilt(sos, y, axis=1).astype(np.float32)


def split_bands(y: np.ndarray, sr: int, edges_hz: list[float], order: int = 4) -> list[np.ndarray]:
    """Split into len(edges)+1 bands with zero-phase crossovers; the bands sum
    back to the input."""
    y = as_2d(y)
    bands: list[np.ndarray] = []
    rest = y
    for fc in edges_hz:
        low = zero_phase_lowpass(rest, sr, fc, order)
        bands.append(low)
        rest = rest - low
    bands.append(rest.astype(np.float32))
    return bands


def fir_from_band_curve(
    sr: int, centres: list[float], gains_db: list[float], numtaps: int | None = None
) -> np.ndarray:
    """Linear-phase FIR whose magnitude follows `gains_db` at the octave band
    centres, interpolated in log frequency (scipy.signal.firwin2). Bands at or
    above Nyquist are dropped; the curve is held flat beyond the last usable
    centre."""
    if numtaps is None:
        numtaps = int(round(0.15 * sr)) | 1
    nyq = sr / 2.0
    pts = [(c, g) for c, g in zip(centres, gains_db, strict=False) if c < nyq * 0.95]
    if not pts:
        return np.array([1.0], dtype=np.float64)
    cs = np.array([p[0] for p in pts])
    gs = np.array([p[1] for p in pts])
    # dense grid, log-frequency interpolation, flat beyond the ends
    grid = np.concatenate([[0.0], np.geomspace(10.0, nyq, 512)[:-1], [nyq]])
    logc = np.log(cs)
    logg = np.log(np.maximum(grid[1:-1], 1e-3))
    interp = np.interp(logg, logc, gs, left=gs[0], right=gs[-1])
    curve_db = np.concatenate([[gs[0]], interp, [gs[-1]]])
    gain = 10.0 ** (curve_db / 20.0)
    return signal.firwin2(numtaps, grid, gain, fs=sr)


def apply_fir_zero_delay(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve with a linear-phase FIR and remove its group delay so the
    output is time-aligned and the same length as the input."""
    y = as_2d(y)
    n = y.shape[1]
    if h.size <= 1:
        return (y * float(h[0]) if h.size == 1 else y).astype(np.float32)
    delay = (h.size - 1) // 2
    full = signal.fftconvolve(y.astype(np.float64), h[None, :], mode="full", axes=1)
    return np.ascontiguousarray(full[:, delay:delay + n]).astype(np.float32)


def rms_db(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return -120.0
    return float(10.0 * np.log10(max(np.mean(y ** 2), 1e-12)))


def frame_rms_db(y: np.ndarray, sr: int, frame_s: float = 0.05) -> np.ndarray:
    """RMS per non-overlapping frame of the mono mix, dBFS."""
    x = mono(y).astype(np.float64)
    n = max(1, int(round(frame_s * sr)))
    m = x.size // n
    if m == 0:
        return np.array([rms_db(x)])
    frames = x[: m * n].reshape(m, n)
    return 10.0 * np.log10(np.maximum(np.mean(frames ** 2, axis=1), 1e-12))


def noise_floor_db(y: np.ndarray, sr: int, frame_s: float = 0.05, fraction: float = 0.05) -> float:
    """Level of the quietest `fraction` of frames (§4.8 `noise_floor_db`): the
    mean power of those frames, in dBFS."""
    fr = frame_rms_db(y, sr, frame_s)
    k = max(1, int(round(fr.size * fraction)))
    quiet = np.sort(fr)[:k]
    return float(10.0 * np.log10(np.mean(10.0 ** (quiet / 10.0))))


def mid_side(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = as_2d(y)
    if y.shape[0] < 2:
        return y[0].copy(), np.zeros_like(y[0])
    return (y[0] + y[1]) * 0.5, (y[0] - y[1]) * 0.5


def width_bands(y: np.ndarray, sr: int, edges_hz: tuple[float, float] = (200.0, 3000.0)) -> list[float]:
    """Side/Mid energy ratio in <200 Hz, 200 Hz–3 kHz, >3 kHz (§4.8)."""
    y = as_2d(y)
    if y.shape[0] < 2:
        return [0.0, 0.0, 0.0]
    m, s = mid_side(y)
    out: list[float] = []
    lims = [(0.0, edges_hz[0]), (edges_hz[0], edges_hz[1]), (edges_hz[1], sr / 2.0)]
    for lo, hi in lims:
        em = band_energy(m, sr, lo, hi)
        es = band_energy(s, sr, lo, hi)
        out.append(es / em if em > 1e-15 else 0.0)
    return out
