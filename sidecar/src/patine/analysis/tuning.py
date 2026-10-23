"""§4.2 Tuning offset: deviation from A440 in cents over the whole file.

`librosa.estimate_tuning` is the floor. It pools every pitch track into one
histogram, so on dense material a few loud, slightly out-of-tune partials can
drag the estimate. This version keeps the same idea — a pitch-class residual
histogram against a shifted reference — but:

* pitch candidates come from `librosa.piptrack` over several windows and each
  window contributes with equal total weight, so a loud passage cannot own
  the histogram;
* candidates are weighted by salience (their magnitude), and only the salient
  half of each window is kept;
* the histogram is circular (±50 cents wrap) and lightly smoothed before the
  peak is read, with parabolic interpolation for sub-cent precision.

The result is capped to ±50 cents by construction."""

from __future__ import annotations

import logging

import librosa
import numpy as np

log = logging.getLogger(__name__)

RESOLUTION_CENTS = 1.0
_N_BINS = int(round(100.0 / RESOLUTION_CENTS))


def tuning_histogram(
    y: np.ndarray,
    sr: int,
    *,
    window_s: float = 20.0,
    fmin: float = 55.0,
    fmax: float = 2200.0,
    n_fft: int = 2048,
    hop_length: int = 1024,
) -> np.ndarray:
    """Circular histogram (100 bins of 1 cent, bin 0 = exactly in tune) of
    pitch residuals, salience-weighted and window-normalised."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    hist = np.zeros(_N_BINS, dtype=np.float64)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    win = max(n_fft, int(window_s * sr))
    for start in range(0, y.size, win):
        seg = y[start : start + win]
        if seg.size < n_fft:
            continue
        pitches, mags = librosa.piptrack(
            y=seg, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, threshold=0.1
        )
        sel = pitches > 0
        if not np.any(sel):
            continue
        p = pitches[sel]
        m = mags[sel]
        cut = np.median(m)
        keep = m >= cut
        p, m = p[keep], m[keep]
        if p.size == 0 or float(m.sum()) <= 0.0:
            continue
        residual = np.mod(12.0 * np.log2(p / 440.0), 1.0) * 100.0  # cents in [0, 100)
        bins = np.mod(np.round(residual / RESOLUTION_CENTS).astype(int), _N_BINS)
        w = m / float(m.sum())  # equal weight per window
        hist += np.bincount(bins, weights=w, minlength=_N_BINS)
    return hist


def estimate_tuning_cents(y: np.ndarray, sr: int, **kwargs: float) -> float:
    """Deviation from A440 in cents, in (-50, 50]. 0.0 when nothing pitched is found."""
    hist = tuning_histogram(y, sr, **kwargs)
    total = float(hist.sum())
    if total <= 0.0:
        log.debug("tuning: no pitched content, assuming A440")
        return 0.0
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    kernel /= kernel.sum()
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    smooth = np.convolve(padded, kernel, mode="valid")
    peak = int(np.argmax(smooth))
    # parabolic interpolation around the peak (circular)
    left = smooth[(peak - 1) % _N_BINS]
    mid = smooth[peak]
    right = smooth[(peak + 1) % _N_BINS]
    denom = left - 2.0 * mid + right
    offset = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
    cents = (peak + offset) * RESOLUTION_CENTS
    if cents > 50.0:
        cents -= 100.0
    cents = float(np.clip(cents, -50.0, 50.0))
    sharpness = float(smooth[peak] / max(np.mean(smooth), 1e-12))
    log.debug("tuning: %+.1f cents (peak/mean %.2f)", cents, sharpness)
    return cents
