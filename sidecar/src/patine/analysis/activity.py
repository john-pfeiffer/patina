"""§4.7 Busyness & register: browsing aids so a busy region can be paired
with a sparse one. They constrain nothing.

* `busyness` — melodic activity, 0..1: the rate at which the dominant pitch
  class of what *moves* in the upper register (C4–B7 chroma minus its
  median) changes, plus the average frame-to-frame movement; broadband
  transients and an empty upper register score 0.
* `register_centroid_hz` — harmonic register centroid: the magnitude-weighted
  log-frequency centroid of the CQT, in Hz, so it reads like a pitch rather
  than a brightness measure."""

from __future__ import annotations

import numpy as np

from .chroma import COMPRESSION, ChromaFeatures

_EPS = 1e-9


def busyness(
    feats: ChromaFeatures,
    *,
    gate: float = 0.05,
    peakiness: float = 0.35,
    change_threshold: float = 0.15,
) -> float:
    """Melodic activity 0..1. The sustained harmony is removed by subtracting
    the median upper-register chroma; what is left is "what moves". A frame
    counts as melodic when its residual is at least `gate` of the sustained
    level and concentrated on one pitch class (`peakiness`), which keeps
    broadband transients (drums, clicks) out. The score is the rate of
    pitch-class changes between melodic frames, plus their mean movement."""
    x = feats.treble.astype(np.float64)
    t = x.shape[1]
    if t < 2:
        return 0.0
    median = np.median(x, axis=1, keepdims=True)
    sustained = float(median.sum())
    if sustained <= _EPS and float(x.max()) <= _EPS:
        return 0.0
    resid = np.maximum(0.0, x - median)
    re = resid.sum(axis=0)
    share = resid.max(axis=0) / (re + _EPS)
    melodic = (re > gate * max(sustained, float(re.max()) * 0.1)) & (share > peakiness)
    both = melodic[1:] & melodic[:-1]
    if not np.any(both):
        return 0.0
    rn = resid / (np.linalg.norm(resid, axis=0) + _EPS)
    dist = 1.0 - np.sum(rn[:, 1:] * rn[:, :-1], axis=0)
    pc = np.argmax(resid, axis=0)
    changes = both & (pc[1:] != pc[:-1]) & (dist > change_threshold)
    rate = float(changes.sum()) / max(t * feats.hop_s, _EPS)
    movement = float(np.clip(dist[both].mean() / 0.3, 0.0, 1.0))
    return float(np.clip(0.7 * (1.0 - np.exp(-rate / 4.0)) + 0.3 * movement, 0.0, 1.0))


def register_centroid_hz(feats: ChromaFeatures) -> float:
    if feats.cqt.shape[1] == 0:
        return 0.0
    mag = np.power(feats.cqt.astype(np.float64), 1.0 / COMPRESSION)  # undo the compression
    w = mag.sum(axis=1)
    total = float(w.sum())
    if total <= _EPS:
        return 0.0
    return float(np.exp(np.sum(w * np.log(feats.freqs)) / total))
