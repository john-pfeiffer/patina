"""§4.7 Busyness & register: browsing aids so a busy region can be paired
with a sparse one. They constrain nothing.

* `busyness` — melodic activity, 0..1: the rate at which the dominant pitch
  class of the upper register (C4–B7 chroma) changes, plus the average
  frame-to-frame chroma movement, gated so an empty upper register scores 0.
* `register_centroid_hz` — harmonic register centroid: the magnitude-weighted
  log-frequency centroid of the CQT, in Hz, so it reads like a pitch rather
  than a brightness measure."""

from __future__ import annotations

import numpy as np

from .chroma import COMPRESSION, ChromaFeatures

_EPS = 1e-9


def busyness(feats: ChromaFeatures, *, gate_db: float = -30.0, change_threshold: float = 0.15) -> float:
    x = feats.treble.astype(np.float64)
    t = x.shape[1]
    if t < 2:
        return 0.0
    e = x.sum(axis=0)
    ref = float(e.max())
    if ref <= _EPS:
        return 0.0
    active = e > ref * 10.0 ** (gate_db / 20.0)
    xn = x / (np.linalg.norm(x, axis=0) + _EPS)
    pc = np.argmax(x, axis=0)
    both = active[1:] & active[:-1]
    if not np.any(both):
        return 0.0
    dist = 1.0 - np.sum(xn[:, 1:] * xn[:, :-1], axis=0)
    changes = both & (pc[1:] != pc[:-1]) & (dist > change_threshold)
    duration = t * feats.hop_s
    rate = float(changes.sum()) / max(duration, _EPS)
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
