"""§4.6 Key: whole-file key estimate with confidence, plus a `key_context` per
region — the key the region sits in, estimated over a window around it so it
can differ from the file key in modulating material.

Krumhansl–Kessler profiles correlated with energy-weighted mean chroma. The
bass-weighted chroma is used because roots carry most of the key evidence."""

from __future__ import annotations

import numpy as np

from ..theory import Key
from .chroma import ChromaFeatures

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _profiles() -> np.ndarray:
    rows = [np.roll(MAJOR_PROFILE, t) for t in range(12)] + [np.roll(MINOR_PROFILE, t) for t in range(12)]
    p = np.stack(rows)
    p -= p.mean(axis=1, keepdims=True)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


PROFILES = _profiles()  # (24, 12), zero-mean unit-norm rows


def key_name(index: int) -> str:
    return Key(int(index % 12), "major" if index < 12 else "minor").name()


def key_scores(mean_chroma: np.ndarray) -> np.ndarray:
    """Pearson correlation of `mean_chroma` with the 24 key profiles
    (0..11 major C..B, 12..23 minor C..B). Zeros for an empty chroma."""
    x = np.asarray(mean_chroma, dtype=np.float64)
    x = x - x.mean()
    n = float(np.linalg.norm(x))
    if n <= 1e-9:
        return np.zeros(24)
    return PROFILES @ (x / n)


def weighted_mean_chroma(chroma: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Energy-weighted mean of (12, T) chroma; silent frames contribute nothing."""
    if chroma.shape[1] == 0:
        return np.zeros(12)
    w = chroma.sum(axis=0).astype(np.float64)
    if weights is not None:
        w = w * np.asarray(weights, dtype=np.float64)
    if float(w.sum()) <= 1e-12:
        return chroma.mean(axis=1).astype(np.float64)
    return (chroma.astype(np.float64) * w[None, :]).sum(axis=1) / float(w.sum())


def estimate_key(feats: ChromaFeatures, *, frame_weights: np.ndarray | None = None) -> tuple[str | None, float, np.ndarray]:
    """(key name like 'F# minor' or None, confidence 0..1, the 24 scores)."""
    mean = weighted_mean_chroma(feats.weighted, frame_weights)
    scores = key_scores(mean)
    if not np.any(scores):
        return None, 0.0, scores
    order = np.argsort(scores)[::-1]
    best, second = float(scores[order[0]]), float(scores[order[1]])
    if best <= 0.0:
        return None, 0.0, scores
    margin = float(np.clip((best - second) / 0.15, 0.2, 1.0))
    confidence = float(np.clip(best, 0.0, 1.0) * margin)
    return key_name(int(order[0])), confidence, scores


def key_context(
    chroma: np.ndarray,
    times: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    global_scores: np.ndarray | None = None,
    window_s: float = 8.0,
    global_weight: float = 0.3,
) -> str | None:
    """Local key over [start - window, end + window] of a (12, T) chroma,
    with a mild prior toward the whole-file key so short regions do not flip."""
    t = np.asarray(times, dtype=np.float64)
    sel = (t >= start_s - window_s) & (t < end_s + window_s)
    if not np.any(sel):
        sel = np.ones(t.size, dtype=bool)
    mean = weighted_mean_chroma(chroma[:, sel])
    scores = key_scores(mean)
    if global_scores is not None and np.any(global_scores):
        scores = scores + global_weight * np.asarray(global_scores, dtype=np.float64)
    if not np.any(scores) or float(scores.max()) <= 0.0:
        if global_scores is not None and np.any(global_scores):
            return key_name(int(np.argmax(global_scores)))
        return None
    return key_name(int(np.argmax(scores)))
