"""§4.6 Key: whole-file key estimate with confidence, plus a `key_context` per
region — the key the region sits in, estimated over a window around it so it
can differ from the file key in modulating material.

Krumhansl–Kessler profiles correlated with a pitch-class histogram. The
histogram is built from the frame-wise chord labels (§4.5) — duration-weighted
chord-tone templates — blended with the energy-weighted mean chroma. The chord
labels are free of harmonics and melody noise (a loud 5th partial otherwise
turns every minor key major); the chroma keeps the estimate alive on material
the chord model called no-chord. A diatonic term (the share of the labelled
chords whose triad sits inside each candidate key) separates closely related
keys such as C and G major, which the profiles alone confuse."""

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


ChordFrame = tuple[str | None, str | None, float]

CHORD_WEIGHT = 0.7      # share of the chord-label histogram in the blend
DIATONIC_WEIGHT = 0.5   # weight of the diatonic-chord share
_ROOT_EMPHASIS = 1.5    # the root of a chord says more about the key than its fifth


def chord_tone_histogram(chord_frames: list[ChordFrame] | np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    """(12,) pitch-class histogram of the labelled frames (no-chord frames add
    nothing). `sel` restricts the frames."""
    from ..theory import QUALITY_INTERVALS, parse_note

    hist = np.zeros(12)
    for i, (root, quality, conf) in enumerate(chord_frames):
        if root is None or quality is None or (sel is not None and not sel[i]):
            continue
        pc, _ = parse_note(root)
        w = 0.5 + 0.5 * float(conf)
        for iv in QUALITY_INTERVALS.get(quality, (0, 4, 7)):
            hist[(pc + iv) % 12] += w * (_ROOT_EMPHASIS if iv == 0 else 1.0)
    return hist


def diatonic_share(chord_frames: list[ChordFrame] | np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    """(24,) share of the labelled frames whose chord triad is diatonic in each key."""
    from ..theory import Chord, is_diatonic, parse_note

    counts: dict[tuple[int, str], int] = {}
    for i, (root, quality, _conf) in enumerate(chord_frames):
        if root is None or quality is None or (sel is not None and not sel[i]):
            continue
        k = (parse_note(root)[0], quality)
        counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return np.zeros(24)
    out = np.zeros(24)
    for idx in range(24):
        key = Key(idx % 12, "major" if idx < 12 else "minor")
        out[idx] = sum(n for (pc, q), n in counts.items() if is_diatonic(Chord(pc, q), key)) / total
    return out


def blended_scores(chroma: np.ndarray, chord_frames: list[ChordFrame] | np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    """24 key scores: chord-label histogram and chroma correlations blended,
    plus the diatonic share of the labelled chords."""
    chroma_sel = chroma if sel is None else chroma[:, sel]
    s_chroma = key_scores(weighted_mean_chroma(chroma_sel))
    hist = chord_tone_histogram(chord_frames, sel) if len(chord_frames) else np.zeros(12)
    if not np.any(hist):
        return s_chroma
    blend = CHORD_WEIGHT * key_scores(hist) + (1.0 - CHORD_WEIGHT) * s_chroma
    return blend + DIATONIC_WEIGHT * diatonic_share(chord_frames, sel)


def _pick(scores: np.ndarray) -> tuple[str | None, float]:
    if not np.any(scores):
        return None, 0.0
    order = np.argsort(scores)[::-1]
    best, second = float(scores[order[0]]), float(scores[order[1]])
    if best <= 0.0:
        return None, 0.0
    margin = float(np.clip((best - second) / 0.15, 0.2, 1.0))
    return key_name(int(order[0])), float(np.clip(best, 0.0, 1.0) * margin)


def estimate_key(
    feats: ChromaFeatures,
    *,
    chord_frames: list[ChordFrame] | None = None,
    frame_weights: np.ndarray | None = None,
) -> tuple[str | None, float, np.ndarray]:
    """(key name like 'F# minor' or None, confidence 0..1, the 24 scores).
    Without `chord_frames` only the chroma is used."""
    if chord_frames:
        scores = blended_scores(feats.weighted, chord_frames)
    else:
        scores = key_scores(weighted_mean_chroma(feats.weighted, frame_weights))
    name, confidence = _pick(scores)
    return name, confidence, scores


def key_context(
    chroma: np.ndarray,
    times: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    chord_frames: list[ChordFrame] | None = None,
    global_scores: np.ndarray | None = None,
    window_s: float = 8.0,
    global_weight: float = 0.3,
) -> str | None:
    """Local key over [start - window, end + window] of a (12, T) chroma (and
    the matching chord labels when given), with a mild prior toward the
    whole-file key so short regions do not flip."""
    t = np.asarray(times, dtype=np.float64)
    sel = (t >= start_s - window_s) & (t < end_s + window_s)
    if not np.any(sel):
        sel = np.ones(t.size, dtype=bool)
    if chord_frames is not None and len(chord_frames) == t.size:
        scores = blended_scores(chroma, chord_frames, sel)
    else:
        scores = key_scores(weighted_mean_chroma(chroma[:, sel]))
    if global_scores is not None and np.any(global_scores):
        scores = scores + global_weight * np.asarray(global_scores, dtype=np.float64)
    if not np.any(scores) or float(scores.max()) <= 0.0:
        if global_scores is not None and np.any(global_scores):
            return key_name(int(np.argmax(global_scores)))
        return None
    return key_name(int(np.argmax(scores)))
