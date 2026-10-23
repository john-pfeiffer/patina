"""§4.5 Chord estimation: frame-wise over the whole file, then segmented into
harmonically stable regions.

Pipeline
--------
1. A :class:`ChordBackend` turns chroma features into per-frame observation
   probabilities over 25 states: 12 roots × {maj, min} plus a no-chord state.
   The default :class:`TemplateBackend` matches **bass-weighted** chroma
   (see `chroma.py`) against triad templates, so a melody line on top cannot
   flip the estimate the way plain chroma template matching can. A learned
   root+maj/min model plugs in through the same protocol (`learned.py`).
2. The observation probabilities are smoothed with an HMM (Viterbi path plus
   forward–backward posteriors for confidence).
3. Runs of equal state become segments. Segments shorter than one beat merge
   into the neighbour whose chord fits them best; no-chord stretches are not
   regions, and short no-chord dips inside a chord are absorbed.
4. Inside each root segment a second, small HMM over the eight qualities of
   `theory.ANALYSIS_QUALITIES` (maj min 7 m7 maj7 dim sus2 sus4) runs on the
   bass-weighted chroma, so a Cmaj7 → C7 change on the same root still splits
   the region; its runs are merged by the same one-beat rule. Per segment the
   quality is then confirmed by template matching on the mean chroma, the bass
   note is read from the low-register chroma so inversions are visible, and
   `chord_confidence` (0..1) blends the template fit, its margin over the
   runner-up and the HMM posterior.
5. Boundaries snap to a nearby onset (§4.4) so regions start on the attack."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import librosa
import numpy as np

from ..theory import ANALYSIS_QUALITIES, QUALITY_INTERVALS
from .chroma import CHROMA_COMPRESSION, ChromaFeatures, cosine_similarity
from .onsets import snap_to_onset

log = logging.getLogger(__name__)

N_ROOTS = 12
N_CHORD_STATES = 24          # 0..11 major, 12..23 minor
NO_CHORD = 24
N_STATES = 25

_EPS = 1e-9


# --- templates ------------------------------------------------------------------

ROOT_WEIGHT = 1.0
TONE_WEIGHT = 0.85   # the root counts a little more, so with bass-weighted chroma
                     # the note in the bass breaks ties such as Gsus4 vs Csus2


def chord_template(root: int, quality: str) -> np.ndarray:
    """Unit-norm pitch-class template with a mild root emphasis."""
    t = np.zeros(12, dtype=np.float32)
    for iv in QUALITY_INTERVALS[quality]:
        t[(root + iv) % 12] = ROOT_WEIGHT if iv == 0 else TONE_WEIGHT
    return t / np.linalg.norm(t)


def _majmin_templates() -> np.ndarray:
    rows = [chord_template(r, "maj") for r in range(12)] + [chord_template(r, "min") for r in range(12)]
    return np.stack(rows).astype(np.float32)


MAJMIN_TEMPLATES = _majmin_templates()  # (24, 12)


def state_label(state: int) -> tuple[int | None, str | None]:
    """(root pitch class, 'maj'|'min') for a state index; (None, None) for no-chord."""
    if state >= N_CHORD_STATES or state < 0:
        return None, None
    return int(state % 12), "maj" if state < 12 else "min"


def state_index(root: int, quality: str) -> int:
    return int(root % 12) + (12 if quality in ("min", "m7", "dim") else 0)


# --- backends ---------------------------------------------------------------------

class ChordBackend(Protocol):
    """Anything that scores chroma frames against the 25 chord states."""

    name: str

    def observation_probabilities(self, feats: ChromaFeatures) -> np.ndarray:
        """(25, T) probabilities in [0, 1]; columns sum to 1."""
        ...


class TemplateBackend:
    """Cosine similarity between bass-weighted chroma and maj/min triad
    templates, turned into probabilities with a softmax. The no-chord state
    scores high on silent frames (level far below the file's loud frames) and
    on flat, noise-like chroma (drums, room tone)."""

    name = "template"

    def __init__(
        self,
        *,
        beta: float = 12.0,
        silence_db: float = -40.0,
        flatness_threshold: float = 0.65,
    ) -> None:
        self.beta = float(beta)
        self.silence_db = float(silence_db)
        self.flatness_threshold = float(flatness_threshold)

    def similarities(self, feats: ChromaFeatures) -> np.ndarray:
        """(25, T) similarity-like scores in [0, 1] before the softmax."""
        x = feats.weighted.astype(np.float32)
        t = x.shape[1]
        if t == 0:
            return np.zeros((N_STATES, 0), dtype=np.float32)
        norms = np.linalg.norm(x, axis=0) + _EPS
        xn = x / norms
        sims = MAJMIN_TEMPLATES @ xn                                   # (24, T)
        energy = feats.energy.astype(np.float64)
        ref = float(np.percentile(energy, 95)) if t > 1 else float(energy.max())
        if ref <= 0.0:
            level_db = np.full(t, -120.0)
        else:
            level_db = 20.0 * np.log10(np.maximum(energy, _EPS) / ref)
        silence = 1.0 / (1.0 + np.exp(-(self.silence_db - level_db) / 3.0))
        lin = np.power(x, 1.0 / CHROMA_COMPRESSION)        # flatness on the linear chroma
        l1 = lin / (lin.sum(axis=0, keepdims=True) + _EPS)
        geo = np.exp(np.mean(np.log(l1 + 1e-6), axis=0))
        flatness = geo / (np.mean(l1, axis=0) + _EPS)
        flat_score = np.clip((flatness - self.flatness_threshold) / 0.25, 0.0, 1.0)
        n_score = np.maximum(silence, flat_score)
        sim_n = 0.55 + 0.45 * n_score
        return np.vstack([sims, sim_n[None, :]]).astype(np.float32)

    def observation_probabilities(self, feats: ChromaFeatures) -> np.ndarray:
        s = self.similarities(feats).astype(np.float64)
        if s.shape[1] == 0:
            return s
        logits = self.beta * s
        logits -= logits.max(axis=0, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=0, keepdims=True)
        return p


def get_backend(chord_model: str = "auto", models_dir: str | None = None) -> ChordBackend:
    """Resolve `AnalysisOptions.chord_model`. 'auto' and any learned name fall
    back to the template backend when the learned model cannot be loaded."""
    if chord_model == "template":
        return TemplateBackend()
    try:
        from .learned import LearnedChordBackend

        backend = LearnedChordBackend(models_dir)
    except (ImportError, FileNotFoundError) as e:
        level = logging.DEBUG if chord_model == "auto" else logging.WARNING
        log.log(level, "learned chord model unavailable (%s); using template backend", e)
        return TemplateBackend()
    except Exception as e:  # noqa: BLE001 - a broken optional model must not sink the analysis
        log.warning("learned chord model failed to load (%s); using template backend", e)
        return TemplateBackend()
    return backend


# --- HMM smoothing -----------------------------------------------------------------

def transition_matrix(hop_s: float, expected_chord_s: float = 1.5, n: int = N_STATES) -> np.ndarray:
    """Self-transition probability tuned so a chord is expected to last about
    `expected_chord_s`; every other transition is equally likely."""
    p_self = float(np.exp(-hop_s / max(expected_chord_s, 1e-3)))
    p_self = float(np.clip(p_self, 0.5, 0.999))
    a = np.full((n, n), (1.0 - p_self) / (n - 1), dtype=np.float64)
    np.fill_diagonal(a, p_self)
    return a


def viterbi_states(probs: np.ndarray, transition: np.ndarray) -> np.ndarray:
    if probs.shape[1] == 0:
        return np.zeros(0, dtype=int)
    p = np.clip(probs, 1e-12, 1.0)
    return np.asarray(librosa.sequence.viterbi(p, transition), dtype=int)


def posteriors(probs: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Forward–backward state posteriors (25, T), scaled to avoid underflow."""
    s, t = probs.shape
    if t == 0:
        return probs.copy()
    obs = np.clip(probs, 1e-12, 1.0)
    alpha = np.zeros((s, t))
    scale = np.zeros(t)
    alpha[:, 0] = obs[:, 0] / s
    scale[0] = alpha[:, 0].sum()
    alpha[:, 0] /= scale[0]
    at = transition.T
    for i in range(1, t):
        a = (at @ alpha[:, i - 1]) * obs[:, i]
        scale[i] = a.sum() + 1e-300
        alpha[:, i] = a / scale[i]
    beta = np.ones((s, t))
    for i in range(t - 2, -1, -1):
        beta[:, i] = transition @ (obs[:, i + 1] * beta[:, i + 1]) / scale[i + 1]
    post = alpha * beta
    post /= post.sum(axis=0, keepdims=True) + 1e-300
    return post


@dataclass
class FrameChords:
    states: np.ndarray     # (T,) 0..24
    probs: np.ndarray      # (25, T) observation probabilities
    posterior: np.ndarray  # (25, T) smoothed posteriors

    def frame_confidence(self) -> np.ndarray:
        if self.states.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.posterior[self.states, np.arange(self.states.size)].astype(np.float32)


def estimate_frame_chords(feats: ChromaFeatures, backend: ChordBackend, *, expected_chord_s: float = 1.5) -> FrameChords:
    probs = backend.observation_probabilities(feats)
    a = transition_matrix(feats.hop_s, expected_chord_s)
    states = viterbi_states(probs, a)
    post = posteriors(probs, a)
    return FrameChords(states=states, probs=probs, posterior=post)


# --- segmentation -----------------------------------------------------------------

@dataclass
class _Run:
    state: int
    f0: int
    f1: int            # exclusive
    pinned: bool = False

    @property
    def length(self) -> int:
        return self.f1 - self.f0


def _runs(states: np.ndarray) -> list[_Run]:
    runs: list[_Run] = []
    for i, s in enumerate(states.tolist()):
        if runs and runs[-1].state == s:
            runs[-1].f1 = i + 1
        else:
            runs.append(_Run(int(s), i, i + 1))
    return runs


def _coalesce(runs: list[_Run]) -> list[_Run]:
    out: list[_Run] = []
    for r in runs:
        if out and out[-1].state == r.state and out[-1].f1 == r.f0:
            out[-1].f1 = r.f1
            out[-1].pinned = out[-1].pinned or r.pinned
        else:
            out.append(r)
    return out


def _best_neighbour(run: _Run, feats: ChromaFeatures, candidates: list[_Run]) -> _Run:
    mean = feats.slice_frames(run.f0, run.f1).mean_weighted()
    best, best_sim = candidates[0], -2.0
    for nb in candidates:
        root, quality = state_label(nb.state)
        if root is None or quality is None:
            continue
        sim = cosine_similarity(mean, chord_template(root, quality))
        if sim > best_sim:
            best, best_sim = nb, sim
    return best


def merge_short_runs(
    runs: list[_Run],
    feats: ChromaFeatures,
    *,
    min_frames: int,
    gap_frames: int,
    keep_isolated_frames: int,
    drop_frames: int,
) -> list[_Run]:
    """§4.5: segments shorter than one beat (`min_frames`) merge into the
    neighbour whose chord fits them best. No-chord dips shorter than
    `gap_frames` inside a chord are absorbed; longer ones separate regions.
    An isolated short chord (silence on both sides) is kept when it is at
    least `keep_isolated_frames` long, otherwise it becomes no-chord."""
    runs = [_Run(r.state, r.f0, r.f1, r.pinned) for r in runs]
    for _ in range(10 * len(runs) + 10):
        runs = _coalesce(runs)
        changed = False
        # short no-chord dips between chords
        for i, r in enumerate(runs):
            if r.state != NO_CHORD or r.length >= gap_frames or len(runs) < 2:
                continue
            left = runs[i - 1] if i > 0 else None
            right = runs[i + 1] if i + 1 < len(runs) else None
            if left is not None and left.state != NO_CHORD:
                left.f1 = r.f1
                del runs[i]
                changed = True
                break
            if right is not None and right.state != NO_CHORD:
                right.f0 = r.f0
                del runs[i]
                changed = True
                break
        if changed:
            continue
        # shortest short chord run
        cand = [i for i, r in enumerate(runs) if r.state != NO_CHORD and not r.pinned and r.length < min_frames]
        if not cand:
            break
        i = min(cand, key=lambda k: runs[k].length)
        r = runs[i]
        neighbours = [nb for nb in (runs[i - 1] if i > 0 else None, runs[i + 1] if i + 1 < len(runs) else None)
                      if nb is not None and nb.state != NO_CHORD]
        if neighbours:
            r.state = _best_neighbour(r, feats, neighbours).state
        elif r.length >= max(keep_isolated_frames, drop_frames):
            r.pinned = True
        else:
            r.state = NO_CHORD
    return _coalesce(runs)


@dataclass
class ChordSegment:
    start_s: float
    end_s: float
    frame_start: int
    frame_end: int              # exclusive
    root: int
    quality: str
    bass: int
    confidence: float
    fit: float                  # cosine fit of the final template, 0..1

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def refine_quality(
    mean_chroma: np.ndarray,
    root: int,
    base_quality: str,
    *,
    margin: float = 0.03,
    qualities: tuple[str, ...] = ANALYSIS_QUALITIES,
) -> tuple[str, float]:
    """Pick the quality whose template fits the segment's bass-weighted chroma
    best; stay with the HMM's maj/min unless another quality wins by `margin`.
    Returns (quality, fit)."""
    sims = {q: cosine_similarity(mean_chroma, chord_template(root, q)) for q in qualities}
    base = base_quality if base_quality in sims else ("min" if base_quality in ("min", "m7", "dim") else "maj")
    best = max(sims, key=sims.get)
    if best != base and sims[best] > sims[base] + margin:
        return best, float(sims[best])
    return base, float(sims[base])


def estimate_bass(mean_bass_chroma: np.ndarray, root: int, quality: str) -> int:
    """Bass pitch class from the low-register chroma. The root wins ties; a
    chord tone needs to clearly beat the root, a foreign bass (slash chord)
    needs to dominate outright."""
    b = np.asarray(mean_bass_chroma, dtype=np.float64)
    total = float(b.sum())
    if total <= _EPS:
        return int(root)
    b = b / total
    best = int(np.argmax(b))
    if best == root:
        return int(root)
    tones = {(root + iv) % 12 for iv in QUALITY_INTERVALS[quality]}
    if best in tones and b[best] > 1.2 * b[root]:
        return best
    if best not in tones and b[best] > 2.0 * b[root] and b[best] > 0.35:
        return best
    return int(root)


def chord_fit_scores(mean_chroma: np.ndarray, root: int, quality: str) -> tuple[float, float]:
    """(fit of the chosen chord, margin over the best other maj/min root)."""
    fit = cosine_similarity(mean_chroma, chord_template(root, quality))
    xn = mean_chroma / (np.linalg.norm(mean_chroma) + _EPS)
    sims = MAJMIN_TEMPLATES @ xn
    own = state_index(root, quality)
    others = np.delete(sims, own)
    margin = float(fit - others.max()) if others.size else 0.0
    return float(fit), margin


def chord_confidence(fit: float, margin: float, posterior: float) -> float:
    """0..1 blend of template fit, margin over the runner-up and HMM posterior."""
    fit_score = float(np.clip((fit - 0.55) / 0.40, 0.0, 1.0))
    margin_score = float(np.clip(margin / 0.15, 0.0, 1.0))
    return float(np.clip(0.45 * fit_score + 0.25 * margin_score + 0.30 * posterior, 0.0, 1.0))


def label_segment(
    feats: ChromaFeatures,
    f0: int,
    f1: int,
    root: int,
    base_quality: str,
    *,
    posterior: float = 1.0,
    quality_override: str | None = None,
) -> tuple[int, str, int, float, float]:
    """(root, quality, bass, confidence, fit) for frames [f0, f1) given a root
    hypothesis; the root only changes for the sus2/sus4 enharmonic swap."""
    seg = feats.slice_frames(f0, f1)
    mean = seg.mean_weighted()
    if quality_override is not None:
        quality = quality_override
        fit, margin = chord_fit_scores(mean, root, quality)
    else:
        quality, _ = refine_quality(mean, root, base_quality)
        fit, margin = chord_fit_scores(mean, root, quality)
    bass = estimate_bass(seg.mean_bass(), root, quality)
    if quality_override is None:
        root, quality = normalise_sus(root, quality, bass)
    return root, quality, bass, chord_confidence(fit, margin, posterior), fit


def normalise_sus(root: int, quality: str, bass: int) -> tuple[int, str]:
    """Csus2/G and Gsus4 are the same pitch classes; name a sus chord from
    its bass note when the bass is the fifth (sus2) or the fourth (sus4)."""
    if quality == "sus2" and bass == (root + 7) % 12:
        return bass, "sus4"
    if quality == "sus4" and bass == (root + 5) % 12:
        return bass, "sus2"
    return root, quality


def quality_similarities(feats: ChromaFeatures, root: int, qualities: tuple[str, ...] = ANALYSIS_QUALITIES) -> np.ndarray:
    """(len(qualities), T) cosine similarity of each frame with each quality of `root`."""
    x = feats.weighted.astype(np.float32)
    xn = x / (np.linalg.norm(x, axis=0) + _EPS)
    templates = np.stack([chord_template(root, q) for q in qualities])
    return templates @ xn


def split_by_quality(
    feats: ChromaFeatures,
    f0: int,
    f1: int,
    root: int,
    base_quality: str,
    *,
    min_frames: int,
    beta: float = 12.0,
    triad_bonus: float = 0.03,
    expected_chord_s: float = 1.5,
    qualities: tuple[str, ...] = ANALYSIS_QUALITIES,
) -> list[tuple[str, int, int]]:
    """Second-pass HMM over the qualities of one root for frames [f0, f1).
    Returns (quality, start, end) runs no shorter than `min_frames` (except
    when the whole segment is shorter). The pass-one triad gets a small bonus
    so marginal evidence does not turn every chord into a 7th."""
    seg = feats.slice_frames(f0, f1)
    if seg.n_frames == 0:
        return []
    sims = quality_similarities(seg, root, qualities).astype(np.float64)
    if base_quality in qualities:
        sims[qualities.index(base_quality)] += triad_bonus
    logits = beta * sims
    logits -= logits.max(axis=0, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=0, keepdims=True)
    states = viterbi_states(probs, transition_matrix(feats.hop_s, expected_chord_s, n=len(qualities)))
    runs = _runs(states)
    for _ in range(len(runs) + 1):
        runs = _coalesce(runs)
        short = [i for i, r in enumerate(runs) if r.length < min_frames]
        if not short or len(runs) == 1:
            break
        i = min(short, key=lambda k: runs[k].length)
        r = runs[i]
        mean = seg.slice_frames(r.f0, r.f1).mean_weighted()
        best_j, best_sim = -1, -2.0
        for j in (i - 1, i + 1):
            if 0 <= j < len(runs):
                sim = cosine_similarity(mean, chord_template(root, qualities[runs[j].state]))
                if sim > best_sim:
                    best_j, best_sim = j, sim
        r.state = runs[best_j].state
    return [(qualities[r.state], f0 + r.f0, f0 + r.f1) for r in _coalesce(runs)]


def _snap_boundaries(
    starts: list[float],
    ends: list[float],
    onsets: list[float],
    tol: float,
    min_len: float,
) -> tuple[list[float], list[float]]:
    """Move region boundaries onto a nearby onset. Shared boundaries move
    together so tiling survives; a move that would empty a region is skipped."""
    n = len(starts)
    starts, ends = list(starts), list(ends)
    for i in range(n):
        s_new = snap_to_onset(starts[i], onsets, tol)
        prev_shared = i > 0 and abs(ends[i - 1] - starts[i]) < 1e-6
        if s_new >= ends[i] - min_len or (prev_shared and s_new <= starts[i - 1] + min_len):
            continue
        starts[i] = s_new
        if prev_shared:
            ends[i - 1] = s_new
    for i in range(n):
        next_shared = i + 1 < n and abs(ends[i] - starts[i + 1]) < 1e-6
        if next_shared:
            continue  # already handled through the next region's start
        e_new = snap_to_onset(ends[i], onsets, tol)
        if e_new > starts[i] + min_len:
            ends[i] = e_new
    return starts, ends


def segment_chords(
    frames: FrameChords,
    feats: ChromaFeatures,
    *,
    min_duration_s: float,
    onsets: list[float] | None = None,
    snap_tolerance_s: float | None = None,
    min_region_s: float = 0.15,
    keep_isolated_s: float = 0.2,
    gap_s: float = 0.25,
) -> list[ChordSegment]:
    """Turn frame states into tiled, harmonically stable segments (§4.5)."""
    hop_s = feats.hop_s
    if frames.states.size == 0:
        return []
    min_frames = max(1, int(round(min_duration_s / hop_s)))
    runs = merge_short_runs(
        _runs(frames.states),
        feats,
        min_frames=min_frames,
        gap_frames=max(1, int(round(gap_s / hop_s))),
        keep_isolated_frames=max(1, int(round(keep_isolated_s / hop_s))),
        drop_frames=max(1, int(round(min_region_s / hop_s))),
    )
    conf_frames = frames.frame_confidence()
    items: list[tuple[int, str, int, int]] = []        # root, quality, f0, f1
    for r in runs:
        root, base_quality = state_label(r.state)
        if root is None or base_quality is None:
            continue
        for quality, a, b in split_by_quality(feats, r.f0, r.f1, root, base_quality, min_frames=min_frames):
            items.append((root, quality, a, b))
    if not items:
        return []
    starts = [feats.frame_start_s(a) for _, _, a, _ in items]
    ends = [feats.frame_end_s(b - 1) for _, _, _, b in items]
    if onsets:
        # the CQT frame grid plus its low-octave smearing puts a boundary up to
        # about a frame and a half from the attack that caused it
        tol = 1.5 * hop_s if snap_tolerance_s is None else snap_tolerance_s
        starts, ends = _snap_boundaries(starts, ends, onsets, tol, min_region_s * 0.5)
    segments: list[ChordSegment] = []
    for (root, quality, a, b), s, e in zip(items, starts, ends, strict=True):
        if e <= s:
            continue
        post = float(conf_frames[a:b].mean()) if b > a else 0.0
        final_root, final_quality, bass, conf, fit = label_segment(feats, a, b, root, quality, posterior=post)
        segments.append(ChordSegment(
            start_s=float(s), end_s=float(e), frame_start=a, frame_end=b,
            root=final_root, quality=final_quality, bass=bass, confidence=conf, fit=fit,
        ))
    return segments


def dominant_chord(frames: FrameChords) -> tuple[int, str, float] | None:
    """(root, maj|min, mean posterior) of the chord state covering the most
    frames, or None when every frame is no-chord."""
    if frames.states.size == 0:
        return None
    counts = np.bincount(frames.states, minlength=N_STATES)
    counts[NO_CHORD] = 0
    if counts.max() == 0:
        return None
    state = int(np.argmax(counts))
    root, quality = state_label(state)
    assert root is not None and quality is not None
    sel = frames.states == state
    post = float(frames.posterior[state, sel].mean()) if np.any(sel) else 0.0
    return root, quality, post
