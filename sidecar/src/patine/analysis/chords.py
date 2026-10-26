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
5. Boundaries snap to a nearby onset (§4.4) so regions start on the attack.

Recall. Three things keep tonal samples from coming back with no regions:
the observations the HMM sees are averaged over about one beat
(`smooth_observations`), so an arpeggio or a melody outlining a chord reads
as that chord instead of one chord per note; the no-chord state only wins on
frames that are both quiet-or-flat *and* not concentrated on a few pitch
classes (`TemplateBackend.gates`); and when the whole file still decodes as
no-chord, `fallback_segment` labels its longest tonal span from the mean
chroma with a capped confidence rather than dropping the file. A bare root
or root+fifth (a plucked note, a bass line, a power chord) is reported as
quality "5" (`bare_root_quality`) instead of an invented third."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Protocol

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from ..theory import ANALYSIS_QUALITIES, QUALITY_INTERVALS
from .chroma import ChromaFeatures, cosine_similarity
from .onsets import snap_to_onset

log = logging.getLogger(__name__)

N_ROOTS = 12
N_CHORD_STATES = 24          # 0..11 major, 12..23 minor
NO_CHORD = 24
N_STATES = 25

_EPS = 1e-9

# No-chord score = floor + (1 - floor) * gate. Every ungated frame therefore
# favours a chord by (best fit - floor) * beta nats; 0.45 lets a dyad or a
# triad over a heavy floor win while noise still saturates the gate.
NO_CHORD_FLOOR = 0.45
# "No structure" gate: the best maj/min template fit minus the median fit over
# all 24. Noise, drum loops and sub rumble stay below 0.14 (90th percentile
# per frame), a pad under drums 6 dB louder stays above 0.15 (10th
# percentile), a single plucked note or a clean chord above 0.4. Spectral
# flatness cannot tell these apart: it measures the floor, not the tonalness.
CONTRAST_THRESHOLD = 0.16
CONTRAST_RANGE = 0.06

# Whole-span fallback (see module docstring).
FALLBACK_MIN_FIT = 0.60
FALLBACK_MIN_MARGIN = 0.06
FALLBACK_MIN_SPAN_S = 0.2
FALLBACK_GAP_S = 0.5
FALLBACK_MAX_CONFIDENCE = 0.5

# Bare root rule: no third stronger than this share of the root (on the
# compressed chroma; a note's 5th harmonic puts 0.27-0.36 there), no sus
# tone either, and the root among the leading pitch classes.
BARE_THIRD_RATIO = 0.40
BARE_ROOT_SHARE = 0.6

# An extended quality (7, m7, maj7, sus2, sus4, dim) is only allowed when each
# tone it adds to the maj/min triad stands this far above the floor of the
# pitch classes neither chord uses: under drums or noise a four-note template
# otherwise beats the triad simply by covering more floor.
EXTENSION_FLOOR_RATIO = 1.4


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
    on frames with no harmonic structure (drums, noise, room tone: no template
    fits better than the rest)."""

    name = "template"

    def __init__(
        self,
        *,
        beta: float = 12.0,
        silence_db: float = -40.0,
        no_chord_floor: float = NO_CHORD_FLOOR,
        contrast_threshold: float = CONTRAST_THRESHOLD,
        contrast_range: float = CONTRAST_RANGE,
    ) -> None:
        self.beta = float(beta)
        self.silence_db = float(silence_db)
        self.no_chord_floor = float(no_chord_floor)
        self.contrast_threshold = float(contrast_threshold)
        self.contrast_range = float(contrast_range)

    def gates(self, feats: ChromaFeatures) -> tuple[np.ndarray, np.ndarray]:
        """(silence, no_structure) scores in [0, 1] per frame: quiet frames,
        and frames where no maj/min template stands out from the rest."""
        x = feats.weighted.astype(np.float32)
        t = x.shape[1]
        if t == 0:
            return np.zeros(0), np.zeros(0)
        silence = silence_scores(feats, self.silence_db)
        xn = x / (np.linalg.norm(x, axis=0) + _EPS)
        sims = MAJMIN_TEMPLATES @ xn
        contrast = sims.max(axis=0) - np.median(sims, axis=0)
        structure = np.clip((self.contrast_threshold - contrast) / self.contrast_range, 0.0, 1.0)
        return silence, structure

    def similarities(self, feats: ChromaFeatures) -> np.ndarray:
        """(25, T) similarity-like scores in [0, 1] before the softmax."""
        x = feats.weighted.astype(np.float32)
        t = x.shape[1]
        if t == 0:
            return np.zeros((N_STATES, 0), dtype=np.float32)
        norms = np.linalg.norm(x, axis=0) + _EPS
        xn = x / norms
        sims = MAJMIN_TEMPLATES @ xn                                   # (24, T)
        silence, no_structure = self.gates(feats)
        n_score = np.maximum(silence, no_structure)
        sim_n = self.no_chord_floor + (1.0 - self.no_chord_floor) * n_score
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


def silence_scores(feats: ChromaFeatures, silence_db: float = -40.0) -> np.ndarray:
    """Per-frame 0..1 "this frame is silent": level against the file's loud
    frames (95th percentile) through a sigmoid 3 dB wide around `silence_db`."""
    energy = feats.energy.astype(np.float64)
    t = energy.size
    if t == 0:
        return np.zeros(0)
    ref = float(np.percentile(energy, 95)) if t > 1 else float(energy.max())
    if ref <= 0.0:
        level_db = np.full(t, -120.0)
    else:
        level_db = 20.0 * np.log10(np.maximum(energy, _EPS) / ref)
    return 1.0 / (1.0 + np.exp(-(silence_db - level_db) / 3.0))


def get_backend(
    chord_model: str = "auto", models_dir: str | None = None, *, no_chord_floor: float = NO_CHORD_FLOOR
) -> ChordBackend:
    """Resolve `AnalysisOptions.chord_model`. 'auto' and any learned name fall
    back to the template backend when the learned model cannot be loaded."""
    if chord_model == "template":
        return TemplateBackend(no_chord_floor=no_chord_floor)
    try:
        from .learned import LearnedChordBackend

        backend = LearnedChordBackend(models_dir)
    except (ImportError, FileNotFoundError) as e:
        level = logging.DEBUG if chord_model == "auto" else logging.WARNING
        log.log(level, "learned chord model unavailable (%s); using template backend", e)
        return TemplateBackend(no_chord_floor=no_chord_floor)
    except Exception as e:  # noqa: BLE001 - a broken optional model must not sink the analysis
        log.warning("learned chord model failed to load (%s); using template backend", e)
        return TemplateBackend(no_chord_floor=no_chord_floor)
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


def observation_window_frames(
    hop_s: float,
    beat_period_s: float | None,
    bpm_confidence: float,
    *,
    window_beats: float = 1.0,
    min_frames: int = 3,
    max_frames: int = 9,
    fallback_s: float = 0.5,
    min_confidence: float = 0.3,
) -> int:
    """Odd number of frames the observations are averaged over: about one
    beat when the tracker is confident, `fallback_s` otherwise. A symmetric
    window is 50/50 exactly at a true chord change, so decoded boundaries
    stay unbiased and the beat snap absorbs the blur. 1 disables smoothing."""
    if window_beats <= 0:
        return 1
    if beat_period_s and beat_period_s > 0 and bpm_confidence >= min_confidence:
        seconds = window_beats * beat_period_s
    else:
        seconds = fallback_s
    n = int(round(seconds / max(hop_s, 1e-6)))
    n = max(min_frames, min(max_frames, n))
    return n if n % 2 == 1 else n + 1


def expected_chord_seconds(
    beat_period_s: float | None,
    bpm_confidence: float,
    *,
    beats: float = 2.0,
    lo: float = 0.75,
    hi: float = 3.0,
    default: float = 1.5,
    min_confidence: float = 0.3,
) -> float:
    """HMM stickiness tied to tempo: a chord is expected to last `beats`."""
    if beat_period_s and beat_period_s > 0 and bpm_confidence >= min_confidence:
        return float(np.clip(beats * beat_period_s, lo, hi))
    return default


def smooth_observations(feats: ChromaFeatures, window_frames: int) -> ChromaFeatures:
    """The chord chroma averaged over `window_frames` (centred boxcar); the
    other views are untouched, so segmentation and labelling keep the raw
    frames."""
    if window_frames <= 1 or feats.n_frames < 2:
        return feats
    w = min(int(window_frames), feats.n_frames)
    smoothed = uniform_filter1d(feats.weighted.astype(np.float32), size=w, axis=1, mode="nearest")
    return replace(feats, weighted=smoothed.astype(np.float32))


def estimate_frame_chords(
    feats: ChromaFeatures,
    backend: ChordBackend,
    *,
    expected_chord_s: float = 1.5,
    window_frames: int = 1,
) -> FrameChords:
    probs = backend.observation_probabilities(smooth_observations(feats, window_frames))
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


def triad_of(quality: str) -> str:
    """The maj/min triad an analysis quality is built on."""
    return "min" if quality in ("min", "m7", "dim", "dim7", "m7b5", "m6") else "maj"


def extension_tones_present(
    mean_chroma: np.ndarray, root: int, quality: str, *, ratio: float = EXTENSION_FLOOR_RATIO
) -> bool:
    """Do the tones `quality` adds to its triad stand out from the floor? The
    floor is the median of the pitch classes neither the triad nor the quality
    uses; with no floor (clean material) every quality is allowed and the
    template fit decides as before."""
    base = triad_of(quality)
    added = set(QUALITY_INTERVALS[quality]) - set(QUALITY_INTERVALS[base])
    if not added:
        return True
    c = np.asarray(mean_chroma, dtype=np.float64)
    used = {(root + iv) % 12 for iv in QUALITY_INTERVALS[quality]} | {(root + iv) % 12 for iv in QUALITY_INTERVALS[base]}
    rest = [c[i] for i in range(12) if i not in used]
    floor = float(np.median(rest)) if rest else 0.0
    return all(c[(root + iv) % 12] >= ratio * floor for iv in added)


def refine_quality(
    mean_chroma: np.ndarray,
    root: int,
    base_quality: str,
    *,
    margin: float = 0.03,
    qualities: tuple[str, ...] = ANALYSIS_QUALITIES,
) -> tuple[str, float]:
    """Pick the quality whose template fits the segment's bass-weighted chroma
    best among those whose added tones are really there (see
    `extension_tones_present`); stay with the HMM's maj/min unless another
    quality wins by `margin`. Returns (quality, fit)."""
    allowed = [q for q in qualities if extension_tones_present(mean_chroma, root, q)]
    sims = {q: cosine_similarity(mean_chroma, chord_template(root, q)) for q in allowed}
    base = base_quality if base_quality in sims else triad_of(base_quality)
    if base not in sims:
        sims[base] = cosine_similarity(mean_chroma, chord_template(root, base))
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
    """(fit of the chosen chord, margin over the best maj/min chord on another
    root; for a bare root both maj and min of its own root are "own")."""
    fit = cosine_similarity(mean_chroma, chord_template(root, quality))
    xn = mean_chroma / (np.linalg.norm(mean_chroma) + _EPS)
    sims = MAJMIN_TEMPLATES @ xn
    own = [state_index(root, quality)] if quality != "5" else [state_index(root, "maj"), state_index(root, "min")]
    others = np.delete(sims, own)
    margin = float(fit - others.max()) if others.size else 0.0
    return float(fit), margin


def chord_confidence(fit: float, margin: float, posterior: float) -> float:
    """0..1 blend of template fit, margin over the runner-up and HMM posterior."""
    fit_score = float(np.clip((fit - 0.55) / 0.40, 0.0, 1.0))
    margin_score = float(np.clip(margin / 0.15, 0.0, 1.0))
    return float(np.clip(0.45 * fit_score + 0.25 * margin_score + 0.30 * posterior, 0.0, 1.0))


def bare_root_quality(
    mean_weighted: np.ndarray,
    mean_bass: np.ndarray,
    root: int,
    *,
    allow_reroot: bool = False,
) -> tuple[int, str] | None:
    """(root, "5") when the segment is a bare root or root+fifth: no third,
    no sus tone, and the root among the strongest pitch classes. With
    `allow_reroot` the root may move to the pitch class the "5" template
    fits best (a C+G dyad scores 1.0 from C and 0.76 from G; the bass chroma
    breaks the remaining ties)."""
    c = np.asarray(mean_weighted, dtype=np.float64)
    if c.size != 12 or float(c.max()) <= _EPS:
        return None
    candidates = {int(root)}
    if allow_reroot:
        candidates |= {int(i) for i in np.argsort(c)[-3:]}
    b = np.asarray(mean_bass, dtype=np.float64)

    def score(r: int) -> float:
        base = cosine_similarity(c, chord_template(r, "5"))
        return base + (0.05 * b[r] / (b.max() + _EPS) if b.size == 12 and b.max() > _EPS else 0.0)

    best = max(candidates, key=score)
    third = max(c[(best + 3) % 12], c[(best + 4) % 12])
    sus = max(c[(best + 2) % 12], c[(best + 5) % 12])
    if third >= BARE_THIRD_RATIO * c[best] or sus >= BARE_THIRD_RATIO * c[best]:
        return None
    if c[best] < BARE_ROOT_SHARE * float(c.max()):
        return None
    return best, "5"


def label_segment(
    feats: ChromaFeatures,
    f0: int,
    f1: int,
    root: int,
    base_quality: str,
    *,
    posterior: float = 1.0,
    quality_override: str | None = None,
    allow_reroot: bool = False,
) -> tuple[int, str, int, float, float]:
    """(root, quality, bass, confidence, fit) for frames [f0, f1) given a root
    hypothesis; the root only changes for the sus2/sus4 enharmonic swap and,
    with `allow_reroot`, for a bare root whose fifth was taken for the root."""
    seg = feats.slice_frames(f0, f1)
    mean = seg.mean_weighted()
    if quality_override is not None:
        quality = quality_override
        fit, margin = chord_fit_scores(mean, root, quality)
    else:
        quality, _ = refine_quality(mean, root, base_quality)
        # A bare root looks like a sus chord from the wrong root (a G pluck is
        # "Dsus4" from D), so when the root may move the rule runs regardless
        # of the refined quality; a real sus or 7th chord has the tones that
        # make the rule decline.
        if quality in ("maj", "min") or allow_reroot:
            bare = bare_root_quality(mean, seg.mean_bass(), root, allow_reroot=allow_reroot)
            if bare is not None:
                root, quality = bare
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
    *,
    beats: list[float] | None = None,
    beat_tol: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Move region boundaries onto the beat grid when a tracked beat is within
    `beat_tol`, else onto a nearby onset within `tol`. Chord changes in real
    music land on beats, and a region whose start is a beat plays in phase
    with the progression's grid instead of a frame or two off it. Shared
    boundaries move together so tiling survives; a move that would empty a
    region is skipped."""

    attack_tol = min(0.15, max(0.06, beat_tol / 0.35 * 0.25)) if beat_tol > 0 else 0.0

    def snap(t: float) -> float:
        if beats and beat_tol > 0:
            b = snap_to_onset(t, beats, beat_tol)
            if b != t:
                # On the grid, at the attack: trackers place a beat some tens
                # of ms after the (backtracked) onset that caused it, so land
                # on that onset when one sits within a quarter beat of it.
                return snap_to_onset(b, onsets, attack_tol) if onsets else b
        return snap_to_onset(t, onsets, tol) if onsets else t

    n = len(starts)
    starts, ends = list(starts), list(ends)
    for i in range(n):
        s_new = snap(starts[i])
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
        e_new = snap(ends[i])
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
    beats: list[float] | None = None,
    beat_period_s: float | None = None,
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
    if onsets or beats:
        # the CQT frame grid plus its low-octave smearing puts a boundary up to
        # about a frame and a half from the attack that caused it; a tracked
        # beat within about a third of a beat period wins over the raw onset
        tol = 1.5 * hop_s if snap_tolerance_s is None else snap_tolerance_s
        beat_tol = min(0.3, max(0.08, 0.35 * beat_period_s)) if beat_period_s else 0.0
        starts, ends = _snap_boundaries(
            starts, ends, onsets or [], tol, min_region_s * 0.5, beats=beats, beat_tol=beat_tol
        )
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


# --- whole-span fallback --------------------------------------------------------------

def tonal_span(
    silence: np.ndarray,
    *,
    gap_frames: int,
    min_frames: int,
) -> tuple[int, int] | None:
    """Longest run of frames that are not silent, bridging silent gaps shorter
    than `gap_frames`; None when shorter than `min_frames`. For a one-shot this
    is attack → where the decay crosses the silence level; for a loop, the file."""
    active = np.asarray(silence) < 0.5
    if not np.any(active):
        return None
    runs: list[list[int]] = []
    for i, a in enumerate(active.tolist()):
        if a:
            if runs and i - runs[-1][1] <= gap_frames:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
    best = max(runs, key=lambda r: r[1] - r[0])
    if best[1] - best[0] < min_frames:
        return None
    return int(best[0]), int(best[1])


def mean_chord_hypothesis(
    feats: ChromaFeatures,
    f0: int,
    f1: int,
    *,
    min_fit: float = FALLBACK_MIN_FIT,
    min_margin: float = FALLBACK_MIN_MARGIN,
) -> tuple[int, str, float, float] | None:
    """(root, 'maj'|'min', fit, margin) of the maj/min template that fits the
    energy-weighted mean chroma of frames [f0, f1) best, or None when the fit
    is weak or no root clearly wins (noise, rumble, clusters)."""
    seg = feats.slice_frames(f0, f1)
    if seg.n_frames == 0:
        return None
    x = seg.weighted.astype(np.float64)
    w = x.sum(axis=0)
    if float(w.sum()) <= _EPS:
        return None
    mean = (x * w[None, :]).sum(axis=1) / float(w.sum())
    # Score the chroma above its floor: drums and noise lift every pitch class
    # and squeeze the margin between chords sharing two tones.
    mean = np.maximum(mean - float(np.percentile(mean, 25)), 0.0)
    if float(mean.max()) <= _EPS:
        return None
    xn = mean / (np.linalg.norm(mean) + _EPS)
    sims = MAJMIN_TEMPLATES @ xn
    best = int(np.argmax(sims))
    root, quality = state_label(best)
    assert root is not None and quality is not None
    other_roots = [i for i in range(N_CHORD_STATES) if i % 12 != root]
    margin = float(sims[best] - sims[other_roots].max())
    if float(sims[best]) < min_fit or margin < min_margin:
        return None
    return root, quality, float(sims[best]), margin


def fallback_segment(
    frames: FrameChords,
    feats: ChromaFeatures,
    *,
    silence_db: float = -40.0,
    onsets: list[float] | None = None,
    beats: list[float] | None = None,
    beat_period_s: float | None = None,
    min_region_s: float = 0.15,
    gap_s: float = FALLBACK_GAP_S,
    min_span_s: float = FALLBACK_MIN_SPAN_S,
) -> ChordSegment | None:
    """One region for a file that decoded as all no-chord but carries a tonal
    span (module docstring). Confidence is capped at `FALLBACK_MAX_CONFIDENCE`
    so a recovered region ranks behind any decoded one."""
    if feats.n_frames == 0:
        return None
    hop_s = feats.hop_s
    span = tonal_span(
        silence_scores(feats, silence_db),
        gap_frames=max(1, int(round(gap_s / hop_s))),
        min_frames=max(1, int(round(min_span_s / hop_s))),
    )
    if span is None:
        return None
    f0, f1 = span
    hyp = mean_chord_hypothesis(feats, f0, f1)
    if hyp is None:
        return None
    root, quality, _fit, _margin = hyp
    state = state_index(root, quality)
    post = float(frames.posterior[state, f0:f1].mean()) if frames.posterior.shape[1] >= f1 else 0.0
    root, quality, bass, conf, fit = label_segment(feats, f0, f1, root, quality, posterior=post, allow_reroot=True)
    start, end = feats.frame_start_s(f0), feats.frame_end_s(f1 - 1)
    if onsets or beats:
        beat_tol = min(0.3, max(0.08, 0.35 * beat_period_s)) if beat_period_s else 0.0
        starts, ends = _snap_boundaries([start], [end], onsets or [], 1.5 * hop_s, min_region_s * 0.5,
                                        beats=beats, beat_tol=beat_tol)
        start, end = starts[0], ends[0]
    if end <= start:
        return None
    return ChordSegment(
        start_s=float(start), end_s=float(end), frame_start=f0, frame_end=f1,
        root=root, quality=quality, bass=bass, confidence=min(conf, FALLBACK_MAX_CONFIDENCE), fit=fit,
    )
