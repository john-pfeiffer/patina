"""Analysis pipeline entry points (handoff §4).

Runs once per sample on ingest in a background worker. Every output is stored;
nothing here runs at playback time. Progress is reported through the callback
so the library pane can show it.

Contract used by the API layer and the tests:

    result = analyze_file(path, progress=cb)        # whole file + regions
    region = measure_region(y, sr, start_s, end_s, ctx)   # re-measure a user-edited region

`AnalysisContext` carries the whole-file results a region measurement needs
(beats, onsets, key) so editing one region never re-runs the file.

Stages (§4.1–4.8): decode → tuning → tempo/beats → onsets → chroma → chords
and segmentation → key → per-region busyness/register/metrics → whole-file
metrics. Submodules: `decode`, `tuning`, `tempo`, `onsets`, `chroma`,
`chords` (+ `learned`), `key`, `activity`, `metrics`, `separation`."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..audio import slice_seconds, to_mono
from ..models import TonalMetrics
from ..theory import pc_name
from . import activity, chords, metrics, tempo, tuning
from . import key as keymod
from . import onsets as onsetmod
from .chroma import ANALYSIS_SR, ChromaFeatures, compute_chroma
from .decode import DecodedAudio, decode, resample_mono

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class AnalysisOptions:
    use_source_separation: bool = False   # §4.5 optional flag, off by default
    chord_model: str = "auto"             # 'auto' | 'template' | a learned backend name
    beat_tracker: str = "auto"            # 'auto' | 'librosa' | 'beat_this'
    min_region_beats: float = 1.0         # §4.5 merge segments shorter than one beat
    models_dir: str | Path | None = None  # where learned weights live (ARCHITECTURE.md: <data dir>/models)


@dataclass
class RegionResult:
    start_s: float
    end_s: float
    chord_root: str | None            # 'C'..'B' sharps, None when no chord
    chord_quality: str | None         # one of theory.ANALYSIS_QUALITIES
    bass_note: str | None
    chord_confidence: float
    key_context: str | None           # 'F# minor'
    beats: list[float]                # absolute seconds inside the file
    first_onset_s: float | None
    busyness: float                   # 0..1
    register_centroid_hz: float
    metrics: TonalMetrics


@dataclass
class AnalysisContext:
    """Whole-file state needed to (re)measure any region."""

    sample_rate: int
    tuning_cents: float
    beats: list[float]
    downbeats: list[float]
    onsets: list[float]
    key_estimate: str | None
    frame_times: list[float] = field(default_factory=list)   # chroma frame centres
    chroma: np.ndarray | None = None                          # (12, frames) bass-weighted
    chord_frames: list[tuple[str | None, str | None, float]] = field(default_factory=list)
    key_scores: np.ndarray | None = None                      # (24,) whole-file key correlations
    features: ChromaFeatures | None = None                    # full features when still in memory


@dataclass
class AnalysisResult:
    duration: float
    sample_rate: int
    channels: int
    tuning_cents: float
    detected_bpm: float
    bpm_confidence: float
    beats: list[float]
    downbeats: list[float]
    key_estimate: str | None
    key_confidence: float
    metrics: TonalMetrics
    regions: list[RegionResult]
    context: AnalysisContext


# --- helpers --------------------------------------------------------------------------

def _report(progress: ProgressFn | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(np.clip(fraction, 0.0, 1.0)), message)


def _min_region_seconds(bpm: float, min_region_beats: float) -> float:
    beat_s = 60.0 / bpm if bpm and bpm > 0 else 0.5
    return max(0.15, float(min_region_beats) * beat_s)


def _beats_in(beats: list[float], start_s: float, end_s: float) -> list[float]:
    return [float(b) for b in beats if start_s <= b < end_s]


def _region_features(
    y: np.ndarray, sr: int, start_s: float, end_s: float, ctx: AnalysisContext
) -> ChromaFeatures:
    """Chroma features for one region: sliced from the whole-file features when
    they are still in memory, otherwise recomputed from the audio."""
    if ctx.features is not None and ctx.features.n_frames > 0:
        sliced = ctx.features.slice_seconds(start_s, end_s)
        if sliced is not None:
            return sliced
    mono = to_mono(slice_seconds(y, sr, start_s, end_s))
    mono_a = resample_mono(mono, sr, ANALYSIS_SR)
    return compute_chroma(mono_a, ANALYSIS_SR, tuning_cents=ctx.tuning_cents)


def _key_context_for(ctx: AnalysisContext, feats: ChromaFeatures, start_s: float, end_s: float) -> str | None:
    if ctx.features is not None and ctx.features.n_frames > 0:
        return keymod.key_context(ctx.features.weighted, ctx.features.times, start_s, end_s, global_scores=ctx.key_scores)
    if ctx.chroma is not None and ctx.frame_times and ctx.chroma.shape[1] == len(ctx.frame_times):
        return keymod.key_context(ctx.chroma, np.asarray(ctx.frame_times), start_s, end_s, global_scores=ctx.key_scores)
    local, _, _ = keymod.estimate_key(feats)
    if ctx.key_scores is not None and np.any(ctx.key_scores):
        return keymod.key_context(feats.weighted, feats.times, -1e9, 1e9, global_scores=ctx.key_scores)
    return local or ctx.key_estimate


def _resolve_chord(
    feats: ChromaFeatures,
    chord_root: str | None,
    chord_quality: str | None,
) -> tuple[str | None, str | None, str | None, float]:
    """(root, quality, bass, confidence) for a region's features, honouring overrides."""
    from ..theory import parse_note

    root_pc: int | None = None
    if chord_root:
        root_pc, _ = parse_note(chord_root)
    base_quality = "maj"
    posterior = 1.0
    if root_pc is None:
        frames = chords.estimate_frame_chords(feats, chords.TemplateBackend())
        dom = chords.dominant_chord(frames)
        if dom is None:
            if chord_quality is not None:
                return None, chord_quality, None, 0.0
            return None, None, None, 0.0
        root_pc, base_quality, posterior = dom
    quality, bass, conf, _fit = chords.label_segment(
        feats, 0, feats.n_frames, root_pc, base_quality,
        posterior=posterior, quality_override=chord_quality,
    )
    return pc_name(root_pc), quality, pc_name(bass), conf


def _measure(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    ctx: AnalysisContext,
    feats: ChromaFeatures,
    *,
    chord_root: str | None,
    chord_quality: str | None,
    bass_note: str | None,
    chord_confidence: float,
) -> RegionResult:
    seg = slice_seconds(y, sr, start_s, end_s)
    rel_onsets = [o - start_s for o in ctx.onsets if start_s <= o < end_s]
    region_metrics = metrics.measure_tonal_metrics(seg, sr, onsets_s=rel_onsets)
    return RegionResult(
        start_s=float(start_s),
        end_s=float(end_s),
        chord_root=chord_root,
        chord_quality=chord_quality,
        bass_note=bass_note,
        chord_confidence=float(chord_confidence),
        key_context=_key_context_for(ctx, feats, start_s, end_s),
        beats=_beats_in(ctx.beats, start_s, end_s),
        first_onset_s=onsetmod.first_onset_in(ctx.onsets, start_s, end_s),
        busyness=activity.busyness(feats),
        register_centroid_hz=activity.register_centroid_hz(feats),
        metrics=region_metrics,
    )


# --- entry points -----------------------------------------------------------------------

def analyze_file(
    path: str | Path,
    *,
    progress: ProgressFn | None = None,
    options: AnalysisOptions | None = None,
) -> AnalysisResult:
    """Whole-file analysis (§4). Calls `progress(fraction, message)` as stages finish."""
    opts = options or AnalysisOptions()
    t_start = time.perf_counter()
    name = Path(path).name

    _report(progress, 0.0, "Decoding")
    audio = decode(path)
    return analyze_decoded(audio, progress=progress, options=opts, name=name, t_start=t_start)


def analyze_decoded(
    audio: DecodedAudio,
    *,
    progress: ProgressFn | None = None,
    options: AnalysisOptions | None = None,
    name: str = "<audio>",
    t_start: float | None = None,
) -> AnalysisResult:
    """`analyze_file` on already-decoded audio (same stages, same result)."""
    opts = options or AnalysisOptions()
    t_start = time.perf_counter() if t_start is None else t_start
    sr_a = audio.sr_a
    mono_a = audio.mono_a
    duration = audio.duration

    # §4.2 tuning
    _report(progress, 0.08, "Estimating tuning")
    tuning_cents = tuning.estimate_tuning_cents(mono_a, sr_a) if mono_a.size else 0.0

    # §4.3 tempo & beats
    _report(progress, 0.18, "Tracking beats")
    env = tempo.onset_envelope(mono_a, sr_a)
    tempo_result = tempo.track_beats(mono_a, sr_a, backend=opts.beat_tracker, env=env)
    beats = [b for b in tempo_result.beats if 0.0 <= b < duration]

    # §4.4 onsets
    _report(progress, 0.32, "Detecting onsets")
    onsets = [o for o in onsetmod.detect_onsets(env, sr_a, tempo_result.hop_length) if o < duration]

    # §4.5 optional separation (analysis only)
    chord_source = mono_a
    if opts.use_source_separation:
        _report(progress, 0.36, "Separating sources")
        from .separation import accompaniment_stem

        stem = accompaniment_stem(audio.y, audio.sr)
        if stem is not None:
            chord_source = resample_mono(stem, audio.sr, sr_a)

    # chroma
    _report(progress, 0.42, "Computing chroma")
    feats = compute_chroma(chord_source, sr_a, tuning_cents=tuning_cents)
    feats_mix = feats if chord_source is mono_a else compute_chroma(mono_a, sr_a, tuning_cents=tuning_cents)

    # §4.5 chords
    _report(progress, 0.55, "Estimating chords")
    backend = chords.get_backend(opts.chord_model, None if opts.models_dir is None else str(opts.models_dir))
    frames = chords.estimate_frame_chords(feats, backend)
    min_region_s = _min_region_seconds(tempo_result.bpm, opts.min_region_beats)
    segments = chords.segment_chords(frames, feats, min_duration_s=min_region_s, onsets=onsets)

    downbeats = tempo_result.downbeats
    if not downbeats and tempo_result.backend == "librosa":
        downbeats = tempo.estimate_downbeats(
            beats, env, sr_a, tempo_result.hop_length, chroma=feats.weighted, frame_times=feats.times
        )

    # §4.6 key
    _report(progress, 0.68, "Estimating key")
    not_silent = 1.0 - frames.posterior[chords.NO_CHORD] if frames.posterior.shape[1] else None
    key_estimate, key_confidence, key_scores = keymod.estimate_key(feats, frame_weights=not_silent)

    frame_conf = frames.frame_confidence()
    chord_frames: list[tuple[str | None, str | None, float]] = []
    for s, c in zip(frames.states.tolist(), frame_conf.tolist(), strict=True):
        root, quality = chords.state_label(int(s))
        chord_frames.append((None if root is None else pc_name(root), quality, float(c)))

    ctx = AnalysisContext(
        sample_rate=audio.sr,
        tuning_cents=tuning_cents,
        beats=beats,
        downbeats=downbeats,
        onsets=onsets,
        key_estimate=key_estimate,
        frame_times=[float(t) for t in feats_mix.times],
        chroma=feats_mix.weighted,
        chord_frames=chord_frames,
        key_scores=key_scores,
        features=feats_mix,
    )

    # §4.7–4.8 per region
    regions: list[RegionResult] = []
    n_seg = max(1, len(segments))
    for i, seg in enumerate(segments):
        _report(progress, 0.72 + 0.2 * i / n_seg, f"Measuring region {i + 1}/{len(segments)}")
        region_feats = _region_features(audio.y, audio.sr, seg.start_s, seg.end_s, ctx)
        regions.append(_measure(
            audio.y, audio.sr, seg.start_s, seg.end_s, ctx, region_feats,
            chord_root=pc_name(seg.root), chord_quality=seg.quality,
            bass_note=pc_name(seg.bass), chord_confidence=seg.confidence,
        ))

    # §4.8 whole file
    _report(progress, 0.93, "Measuring file")
    file_metrics = metrics.measure_tonal_metrics(audio.y, audio.sr, onsets_s=onsets)

    _report(progress, 1.0, "Done")
    log.info(
        "%s: %.1fs, %d regions, %.1f bpm (%.2f), key %s (%.2f), tuning %+.1f c, backend %s/%s, %.2fs",
        name, duration, len(regions), tempo_result.bpm, tempo_result.confidence,
        key_estimate, key_confidence, tuning_cents, backend.name, tempo_result.backend,
        time.perf_counter() - t_start,
    )
    return AnalysisResult(
        duration=duration,
        sample_rate=audio.sr,
        channels=audio.channels,
        tuning_cents=tuning_cents,
        detected_bpm=float(tempo_result.bpm),
        bpm_confidence=float(tempo_result.confidence),
        beats=beats,
        downbeats=downbeats,
        key_estimate=key_estimate,
        key_confidence=float(key_confidence),
        metrics=file_metrics,
        regions=regions,
        context=ctx,
    )


def measure_region(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    ctx: AnalysisContext,
    *,
    chord_root: str | None = None,
    chord_quality: str | None = None,
) -> RegionResult:
    """Measure one region of already-decoded audio `y` (channels, frames).
    When `chord_root`/`chord_quality` are given (user override) they are kept
    and only the bass note, confidence and measurements are computed."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    duration = y.shape[1] / float(sr)
    start_s = float(np.clip(start_s, 0.0, duration))
    end_s = float(np.clip(end_s, 0.0, duration))
    if end_s <= start_s:
        raise ValueError(f"empty region [{start_s}, {end_s})")
    feats = _region_features(y, sr, start_s, end_s, ctx)
    root, quality, bass, conf = _resolve_chord(feats, chord_root, chord_quality)
    return _measure(
        y, sr, start_s, end_s, ctx, feats,
        chord_root=root, chord_quality=quality, bass_note=bass, chord_confidence=conf,
    )
