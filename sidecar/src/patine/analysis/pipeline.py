"""Analysis pipeline entry points (handoff §4).

Runs once per sample on ingest in a background worker. Every output is stored;
nothing here runs at playback time. Progress is reported through the callback
so the library pane can show it.

Contract used by the API layer and the tests:

    result = analyze_file(path, progress=cb)        # whole file + regions
    region = measure_region(y, sr, start_s, end_s, ctx)   # re-measure a user-edited region

`AnalysisContext` carries the whole-file results a region measurement needs
(beats, onsets, key) so editing one region never re-runs the file."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..models import TonalMetrics

ProgressFn = Callable[[float, str], None]


@dataclass
class AnalysisOptions:
    use_source_separation: bool = False   # §4.5 optional flag, off by default
    chord_model: str = "auto"             # 'auto' | 'template' | a learned backend name
    beat_tracker: str = "auto"            # 'auto' | 'librosa' | 'beat_this'
    min_region_beats: float = 1.0         # §4.5 merge segments shorter than one beat


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


def analyze_file(
    path: str | Path,
    *,
    progress: ProgressFn | None = None,
    options: AnalysisOptions | None = None,
) -> AnalysisResult:
    raise NotImplementedError("analysis pipeline not implemented yet")


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
    raise NotImplementedError("region measurement not implemented yet")
