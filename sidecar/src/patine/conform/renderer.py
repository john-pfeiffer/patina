"""Conform renderer contract.

    key = cache_key(req)                 # content-addressed (§2 render cache)
    deltas = render_region(req, out_wav) # writes the WAV, returns the delta readout
    deltas = compute_deltas(req)         # readout only, no audio

Processing order per region (§6): Tuning → temporal layer → Loudness → Tilt →
Low-end → Noise → Width. Density and Space are bus-level and never rendered
into region files."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..models import ConformDeltas, Region, TemporalSettings, TonalMetrics, TonalSettings


@dataclass
class AnchorProfile:
    """What a member needs to know about the anchor. Hashed into every member's
    cache key, so switching anchors invalidates every tonal render (§2)."""

    region_id: str
    metrics: TonalMetrics
    tuning_cents: float
    sample_path: str
    sample_hash: str
    start_s: float
    end_s: float


@dataclass
class RenderRequest:
    region: Region
    sample_path: str
    sample_hash: str
    sample_tuning_cents: float
    sample_bpm: float | None
    sample_bpm_confidence: float | None
    sample_beats: list[float]
    anchor: AnchorProfile | None          # None → tonal modules are identity
    tonal: TonalSettings
    temporal: TemporalSettings
    session_bpm: float
    transpose_semitones: int = 0
    is_anchor: bool = False
    extra: dict = field(default_factory=dict)   # reserved; hashed into the key


ProgressFn = Callable[[float, str], None]


def cache_key(req: RenderRequest) -> str:
    """hash(source file, region bounds, temporal settings, tonal settings,
    anchor fingerprint, overrides, transpose, RENDER_VERSION). Stable across
    processes and machines for identical inputs."""
    raise NotImplementedError


def compute_deltas(req: RenderRequest) -> ConformDeltas:
    raise NotImplementedError


def render_region(req: RenderRequest, out_path: str, *, progress: ProgressFn | None = None) -> ConformDeltas:
    raise NotImplementedError
