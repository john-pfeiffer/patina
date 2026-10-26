"""Pydantic models shared by the API, the database layer and the renderer.
Field names follow the handoff data model (§3) one for one."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# --- measurements ------------------------------------------------------------

TILT_BAND_CENTRES_HZ = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


class TonalMetrics(BaseModel):
    """Handoff §4.8. Measured per region and per whole file."""

    loudness_lufs: float | None = None
    tilt_bands: list[float] = Field(default_factory=lambda: [0.0] * 10)
    low_rolloff_hz: float | None = None
    bass_mono_hz: float | None = None
    sub_ratio: float | None = None
    noise_floor_db: float | None = None
    noise_spectrum: list[float] = Field(default_factory=lambda: [0.0] * 10)
    width_bands: list[float] = Field(default_factory=lambda: [0.0] * 3)
    crest_db: float | None = None
    wetness: float | None = None


# --- per-region user state ----------------------------------------------------

class WarpMarker(BaseModel):
    """A manual marker pins source time `source_s` (absolute seconds in the
    file) to `beat` (beats from the region's grid origin)."""

    source_s: float
    beat: float


class UserOverrides(BaseModel):
    """Handoff §3 `user_overrides`, §6 per-region bypass, §7 per-region temporal."""

    keep_preroll: bool = False            # §7.2 — play from region start, not first onset
    nudge_ms: float = 0.0                 # §7.2
    warp_markers: list[WarpMarker] = Field(default_factory=list)  # §7.3 tier 1
    warp_strength: float | None = None    # §7.4 per-region override, 0–100
    stretch_method: Literal["elastic", "repitch"] | None = None   # §7.5 override
    allow_wobble: bool = False            # §7.5 — repitch under non-uniform warp
    tempo_multiplier: float = 1.0         # §7.1 ½× / 2× buttons
    tempo_feel: Literal["quarter", "eighth", "half"] = "quarter"  # §7.1
    bypass: dict[str, bool] = Field(default_factory=dict)         # §6 module → bypassed


# --- settings ----------------------------------------------------------------

ModuleName = Literal["tuning", "loudness", "tilt", "lowend", "noise", "width", "density", "space"]
MODULE_NAMES: tuple[str, ...] = ("tuning", "loudness", "tilt", "lowend", "noise", "width", "density", "space")
REGION_MODULES: tuple[str, ...] = ("tuning", "loudness", "tilt", "lowend", "noise", "width")
BUS_MODULES: tuple[str, ...] = ("density", "space")


class ModuleSetting(BaseModel):
    enabled: bool = True
    amount: float | None = None  # percent where the module has an amount


class TonalSettings(BaseModel):
    """Handoff §6 defaults."""

    tuning: ModuleSetting = Field(default_factory=lambda: ModuleSetting(enabled=True))
    loudness: ModuleSetting = Field(default_factory=lambda: ModuleSetting(enabled=True))
    tilt: ModuleSetting = Field(default_factory=lambda: ModuleSetting(enabled=True, amount=40))
    lowend: ModuleSetting = Field(default_factory=lambda: ModuleSetting(enabled=True))
    noise: ModuleSetting = Field(default_factory=lambda: ModuleSetting(enabled=True))
    width: ModuleSetting = Field(default_factory=lambda: ModuleSetting(enabled=True, amount=100))
    density: ModuleSetting = Field(default_factory=lambda: ModuleSetting(enabled=True, amount=100))
    space: ModuleSetting = Field(default_factory=lambda: ModuleSetting(enabled=False, amount=100))

    def module(self, name: str) -> ModuleSetting:
        return getattr(self, name)


class TemporalSettings(BaseModel):
    """Handoff §7 defaults (set-level)."""

    stretch_method: Literal["elastic", "repitch"] = "elastic"
    warp_strength: float = 85.0
    transient_snap: bool = True
    repitch_mode: Literal["plain", "elastic"] = "plain"   # §5.4 transposition mode
    declick_ms: float = 1.5                               # §7.6


# --- patterns ------------------------------------------------------------------

class Step(BaseModel):
    active: bool = True
    gate: float = 1.0          # fraction of the division the hit stays open
    velocity: float = 1.0
    probability: float = 1.0
    offset_beats: float = 0.0  # beat offset into the region


class Pattern(BaseModel):
    """Handoff §8.2."""

    mode: Literal["hold", "retrigger"] = "hold"
    division: Literal["1/4", "1/8", "1/16"] = "1/8"
    steps: list[Step] = Field(default_factory=list)


# --- entities (API shape) -----------------------------------------------------

class Region(BaseModel):
    id: str
    sample_id: str
    position: int = 0
    start_s: float
    end_s: float
    chord_root: str | None = None
    chord_quality: str | None = None
    bass_note: str | None = None
    chord_confidence: float | None = None
    key_context: str | None = None
    beats: list[float] = Field(default_factory=list)
    first_onset_s: float | None = None
    declared_length_beats: float | None = None
    busyness: float | None = None
    register_centroid_hz: float | None = None
    metrics: TonalMetrics | None = None
    user_overrides: UserOverrides = Field(default_factory=UserOverrides)
    source: Literal["analysis", "user"] = "analysis"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def chord_symbol(self) -> str | None:
        if self.chord_root is None or self.chord_quality is None:
            return None
        from .theory import QUALITY_SUFFIX

        s = self.chord_root + QUALITY_SUFFIX.get(self.chord_quality, "")
        if self.bass_note and self.bass_note != self.chord_root:
            s += "/" + self.bass_note
        return s


class Sample(BaseModel):
    id: str
    path: str
    hash: str
    name: str
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    added_at: str
    tuning_cents: float | None = None
    detected_bpm: float | None = None
    bpm_confidence: float | None = None
    beats: list[float] = Field(default_factory=list)
    downbeats: list[float] = Field(default_factory=list)
    onsets: list[float] = Field(default_factory=list)     # §4.4 transients, whole file
    key_estimate: str | None = None
    key_confidence: float | None = None
    metrics: TonalMetrics | None = None
    analysis_status: Literal["pending", "running", "done", "failed"] = "pending"
    analysis_error: str | None = None
    analysis_version: int | None = None
    analysis_progress: float = 0.0
    regions: list[Region] = Field(default_factory=list)


class SetMember(BaseModel):
    region: Region
    position: int


class Set(BaseModel):
    id: str
    name: str
    anchor_region_id: str | None = None
    session_bpm: float | None = None
    bpm_pinned: bool = False
    tonal_settings: TonalSettings = Field(default_factory=TonalSettings)
    temporal_settings: TemporalSettings = Field(default_factory=TemporalSettings)
    created_at: str
    members: list[SetMember] = Field(default_factory=list)


class Slot(BaseModel):
    id: str
    progression_id: str
    idx: int
    chord_symbol: str
    roman: str | None = None
    chord_root: str | None = None
    chord_quality: str | None = None
    bass_note: str | None = None
    length_beats: float = 4.0
    region_id: str | None = None
    transpose_semitones: int = 0
    locked: bool = False
    pattern: Pattern = Field(default_factory=Pattern)


class Progression(BaseModel):
    id: str
    set_id: str
    name: str
    key: str | None = None
    loop: bool = True
    created_at: str
    slots: list[Slot] = Field(default_factory=list)


class Job(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "done", "failed", "cancelled"]
    progress: float = 0.0
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


# --- renders ---------------------------------------------------------------------

class ConformDeltas(BaseModel):
    """How far a member moved on each module (§6 delta readout). Units differ per
    module and are documented on the field."""

    tuning_cents: float = 0.0            # cents corrected
    loudness_db: float = 0.0             # gain applied
    tilt_db: list[float] = Field(default_factory=lambda: [0.0] * 10)  # per band after amount/cap
    lowend_hpf_hz: float | None = None   # high-pass applied, or None
    lowend_mono_hz: float | None = None  # mono-below applied, or None
    noise_floor_db: float | None = None  # level of the bedded floor, or None
    width_ratio: list[float] = Field(default_factory=lambda: [1.0] * 3)  # side scale per band
    stretch_ratio: float = 1.0           # §7.5 average stretch
    transpose_semitones: int = 0
    bypassed: list[str] = Field(default_factory=list)


class RenderStatus(BaseModel):
    cache_key: str
    region_id: str
    set_id: str
    status: Literal["queued", "rendering", "ready", "failed"]
    url: str | None = None            # /renders/{cache_key}.wav when ready
    error: str | None = None
    deltas: ConformDeltas | None = None
    stale: bool = False               # a newer key exists for this region but is not ready


class Candidate(BaseModel):
    """Handoff §5.2 ranking output."""

    region: Region
    sample_name: str
    sample_id: str = ""
    # Other slots of the same progression whose region comes from this sample
    # (1-based display indices), so the UI can show and the reroll can avoid reuse.
    used_by_slot_idx: list[int] = Field(default_factory=list)
    score: float
    transposition_cost: float
    timbral_distance: float
    confidence_cost: float
    transpose_semitones: int
    label: str                 # 'exact', 'relative', 'same function (dominant)', 'shares 2 tones', '+3 st'


# --- request bodies --------------------------------------------------------------

class IngestRequest(BaseModel):
    paths: list[str]


class RegionPatch(BaseModel):
    start_s: float | None = None
    end_s: float | None = None
    chord_root: str | None = None
    chord_quality: str | None = None
    bass_note: str | None = None
    key_context: str | None = None
    declared_length_beats: float | None = None
    clear_declared_length: bool = False
    user_overrides: UserOverrides | None = None


class RegionCreate(BaseModel):
    start_s: float
    end_s: float
    chord_symbol: str | None = None
    declared_length_beats: float | None = None


class SetCreate(BaseModel):
    name: str = "Untitled set"


class SetPatch(BaseModel):
    name: str | None = None
    anchor_region_id: str | None = None
    session_bpm: float | None = None
    bpm_pinned: bool | None = None
    tonal_settings: TonalSettings | None = None
    temporal_settings: TemporalSettings | None = None


class MemberAdd(BaseModel):
    region_id: str


class ProgressionCreate(BaseModel):
    set_id: str
    name: str = "Progression"
    key: str | None = None
    text: str | None = None          # 'Cm Fm G' or 'i iv V'
    default_length_beats: float = 4.0


class ProgressionPatch(BaseModel):
    name: str | None = None
    key: str | None = None
    loop: bool | None = None
    text: str | None = None          # replaces all slots (keeps assignments where chord unchanged)


class SlotPatch(BaseModel):
    chord_symbol: str | None = None
    roman: str | None = None
    length_beats: float | None = None
    region_id: str | None = None
    clear_region: bool = False
    transpose_semitones: int | None = None
    locked: bool | None = None
    pattern: Pattern | None = None


class RerollRequest(BaseModel):
    amount: float = 100.0            # percent of unlocked slots to change
    strictness: Literal["exact", "function", "two_tones"] = "function"
    seed: int | None = None
    # Never give two slots regions from the same source sample. Locked and
    # unchanged slots claim their samples first; when a slot has no candidate
    # left from an unused sample the rule is relaxed for it and a warning says so.
    unique_samples: bool = False


class RerollResponse(BaseModel):
    progression: Progression
    warnings: list[str] = Field(default_factory=list)


class ParseRequest(BaseModel):
    text: str
    key: str | None = None


class ParsedChord(BaseModel):
    input: str
    chord_symbol: str
    roman: str | None
    root: str
    quality: str
    bass: str | None
    pitch_classes: list[int]
    length_beats: float


class ParseResponse(BaseModel):
    ok: bool
    key: str | None = None
    chords: list[ParsedChord] = Field(default_factory=list)
    error: str | None = None


class CsvImportRequest(BaseModel):
    set_id: str
    csv_text: str
    default_length_beats: float = 4.0


class ExportRequest(BaseModel):
    progression_id: str
    kind: Literal["full", "stems", "kit", "midi"]
    out_dir: str
    loops: int = 1
    # Exports are numbered takes (`<name>_t03_full.wav`) and never overwrite;
    # `overwrite` restores the fixed `<name>_full.wav` names.
    overwrite: bool = False


class SidecarInfo(BaseModel):
    name: str = "patine"
    version: str
    analysis_version: int
    render_version: int
    data_dir: str
    ml_available: bool = False

    @field_validator("version")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("version required")
        return v
