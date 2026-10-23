"""Conform renderer contract.

    key = cache_key(req)                 # content-addressed (§2 render cache)
    deltas = render_region(req, out_wav) # writes the WAV, returns the delta readout
    deltas = compute_deltas(req)         # readout only, no audio

Processing order per region (§6): Tuning → temporal layer → Loudness → Tilt →
Low-end → Noise → Width. Density and Space are bus-level and never rendered
into region files.

The anchor invariant (§6): when `req.is_anchor` every tonal module is identity
and the written file is the snapped source slice after the temporal layer —
with the session at the region's own tempo and a uniform plan, bit-identical
apart from the declick ramps. Tonal modules are also identity when
`req.anchor` is None (no anchor to conform toward), when the module is off in
`TonalSettings`, or when the region bypasses it (`user_overrides.bypass`).

Output WAVs are float32 at the SOURCE sample rate; the UI resamples on load."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .. import RENDER_VERSION
from ..audio import load_audio, write_wav
from ..models import (
    REGION_MODULES,
    ConformDeltas,
    Region,
    TemporalSettings,
    TonalMetrics,
    TonalSettings,
)
from ..temporal.grid import RegionTempo, derive_region_tempo
from ..temporal.stretch import declick, resample_ratio, resolve_method, stretch_plan
from ..temporal.warp import WarpPlan, build_warp_plan, rebase_plan, scale_plan_source
from .loudness import loudness_gain_db, match_loudness
from .lowend import align_lowend, lowend_decision
from .noise import bed_level_db, bed_noise_floor
from .tilt import match_tilt, tilt_correction_db
from .tuning import correct_tuning, tuning_shift_cents
from .width import match_width, width_scales

log = logging.getLogger(__name__)

SLICE_CONTEXT_S = 0.25   # source context around the region handed to the stretcher


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


# --- cache key -------------------------------------------------------------------

def _canonical(obj: object) -> object:
    """JSON-ready, deterministic form: pydantic models dumped, floats kept as
    Python floats (their repr is platform-stable), NaN/inf stringified."""
    if hasattr(obj, "model_dump"):
        return _canonical(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list | tuple):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return str(obj)
        return float(obj)
    if isinstance(obj, np.floating | np.integer):
        return _canonical(obj.item())
    return obj


def anchor_fingerprint(anchor: AnchorProfile | None) -> dict | None:
    if anchor is None:
        return None
    return {
        "region_id": anchor.region_id,
        "metrics": _canonical(anchor.metrics),
        "tuning_cents": float(anchor.tuning_cents),
        "sample_hash": anchor.sample_hash,
        "start_s": float(anchor.start_s),
        "end_s": float(anchor.end_s),
    }


def cache_key(req: RenderRequest) -> str:
    """hash(source file, region bounds, temporal settings, tonal settings,
    anchor fingerprint, overrides, transpose, RENDER_VERSION). Stable across
    processes and machines for identical inputs."""
    r = req.region
    payload = {
        "render_version": RENDER_VERSION,
        "sample_hash": req.sample_hash,
        "region": {
            "start_s": float(r.start_s),
            "end_s": float(r.end_s),
            "first_onset_s": r.first_onset_s,
            "declared_length_beats": r.declared_length_beats,
            "beats": [float(b) for b in r.beats],
            "metrics": _canonical(r.metrics) if r.metrics is not None else None,
        },
        "overrides": _canonical(r.user_overrides),
        "sample": {
            "tuning_cents": float(req.sample_tuning_cents),
            "bpm": req.sample_bpm,
            "bpm_confidence": req.sample_bpm_confidence,
            "beats": [float(b) for b in req.sample_beats],
        },
        "temporal": _canonical(req.temporal),
        "session_bpm": float(req.session_bpm),
        "tonal": _canonical(req.tonal),
        "anchor": anchor_fingerprint(req.anchor),
        "transpose_semitones": int(req.transpose_semitones),
        "is_anchor": bool(req.is_anchor),
        "extra": _canonical(req.extra),
    }
    blob = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


# --- module gating ----------------------------------------------------------------

def _active_modules(req: RenderRequest) -> tuple[set[str], list[str]]:
    """(modules that run, modules listed as bypassed in the readout)."""
    bypassed: list[str] = []
    active: set[str] = set()
    ov = req.region.user_overrides
    for name in REGION_MODULES:
        enabled = req.tonal.module(name).enabled
        user_bypass = bool(ov.bypass.get(name, False))
        if not enabled or user_bypass:
            bypassed.append(name)
            continue
        if req.is_anchor or req.anchor is None:
            continue  # identity, but not the user's doing: not listed as bypassed
        active.add(name)
    return active, bypassed


def _temporal_plan(req: RenderRequest) -> tuple[RegionTempo, WarpPlan]:
    tempo = derive_region_tempo(
        req.region,
        req.sample_bpm,
        req.sample_bpm_confidence,
        req.sample_beats,
        transient_snap=req.temporal.transient_snap,
    )
    ov = req.region.user_overrides
    strength = ov.warp_strength if ov.warp_strength is not None else req.temporal.warp_strength
    plan = build_warp_plan(
        req.region, tempo, req.session_bpm, strength, ov, transient_snap=req.temporal.transient_snap
    )
    return tempo, plan


def _tuning_shift(req: RenderRequest, active: set[str]) -> float:
    if "tuning" not in active:
        return 0.0
    anchor_cents = req.anchor.tuning_cents if req.anchor is not None else None
    return tuning_shift_cents(req.sample_tuning_cents, anchor_cents)


# --- deltas ---------------------------------------------------------------------------

def compute_deltas(req: RenderRequest) -> ConformDeltas:
    """The delta readout from metrics alone (§6), no audio touched. Loudness is
    the metric gap; `render_region` reports the gain it actually applied."""
    active, bypassed = _active_modules(req)
    _, plan = _temporal_plan(req)
    d = ConformDeltas(
        stretch_ratio=float(plan.average_ratio),
        transpose_semitones=int(req.transpose_semitones),
        bypassed=bypassed,
    )
    d.tuning_cents = _tuning_shift(req, active)
    if req.anchor is None:
        return d
    rm = req.region.metrics or TonalMetrics()
    am = req.anchor.metrics
    if "loudness" in active and rm.loudness_lufs is not None:
        d.loudness_db = loudness_gain_db(rm.loudness_lufs, am.loudness_lufs)
    if "tilt" in active:
        d.tilt_db = tilt_correction_db(rm.tilt_bands, am.tilt_bands, req.tonal.tilt.amount)
    if "lowend" in active:
        d.lowend_hpf_hz, d.lowend_mono_hz = lowend_decision(rm, am, 2)
    if "noise" in active:
        d.noise_floor_db = bed_level_db(rm.noise_floor_db, am.noise_floor_db)
    if "width" in active:
        d.width_ratio = width_scales(rm.width_bands, am.width_bands, req.tonal.width.amount)
    return d


# --- render ---------------------------------------------------------------------------

def _progress(cb: ProgressFn | None, fraction: float, message: str) -> None:
    if cb is not None:
        cb(fraction, message)


def _load_anchor_audio(req: RenderRequest, y: np.ndarray, sr: int) -> np.ndarray | None:
    a = req.anchor
    if a is None:
        return None
    try:
        if a.sample_path == req.sample_path:
            ya, sra = y, sr
        else:
            ya, sra = load_audio(a.sample_path)
    except Exception as e:  # noqa: BLE001
        log.warning("noise: cannot load anchor audio %s: %s", a.sample_path, e)
        return None
    start = max(0, int(round(a.start_s * sra)))
    end = min(ya.shape[1], int(round(a.end_s * sra)))
    piece = ya[:, start:end]
    if sra != sr and piece.shape[1] > 0:
        piece = resample_ratio(piece, sra, sra / sr)
    return np.ascontiguousarray(piece)


def render_region(req: RenderRequest, out_path: str, *, progress: ProgressFn | None = None) -> ConformDeltas:
    """Render `req.region` conformed to the anchor at the session tempo (§6, §7)
    into `out_path` (float32 WAV, source sample rate). Returns the deltas."""
    region = req.region
    active, bypassed = _active_modules(req)
    deltas = ConformDeltas(transpose_semitones=int(req.transpose_semitones), bypassed=bypassed)

    _progress(progress, 0.0, "load")
    y, sr = load_audio(req.sample_path)
    y = np.ascontiguousarray(y, dtype=np.float32)
    tempo, plan = _temporal_plan(req)
    deltas.stretch_ratio = float(plan.average_ratio)

    # Slice with context for the stretcher; the slice origin sits on a whole sample so
    # the rebased plan rounds exactly like the file-absolute one would.
    a_idx = max(0, int(round((plan.play_start_s - SLICE_CONTEXT_S) * sr)))
    b_idx = min(y.shape[1], int(round((region.end_s + SLICE_CONTEXT_S) * sr)))
    piece = np.ascontiguousarray(y[:, a_idx:b_idx])
    local = rebase_plan(plan, a_idx / sr)

    # Tuning (§6, runs first; its length change is absorbed by the stretch)
    shift = _tuning_shift(req, active)
    if abs(shift) >= 0.05:
        _progress(progress, 0.1, "tuning")
        piece, speed = correct_tuning(piece, sr, shift)
        local = scale_plan_source(local, 1.0 / speed)
        deltas.tuning_cents = shift

    # Plain transposition = resample, compensated by folding the factor into the plan (§5.4)
    elastic_transpose = 0.0
    st = int(req.transpose_semitones)
    if st != 0:
        if req.temporal.repitch_mode == "plain":
            f = float(2.0 ** (st / 12.0))
            piece = resample_ratio(piece, sr, f)
            local = scale_plan_source(local, 1.0 / f)
        else:
            elastic_transpose = float(st)

    # Temporal layer (§7)
    _progress(progress, 0.2, "stretch")
    method, forced = resolve_method(req.temporal, region.user_overrides, local)
    if forced:
        log.info("region %s: non-uniform warp, repitch forced to elastic (§7.5)", region.id)
    out = stretch_plan(piece, sr, local, method=method, transpose_semitones=elastic_transpose, declick_ms=0.0)

    # Tonal modules (§6), identity for the anchor
    if active:
        rm = region.metrics or TonalMetrics()
        am = req.anchor.metrics  # type: ignore[union-attr]  # active is empty without an anchor
        if "loudness" in active:
            _progress(progress, 0.5, "loudness")
            out, deltas.loudness_db = match_loudness(out, sr, am.loudness_lufs)
        if "tilt" in active:
            _progress(progress, 0.6, "tilt")
            out, deltas.tilt_db = match_tilt(out, sr, rm.tilt_bands, am.tilt_bands, req.tonal.tilt.amount)
        if "lowend" in active:
            _progress(progress, 0.7, "lowend")
            out, deltas.lowend_hpf_hz, deltas.lowend_mono_hz = align_lowend(out, sr, rm, am)
        if "noise" in active:
            _progress(progress, 0.8, "noise")
            anchor_audio = None
            if bed_level_db(rm.noise_floor_db, am.noise_floor_db) is not None:
                anchor_audio = _load_anchor_audio(req, y, sr)
            out, deltas.noise_floor_db = bed_noise_floor(
                out, sr, rm.noise_floor_db, am.noise_floor_db, anchor_audio, am.noise_spectrum
            )
        if "width" in active:
            _progress(progress, 0.9, "width")
            out, deltas.width_ratio = match_width(out, sr, rm.width_bands, am.width_bands, req.tonal.width.amount)

    out = declick(out, sr, req.temporal.declick_ms)   # §7.6 hard cuts, last so nothing smears them
    write_wav(out_path, out, sr)
    _progress(progress, 1.0, "done")
    return deltas
