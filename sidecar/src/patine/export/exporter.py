"""Export contract (handoff §8.4).

    files = export_progression(ExportJob(...))

`renders` maps slot_id → path of that slot's *ready* conformed WAV (keyed by
slot, not region: two slots can share one region at different transpositions
and therefore different render files). `deltas` maps slot_id → ConformDeltas
dict. `regions` maps region_id → Region. The API layer guarantees every
assigned slot has a render before calling.

Full and stem renders reproduce the UI playback model (§8.2–8.3): hard cuts
with 1–2 ms declick ramps and no crossfades; hold = the region's render (which
is already the whole region at the session tempo) played from its start for
the slot's length — truncated when longer, followed by silence when shorter,
never looped; retrigger = restart the render from its start on every step of
the division, with per-step gate, velocity, probability and beat offset. The
set bus (Density comp; Space send when on) applies to the full mix only.

Kinds (`<take>` is `<name>_tNN`, see below):
  * full  → `<take>_full.wav` at `sample_rate`
  * stems → one full-length, time-aligned `<take>_slot<idx>_<chord>.wav` per slot
  * kit   → `<take>_kit/<idx>_<chord>.wav` (the conformed WAVs copied as they are)
            + `patine.json` with chord, key, tempo, beat markers and deltas
  * midi  → `<take>.mid`, one track, chord notes from `theory.chord_midi_notes`

Takes. Exports never overwrite: every export of a progression into a folder
gets the next free take number (`Test_Prog_t01_full.wav`, `Test_Prog_t02_full.wav`
after a reroll, …) and writes `<take>.json`, a record of what was exported
(chords, regions, sample paths, bounds, transposition, set settings, deltas)
so a take can be found again or rebuilt. `ExportJob.overwrite=True` restores
the fixed names (`<name>_full.wav`) for scripted use.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import mido
import numpy as np
import soundfile as sf

from ..audio import write_wav
from ..conform.bus import bus_compressor, reverb, space_send_level
from ..models import Pattern, Progression, Region, Set, Slot, Step
from ..temporal.stretch import declick, resample_ratio
from ..theory import TheoryError, chord_midi_notes, parse_chord_symbol

log = logging.getLogger(__name__)

DEFAULT_BPM = 120.0
DECLICK_MS = 1.5
DIVISION_BEATS: dict[str, float] = {"1/4": 1.0, "1/8": 0.5, "1/16": 0.25}
MIDI_TICKS_PER_BEAT = 480
MIDI_VELOCITY = 100


@dataclass
class ExportJob:
    kind: Literal["full", "stems", "kit", "midi"]
    progression: Progression
    set_: Set
    regions: dict[str, Region]
    renders: dict[str, str]                   # slot_id → wav path
    deltas: dict[str, dict] = field(default_factory=dict)  # slot_id → ConformDeltas dict
    out_dir: Path = Path(".")
    loops: int = 1
    sample_rate: int = 48000
    seed: int | None = None                   # for probability steps
    sample_paths: dict[str, str] = field(default_factory=dict)  # sample_id → source path, for the record
    take: int | None = None                   # None: next free take in out_dir
    overwrite: bool = False                   # fixed `<name>_full.wav` names, no take, no record


@dataclass
class ExportResult:
    files: list[Path]
    take: int | None
    prefix: str


TAKE_RE = re.compile(r"^(?P<base>.+?)_t(?P<n>\d{2,})(?:[_.]|$)")


def next_take(out_dir: Path, base: str) -> int:
    """The first take number not used by any file or folder `<base>_tNN…` in `out_dir`."""
    used = 0
    try:
        names = [p.name for p in Path(out_dir).iterdir()]
    except OSError:
        names = []
    for name in names:
        m = TAKE_RE.match(name)
        if m and m.group("base") == base:
            used = max(used, int(m.group("n")))
    return used + 1


def export_progression(job: ExportJob) -> list[Path]:
    """Files written, the take record last (none for `overwrite` exports)."""
    return export_take(job).files


def export_take(job: ExportJob) -> ExportResult:
    out_dir = Path(job.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _slug(job.progression.name)
    if job.overwrite:
        take: int | None = None
        prefix = base
    else:
        take = job.take if job.take is not None else next_take(out_dir, base)
        prefix = f"{base}_t{take:02d}"
    if job.kind == "full":
        files = [_export_full(job, out_dir, prefix)]
    elif job.kind == "stems":
        files = _export_stems(job, out_dir, prefix)
    elif job.kind == "kit":
        files = _export_kit(job, out_dir, prefix, take)
    elif job.kind == "midi":
        files = [_export_midi(job, out_dir, prefix)]
    else:
        raise ValueError(f"unknown export kind {job.kind!r}")
    if take is not None and job.kind != "kit":
        files.append(_write_take_record(job, out_dir, prefix, take, files))
    return ExportResult(files=files, take=take, prefix=prefix)


def take_record(job: ExportJob, take: int | None, files: list[Path]) -> dict:
    """What this export was made from: enough to find or rebuild it later."""
    bpm = _session_bpm(job)
    slots = []
    for slot in sorted(job.progression.slots, key=lambda s: s.idx):
        region = job.regions.get(slot.region_id or "")
        slots.append({
            "index": slot.idx,
            "chord": slot.chord_symbol,
            "roman": slot.roman,
            "length_beats": slot.length_beats,
            "pattern": slot.pattern.model_dump(),
            "transpose_semitones": slot.transpose_semitones,
            "locked": slot.locked,
            "region_id": slot.region_id,
            "sample_id": region.sample_id if region is not None else None,
            "sample_path": job.sample_paths.get(region.sample_id) if region is not None else None,
            "start_s": region.start_s if region is not None else None,
            "end_s": region.end_s if region is not None else None,
            "source_chord": region.chord_symbol if region is not None else None,
            "deltas": job.deltas.get(slot.id),
        })
    return {
        "patine": "take",
        "name": job.progression.name,
        "take": take,
        "kind": job.kind,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "key": job.progression.key,
        "tempo": bpm,
        "loops": int(job.loops),
        "set": {
            "name": job.set_.name,
            "anchor_region_id": job.set_.anchor_region_id,
            "tonal_settings": job.set_.tonal_settings.model_dump(),
            "temporal_settings": job.set_.temporal_settings.model_dump(),
        },
        "slots": slots,
        "files": [f.name for f in files],
    }


def _write_take_record(job: ExportJob, out_dir: Path, prefix: str, take: int, files: list[Path]) -> Path:
    path = out_dir / f"{prefix}.json"
    path.write_text(json.dumps(take_record(job, take, files), indent=2), encoding="utf-8")
    return path


# --- helpers --------------------------------------------------------------------------

def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9#+-]+", "_", text.strip()).strip("_")
    return s or "untitled"


def _session_bpm(job: ExportJob) -> float:
    bpm = job.set_.session_bpm
    if not bpm or bpm <= 0:
        log.warning("export: set %s has no session bpm, using %.0f", job.set_.id, DEFAULT_BPM)
        return DEFAULT_BPM
    return float(bpm)


def _load_render(path: str, sr: int, channels: int) -> np.ndarray:
    """Load a render as float32 (channels, frames) at `sr`, matching `channels`."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    y = np.ascontiguousarray(data.T)
    if file_sr != sr:
        y = resample_ratio(y, int(file_sr), float(file_sr) / float(sr))
    if y.shape[0] < channels:
        y = np.repeat(y[:1], channels, axis=0) if y.shape[0] == 1 else np.pad(y, ((0, channels - y.shape[0]), (0, 0)))
    elif y.shape[0] > channels:
        y = y.mean(axis=0, keepdims=True) if channels == 1 else y[:channels]
    return np.ascontiguousarray(y, dtype=np.float32)


def _render_channels(job: ExportJob) -> int:
    for slot in job.progression.slots:
        path = job.renders.get(slot.id)
        if path:
            try:
                return max(1, int(sf.info(path).channels))
            except Exception:  # noqa: BLE001
                continue
    return 2


def _slot_frames(slot: Slot, beat_s: float, sr: int) -> int:
    return max(0, int(round(slot.length_beats * beat_s * sr)))


def render_slot(
    render: np.ndarray | None,
    slot: Slot,
    beat_s: float,
    sr: int,
    rng: np.random.Generator,
    channels: int,
    declick_ms: float = DECLICK_MS,
) -> np.ndarray:
    """One slot's audio (channels, slot frames) following its pattern (§8.2)."""
    n = _slot_frames(slot, beat_s, sr)
    out = np.zeros((channels, n), dtype=np.float32)
    if render is None or n == 0 or render.shape[1] == 0:
        return out
    pat: Pattern = slot.pattern
    if pat.mode == "hold":
        # Hold fills the slot: a region shorter than the slot restarts from its
        # snap point (hard cut) until the slot ends, never trailing silence.
        pos = 0
        total = render.shape[1]
        while pos < n:
            m = min(total, n - pos)
            out[:, pos:pos + m] = declick(render[:, :m], sr, declick_ms)
            pos += m
        return out

    step_beats = DIVISION_BEATS.get(pat.division, 0.5)
    steps = pat.steps or [Step()]
    n_steps = int(np.ceil(slot.length_beats / step_beats - 1e-9))
    for k in range(n_steps):
        st = steps[k % len(steps)]
        if not st.active:
            continue
        if st.probability < 1.0 and rng.random() > st.probability:
            continue
        start = int(round(k * step_beats * beat_s * sr))
        gate = min(max(float(st.gate), 0.0), 1.0)
        length = int(round(gate * step_beats * beat_s * sr))
        end = min(start + length, n)
        if end <= start:
            continue
        offset = int(round(max(0.0, float(st.offset_beats)) * beat_s * sr))
        src = render[:, offset:offset + (end - start)]
        if src.shape[1] == 0:
            continue
        seg = declick(src * np.float32(st.velocity), sr, declick_ms)
        out[:, start:start + seg.shape[1]] = seg
    return out


def _timeline(job: ExportJob, sr: int) -> tuple[list[np.ndarray], int, float, int]:
    """Per-slot audio for one pass, plus (pass frames, beat seconds, channels)."""
    bpm = _session_bpm(job)
    beat_s = 60.0 / bpm
    channels = _render_channels(job)
    rng = np.random.default_rng(job.seed)
    cache: dict[str, np.ndarray] = {}
    slots_audio: list[np.ndarray] = []
    for slot in sorted(job.progression.slots, key=lambda s: s.idx):
        render = None
        path = job.renders.get(slot.id)
        if path:
            if path not in cache:
                cache[path] = _load_render(path, sr, channels)
            render = cache[path]
        elif slot.region_id:
            log.warning("export: slot %s has no render; exporting silence", slot.id)
        slots_audio.append(render_slot(render, slot, beat_s, sr, rng, channels))
    pass_frames = sum(a.shape[1] for a in slots_audio)
    return slots_audio, pass_frames, beat_s, channels


def _loop(y: np.ndarray, loops: int) -> np.ndarray:
    loops = max(1, int(loops))
    return np.concatenate([y] * loops, axis=1) if loops > 1 else y


def _apply_bus(job: ExportJob, mix: np.ndarray, slots_audio: list[np.ndarray], sr: int) -> np.ndarray:
    """§6 Density and Space on the full mix (§8.3 set bus)."""
    tonal = job.set_.tonal_settings
    out = mix
    if tonal.space.enabled:
        anchor = job.regions.get(job.set_.anchor_region_id or "")
        anchor_wet = anchor.metrics.wetness if anchor is not None and anchor.metrics else None
        amount = (tonal.space.amount if tonal.space.amount is not None else 100.0) / 100.0
        send = np.zeros_like(mix)
        pass_frames = sum(a.shape[1] for a in slots_audio)
        pos = 0
        any_send = False
        for slot, audio in zip(sorted(job.progression.slots, key=lambda s: s.idx), slots_audio, strict=False):
            region = job.regions.get(slot.region_id or "")
            wet = region.metrics.wetness if region is not None and region.metrics else None
            level = space_send_level(wet, anchor_wet) * amount
            n = audio.shape[1]
            if level > 0 and n:
                for k in range(max(1, int(job.loops))):   # every loop pass of this slot
                    at = k * pass_frames + pos
                    send[:, at:at + n] += audio * np.float32(level)
                any_send = True
            pos += n
        if any_send:
            out = out + reverb(send, sr)
    if tonal.density.enabled:
        amount = (tonal.density.amount if tonal.density.amount is not None else 100.0) / 100.0
        if amount > 0:
            out, _ = bus_compressor(out, sr, target_gr_db=2.5 * amount)
    return out.astype(np.float32)


# --- kinds ---------------------------------------------------------------------------------

def _export_full(job: ExportJob, out_dir: Path, prefix: str) -> Path:
    sr = int(job.sample_rate)
    slots_audio, pass_frames, _, channels = _timeline(job, sr)
    one_pass = np.concatenate(slots_audio, axis=1) if slots_audio else np.zeros((channels, 0), np.float32)
    mix = _loop(one_pass, job.loops)
    mix = _apply_bus(job, mix, slots_audio, sr)
    path = out_dir / f"{prefix}_full.wav"
    write_wav(path, mix, sr)
    return path


def _export_stems(job: ExportJob, out_dir: Path, prefix: str) -> list[Path]:
    sr = int(job.sample_rate)
    slots_audio, pass_frames, _, channels = _timeline(job, sr)
    files: list[Path] = []
    pos = 0
    base = prefix
    for slot, audio in zip(sorted(job.progression.slots, key=lambda s: s.idx), slots_audio, strict=False):
        one_pass = np.zeros((channels, pass_frames), dtype=np.float32)
        one_pass[:, pos:pos + audio.shape[1]] = audio
        pos += audio.shape[1]
        stem = _loop(one_pass, job.loops)
        path = out_dir / f"{base}_slot{slot.idx:02d}_{_slug(slot.chord_symbol)}.wav"
        write_wav(path, stem, sr)
        files.append(path)
    return files


def _beat_markers(region: Region | None, slot: Slot, render_path: str | None, bpm: float) -> list[dict]:
    """Region beats re-expressed at the session tempo: the render's sample 0
    is the region's snap point (beat 0) and every grid beat lands k beats
    later at the session tempo (§7.3 at full warp strength)."""
    beat_s = 60.0 / bpm
    length_s = None
    if render_path:
        try:
            info = sf.info(render_path)
            length_s = float(info.frames) / float(info.samplerate)
        except Exception:  # noqa: BLE001
            length_s = None
    if length_s is None:
        length_s = slot.length_beats * beat_s
    markers = []
    k = 0
    while k * beat_s <= length_s + 1e-6:
        markers.append({"beat": k, "time_s": round(k * beat_s, 6)})
        k += 1
    return markers


def _export_kit(job: ExportJob, out_dir: Path, prefix: str, take: int | None) -> list[Path]:
    bpm = _session_bpm(job)
    kit_dir = out_dir / f"{prefix}_kit"
    kit_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    entries: list[dict] = []
    for slot in sorted(job.progression.slots, key=lambda s: s.idx):
        region = job.regions.get(slot.region_id or "")
        path = job.renders.get(slot.id)
        entry: dict = {
            "index": slot.idx,
            "chord": slot.chord_symbol,
            "roman": slot.roman,
            "length_beats": slot.length_beats,
            "region_id": slot.region_id,
            "transpose_semitones": slot.transpose_semitones,
            "file": None,
            "source_chord": region.chord_symbol if region is not None else None,
            "key_context": region.key_context if region is not None else None,
            "beats": _beat_markers(region, slot, path, bpm) if path else [],
            "deltas": job.deltas.get(slot.id),
        }
        if path and Path(path).exists():
            dest = kit_dir / f"{slot.idx:02d}_{_slug(slot.chord_symbol)}.wav"
            shutil.copyfile(path, dest)
            files.append(dest)
            entry["file"] = dest.name
        entries.append(entry)
    meta = {
        "name": job.progression.name,
        "take": take,
        "key": job.progression.key,
        "tempo": bpm,
        "set": job.set_.name,
        "anchor_region_id": job.set_.anchor_region_id,
        "slots": entries,
        "record": take_record(job, take, files),
    }
    meta_path = kit_dir / "patine.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files.append(meta_path)
    return files


def slot_midi_notes(slot: Slot) -> list[int]:
    try:
        chord = parse_chord_symbol(slot.chord_symbol)
    except TheoryError:
        log.warning("midi: cannot parse chord %r in slot %s; resting", slot.chord_symbol, slot.id)
        return []
    return chord_midi_notes(chord)


def _export_midi(job: ExportJob, out_dir: Path, prefix: str) -> Path:
    bpm = _session_bpm(job)
    mid = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=job.progression.name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    pending = 0  # ticks of rest to attach to the next event
    slots = sorted(job.progression.slots, key=lambda s: s.idx)
    for _ in range(max(1, int(job.loops))):
        for slot in slots:
            ticks = int(round(slot.length_beats * MIDI_TICKS_PER_BEAT))
            notes = slot_midi_notes(slot)
            if not notes or ticks <= 0:
                pending += max(0, ticks)
                continue
            for i, n in enumerate(notes):
                track.append(mido.Message("note_on", note=n, velocity=MIDI_VELOCITY, time=pending if i == 0 else 0))
                pending = 0
            for i, n in enumerate(notes):
                track.append(mido.Message("note_off", note=n, velocity=0, time=ticks if i == 0 else 0))
    track.append(mido.MetaMessage("end_of_track", time=pending))
    path = out_dir / f"{prefix}.mid"
    mid.save(str(path))
    return path
