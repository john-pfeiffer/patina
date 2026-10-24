"""Progression endpoints (handoff §8.1–8.2, §5.2–5.5)."""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from ..models import (
    Candidate,
    CsvImportRequest,
    Progression,
    ProgressionCreate,
    ProgressionPatch,
    RenderStatus,
    RerollRequest,
    Slot,
    SlotPatch,
)
from ..selection import rank_candidates, region_chord, reroll
from ..theory import (
    PC_NAMES_SHARP,
    Chord,
    Key,
    TheoryError,
    chord_to_roman,
    parse_chord_symbol,
    parse_key,
    parse_progression_text,
    parse_roman,
)
from . import services
from .deps import ctx

router = APIRouter(tags=["progressions"])


def _key(text: str | None) -> Key | None:
    if not text:
        return None
    try:
        return parse_key(text)
    except TheoryError as e:
        raise HTTPException(400, str(e)) from e


def _slot_specs(text: str, key: Key | None, default_len: float) -> list[dict]:
    try:
        specs = parse_progression_text(text, key, default_len)
    except TheoryError as e:
        raise HTTPException(400, str(e)) from e
    out = []
    for s in specs:
        out.append({
            "chord_symbol": s.chord.symbol(key.prefer_flats if key else False),
            "roman": s.roman,
            "chord_root": PC_NAMES_SHARP[s.chord.root],
            "chord_quality": s.chord.quality,
            "bass_note": PC_NAMES_SHARP[s.chord.bass] if s.chord.bass is not None else None,
            "length_beats": s.length_beats,
        })
    return out


@router.post("/progressions", response_model=Progression)
def create_progression(body: ProgressionCreate, state=Depends(ctx)) -> Progression:
    state.repo.get_set(body.set_id)
    key = _key(body.key)
    prog = state.repo.insert_progression(body.set_id, body.name, key.name() if key else None)
    if body.text:
        state.repo.replace_slots(prog.id, _slot_specs(body.text, key, body.default_length_beats))
    return state.repo.get_progression(prog.id)


@router.get("/progressions", response_model=list[Progression])
def list_progressions(set_id: str | None = None, state=Depends(ctx)) -> list[Progression]:
    return state.repo.list_progressions(set_id)


@router.get("/progressions/{prog_id}", response_model=Progression)
def get_progression(prog_id: str, state=Depends(ctx)) -> Progression:
    return state.repo.get_progression(prog_id)


@router.patch("/progressions/{prog_id}", response_model=Progression)
def patch_progression(prog_id: str, body: ProgressionPatch, state=Depends(ctx)) -> Progression:
    prog = state.repo.get_progression(prog_id)
    fields: dict = {}
    key = _key(body.key) if body.key is not None else _key(prog.key)
    if body.key is not None:
        fields["key"] = key.name() if key else None
    if body.name is not None:
        fields["name"] = body.name
    if body.loop is not None:
        fields["loop"] = body.loop
    prog = state.repo.update_progression(prog_id, **fields)
    if body.text is not None:
        state.repo.replace_slots(prog_id, _slot_specs(body.text, key, 4.0))
    elif body.key is not None:
        # Re-label romans against the new key.
        for s in prog.slots:
            c = _slot_chord(s)
            if c is not None:
                state.repo.update_slot(s.id, roman=chord_to_roman(c, key) if key else None)
    return state.repo.get_progression(prog_id)


@router.delete("/progressions/{prog_id}", status_code=204)
def delete_progression(prog_id: str, state=Depends(ctx)) -> Response:
    state.repo.get_progression(prog_id)
    state.repo.delete_progression(prog_id)
    return Response(status_code=204)


@router.get("/progressions/{prog_id}/renders", response_model=dict[str, RenderStatus])
def progression_renders(prog_id: str, state=Depends(ctx)) -> dict[str, RenderStatus]:
    return services.progression_renders(state, prog_id)


def _slot_chord(slot: Slot) -> Chord | None:
    if slot.chord_root is None or slot.chord_quality is None:
        return None
    bass = PC_NAMES_SHARP.index(slot.bass_note) if slot.bass_note in PC_NAMES_SHARP else None
    return Chord(PC_NAMES_SHARP.index(slot.chord_root), slot.chord_quality, bass)


@router.patch("/slots/{slot_id}", response_model=Slot)
def patch_slot(slot_id: str, body: SlotPatch, state=Depends(ctx)) -> Slot:
    slot = state.repo.get_slot(slot_id)
    prog = state.repo.get_progression(slot.progression_id)
    key = _key(prog.key)
    fields: dict = {}
    try:
        if body.roman is not None:
            if key is None:
                raise HTTPException(400, "roman numerals need a progression key")
            c = parse_roman(body.roman, key)
            fields.update(chord_symbol=c.symbol(key.prefer_flats), roman=body.roman,
                          chord_root=PC_NAMES_SHARP[c.root], chord_quality=c.quality, bass_note=None)
        elif body.chord_symbol is not None:
            c = parse_chord_symbol(body.chord_symbol)
            fields.update(chord_symbol=body.chord_symbol.strip(), roman=chord_to_roman(c, key) if key else None,
                          chord_root=PC_NAMES_SHARP[c.root], chord_quality=c.quality,
                          bass_note=PC_NAMES_SHARP[c.bass] if c.bass is not None else None)
    except TheoryError as e:
        raise HTTPException(400, str(e)) from e
    if body.length_beats is not None:
        if body.length_beats <= 0:
            raise HTTPException(400, "length_beats must be positive")
        fields["length_beats"] = body.length_beats
    if body.clear_region:
        fields["region_id"] = None
        fields["transpose_semitones"] = 0
    elif body.region_id is not None:
        state.repo.get_region(body.region_id)
        fields["region_id"] = body.region_id
        fields["transpose_semitones"] = body.transpose_semitones or 0
        _ensure_member(state, prog.set_id, body.region_id)
    elif body.transpose_semitones is not None:
        fields["transpose_semitones"] = body.transpose_semitones
    if body.locked is not None:
        fields["locked"] = body.locked
    if body.pattern is not None:
        fields["pattern"] = body.pattern
    return state.repo.update_slot(slot_id, **fields)


def _ensure_member(state, set_id: str, region_id: str) -> None:
    """A region that plays in a slot is part of the set: it is conformed toward
    the anchor, so it belongs in the member list with its delta readout. The
    first member of an empty set becomes the anchor (§6 anchor rules)."""
    set_ = state.repo.get_set(set_id)
    if any(m.region.id == region_id for m in set_.members):
        return
    set_ = state.repo.add_member(set_id, region_id)
    if set_.anchor_region_id is None:
        state.repo.update_set(set_id, anchor_region_id=region_id)
    services.invalidate_set(state, set_id)


def _anchor_metrics(state, set_):
    if not set_.anchor_region_id:
        return None
    try:
        return state.repo.get_region(set_.anchor_region_id).metrics
    except KeyError:
        return None


@router.get("/slots/{slot_id}/candidates", response_model=list[Candidate])
def candidates(
    slot_id: str,
    strictness: str = Query("function", pattern="^(exact|function|two_tones)$"),
    limit: int = Query(50, ge=1, le=500),
    state=Depends(ctx),
) -> list[Candidate]:
    slot = state.repo.get_slot(slot_id)
    prog = state.repo.get_progression(slot.progression_id)
    set_ = state.repo.get_set(prog.set_id)
    target = _slot_chord(slot)
    if target is None:
        return []
    key = _key(prog.key)
    pool = [(r, s) for r, s in state.repo.regions_with_samples() if region_chord(r) is not None]
    return rank_candidates(target, key, pool, _anchor_metrics(state, set_), strictness, limit=limit)  # type: ignore[arg-type]


@router.post("/progressions/{prog_id}/reroll", response_model=Progression)
def reroll_progression(prog_id: str, body: RerollRequest, state=Depends(ctx)) -> Progression:
    prog = state.repo.get_progression(prog_id)
    set_ = state.repo.get_set(prog.set_id)
    key = _key(prog.key)
    pool = [(r, s) for r, s in state.repo.regions_with_samples() if region_chord(r) is not None]
    slots = []
    for s in prog.slots:
        c = _slot_chord(s)
        if c is not None:
            slots.append((s.id, c, s.region_id, s.locked))
    changes = reroll(slots, key, pool, _anchor_metrics(state, set_), body.strictness, body.amount, seed=body.seed)
    for slot_id, pick in changes.items():
        if pick is None:
            continue
        region_id, transpose = pick
        state.repo.update_slot(slot_id, region_id=region_id, transpose_semitones=transpose)
        _ensure_member(state, prog.set_id, region_id)
    return state.repo.get_progression(prog_id)


@router.post("/progressions/import-csv", response_model=list[Progression])
def import_csv(body: CsvImportRequest, state=Depends(ctx)) -> list[Progression]:
    state.repo.get_set(body.set_id)
    reader = csv.DictReader(io.StringIO(body.csv_text.strip()))
    if not reader.fieldnames:
        raise HTTPException(400, "CSV needs a header row with chords, roman, key, bpm")
    cols = {c.strip().lower(): c for c in reader.fieldnames}
    out: list[Progression] = []
    for i, row in enumerate(reader, start=1):
        chords = (row.get(cols.get("chords", ""), "") or "").strip()
        roman = (row.get(cols.get("roman", ""), "") or "").strip()
        key_text = (row.get(cols.get("key", ""), "") or "").strip()
        bpm = (row.get(cols.get("bpm", ""), "") or "").strip()
        key = _key(key_text) if key_text else None
        text = chords or roman
        if not text:
            continue
        try:
            specs = _slot_specs(text.replace("-", " ").replace("|", " "), key, body.default_length_beats)
        except HTTPException as e:
            raise HTTPException(400, f"row {i}: {e.detail}") from e
        name = " ".join(s["chord_symbol"] for s in specs)
        if key:
            name += f" ({key.name()})"
        if bpm:
            name += f" {bpm} bpm"
        prog = state.repo.insert_progression(body.set_id, name, key.name() if key else None)
        state.repo.replace_slots(prog.id, specs)
        out.append(state.repo.get_progression(prog.id))
    return out
