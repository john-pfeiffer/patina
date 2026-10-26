"""Library endpoints (handoff §5.1): samples, regions, audition audio."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse

from ..audio import load_audio, slice_seconds, to_mono
from ..models import IngestRequest, Job, Region, RegionCreate, RegionPatch, Sample
from ..theory import PC_NAMES_SHARP, parse_chord_symbol
from . import services
from .deps import ctx

router = APIRouter(tags=["library"])


@router.post("/samples")
def ingest(body: IngestRequest, state=Depends(ctx)) -> dict:
    samples, jobs = services.ingest(state, body.paths)
    return {"samples": [s.model_dump() for s in samples], "jobs": [j.model_dump() for j in jobs]}


@router.get("/samples", response_model=list[Sample])
def list_samples(state=Depends(ctx)) -> list[Sample]:
    return state.repo.list_samples()


@router.get("/samples/{sample_id}", response_model=Sample)
def get_sample(sample_id: str, state=Depends(ctx)) -> Sample:
    return state.repo.get_sample(sample_id)


@router.delete("/samples/{sample_id}", status_code=204)
def delete_sample(sample_id: str, state=Depends(ctx)) -> Response:
    sample = state.repo.get_sample(sample_id)
    set_ids = {sid for r in sample.regions for sid in state.repo.sets_containing(r.id)}
    state.repo.delete_sample(sample_id)
    state.events.publish("sample.removed", {"sample_id": sample_id})
    for sid in set_ids:
        services.invalidate_set(state, sid)
    return Response(status_code=204)


@router.post("/samples/{sample_id}/reanalyze", response_model=Job)
def reanalyze(sample_id: str, state=Depends(ctx)) -> Job:
    state.repo.get_sample(sample_id)
    return services.queue_analysis(state, sample_id)


@router.get("/samples/{sample_id}/peaks")
def peaks(sample_id: str, bins: int = Query(800, ge=10, le=10000), state=Depends(ctx)) -> dict:
    sample = state.repo.get_sample(sample_id)
    y, sr = load_audio(sample.path)
    mono = np.abs(to_mono(y))
    n = mono.shape[0]
    if n == 0:
        return {"bins": 0, "duration": 0.0, "peaks": []}
    edges = np.linspace(0, n, bins + 1).astype(int)
    out = [float(mono[a:b].max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return {"bins": bins, "duration": n / sr, "peaks": out}


@router.get("/samples/{sample_id}/audio")
def sample_audio(sample_id: str, state=Depends(ctx)) -> FileResponse:
    sample = state.repo.get_sample(sample_id)
    path = Path(sample.path)
    if not path.exists():
        raise HTTPException(404, "source file missing")
    return FileResponse(str(path), filename=path.name)


def _validate_chord_fields(root: str | None, quality: str | None, bass: str | None) -> None:
    for name, v in (("chord_root", root), ("bass_note", bass)):
        if v is not None and v not in PC_NAMES_SHARP:
            raise HTTPException(400, f"{name} must be one of {PC_NAMES_SHARP}")
    if quality is not None:
        from ..theory import QUALITY_INTERVALS

        if quality not in QUALITY_INTERVALS:
            raise HTTPException(400, f"unknown chord_quality {quality!r}")


@router.post("/samples/{sample_id}/regions", response_model=Region)
def create_region(sample_id: str, body: RegionCreate, state=Depends(ctx)) -> Region:
    sample = state.repo.get_sample(sample_id)
    if body.end_s <= body.start_s:
        raise HTTPException(400, "end_s must be after start_s")
    if sample.analysis_status != "done":
        raise HTTPException(409, "sample not analysed yet")
    root = quality = None
    if body.chord_symbol:
        c = parse_chord_symbol(body.chord_symbol)
        root, quality = PC_NAMES_SHARP[c.root], c.quality
    pos = len(sample.regions)
    region = state.repo.insert_region(
        sample_id,
        {"start_s": body.start_s, "end_s": body.end_s, "position": pos, "declared_length_beats": body.declared_length_beats},
        source="user",
    )
    region = services.remeasure_region(state, region, chord_root=root, chord_quality=quality)
    state.events.publish("region.updated", {"region_id": region.id, "sample_id": sample_id})
    return region


@router.get("/regions/{region_id}", response_model=Region)
def get_region(region_id: str, state=Depends(ctx)) -> Region:
    return state.repo.get_region(region_id)


@router.patch("/regions/{region_id}", response_model=Region)
def patch_region(region_id: str, body: RegionPatch, state=Depends(ctx)) -> Region:
    region = state.repo.get_region(region_id)
    _validate_chord_fields(body.chord_root, body.chord_quality, body.bass_note)
    fields: dict = {}
    bounds_changed = False
    if body.start_s is not None or body.end_s is not None:
        start = body.start_s if body.start_s is not None else region.start_s
        end = body.end_s if body.end_s is not None else region.end_s
        if end <= start:
            raise HTTPException(400, "end_s must be after start_s")
        fields.update(start_s=start, end_s=end)
        bounds_changed = True
    for k in ("chord_root", "chord_quality", "bass_note", "key_context"):
        v = getattr(body, k)
        if v is not None:
            fields[k] = v
    if body.clear_declared_length:
        fields["declared_length_beats"] = None
    elif body.declared_length_beats is not None:
        if body.declared_length_beats <= 0:
            raise HTTPException(400, "declared_length_beats must be positive")
        fields["declared_length_beats"] = body.declared_length_beats
    if body.user_overrides is not None:
        fields["user_overrides"] = body.user_overrides
    # A region the user has shaped is theirs: it survives reanalysis whole
    # (Repo.replace_analysis_regions) instead of being re-cut by the pipeline.
    if fields and region.source == "analysis":
        fields["source"] = "user"
    region = state.repo.update_region(region_id, **fields)
    if bounds_changed:
        region = services.remeasure_region(
            state, region, chord_root=fields.get("chord_root", region.chord_root),
            chord_quality=fields.get("chord_quality", region.chord_quality),
        )
    state.events.publish("region.updated", {"region_id": region.id, "sample_id": region.sample_id})
    services.invalidate_region(state, region.id)
    return region


@router.delete("/regions/{region_id}", status_code=204)
def delete_region(region_id: str, state=Depends(ctx)) -> Response:
    region = state.repo.get_region(region_id)
    set_ids = state.repo.sets_containing(region_id)
    state.repo.delete_region(region_id)
    state.events.publish("region.updated", {"region_id": region_id, "sample_id": region.sample_id, "removed": True})
    for sid in set_ids:
        services.invalidate_set(state, sid)
    return Response(status_code=204)


@router.get("/regions/{region_id}/audio")
def region_audio(region_id: str, state=Depends(ctx)) -> Response:
    import soundfile as sf

    region = state.repo.get_region(region_id)
    sample = state.repo.get_sample(region.sample_id)
    y, sr = load_audio(sample.path)
    seg = slice_seconds(y, sr, region.start_s, region.end_s)
    buf = io.BytesIO()
    sf.write(buf, seg.T, sr, format="WAV", subtype="FLOAT")
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"Content-Disposition": f'inline; filename="{region_id}.wav"'})
