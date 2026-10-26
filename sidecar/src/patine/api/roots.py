"""Library roots: folders the library scans recursively (a Splice library, a
pack folder). Adding a root scans it; scans index new files, re-queue changed
ones and mark vanished ones `missing`. Analysis of what a scan finds runs in
the background at low priority and can be paused from the settings."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from ..models import Job, LibraryRoot, LibraryRootCreate
from . import services
from .deps import ctx

router = APIRouter(tags=["roots"])


@router.get("/roots", response_model=list[LibraryRoot])
def list_roots(state=Depends(ctx)) -> list[LibraryRoot]:
    return state.repo.list_roots()


@router.post("/roots")
def add_root(body: LibraryRootCreate, state=Depends(ctx)) -> dict:
    try:
        root, job = services.add_root(state, body.path, body.name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"root": root.model_dump(), "job": job.model_dump()}


@router.get("/roots/{root_id}", response_model=LibraryRoot)
def get_root(root_id: str, state=Depends(ctx)) -> LibraryRoot:
    return state.repo.get_root(root_id)


@router.post("/roots/scan", response_model=list[Job])
def scan_all(state=Depends(ctx)) -> list[Job]:
    return services.scan_all_roots(state, priority=4)


@router.post("/roots/{root_id}/scan", response_model=Job)
def scan_root(root_id: str, state=Depends(ctx)) -> Job:
    state.repo.get_root(root_id)
    return services.queue_scan(state, root_id)


@router.delete("/roots/{root_id}", status_code=204)
def delete_root(
    root_id: str,
    delete_samples: bool = Query(False, description="also remove the samples found under this root"),
    state=Depends(ctx),
) -> Response:
    state.repo.get_root(root_id)
    affected_sets: set[str] = set()
    if delete_samples:
        for s in state.repo.samples_for_root(root_id):
            full = state.repo.get_sample(s.id)
            affected_sets |= {sid for r in full.regions for sid in state.repo.sets_containing(r.id)}
    removed = state.repo.delete_root(root_id, delete_samples=delete_samples)
    for sid in removed:
        state.events.publish("sample.removed", {"sample_id": sid})
    for set_id in affected_sets:
        services.invalidate_set(state, set_id)
    state.events.publish("root.updated", {"root_id": root_id, "status": "removed", "removed_samples": len(removed)})
    return Response(status_code=204)
