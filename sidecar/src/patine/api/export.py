"""Export (handoff §8.4) runs as a job; the result lists the files written."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..models import ExportRequest, Job
from .deps import ctx

router = APIRouter(tags=["export"])


@router.post("/export", response_model=Job)
def export(body: ExportRequest, state=Depends(ctx)) -> Job:
    prog = state.repo.get_progression(body.progression_id)
    if not any(s.region_id for s in prog.slots) and body.kind != "midi":
        raise HTTPException(409, "no slot has a region assigned")
    if not body.out_dir:
        raise HTTPException(400, "out_dir required")
    return state.worker.submit("export", body.model_dump(), priority=3)
