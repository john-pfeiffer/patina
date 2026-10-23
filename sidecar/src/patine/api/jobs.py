from __future__ import annotations

from fastapi import APIRouter, Depends

from ..models import Job
from .deps import ctx

router = APIRouter(tags=["jobs"])


@router.get("/jobs", response_model=list[Job])
def list_jobs(status: str | None = None, state=Depends(ctx)) -> list[Job]:
    return state.worker.list(status)


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str, state=Depends(ctx)) -> Job:
    return state.worker.get(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=Job)
def cancel_job(job_id: str, state=Depends(ctx)) -> Job:
    return state.worker.cancel(job_id)
