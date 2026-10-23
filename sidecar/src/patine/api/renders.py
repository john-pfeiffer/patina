"""Serve ready renders from the cache."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from .deps import ctx

router = APIRouter(tags=["renders"])


@router.get("/renders/{cache_key}.wav")
def render_file(cache_key: str, state=Depends(ctx)) -> FileResponse:
    path = state.repo.render_path(cache_key)
    if path is None or not Path(path).exists():
        raise HTTPException(404, "render not ready")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "max-age=31536000, immutable"})
