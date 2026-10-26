"""Sidecar settings (Preferences): analysis options and the analysis pause."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..models import Settings, SettingsPatch
from . import services
from .deps import ctx

router = APIRouter(tags=["settings"])


@router.get("/settings", response_model=Settings)
def get_settings(state=Depends(ctx)) -> Settings:
    return state.repo.get_settings()


@router.patch("/settings", response_model=Settings)
def patch_settings(body: SettingsPatch, state=Depends(ctx)) -> Settings:
    state.repo.set_settings(body.model_dump(exclude_none=True))
    current = services.apply_settings(state)
    state.events.publish("settings.updated", current.model_dump())
    return current
