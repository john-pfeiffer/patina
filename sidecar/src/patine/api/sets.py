"""Set endpoints (handoff §6–7): anchor, members, settings, renders, deltas."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from ..models import MemberAdd, RenderStatus, Set, SetCreate, SetPatch
from . import services
from .deps import ctx

router = APIRouter(tags=["sets"])


@router.post("/sets", response_model=Set)
def create_set(body: SetCreate, state=Depends(ctx)) -> Set:
    return state.repo.insert_set(body.name)


@router.get("/sets", response_model=list[Set])
def list_sets(state=Depends(ctx)) -> list[Set]:
    return state.repo.list_sets()


@router.get("/sets/{set_id}", response_model=Set)
def get_set(set_id: str, state=Depends(ctx)) -> Set:
    return state.repo.get_set(set_id)


@router.patch("/sets/{set_id}", response_model=Set)
def patch_set(set_id: str, body: SetPatch, state=Depends(ctx)) -> Set:
    set_ = state.repo.get_set(set_id)
    fields: dict = {}
    if body.name is not None:
        fields["name"] = body.name
    if body.anchor_region_id is not None:
        if body.anchor_region_id not in {m.region.id for m in set_.members}:
            raise HTTPException(400, "anchor must be a member of the set")
        fields["anchor_region_id"] = body.anchor_region_id
    if body.bpm_pinned is not None:
        fields["bpm_pinned"] = body.bpm_pinned
    if body.session_bpm is not None:
        if body.session_bpm <= 0:
            raise HTTPException(400, "session_bpm must be positive")
        fields["session_bpm"] = body.session_bpm
        if body.bpm_pinned is None:
            fields["bpm_pinned"] = True   # typing a bpm pins it (§7)
    if body.tonal_settings is not None:
        fields["tonal_settings"] = body.tonal_settings
    if body.temporal_settings is not None:
        fields["temporal_settings"] = body.temporal_settings
    set_ = state.repo.update_set(set_id, **fields)
    if not set_.bpm_pinned:
        services.effective_session_bpm(state, set_)
        set_ = state.repo.get_set(set_id)
    services.invalidate_set(state, set_id)
    return set_


@router.delete("/sets/{set_id}", status_code=204)
def delete_set(set_id: str, state=Depends(ctx)) -> Response:
    state.repo.get_set(set_id)
    state.repo.delete_set(set_id)
    return Response(status_code=204)


@router.post("/sets/{set_id}/members", response_model=Set)
def add_member(set_id: str, body: MemberAdd, state=Depends(ctx)) -> Set:
    set_ = state.repo.get_set(set_id)
    state.repo.get_region(body.region_id)
    set_ = state.repo.add_member(set_id, body.region_id)
    if set_.anchor_region_id is None:
        set_ = state.repo.update_set(set_id, anchor_region_id=body.region_id)   # §6 default anchor
    if not set_.bpm_pinned:
        services.effective_session_bpm(state, set_)
        set_ = state.repo.get_set(set_id)
    services.invalidate_set(state, set_id)
    return set_


@router.delete("/sets/{set_id}/members/{region_id}", response_model=Set)
def remove_member(set_id: str, region_id: str, state=Depends(ctx)) -> Set:
    set_ = state.repo.get_set(set_id)
    set_ = state.repo.remove_member(set_id, region_id)
    if set_.anchor_region_id == region_id:
        new_anchor = set_.members[0].region.id if set_.members else None
        set_ = state.repo.update_set(set_id, anchor_region_id=new_anchor)
    services.invalidate_set(state, set_id)
    return set_


@router.get("/sets/{set_id}/deltas")
def deltas(set_id: str, state=Depends(ctx)) -> dict:
    return services.set_deltas(state, state.repo.get_set(set_id))


@router.get("/sets/{set_id}/renders", response_model=list[RenderStatus])
def renders(set_id: str, state=Depends(ctx)) -> list[RenderStatus]:
    return services.set_renders(state, state.repo.get_set(set_id))


class PreviewRequest(BaseModel):
    region_id: str
    transpose_semitones: int = 0


@router.post("/sets/{set_id}/renders/preview", response_model=RenderStatus)
def preview(set_id: str, body: PreviewRequest, state=Depends(ctx)) -> RenderStatus:
    set_ = state.repo.get_set(set_id)
    region = state.repo.get_region(body.region_id)
    return services.ensure_render(state, set_, region, body.transpose_semitones, priority=2)
