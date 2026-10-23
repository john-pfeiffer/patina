from __future__ import annotations

from fastapi import Request

from ..server import AppState


def ctx(request: Request) -> AppState:
    return request.app.state.ctx
