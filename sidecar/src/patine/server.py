"""FastAPI application factory. Token middleware, CORS for the Tauri webview
origins and the Vite dev server, SSE events, render file serving."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import ANALYSIS_VERSION, RENDER_VERSION, __version__
from .db import Database
from .events import EventBus
from .jobs import JobNotFound, JobWorker
from .models import SidecarInfo
from .paths import DataPaths, default_data_dir
from .repo import NotFound, Repo
from .theory import TheoryError

log = logging.getLogger(__name__)

TOKEN_HEADER = "x-patine-token"
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class AppState:
    """Everything the routers need, attached to `app.state.ctx`."""

    def __init__(self, paths: DataPaths, token: str, workers: int = 2):
        self.paths = paths.ensure()
        self.token = token
        self.db = Database(paths.db)
        self.repo = Repo(self.db)
        self.events = EventBus()
        self.worker = JobWorker(self.db, self.events, threads=workers)

    def close(self) -> None:
        self.worker.stop()
        self.db.close_all()

    def ml_available(self) -> bool:
        try:
            import torch  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True


def create_app(state: AppState | None = None) -> FastAPI:
    if state is None:
        data_dir = Path(os.environ.get("PATINE_DATA_DIR") or default_data_dir())
        token = os.environ.get("PATINE_TOKEN", "dev")
        workers = int(os.environ.get("PATINE_WORKERS", "2"))
        state = AppState(DataPaths(data_dir), token, workers)

    from .api import export, jobs, library, progressions, renders, services, sets, theory
    from .api.services import register_job_handlers

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        register_job_handlers(state)
        state.worker.start()
        services.requeue_stale_analyses(state)
        log.info("Patine sidecar %s serving data dir %s", __version__, state.paths.root)
        try:
            yield
        finally:
            state.close()

    app = FastAPI(title="Patine sidecar", version=__version__, lifespan=lifespan)
    app.state.ctx = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # bound to 127.0.0.1 and token-gated; origins are the webview and Vite
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def require_token(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        if request.method == "OPTIONS" or path in PUBLIC_PATHS:
            return await call_next(request)
        supplied = request.headers.get(TOKEN_HEADER) or request.query_params.get("token")
        if supplied != state.token:
            return JSONResponse({"detail": "missing or invalid X-Patine-Token"}, status_code=401)
        return await call_next(request)

    @app.exception_handler(NotFound)
    @app.exception_handler(JobNotFound)
    async def _not_found(_: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"detail": f"not found: {exc.args[0] if exc.args else ''}"}, status_code=404)

    @app.exception_handler(TheoryError)
    async def _theory_error(_: Request, exc: TheoryError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/health", response_model=SidecarInfo)
    async def health() -> SidecarInfo:
        return SidecarInfo(
            version=__version__,
            analysis_version=ANALYSIS_VERSION,
            render_version=RENDER_VERSION,
            data_dir=str(state.paths.root),
            ml_available=state.ml_available(),
        )

    @app.get("/events")
    async def events(request: Request) -> StreamingResponse:
        q = state.events.subscribe()

        async def gen() -> AsyncIterator[str]:
            try:
                yield "event: hello\ndata: {}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.to_thread(q.get, True, 15.0)
                        yield ev.encode()
                    except queue.Empty:
                        yield "event: ping\ndata: {}\n\n"
            finally:
                state.events.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.include_router(library.router)
    app.include_router(sets.router)
    app.include_router(progressions.router)
    app.include_router(theory.router)
    app.include_router(renders.router)
    app.include_router(export.router)
    app.include_router(jobs.router)
    return app
