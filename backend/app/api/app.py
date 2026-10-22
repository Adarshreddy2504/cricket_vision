"""Build the FastAPI application.

The app is created by a factory so tests can point it at a temporary data root
and inject a deterministic pipeline. The production module-level app lives in
`backend/main.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .jobs import JobRegistry, cleanup_stale_temp
from .routes import analysis_router, media_router, system_router

log = logging.getLogger(__name__)


def _default_pipeline_factory(settings: Settings):
    from ..pipeline.analysis import AnalysisPipeline

    return AnalysisPipeline(settings=settings)


def create_app(
    *,
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings], object] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = pipeline_factory or _default_pipeline_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_dirs()
        cleanup_stale_temp(settings)
        pipeline = factory(settings)
        registry = JobRegistry(settings, pipeline)
        registry.load_existing()
        app.state.settings = settings
        app.state.pipeline_factory = factory
        app.state.pipeline = pipeline
        app.state.registry = registry
        yield

    app = FastAPI(
        title="CricketVision API",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
    )
    app.include_router(system_router, prefix="/api")
    app.include_router(analysis_router, prefix="/api")
    app.include_router(media_router, prefix="/api")
    return app


__all__ = ["create_app"]
