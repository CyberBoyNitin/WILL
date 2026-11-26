"""FastAPI application factory."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import AppConfig, load_config
from backend.server.routes import (
    benchmarks,
    checkpoints,
    datasets,
    experiments,
    memory,
    openai as openai_routes,
    playground,
    system,
    training,
)
from backend.server.services import Services


def create_app(
    cfg: AppConfig | None = None,
    services: Services | None = None,
) -> FastAPI:
    cfg = cfg or load_config()
    services = services or Services.create(
        cfg, max_workers=int(os.environ.get("WILL_MAX_WORKERS", "1"))
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        services.shutdown()

    app = FastAPI(
        title="WILL Research API",
        version="0.3.0",
        description=(
            "Local research API for the WILL adaptive-recurrent model. "
            "All metrics come from real model execution."
        ),
        lifespan=lifespan,
    )
    app.state.services = services

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def api_key_guard(request: Request, call_next):
        api_key = cfg.server.api_key
        path = request.url.path
        if api_key and path.startswith(("/api", "/v1")) and path != "/api/health":
            supplied = request.headers.get("x-api-key")
            if not supplied:
                auth = request.headers.get("authorization", "")
                if auth.lower().startswith("bearer "):
                    supplied = auth[7:]
            if not supplied and path.endswith("/events"):
                supplied = request.query_params.get("api_key")
            if supplied != api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "invalid or missing API key"},
                )
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def key_error_handler(_request: Request, exc: KeyError):
        detail = exc.args[0] if exc.args else "not found"
        return JSONResponse(status_code=404, content={"detail": str(detail)})

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_request: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(PermissionError)
    async def permission_handler(_request: Request, exc: PermissionError):
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    app.include_router(system.router)
    app.include_router(checkpoints.router)
    app.include_router(training.router)
    app.include_router(benchmarks.router)
    app.include_router(memory.router)
    app.include_router(datasets.router)
    app.include_router(playground.router)
    app.include_router(experiments.router)
    app.include_router(openai_routes.router)

    @app.get("/")
    def root() -> dict:
        return {
            "name": "WILL",
            "version": "0.3.0",
            "docs": "/docs",
            "health": "/api/health",
            "openai_compatible": "/v1/chat/completions",
            "telemetry": "/v1/telemetry",
            "description": "Adaptive recurrent computation with session-local fast weights.",
        }

    return app
