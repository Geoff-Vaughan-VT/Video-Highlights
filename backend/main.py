from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from .config import settings
from .database import init_db, session_scope
from .logging_utils import configure_runtime_logging
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from .routers import (
    admin_global,
    admin_tenant,
    agent,
    auth,
    events,
    feedback,
    health,
    jobs,
    matches,
    roster,
    sharing,
    studio,
    training,
)
from .services.job_runner import recover_interrupted_inline_jobs
from .tenant import ensure_seed_tenants
from .utils import generate_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with session_scope() as session:
        ensure_seed_tenants(session)
        recover_interrupted_inline_jobs(session)
    yield


def create_app() -> FastAPI:
    configure_runtime_logging()
    app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = generate_id("req")
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):  # type: ignore[no-untyped-def]
        details: List[Dict[str, Any]] = []
        for item in exc.errors():
            loc = ".".join(str(part) for part in item.get("loc", []))
            details.append({"field": loc, "issue": item.get("msg", "invalid")})

        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):  # type: ignore[no-untyped-def]
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        details = [] if isinstance(exc.detail, str) else [{"issue": str(exc.detail)}]
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": message,
                    "details": details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):  # type: ignore[no-untyped-def]
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": [{"issue": str(exc)}],
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    app.include_router(health.router, prefix="/v1")
    app.include_router(auth.router, prefix="/v1")
    app.include_router(admin_global.router, prefix="/v1")
    app.include_router(admin_tenant.router, prefix="/v1")
    # roster must register before matches: its static /matches/roster-template.csv
    # route would otherwise be shadowed by matches' /matches/{match_id}.
    app.include_router(roster.router, prefix="/v1")
    app.include_router(sharing.router, prefix="/v1")
    app.include_router(matches.router, prefix="/v1")
    app.include_router(jobs.router, prefix="/v1")
    app.include_router(events.router, prefix="/v1")
    app.include_router(feedback.router, prefix="/v1")
    app.include_router(training.router, prefix="/v1")
    app.include_router(agent.router, prefix="/v1")
    app.include_router(studio.router, prefix="/v1")

    # Built-in web UI: a self-contained SPA served at "/" (no separate
    # frontend server, no build step).
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="studio-ui")

    return app


app = create_app()
