from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.constants import http_status
from app.middleware.auth import authorization_middleware
from app.router import router as api_router
from app.runtime.warmup import (
    validate_required_service_config,
    warmup_resident_runtimes,
)
from app.utils.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging("INFO")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        validate_required_service_config(settings)
        warmup_resident_runtimes(settings)
        yield

    app = FastAPI(
        title="glamify-image-ai",
        debug=True,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.middleware("http")(authorization_middleware)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=http_status.UNPROCESSABLE_CONTENT,
            content={
                "status": http_status.UNPROCESSABLE_CONTENT,
                "message": "Invalid request.",
                "data": None,
            },
        )

    app.include_router(api_router)

    return app


app = create_app()
