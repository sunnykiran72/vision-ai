from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.middleware.auth import authorization_middleware
from app.router import router as api_router
from app.runtime.warmup import warmup_resident_runtimes
from app.utils.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging("INFO")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if settings.startup_warmup_enabled:
            warmup_resident_runtimes(settings)
        yield

    app = FastAPI(
        title="glamify-vision-ai",
        debug=True,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.middleware("http")(authorization_middleware)
    app.include_router(api_router)

    return app


app = create_app()
