from fastapi import FastAPI

from app.config import get_settings
from app.router import router as api_router
from app.utils.logging import configure_logging


def create_app() -> FastAPI:
    get_settings();
    configure_logging("INFO");
    app = FastAPI(
        title="Glamify AI",
        debug=True,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(api_router)
    return app


app = create_app()
