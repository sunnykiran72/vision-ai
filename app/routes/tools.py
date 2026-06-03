from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/tools/api-console", include_in_schema=False)
async def api_console() -> FileResponse:
    console_path = Path(__file__).resolve().parents[2] / "tools" / "api-console.html"
    return FileResponse(console_path)
