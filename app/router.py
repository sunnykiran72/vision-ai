from fastapi import APIRouter

from app.routes.health import router as health_router
from app.routes.tools import router as tools_router
from app.routes.tryon import router as tryon_router
from app.routes.upscale import router as upscale_router
from app.routes.user_validation import router as user_validation_router
from app.routes.wardrobe import router as wardrobe_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(tools_router, tags=["tools"])
router.include_router(wardrobe_router, tags=["wardrobe"])
router.include_router(user_validation_router, tags=["user-validation"])
router.include_router(tryon_router, tags=["tryon"])
router.include_router(upscale_router, tags=["upscale"])
