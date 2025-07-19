from fastapi import APIRouter

from .health import router as health_router
from .process import router as process_router
from .models import router as models_router

router = APIRouter()

router.include_router(health_router, prefix="/health", tags=["health"])
router.include_router(process_router, prefix="/process", tags=["processing"])
router.include_router(models_router, prefix="/models", tags=["models"]) 