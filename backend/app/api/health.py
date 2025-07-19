from fastapi import APIRouter
from datetime import datetime
import httpx

from app.core.config import settings

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Check API health status and dependencies.
    """
    # Check Functions service
    functions_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{settings.FUNCTIONS_BASE_URL}/api/health")
            functions_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        functions_status = "unhealthy"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "services": {
            "functions": functions_status,
            "storage": "healthy"  # TODO: Implement actual storage check
        }
    } 