from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_available_models():
    """
    Get list of available AI models and their specifications.
    """
    models = [
        {
            "id": "u2net",
            "name": "U2-Net",
            "description": "Balanced accuracy and speed for general use",
            "input_size": [320, 320],
            "processing_time_avg": 8.5,
            "accuracy_score": 0.92,
            "recommended_for": ["portraits", "objects", "general_use"]
        },
        {
            "id": "u2netp",
            "name": "U2-Net (Portable)",
            "description": "Faster processing with good accuracy",
            "input_size": [320, 320],
            "processing_time_avg": 5.2,
            "accuracy_score": 0.89,
            "recommended_for": ["mobile", "quick_processing", "batch_jobs"]
        },
        {
            "id": "isnet",
            "name": "IS-Net",
            "description": "High accuracy for complex images",
            "input_size": [1024, 1024],
            "processing_time_avg": 15.2,
            "accuracy_score": 0.96,
            "recommended_for": ["complex_scenes", "fine_details", "professional_use"]
        }
    ]
    
    return {
        "models": models,
        "default_model": "u2net"
    } 