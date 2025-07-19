from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Application
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    ENVIRONMENT: str = "development"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Azure
    AZURE_STORAGE_ACCOUNT_NAME: str = ""
    AZURE_CLIENT_ID: str = ""
    AZURE_CLIENT_SECRET: str = ""
    AZURE_TENANT_ID: str = ""
    AZURE_SUBSCRIPTION_ID: str = ""
    
    # Functions
    FUNCTIONS_BASE_URL: str = "http://localhost:7071"
    
    # ML Configuration
    MODEL_CONTAINER_NAME: str = "models"
    DEFAULT_MODEL_TYPE: str = "u2net"
    MAX_IMAGE_SIZE_MB: int = 10
    PROCESSING_TIMEOUT_SECONDS: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 