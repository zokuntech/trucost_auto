from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TruCost Auto"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]  # Change in production
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "jpg", "jpeg", "png", "txt"]
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini" # Or another model like gpt-4-turbo
    
    # Search API Key (Placeholder - Needed for real parts research URL finding)
    # e.g., for Google Custom Search, Bing Web Search, SerpApi, Tavily AI, etc.
    SEARCH_API_KEY: Optional[str] = None
    
    # Google Custom Search API (for Parts Research URL finding)
    GOOGLE_CSE_ID: Optional[str] = None # Your Programmable Search Engine ID (CX)
    
    # Celery / Redis Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings() 