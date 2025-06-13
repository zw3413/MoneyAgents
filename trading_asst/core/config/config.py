from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import os
from pathlib import Path

# Get the project root directory (where .env file is located)
ROOT_DIR = Path(__file__).parent.parent.parent.parent

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    
    # Service Configuration
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database Configuration
    FIRESTORE_COLLECTION: str = "trading_analysis"
    GCS_BUCKET_NAME: Optional[str] = None
    
    # Model Configuration
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Agent Configuration
    # FUNDAMENTAL_AGENT_MODEL: str = "gpt-4"
    # TECHNICAL_AGENT_MODEL: str = "gpt-4"
    # NEWS_AGENT_MODEL: str = "claude-v1"
    # SENTIMENT_AGENT_MODEL: str = "text-bison@001"

    FUNDAMENTAL_AGENT_MODEL: str = "gpt-4o-mini"
    TECHNICAL_AGENT_MODEL: str = "gpt-4o-mini"
    #NEWS_AGENT_MODEL: str = "claude-v1"
    NEWS_AGENT_MODEL: str = "gpt-4o-mini"
    #SENTIMENT_AGENT_MODEL: str = "text-bison@001"
    SENTIMENT_AGENT_MODEL: str = "gpt-4o-mini"
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        env_file = ROOT_DIR / ".env"
        env_file_encoding = None  # This makes .env file optional
        validate_default = True  # Still validate the values we do get

@lru_cache()
def get_settings() -> Settings:
    return Settings() 