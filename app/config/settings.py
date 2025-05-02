"""
Application settings.
"""
import os
from typing import Dict, Any
from pydantic_settings import BaseSettings
from dataclasses import dataclass
from pydantic import field_validator


class DocumentSettings(BaseSettings):
    """Document settings."""
    DATA_DIR: str = "data"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200


class ApiSettings(BaseSettings):
    """API settings."""
    HOST: str = "127.0.0.1"
    PORT: int = 8000


class VectorDbSettings(BaseSettings):
    """Vector database settings."""
    PERSIST_DIRECTORY: str = "vectorstore"


class Settings(BaseSettings):
    """Application settings."""
    PROJECT_NAME: str = "RAGStackXL"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    BASE_DIR: str = "."
    
    # Component settings
    DOCUMENT: DocumentSettings = DocumentSettings()
    API: ApiSettings = ApiSettings()
    VECTORDB: VectorDbSettings = VectorDbSettings()
    
    @field_validator("ENVIRONMENT")
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v.lower()
    
    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.DOCUMENT.DATA_DIR, exist_ok=True)
        os.makedirs(self.VECTORDB.PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(os.path.join(self.BASE_DIR, "logs"), exist_ok=True)


# Create global settings instance
settings = Settings() 