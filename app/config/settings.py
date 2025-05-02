"""
Configuration management for RAGStackXL.
Uses Pydantic for settings validation and .env file for environment variables.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class LoggingSettings(BaseSettings):
    """Logging configuration."""
    LEVEL: str = Field("INFO", description="Logging level")
    FORMAT: str = Field(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Logging format"
    )
    ROTATION: str = Field("20 MB", description="Log rotation size")
    RETENTION: str = Field("7 days", description="Log retention period")
    JSON: bool = Field(False, description="Whether to output logs in JSON format")

class OpenAISettings(BaseSettings):
    """OpenAI API configuration."""
    API_KEY: Optional[str] = Field(None, description="OpenAI API key")
    MODEL: str = Field("gpt-3.5-turbo", description="Default OpenAI model")
    EMBEDDING_MODEL: str = Field("text-embedding-ada-002", description="OpenAI embedding model")
    MAX_TOKENS: int = Field(1000, description="Maximum tokens for completion")
    TEMPERATURE: float = Field(0.0, description="Temperature for generation")

class AnthropicSettings(BaseSettings):
    """Anthropic API configuration."""
    API_KEY: Optional[str] = Field(None, description="Anthropic API key")
    MODEL: str = Field("claude-2", description="Default Anthropic model")
    MAX_TOKENS: int = Field(1000, description="Maximum tokens for completion")
    TEMPERATURE: float = Field(0.0, description="Temperature for generation")

class VectorDBSettings(BaseSettings):
    """Vector database configuration."""
    PROVIDER: str = Field("chroma", description="Vector DB provider (chroma, qdrant, etc.)")
    CONNECTION_STRING: Optional[str] = Field(None, description="Connection string for the vector database")
    COLLECTION_NAME: str = Field("ragstackxl", description="Default collection name")
    PERSIST_DIRECTORY: str = Field("./data/vectordb", description="Directory to persist vector database")

class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    PROVIDER: str = Field("openai", description="Embedding provider (openai, sentence-transformers, etc.)")
    MODEL_NAME: str = Field("text-embedding-ada-002", description="Embedding model name")
    EMBEDDING_DIMENSION: int = Field(1536, description="Dimension of embeddings")
    BATCH_SIZE: int = Field(32, description="Batch size for embedding generation")

class DocumentSettings(BaseSettings):
    """Document processing configuration."""
    CHUNK_SIZE: int = Field(1000, description="Default chunk size for text splitting")
    CHUNK_OVERLAP: int = Field(200, description="Default chunk overlap for text splitting")
    MAX_DOCS_PER_QUERY: int = Field(5, description="Maximum number of documents to retrieve per query")
    DATA_DIR: str = Field("./data/documents", description="Directory for document storage")

class AgentSettings(BaseSettings):
    """Agent system configuration."""
    MAX_ITERATIONS: int = Field(10, description="Maximum iterations for agent execution")
    MEMORY_SIZE: int = Field(10, description="Size of conversation memory in messages")
    TOOLS_ENABLED: List[str] = Field(["web_search", "calculator"], description="Enabled agent tools")
    VERBOSE: bool = Field(False, description="Whether to log detailed agent steps")

class APISettings(BaseSettings):
    """API configuration."""
    HOST: str = Field("0.0.0.0", description="API host")
    PORT: int = Field(8000, description="API port")
    WORKERS: int = Field(4, description="Number of API workers")
    DEBUG: bool = Field(False, description="Enable debug mode")
    CORS_ORIGINS: List[str] = Field(["*"], description="Allowed CORS origins")

class Settings(BaseSettings):
    """Main settings class for RAGStackXL."""
    # General settings
    PROJECT_NAME: str = Field("RAGStackXL", description="Project name")
    VERSION: str = Field("0.1.0", description="Project version")
    ENVIRONMENT: str = Field("development", description="Environment (development, staging, production)")
    BASE_DIR: str = Field(str(Path(__file__).resolve().parent.parent.parent), description="Base directory")
    
    # Component settings
    LOGGING: LoggingSettings = Field(default_factory=LoggingSettings, description="Logging settings")
    OPENAI: OpenAISettings = Field(default_factory=OpenAISettings, description="OpenAI settings")
    ANTHROPIC: AnthropicSettings = Field(default_factory=AnthropicSettings, description="Anthropic settings")
    VECTORDB: VectorDBSettings = Field(default_factory=VectorDBSettings, description="Vector database settings")
    EMBEDDING: EmbeddingSettings = Field(default_factory=EmbeddingSettings, description="Embedding settings")
    DOCUMENT: DocumentSettings = Field(default_factory=DocumentSettings, description="Document processing settings")
    AGENT: AgentSettings = Field(default_factory=AgentSettings, description="Agent settings")
    API: APISettings = Field(default_factory=APISettings, description="API settings")
    
    # For loading .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
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

# Create a global settings instance
settings = Settings()
settings.create_directories() 