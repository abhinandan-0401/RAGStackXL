"""
Application settings.
"""
import os
from typing import Dict, Any, Optional
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


class FaissSettings(BaseSettings):
    """FAISS vector database settings."""
    # No special settings for FAISS
    pass


class QdrantSettings(BaseSettings):
    """Qdrant vector database settings."""
    URL: Optional[str] = None
    API_KEY: Optional[str] = None
    PREFER_GRPC: bool = True


class WeaviateSettings(BaseSettings):
    """Weaviate vector database settings."""
    URL: str = "http://localhost:8080"
    API_KEY: Optional[str] = None


class PineconeSettings(BaseSettings):
    """Pinecone vector database settings."""
    API_KEY: Optional[str] = None
    ENVIRONMENT: str = "us-west1-gcp"


class MilvusSettings(BaseSettings):
    """Milvus vector database settings."""
    URI: str = "http://localhost:19530"
    TOKEN: Optional[str] = None


class VectorDbSettings(BaseSettings):
    """Vector database settings."""
    PROVIDER: str = "faiss"  # Default provider
    COLLECTION_NAME: str = "documents"
    EMBEDDING_DIMENSION: int = 1536  # Default for OpenAI embeddings
    DISTANCE_METRIC: str = "cosine"
    PERSIST_DIRECTORY: str = "vectorstore"
    
    # Provider-specific settings
    FAISS: FaissSettings = FaissSettings()
    QDRANT: QdrantSettings = QdrantSettings()
    WEAVIATE: WeaviateSettings = WeaviateSettings()
    PINECONE: PineconeSettings = PineconeSettings()
    MILVUS: MilvusSettings = MilvusSettings()


class OpenAIEmbeddingSettings(BaseSettings):
    """OpenAI embedding settings."""
    API_KEY: Optional[str] = None
    API_BASE: Optional[str] = None
    API_VERSION: Optional[str] = None
    TIMEOUT: int = 60


class SBERTEmbeddingSettings(BaseSettings):
    """SBERT embedding settings."""
    DEVICE: str = "cpu"
    BATCH_SIZE: int = 32
    SHOW_PROGRESS_BAR: bool = False


class HuggingFaceEmbeddingSettings(BaseSettings):
    """HuggingFace embedding settings."""
    TOKEN: Optional[str] = None
    REVISION: str = "main"
    DEVICE: str = "cpu"


class CohereEmbeddingSettings(BaseSettings):
    """Cohere embedding settings."""
    API_KEY: Optional[str] = None
    TIMEOUT: int = 60


class FastEmbedEmbeddingSettings(BaseSettings):
    """FastEmbed embedding settings."""
    CACHE_DIR: Optional[str] = None
    THREADS: int = 4
    BATCH_SIZE: int = 32


class EmbeddingSettings(BaseSettings):
    """Embedding settings."""
    MODEL_TYPE: str = "fastembed"  # Default to fastembed as it requires no API keys
    MODEL_NAME: str = "default"  # Model name (specific to each provider)
    DIMENSION: int = 768  # Default embedding dimension
    NORMALIZE: bool = True  # Whether to normalize embedding vectors
    
    # Provider-specific settings
    OPENAI: OpenAIEmbeddingSettings = OpenAIEmbeddingSettings()
    SBERT: SBERTEmbeddingSettings = SBERTEmbeddingSettings()
    HUGGINGFACE: HuggingFaceEmbeddingSettings = HuggingFaceEmbeddingSettings()
    COHERE: CohereEmbeddingSettings = CohereEmbeddingSettings()
    FASTEMBED: FastEmbedEmbeddingSettings = FastEmbedEmbeddingSettings()


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
    EMBEDDING: EmbeddingSettings = EmbeddingSettings()
    
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