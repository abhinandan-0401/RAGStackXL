"""
Embedding interfaces for RAGStackXL.

This module defines the interfaces and abstractions for embedding models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Union
import asyncio

from app.core.interfaces import RagDocument
from app.utils.logging import log


class EmbeddingModelType(str, Enum):
    """Types of embedding models supported by the system."""
    OPENAI = "openai"
    SBERT = "sbert"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    FASTEMBED = "fastembed"
    CUSTOM = "custom"


class EmbeddingDimension(Enum):
    """Common embedding dimensions for popular models."""
    OPENAI_ADA_002 = 1536
    OPENAI_TEXT_EMBED_3_SMALL = 1536
    OPENAI_TEXT_EMBED_3_LARGE = 3072
    BERT_BASE = 768
    BERT_LARGE = 1024
    SBERT_MPNET_BASE = 768
    COHERE_ENGLISH = 1024
    COHERE_MULTILINGUAL = 768
    FASTTEXT = 300
    FASTEMBED_SMALL = 384
    FASTEMBED_BASE = 768
    FASTEMBED_LARGE = 1024


class EmbeddingConfig:
    """Configuration for embedding models."""
    
    def __init__(
        self,
        model_name: str,
        model_type: Union[EmbeddingModelType, str],
        dimension: int,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize embedding configuration.
        
        Args:
            model_name: Name of the embedding model
            model_type: Type of the embedding model
            dimension: Dimension of the embedding vectors
            normalize: Whether to normalize embedding vectors
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.model_type = model_type if isinstance(model_type, EmbeddingModelType) else EmbeddingModelType(model_type)
        self.dimension = dimension
        self.normalize = normalize
        self.additional_config = kwargs
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_config.get(key, default)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding model.
        
        Args:
            config: Configuration for the embedding model
        """
        self.config = config
        log.info(f"Initializing {self.__class__.__name__} with model: {config.model_name}")
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector for the query
        """
        pass
    
    @abstractmethod
    async def embed_documents(self, documents: List[Union[str, RagDocument]]) -> List[List[float]]:
        """
        Embed multiple documents or strings.
        
        Args:
            documents: List of documents or strings to embed
            
        Returns:
            List of embedding vectors for the documents
        """
        pass
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector
        """
        if not self.config.normalize:
            return vector
            
        import numpy as np
        norm = np.linalg.norm(vector)
        if norm > 0:
            return (np.array(vector) / norm).tolist()
        return vector
    
    def _validate_dimension(self, vector: List[float]) -> bool:
        """
        Validate that a vector has the expected dimension.
        
        Args:
            vector: Vector to validate
            
        Returns:
            True if the vector has the expected dimension, False otherwise
        """
        return len(vector) == self.config.dimension


class EmbeddingModelFactory:
    """Factory for creating embedding models."""
    
    _registry = {}
    
    @classmethod
    def register(cls, model_type: EmbeddingModelType, model_class):
        """
        Register an embedding model class for a model type.
        
        Args:
            model_type: Type of the embedding model
            model_class: Class implementing the embedding model
        """
        cls._registry[model_type] = model_class
    
    @classmethod
    def create(cls, config: EmbeddingConfig) -> EmbeddingModel:
        """
        Create an embedding model instance.
        
        Args:
            config: Configuration for the embedding model
            
        Returns:
            Embedding model instance
        """
        model_type = config.model_type
        if model_type not in cls._registry:
            raise ValueError(f"Embedding model type '{model_type.value}' is not registered")
            
        model_class = cls._registry[model_type]
        return model_class(config)
    
    @classmethod
    def get_types(cls) -> List[EmbeddingModelType]:
        """
        Get list of registered model types.
        
        Returns:
            List of registered model types
        """
        return list(cls._registry.keys()) 