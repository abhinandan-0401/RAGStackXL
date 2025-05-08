"""
Retrieval interfaces for RAGStackXL.

This module defines the interfaces and abstractions for retrieval mechanisms.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Protocol, Callable, Tuple
import asyncio

from app.core.interfaces import RagDocument, DocumentMetadata, EventType, event_bus
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.utils.logging import log


class RetrieverType(str, Enum):
    """Types of retrievers supported by the system."""
    BASIC = "basic"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    MMR = "mmr"
    CONTEXTUAL = "contextual"
    QUERY_EXPANSION = "query_expansion"
    RERANKING = "reranking"
    CUSTOM = "custom"


class QueryReformulation(Protocol):
    """Protocol for query reformulation strategies."""
    
    async def reformulate(self, query: str) -> List[str]:
        """
        Reformulate a query into multiple query variations.
        
        Args:
            query: Original query string
            
        Returns:
            List of reformulated queries
        """
        pass


class Reranker(Protocol):
    """Protocol for reranking strategies."""
    
    async def rerank(
        self, 
        query: str,
        documents: List[RagDocument],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[RagDocument, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Query string
            documents: List of retrieved documents
            scores: Optional original relevance scores
            
        Returns:
            Reranked list of (document, score) tuples
        """
        pass


class RetrieverConfig:
    """Configuration for retrieval mechanisms."""
    
    def __init__(
        self,
        retriever_type: Union[RetrieverType, str],
        top_k: int = 4,
        filter_fn: Optional[Callable[[RagDocument], bool]] = None,
        use_metadata_filter: bool = True,
        include_metadata: bool = True,
        **kwargs
    ):
        """
        Initialize retriever configuration.
        
        Args:
            retriever_type: Type of retriever
            top_k: Number of documents to retrieve
            filter_fn: Optional function to filter documents
            use_metadata_filter: Whether to use metadata for filtering
            include_metadata: Whether to include metadata in results
            **kwargs: Additional retriever-specific parameters
        """
        self.retriever_type = retriever_type if isinstance(retriever_type, RetrieverType) else RetrieverType(retriever_type)
        self.top_k = top_k
        self.filter_fn = filter_fn
        self.use_metadata_filter = use_metadata_filter
        self.include_metadata = include_metadata
        self.additional_config = kwargs
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_config.get(key, default)


class Retriever(ABC):
    """Abstract base class for retrieval mechanisms."""
    
    def __init__(
        self,
        vectordb: VectorDB,
        embedding_model: EmbeddingModel,
        config: RetrieverConfig
    ):
        """
        Initialize the retriever.
        
        Args:
            vectordb: Vector database to retrieve from
            embedding_model: Embedding model for queries
            config: Retriever configuration
        """
        self.vectordb = vectordb
        self.embedding_model = embedding_model
        self.config = config
        log.info(f"Initializing {self.__class__.__name__} retriever")
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides config)
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        pass
    
    def _prepare_filter(
        self,
        filter: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare filter for retrieval.
        
        Args:
            filter: Optional metadata filter
            
        Returns:
            Prepared filter
        """
        if not self.config.use_metadata_filter:
            return None
        return filter
    
    def _publish_retrieval_event(
        self,
        query: str,
        results: List[SearchResult]
    ) -> None:
        """
        Publish event for retrieval completion.
        
        Args:
            query: Original query
            results: Retrieved results
        """
        event_bus.publish(
            event_type=EventType.QUERY_EXECUTED,
            payload={
                "query": query,
                "results_count": len(results),
                "retriever_type": self.config.retriever_type.value
            }
        )


class RetrieverFactory:
    """Factory for creating retrievers."""
    
    _registry = {}
    
    @classmethod
    def register(cls, retriever_type: RetrieverType, retriever_class):
        """
        Register a retriever class for a retriever type.
        
        Args:
            retriever_type: Type of the retriever
            retriever_class: Class implementing the retriever
        """
        cls._registry[retriever_type] = retriever_class
    
    @classmethod
    def create(
        cls,
        vectordb: VectorDB,
        embedding_model: EmbeddingModel,
        config: RetrieverConfig
    ) -> Retriever:
        """
        Create a retriever instance.
        
        Args:
            vectordb: Vector database to retrieve from
            embedding_model: Embedding model for queries
            config: Retriever configuration
            
        Returns:
            Retriever instance
        """
        retriever_type = config.retriever_type
        if retriever_type not in cls._registry:
            raise ValueError(f"Retriever type '{retriever_type.value}' is not registered")
            
        retriever_class = cls._registry[retriever_type]
        return retriever_class(vectordb, embedding_model, config)
    
    @classmethod
    def get_types(cls) -> List[RetrieverType]:
        """
        Get list of registered retriever types.
        
        Returns:
            List of registered retriever types
        """
        return list(cls._registry.keys()) 