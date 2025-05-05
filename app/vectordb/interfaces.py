from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import uuid

from app.core.interfaces import RagDocument, DocumentMetadata, EventType
from app.core.interfaces import event_bus
from app.utils.logging import log


class VectorDBProvider(str, Enum):
    """Vector database providers supported by the system."""
    CHROMA = "chroma"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    MILVUS = "milvus"
    FAISS = "faiss"  # Added for completeness, can be implemented later


class SearchResult:
    """Result from a vector search operation."""
    
    def __init__(
        self,
        document: RagDocument,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.document = document
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, doc_id={self.document.doc_id})"


class VectorDBConfig:
    """Configuration for vector databases."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_dimension: int = 1536,  # Default for OpenAI embeddings
        distance_metric: str = "cosine",
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        self.persist_directory = persist_directory
        self.additional_config = kwargs
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_config.get(key, default)


class VectorDB(ABC):
    """Abstract base class for vector database implementations."""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.collection_name = config.collection_name
        log.info(f"Initializing {self.__class__.__name__} with collection: {self.collection_name}")
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[RagDocument],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents to add
            embeddings: List of embeddings corresponding to the documents
            ids: Optional list of IDs for the documents
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[RagDocument]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform a similarity search.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    async def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[RagDocument, float]]:
        """
        Perform a similarity search with scores.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        pass
    
    @abstractmethod
    async def clear_collection(self) -> bool:
        """
        Clear the collection.
        
        Returns:
            True if collection was cleared, False otherwise
        """
        pass
    
    @abstractmethod
    async def persist(self) -> bool:
        """
        Persist the database to disk.
        
        Returns:
            True if database was persisted, False otherwise
        """
        pass
    
    def _generate_doc_id(self) -> str:
        """Generate a unique document ID."""
        return str(uuid.uuid4())
    
    def _validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """Validate embeddings dimensions."""
        if not embeddings:
            return False
        
        expected_dim = self.config.embedding_dimension
        for emb in embeddings:
            if len(emb) != expected_dim:
                log.error(f"Embedding dimension mismatch: expected {expected_dim}, got {len(emb)}")
                return False
        
        return True
    
    def _notify_documents_added(self, doc_ids: List[str]) -> None:
        """Notify that documents were added to the database."""
        event_bus.publish(
            event_type=EventType.DOCUMENT_ADDED,
            payload={"doc_ids": doc_ids, "collection_name": self.collection_name}
        )


class VectorDBFactory:
    """Factory for creating vector database instances."""
    
    _registry = {}
    
    @classmethod
    def register(cls, provider: VectorDBProvider, db_class):
        """Register a vector database class for a provider."""
        cls._registry[provider] = db_class
    
    @classmethod
    def create(cls, provider: VectorDBProvider, config: VectorDBConfig) -> VectorDB:
        """Create a vector database instance."""
        if provider not in cls._registry:
            raise ValueError(f"Vector database provider '{provider}' is not registered")
        
        db_class = cls._registry[provider]
        log.info(f"Creating {provider.value} vector database")
        return db_class(config)
    
    @classmethod
    def get_providers(cls) -> List[VectorDBProvider]:
        """Get list of registered providers."""
        return list(cls._registry.keys()) 