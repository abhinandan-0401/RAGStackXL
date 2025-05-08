"""
Vector database module for RAGStackXL.

This module provides a flexible abstraction layer for various vector database backends.
Supported vector databases include:
- FAISS: Efficient local vector database for similarity search
- Qdrant: Balanced performance and ease of use
- Weaviate: GraphQL support with hybrid search capabilities
- Pinecone: Managed service with real-time features
- Milvus: High-scale solution with GPU acceleration

Usage:
    from app.vectordb import create_vectordb, VectorDBProvider
    
    # Create a vector database using default settings
    vector_db = create_vectordb()
    
    # Or specify a different provider
    vector_db = create_vectordb(provider="qdrant")
    
    # With custom configuration
    vector_db = create_vectordb(
        provider="faiss",
        collection_name="my_documents",
        embedding_dimension=1536,
        persist_directory="./vector_data"
    )
    
    # Use the vector database
    await vector_db.add_documents(documents, embeddings)
    results = await vector_db.similarity_search(query_embedding, k=5)
"""

from app.vectordb.interfaces import (
    VectorDB,
    VectorDBConfig,
    VectorDBProvider,
    VectorDBFactory,
    SearchResult
)

# Import specific implementations
from app.vectordb.faiss import FaissVectorDB
from app.vectordb.qdrant import QdrantVectorDB
# from app.vectordb.weaviate import WeaviateVectorDB  # Temporarily commented out
# from app.vectordb.pinecone import PineconeVectorDB
# from app.vectordb.milvus import MilvusVectorDB

# Import factory functions
from app.vectordb.factory import create_vectordb

# Import utility functions
from app.vectordb.utils import add_vectordb_args, create_vectordb_from_args, format_search_results

__all__ = [
    # Core interfaces
    "VectorDB",
    "VectorDBConfig",
    "VectorDBProvider",
    "VectorDBFactory",
    "SearchResult",
    
    # Implementations
    "FaissVectorDB",
    "QdrantVectorDB",
    # "WeaviateVectorDB",  # Temporarily commented out
    # "PineconeVectorDB",
    # "MilvusVectorDB",
    
    # Factory functions
    "create_vectordb",
    
    # Utility functions
    "add_vectordb_args",
    "create_vectordb_from_args",
    "format_search_results",
] 