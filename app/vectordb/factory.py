"""
Vector database factory module.

This module provides utilities for creating vector database instances
from configuration settings.
"""

import os
from typing import Optional, Dict, Any, Type

from app.config.settings import settings
from app.utils.logging import log
from app.vectordb.interfaces import VectorDB, VectorDBConfig, VectorDBProvider, VectorDBFactory


def create_vectordb(
    provider: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    distance_metric: Optional[str] = None,
    persist_directory: Optional[str] = None,
    **kwargs
) -> VectorDB:
    """
    Create a vector database instance from settings.
    
    Args:
        provider: Vector database provider name (if None, uses settings.VECTORDB.PROVIDER)
        collection_name: Collection name (if None, uses settings.VECTORDB.COLLECTION_NAME)
        embedding_dimension: Embedding dimension (if None, uses settings.VECTORDB.EMBEDDING_DIMENSION)
        distance_metric: Distance metric (if None, uses settings.VECTORDB.DISTANCE_METRIC)
        persist_directory: Persist directory (if None, uses settings.VECTORDB.PERSIST_DIRECTORY)
        **kwargs: Additional provider-specific configuration options
        
    Returns:
        Vector database instance
    """
    # Use provided values or fall back to settings
    provider_name = provider or settings.VECTORDB.PROVIDER
    collection = collection_name or settings.VECTORDB.COLLECTION_NAME
    dimension = embedding_dimension or settings.VECTORDB.EMBEDDING_DIMENSION
    metric = distance_metric or settings.VECTORDB.DISTANCE_METRIC
    
    # Handle persist directory
    if persist_directory is None and settings.VECTORDB.PERSIST_DIRECTORY:
        persist_directory = settings.VECTORDB.PERSIST_DIRECTORY
        
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
    
    # Create provider-specific configuration
    provider_config = _get_provider_config(provider_name, kwargs)
    
    # Create vector database config
    config = VectorDBConfig(
        collection_name=collection,
        embedding_dimension=dimension,
        distance_metric=metric,
        persist_directory=persist_directory,
        **provider_config
    )
    
    # Get provider enum from string
    try:
        provider_enum = VectorDBProvider(provider_name.lower())
    except ValueError:
        log.error(f"Unknown vector database provider: {provider_name}")
        available_providers = [p.value for p in VectorDBProvider]
        log.info(f"Available providers: {', '.join(available_providers)}")
        raise ValueError(f"Unknown vector database provider: {provider_name}. Available: {available_providers}")
    
    # Create vector database
    vector_db = VectorDBFactory.create(provider_enum, config)
    
    log.info(f"Created {provider_name} vector database with collection '{collection}'")
    return vector_db


def _get_provider_config(provider_name: str, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Get provider-specific configuration from settings."""
    provider_config = {}
    
    provider_name = provider_name.lower()
    
    # Add provider-specific settings from config
    if provider_name == "faiss":
        # No special settings for FAISS
        pass
    elif provider_name == "qdrant":
        # Get Qdrant-specific settings
        provider_config["url"] = settings.VECTORDB.QDRANT.URL
        provider_config["api_key"] = settings.VECTORDB.QDRANT.API_KEY
        provider_config["prefer_grpc"] = settings.VECTORDB.QDRANT.PREFER_GRPC
    elif provider_name == "weaviate":
        # Get Weaviate-specific settings
        provider_config["url"] = settings.VECTORDB.WEAVIATE.URL
        provider_config["api_key"] = settings.VECTORDB.WEAVIATE.API_KEY
    elif provider_name == "pinecone":
        # Get Pinecone-specific settings
        provider_config["api_key"] = settings.VECTORDB.PINECONE.API_KEY
        provider_config["environment"] = settings.VECTORDB.PINECONE.ENVIRONMENT
    elif provider_name == "milvus":
        # Get Milvus-specific settings
        provider_config["uri"] = settings.VECTORDB.MILVUS.URI
        provider_config["token"] = settings.VECTORDB.MILVUS.TOKEN
    
    # User kwargs override settings
    provider_config.update(user_kwargs)
    
    return provider_config 