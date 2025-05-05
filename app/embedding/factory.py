"""
Embedding model factory for RAGStackXL.

This module provides utilities for creating embedding model instances from
configuration settings.
"""

from typing import Dict, Any, Optional

from app.config.settings import settings
from app.utils.logging import log
from app.embedding.interfaces import EmbeddingModel, EmbeddingConfig, EmbeddingModelType, EmbeddingModelFactory


def create_embedding_model(
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    dimension: Optional[int] = None,
    normalize: Optional[bool] = None,
    **kwargs
) -> EmbeddingModel:
    """
    Create an embedding model instance from settings.
    
    Args:
        model_name: Model name (if None, uses settings.EMBEDDING.MODEL_NAME)
        model_type: Model type (if None, uses settings.EMBEDDING.MODEL_TYPE)
        dimension: Embedding dimension (if None, uses settings.EMBEDDING.DIMENSION)
        normalize: Whether to normalize embeddings (if None, uses settings.EMBEDDING.NORMALIZE)
        **kwargs: Additional model-specific settings
        
    Returns:
        Embedding model instance
    """
    # Use provided values or fall back to settings
    model_name = model_name or settings.EMBEDDING.MODEL_NAME
    model_type_str = model_type or settings.EMBEDDING.MODEL_TYPE
    dimension_val = dimension or settings.EMBEDDING.DIMENSION
    normalize_val = normalize if normalize is not None else settings.EMBEDDING.NORMALIZE
    
    # Get provider-specific additional settings
    model_config = _get_model_config(model_type_str, kwargs)
    
    try:
        # Create model type enum
        model_type_enum = EmbeddingModelType(model_type_str.lower())
    except ValueError:
        log.error(f"Unknown embedding model type: {model_type_str}")
        available_types = [t.value for t in EmbeddingModelType]
        log.info(f"Available types: {', '.join(available_types)}")
        raise ValueError(f"Unknown embedding model type: {model_type_str}. Available: {available_types}")
    
    # Create embedding configuration
    config = EmbeddingConfig(
        model_name=model_name,
        model_type=model_type_enum,
        dimension=dimension_val,
        normalize=normalize_val,
        **model_config
    )
    
    # Create embedding model
    embedding_model = EmbeddingModelFactory.create(config)
    
    log.info(f"Created {model_type_str} embedding model: {model_name}")
    return embedding_model


def _get_model_config(model_type: str, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Get model-specific configuration from settings."""
    model_config = {}
    model_type = model_type.lower()
    
    # Add model-specific settings from config
    if model_type == "openai":
        model_config["api_key"] = settings.EMBEDDING.OPENAI.API_KEY
        model_config["api_base"] = settings.EMBEDDING.OPENAI.API_BASE
        model_config["api_version"] = settings.EMBEDDING.OPENAI.API_VERSION
        model_config["timeout"] = settings.EMBEDDING.OPENAI.TIMEOUT
    elif model_type == "sbert":
        model_config["device"] = settings.EMBEDDING.SBERT.DEVICE
        model_config["batch_size"] = settings.EMBEDDING.SBERT.BATCH_SIZE
        model_config["show_progress_bar"] = settings.EMBEDDING.SBERT.SHOW_PROGRESS_BAR
    elif model_type == "huggingface":
        model_config["token"] = settings.EMBEDDING.HUGGINGFACE.TOKEN
        model_config["revision"] = settings.EMBEDDING.HUGGINGFACE.REVISION
        model_config["device"] = settings.EMBEDDING.HUGGINGFACE.DEVICE
    elif model_type == "cohere":
        model_config["api_key"] = settings.EMBEDDING.COHERE.API_KEY
        model_config["timeout"] = settings.EMBEDDING.COHERE.TIMEOUT
    elif model_type == "fastembed":
        model_config["cache_dir"] = settings.EMBEDDING.FASTEMBED.CACHE_DIR
        model_config["threads"] = settings.EMBEDDING.FASTEMBED.THREADS
        model_config["batch_size"] = settings.EMBEDDING.FASTEMBED.BATCH_SIZE
    
    # User kwargs override settings
    model_config.update(user_kwargs)
    
    return model_config 