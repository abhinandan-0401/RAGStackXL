"""
FastEmbed implementation for RAGStackXL.

This module provides an implementation of the EmbeddingModel interface using
FastEmbed models, which are lightweight, efficient embedding models.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np

from app.core.interfaces import RagDocument
from app.embedding.interfaces import EmbeddingModel, EmbeddingConfig, EmbeddingModelType, EmbeddingModelFactory
from app.utils.logging import log


class FastEmbedModel(EmbeddingModel):
    """FastEmbed model implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the FastEmbed model.
        
        Args:
            config: Configuration for the model
        """
        super().__init__(config)
        
        # Check if fastembed is installed
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "The fastembed package is not installed. "
                "Please install it with `pip install fastembed`."
            )
        
        # Get model configuration
        self.model_name = config.model_name
        self.cache_dir = config.get("cache_dir", None)
        self.batch_size = config.get("batch_size", 32)
        self.threads = config.get("threads", 4)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the FastEmbed model."""
        from fastembed import TextEmbedding
        
        try:
            log.info(f"Loading FastEmbed model: {self.model_name}")
            
            # Use model name as "model" attribute in FastEmbed or use default
            model_name = self.model_name
            if model_name == "default":
                model_name = None  # FastEmbed will use the default model
            
            self.model = TextEmbedding(
                model_name=model_name,
                threads=self.threads,
                cache_dir=self.cache_dir
            )
            
            # Validate dimensions
            dummy_embedding = next(self.model.embed(["test"]))
            actual_dim = len(dummy_embedding)
            
            if actual_dim != self.config.dimension:
                log.warning(
                    f"Model dimension ({actual_dim}) does not match configured dimension "
                    f"({self.config.dimension}). Using actual dimension."
                )
                self.config.dimension = actual_dim
            
            log.info(f"FastEmbed model loaded with dimension: {self.config.dimension}")
        except Exception as e:
            log.error(f"Error loading FastEmbed model: {e}")
            raise RuntimeError(f"Failed to load FastEmbed model: {e}")
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector for the query
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        loop = asyncio.get_event_loop()
        
        def _embed():
            # FastEmbed returns a generator, we need to take the first item
            return next(self.model.embed([query])).tolist()
        
        # Run in a thread to avoid blocking the event loop
        vector = await loop.run_in_executor(None, _embed)
        
        # Normalize if required
        if self.config.normalize:
            vector = self._normalize_vector(vector)
        
        return vector
    
    async def embed_documents(self, documents: List[Union[str, RagDocument]]) -> List[List[float]]:
        """
        Embed multiple documents or strings.
        
        Args:
            documents: List of documents or strings to embed
            
        Returns:
            List of embedding vectors for the documents
        """
        if not documents:
            return []
        
        # Process documents to extract text
        texts = []
        for doc in documents:
            if isinstance(doc, RagDocument):
                texts.append(doc.content)
            elif isinstance(doc, str):
                texts.append(doc)
            else:
                raise TypeError(f"Unsupported document type: {type(doc)}")
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        if not non_empty_texts:
            return []
        
        loop = asyncio.get_event_loop()
        
        def _batch_embed():
            # Embed all texts
            embeddings = list(self.model.embed(non_empty_texts, batch_size=self.batch_size))
            return [embedding.tolist() for embedding in embeddings]
        
        # Run in a thread to avoid blocking the event loop
        vectors = await loop.run_in_executor(None, _batch_embed)
        
        # Normalize if required
        if self.config.normalize:
            vectors = [self._normalize_vector(vector) for vector in vectors]
        
        return vectors


# Register with the factory
EmbeddingModelFactory.register(EmbeddingModelType.FASTEMBED, FastEmbedModel) 