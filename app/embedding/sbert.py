"""
SentenceTransformers (SBERT) embedding model implementation for RAGStackXL.

This module provides an implementation of the EmbeddingModel interface using
SentenceTransformers (SBERT) models from HuggingFace.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np

from app.core.interfaces import RagDocument
from app.embedding.interfaces import EmbeddingModel, EmbeddingConfig, EmbeddingModelType, EmbeddingModelFactory
from app.utils.logging import log


class SentenceTransformerEmbedding(EmbeddingModel):
    """SentenceTransformer (SBERT) embedding model implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the SentenceTransformer embedding model.
        
        Args:
            config: Configuration for the model
        """
        super().__init__(config)
        
        # Check if sentence-transformers is installed
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The sentence-transformers package is not installed. "
                "Please install it with `pip install sentence-transformers`."
            )
        
        # Get model name and other configurations
        self.model_name = config.model_name
        self.batch_size = config.get("batch_size", 32)
        self.device = config.get("device", "cpu")  # 'cpu', 'cuda', 'cuda:0', etc.
        self.show_progress_bar = config.get("show_progress_bar", False)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        from sentence_transformers import SentenceTransformer
        
        try:
            log.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Validate the model's dimension
            dummy_embedding = self.model.encode(["test"], show_progress_bar=False)[0]
            actual_dim = len(dummy_embedding)
            
            if actual_dim != self.config.dimension:
                log.warning(
                    f"Model dimension ({actual_dim}) does not match configured dimension "
                    f"({self.config.dimension}). Using actual dimension."
                )
                self.config.dimension = actual_dim
            
            log.info(f"SentenceTransformer model loaded with dimension: {self.config.dimension}")
        except Exception as e:
            log.error(f"Error loading SentenceTransformer model: {e}")
            raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")
    
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
            embedding = self.model.encode(query, show_progress_bar=False)
            return embedding.tolist()
        
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
            # Encode in batches for efficiency
            all_embeddings = []
            
            for i in range(0, len(non_empty_texts), self.batch_size):
                batch = non_empty_texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=self.show_progress_bar
                )
                all_embeddings.extend(batch_embeddings.tolist())
            
            return all_embeddings
        
        # Run in a thread to avoid blocking the event loop
        vectors = await loop.run_in_executor(None, _batch_embed)
        
        # Normalize if required
        if self.config.normalize:
            vectors = [self._normalize_vector(vector) for vector in vectors]
        
        return vectors


# Register with the factory
EmbeddingModelFactory.register(EmbeddingModelType.SBERT, SentenceTransformerEmbedding) 