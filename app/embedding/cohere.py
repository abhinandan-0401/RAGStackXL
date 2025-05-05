"""
Cohere embedding model implementation for RAGStackXL.

This module provides an implementation of the EmbeddingModel interface using
Cohere's embedding API.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np

from app.core.interfaces import RagDocument
from app.embedding.interfaces import EmbeddingModel, EmbeddingConfig, EmbeddingModelType, EmbeddingModelFactory
from app.utils.logging import log


class CohereEmbedding(EmbeddingModel):
    """Cohere embedding model implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the Cohere embedding model.
        
        Args:
            config: Configuration for the model
        """
        super().__init__(config)
        
        # Check if cohere is installed
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "The cohere package is not installed. "
                "Please install it with `pip install cohere`."
            )
        
        # Get model configuration
        self.model_name = config.model_name
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 60)
        self.batch_size = config.get("batch_size", 32)
        
        # Set up the Cohere client
        self._setup_client()
    
    def _setup_client(self):
        """Set up the Cohere client."""
        import cohere
        
        try:
            if not self.api_key:
                raise ValueError("Cohere API key is required. Please provide 'api_key' in the configuration.")
            
            self.client = cohere.Client(api_key=self.api_key, timeout=self.timeout)
            log.info(f"Cohere client initialized with model: {self.model_name}")
        except Exception as e:
            log.error(f"Error setting up Cohere client: {e}")
            raise RuntimeError(f"Failed to set up Cohere client: {e}")
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string using Cohere's API.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector for the query
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        loop = asyncio.get_event_loop()
        
        def _embed():
            try:
                # Get embedding from Cohere API
                response = self.client.embed(
                    texts=[query],
                    model=self.model_name,
                    input_type="search_query"  # For query embedding
                )
                # Extract the embedding vector
                embedding = response.embeddings[0]
                return embedding
            except Exception as e:
                log.error(f"Error getting embedding from Cohere: {e}")
                raise RuntimeError(f"Failed to get embedding from Cohere: {e}")
        
        # Run in a thread to avoid blocking the event loop
        vector = await loop.run_in_executor(None, _embed)
        
        # Normalize if required
        if self.config.normalize:
            vector = self._normalize_vector(vector)
        
        return vector
    
    async def embed_documents(self, documents: List[Union[str, RagDocument]]) -> List[List[float]]:
        """
        Embed multiple documents or strings using Cohere's API.
        
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
            all_embeddings = []
            
            # Process in batches due to API limits
            for i in range(0, len(non_empty_texts), self.batch_size):
                batch_texts = non_empty_texts[i:i + self.batch_size]
                
                try:
                    # Get embeddings from Cohere API
                    response = self.client.embed(
                        texts=batch_texts,
                        model=self.model_name,
                        input_type="search_document"  # For document embedding
                    )
                    # Extract embeddings
                    batch_embeddings = response.embeddings
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    log.error(f"Error getting embeddings from Cohere: {e}")
                    raise RuntimeError(f"Failed to get embeddings from Cohere: {e}")
            
            return all_embeddings
        
        # Run in a thread to avoid blocking the event loop
        vectors = await loop.run_in_executor(None, _batch_embed)
        
        # Normalize if required
        if self.config.normalize:
            vectors = [self._normalize_vector(vector) for vector in vectors]
        
        return vectors


# Register with the factory
EmbeddingModelFactory.register(EmbeddingModelType.COHERE, CohereEmbedding) 