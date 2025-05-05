"""
OpenAI embedding model implementation for RAGStackXL.

This module provides an implementation of the EmbeddingModel interface using
OpenAI's embedding API.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np

from app.core.interfaces import RagDocument
from app.embedding.interfaces import EmbeddingModel, EmbeddingConfig, EmbeddingModelType, EmbeddingModelFactory
from app.utils.logging import log


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI API embedding model implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the OpenAI embedding model.
        
        Args:
            config: Configuration for the model
        """
        super().__init__(config)
        
        # Check if openai is installed
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The openai package is not installed. "
                "Please install it with `pip install openai`."
            )
        
        # Get model configuration
        self.model_name = config.model_name
        self.api_key = config.get("api_key")
        self.api_base = config.get("api_base")
        self.api_version = config.get("api_version")
        self.timeout = config.get("timeout", 60)
        
        # Set up the OpenAI client
        self._setup_client()
    
    def _setup_client(self):
        """Set up the OpenAI client."""
        import openai
        
        try:
            # Check if we're using the new or old API structure
            if hasattr(openai, "OpenAI"):
                # New OpenAI client (v1.0.0+)
                client_kwargs = {}
                
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                
                if self.api_base:
                    client_kwargs["base_url"] = self.api_base
                
                if self.timeout:
                    client_kwargs["timeout"] = self.timeout
                
                self.client = openai.OpenAI(**client_kwargs)
                log.info("Using OpenAI API client v1.0+")
            else:
                # Old OpenAI client (before v1.0.0)
                if self.api_key:
                    openai.api_key = self.api_key
                
                if self.api_base:
                    openai.api_base = self.api_base
                
                if self.api_version:
                    openai.api_version = self.api_version
                
                if self.timeout:
                    openai.timeout = self.timeout
                
                self.client = openai
                log.info("Using OpenAI API client v0.x")
            
            log.info(f"OpenAI API client configured with model: {self.model_name}")
        except Exception as e:
            log.error(f"Error setting up OpenAI client: {e}")
            raise RuntimeError(f"Failed to set up OpenAI client: {e}")
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string using OpenAI's embeddings API.
        
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
                # Check if we're using the new or old API structure
                if hasattr(self.client, "embeddings"):
                    # New OpenAI client (v1.0.0+)
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=query
                    )
                    embedding = response.data[0].embedding
                else:
                    # Old OpenAI client (before v1.0.0)
                    response = self.client.Embedding.create(
                        model=self.model_name,
                        input=query
                    )
                    embedding = response["data"][0]["embedding"]
                
                return embedding
            except Exception as e:
                log.error(f"Error getting embedding from OpenAI: {e}")
                raise RuntimeError(f"Failed to get embedding from OpenAI: {e}")
        
        # Run in a thread to avoid blocking the event loop
        vector = await loop.run_in_executor(None, _embed)
        
        # Normalize if required
        if self.config.normalize:
            vector = self._normalize_vector(vector)
        
        return vector
    
    async def embed_documents(self, documents: List[Union[str, RagDocument]]) -> List[List[float]]:
        """
        Embed multiple documents or strings using OpenAI's embeddings API.
        
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
            try:
                # Check if we're using the new or old API structure
                if hasattr(self.client, "embeddings"):
                    # New OpenAI client (v1.0.0+)
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=non_empty_texts
                    )
                    # Sort embeddings by index to maintain order
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    embeddings = [item.embedding for item in sorted_data]
                else:
                    # Old OpenAI client (before v1.0.0)
                    response = self.client.Embedding.create(
                        model=self.model_name,
                        input=non_empty_texts
                    )
                    # Sort embeddings by index to maintain order
                    sorted_data = sorted(response["data"], key=lambda x: x["index"])
                    embeddings = [item["embedding"] for item in sorted_data]
                
                return embeddings
            except Exception as e:
                log.error(f"Error getting embeddings from OpenAI: {e}")
                raise RuntimeError(f"Failed to get embeddings from OpenAI: {e}")
        
        # Run in a thread to avoid blocking the event loop
        vectors = await loop.run_in_executor(None, _batch_embed)
        
        # Normalize if required
        if self.config.normalize:
            vectors = [self._normalize_vector(vector) for vector in vectors]
        
        return vectors


# Register with the factory
EmbeddingModelFactory.register(EmbeddingModelType.OPENAI, OpenAIEmbedding) 