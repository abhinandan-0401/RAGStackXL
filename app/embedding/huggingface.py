"""
HuggingFace embedding model implementation for RAGStackXL.

This module provides an implementation of the EmbeddingModel interface using
HuggingFace Transformers models.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np

from app.core.interfaces import RagDocument
from app.embedding.interfaces import EmbeddingModel, EmbeddingConfig, EmbeddingModelType, EmbeddingModelFactory
from app.utils.logging import log


class HuggingFaceEmbedding(EmbeddingModel):
    """HuggingFace embedding model implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the HuggingFace embedding model.
        
        Args:
            config: Configuration for the model
        """
        super().__init__(config)
        
        # Check if transformers is installed
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "The transformers package is not installed. "
                "Please install it with `pip install transformers`."
            )
        
        # Get model configuration
        self.model_name = config.model_name
        self.token = config.get("token")
        self.revision = config.get("revision", "main")
        self.device = config.get("device", "cpu")  # 'cpu', 'cuda', 'cuda:0', etc.
        self.batch_size = config.get("batch_size", 32)
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        from transformers import AutoTokenizer, AutoModel
        
        try:
            log.info(f"Loading HuggingFace model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.token,
                revision=self.revision
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                token=self.token,
                revision=self.revision
            )
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Validate dimensions with a test input
            dummy_embedding = self._get_embedding("test")
            actual_dim = len(dummy_embedding)
            
            if actual_dim != self.config.dimension:
                log.warning(
                    f"Model dimension ({actual_dim}) does not match configured dimension "
                    f"({self.config.dimension}). Using actual dimension."
                )
                self.config.dimension = actual_dim
            
            log.info(f"HuggingFace model loaded with dimension: {self.config.dimension}")
        except Exception as e:
            log.error(f"Error loading HuggingFace model: {e}")
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using HuggingFace model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        import torch
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the [CLS] token representation as the embedding
        # This is standard practice for sentence embeddings with BERT-like models
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding.tolist()
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string using HuggingFace model.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector for the query
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        loop = asyncio.get_event_loop()
        
        # Run in a thread to avoid blocking the event loop
        vector = await loop.run_in_executor(None, lambda: self._get_embedding(query))
        
        # Normalize if required
        if self.config.normalize:
            vector = self._normalize_vector(vector)
        
        return vector
    
    async def embed_documents(self, documents: List[Union[str, RagDocument]]) -> List[List[float]]:
        """
        Embed multiple documents or strings using HuggingFace model.
        
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
            import torch
            
            all_embeddings = []
            
            for i in range(0, len(non_empty_texts), self.batch_size):
                batch_texts = non_empty_texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to device
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                
                # Get model output
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Use the [CLS] token representation as the embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
            
            return all_embeddings
        
        # Run in a thread to avoid blocking the event loop
        vectors = await loop.run_in_executor(None, _batch_embed)
        
        # Normalize if required
        if self.config.normalize:
            vectors = [self._normalize_vector(vector) for vector in vectors]
        
        return vectors


# Register with the factory
EmbeddingModelFactory.register(EmbeddingModelType.HUGGINGFACE, HuggingFaceEmbedding) 