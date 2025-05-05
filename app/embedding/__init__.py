"""
Embedding module for RAGStackXL.

This module provides tools for creating embeddings from text documents and queries.
"""

from app.embedding.interfaces import (
    EmbeddingModel,
    EmbeddingConfig,
    EmbeddingModelType,
    EmbeddingDimension,
    EmbeddingModelFactory,
)

from app.embedding.factory import create_embedding_model

# Import all model implementations
from app.embedding.sbert import SentenceTransformerEmbedding
from app.embedding.fastembed import FastEmbedModel
from app.embedding.openai import OpenAIEmbedding
from app.embedding.huggingface import HuggingFaceEmbedding
from app.embedding.cohere import CohereEmbedding

__all__ = [
    "EmbeddingModel",
    "EmbeddingConfig",
    "EmbeddingModelType",
    "EmbeddingDimension",
    "EmbeddingModelFactory",
    "create_embedding_model",
    "SentenceTransformerEmbedding",
    "FastEmbedModel", 
    "OpenAIEmbedding",
    "HuggingFaceEmbedding",
    "CohereEmbedding",
] 