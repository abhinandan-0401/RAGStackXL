"""
Text chunking package.

This package provides text splitters that break documents into chunks
for embedding and retrieval.
"""

from app.document_processing.chunking.base import BaseTextSplitter, text_splitter_registry
from app.document_processing.chunking.character import CharacterTextSplitter
from app.document_processing.chunking.semantic import SemanticTextSplitter, RecursiveSemanticSplitter

__all__ = [
    'BaseTextSplitter',
    'text_splitter_registry',
    'CharacterTextSplitter',
    'SemanticTextSplitter',
    'RecursiveSemanticSplitter',
] 