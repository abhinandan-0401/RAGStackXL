"""
Document processing package.

This package provides document loaders and text splitters for
ingesting and processing documents for RAG.
"""

from app.document_processing.loaders import (
    BaseDocumentLoader,
    document_loader_registry,
    TextDocumentLoader,
    PDFDocumentLoader,
    DocxDocumentLoader,
    HTMLDocumentLoader,
    MarkdownDocumentLoader,
)

from app.document_processing.chunking import (
    BaseTextSplitter,
    text_splitter_registry,
    CharacterTextSplitter,
    SemanticTextSplitter,
    RecursiveSemanticSplitter,
)

__all__ = [
    # Document loaders
    'BaseDocumentLoader',
    'document_loader_registry',
    'TextDocumentLoader',
    'PDFDocumentLoader',
    'DocxDocumentLoader',
    'HTMLDocumentLoader',
    'MarkdownDocumentLoader',
    
    # Text splitters
    'BaseTextSplitter',
    'text_splitter_registry',
    'CharacterTextSplitter',
    'SemanticTextSplitter',
    'RecursiveSemanticSplitter',
] 