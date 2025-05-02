"""
Document processing package.

This package provides document loaders and text splitters for
ingesting and processing documents for RAG.
"""

from app.document_processing.loaders import (
    BaseDocumentLoader,
    DocumentLoaderRegistry,
    document_loader_registry,
    TextDocumentLoader,
    PDFDocumentLoader,
    DocxDocumentLoader,
    HTMLDocumentLoader,
    MarkdownDocumentLoader,
)

from app.document_processing.chunking import (
    BaseTextSplitter,
    TextSplitterRegistry,
    text_splitter_registry,
    CharacterTextSplitter,
    SemanticTextSplitter,
    RecursiveSemanticSplitter,
)

__all__ = [
    # Document loaders
    'BaseDocumentLoader',
    'DocumentLoaderRegistry',
    'document_loader_registry',
    'TextDocumentLoader',
    'PDFDocumentLoader',
    'DocxDocumentLoader',
    'HTMLDocumentLoader',
    'MarkdownDocumentLoader',
    
    # Text splitters
    'BaseTextSplitter',
    'TextSplitterRegistry',
    'text_splitter_registry',
    'CharacterTextSplitter',
    'SemanticTextSplitter',
    'RecursiveSemanticSplitter',
] 