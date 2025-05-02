"""
Document loaders package.

This package provides document loaders for various file formats.
"""

from app.document_processing.loaders.base import BaseDocumentLoader, DocumentLoaderRegistry, document_loader_registry
from app.document_processing.loaders.text import TextDocumentLoader
from app.document_processing.loaders.pdf import PDFDocumentLoader
from app.document_processing.loaders.docx import DocxDocumentLoader
from app.document_processing.loaders.html import HTMLDocumentLoader
from app.document_processing.loaders.markdown import MarkdownDocumentLoader

__all__ = [
    'BaseDocumentLoader',
    'DocumentLoaderRegistry',
    'document_loader_registry',
    'TextDocumentLoader',
    'PDFDocumentLoader',
    'DocxDocumentLoader',
    'HTMLDocumentLoader',
    'MarkdownDocumentLoader',
] 