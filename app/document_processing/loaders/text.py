"""
Text document loader.

This module provides a document loader for text files.
"""
import os
from pathlib import Path
from typing import List, Optional
import uuid

from app.core.interfaces import RagDocument, DocumentType
from app.document_processing.loaders.base import BaseDocumentLoader, document_loader_registry
from app.utils.logging import log


class TextDocumentLoader(BaseDocumentLoader):
    """Document loader for text files."""
    
    def __init__(self):
        """Initialize the text document loader."""
        super().__init__()
        self.supported_extensions = {"txt"}
    
    def _load_file(self, file_path: str) -> List[RagDocument]:
        """
        Load a text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            List containing a single RagDocument with the file content.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        try:
            log.debug(f"Loading text file: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Create metadata
            metadata = self._create_metadata(file_path)
            
            # Create document
            doc = RagDocument(
                content=content,
                metadata=metadata,
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentType.TEXT
            )
            
            return [doc]
        
        except Exception as e:
            log.error(f"Error loading text file {file_path}: {e}")
            raise


# Register the loader
document_loader_registry.register(TextDocumentLoader()) 