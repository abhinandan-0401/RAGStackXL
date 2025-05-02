"""
PDF document loader.

This module provides a document loader for PDF files.
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

from pypdf import PdfReader
from app.core.interfaces import RagDocument, DocumentType
from app.document_processing.loaders.base import BaseDocumentLoader, document_loader_registry
from app.utils.logging import log


class PDFDocumentLoader(BaseDocumentLoader):
    """Document loader for PDF files."""
    
    def __init__(self, extract_images: bool = False):
        """
        Initialize the PDF document loader.
        
        Args:
            extract_images: Whether to extract images from the PDF.
                            This is not yet implemented.
        """
        super().__init__()
        self.supported_extensions = {"pdf"}
        self.extract_images = extract_images
    
    def _load_file(self, file_path: str) -> List[RagDocument]:
        """
        Load a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of RagDocument objects, one per page.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        try:
            log.debug(f"Loading PDF file: {file_path}")
            
            # Read the PDF file
            reader = PdfReader(file_path)
            
            # Extract metadata from the PDF
            pdf_info = reader.metadata
            pdf_metadata = {}
            
            if pdf_info:
                for key, value in pdf_info.items():
                    if key.startswith("/"):
                        key = key[1:]  # Remove leading slash
                    pdf_metadata[f"pdf_{key.lower()}"] = value
            
            documents = []
            total_pages = len(reader.pages)
            
            # Process each page
            for i, page in enumerate(reader.pages):
                # Extract text from the page
                text = page.extract_text()
                
                if not text.strip():
                    log.warning(f"Page {i+1} in {file_path} contains no text, skipping")
                    continue
                
                # Create metadata for this page
                metadata = self._create_metadata(
                    source_path=file_path,
                    page_number=i + 1,
                    total_pages=total_pages,
                    **pdf_metadata
                )
                
                # Create a document for this page
                doc = RagDocument(
                    content=text,
                    metadata=metadata,
                    doc_id=f"{uuid.uuid4()}",
                    doc_type=DocumentType.PDF
                )
                
                documents.append(doc)
            
            log.info(f"Loaded {len(documents)} pages from PDF: {file_path}")
            return documents
        
        except Exception as e:
            log.error(f"Error loading PDF file {file_path}: {e}")
            raise


# Register the loader
document_loader_registry.register(PDFDocumentLoader()) 