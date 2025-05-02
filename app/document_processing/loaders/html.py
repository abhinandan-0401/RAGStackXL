"""
HTML document loader.

This module provides a document loader for HTML files.
"""
import os
from pathlib import Path
from typing import List, Optional
import uuid

from bs4 import BeautifulSoup
from app.core.interfaces import RagDocument, DocumentType
from app.document_processing.loaders.base import BaseDocumentLoader, document_loader_registry
from app.utils.logging import log


class HTMLDocumentLoader(BaseDocumentLoader):
    """Document loader for HTML files."""
    
    def __init__(self, extract_images: bool = False, include_tables: bool = True):
        """
        Initialize the HTML document loader.
        
        Args:
            extract_images: Whether to extract image descriptions.
                           Not yet fully implemented.
            include_tables: Whether to include tables in the extraction.
        """
        super().__init__()
        self.supported_extensions = {"html", "htm", "xhtml"}
        self.extract_images = extract_images
        self.include_tables = include_tables
    
    def _load_file(self, file_path: str) -> List[RagDocument]:
        """
        Load an HTML file.
        
        Args:
            file_path: Path to the HTML file.
            
        Returns:
            List containing a single RagDocument with the HTML content.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        try:
            log.debug(f"Loading HTML file: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract metadata
            html_metadata = {}
            
            # Extract title
            title_tag = soup.find("title")
            if title_tag and title_tag.text.strip():
                html_metadata["html_title"] = title_tag.text.strip()
            
            # Extract meta tags
            for meta in soup.find_all("meta"):
                name = meta.get("name") or meta.get("property")
                content = meta.get("content")
                
                if name and content:
                    html_metadata[f"html_meta_{name.lower().replace(':', '_')}"] = content
            
            # Clean the HTML for text extraction
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            content_parts = []
            
            # Add the title if it exists
            if "html_title" in html_metadata:
                content_parts.append(f"# {html_metadata['html_title']}")
            
            # Handle headings
            for i in range(1, 7):  # h1 to h6
                for heading in soup.find_all(f'h{i}'):
                    if heading.text.strip():
                        content_parts.append(f"{'#' * i} {heading.text.strip()}")
            
            # Extract paragraphs
            for p in soup.find_all('p'):
                if p.text.strip():
                    content_parts.append(p.text.strip())
            
            # Extract lists
            for ul in soup.find_all('ul'):
                for li in ul.find_all('li'):
                    if li.text.strip():
                        content_parts.append(f"* {li.text.strip()}")
            
            for ol in soup.find_all('ol'):
                for i, li in enumerate(ol.find_all('li')):
                    if li.text.strip():
                        content_parts.append(f"{i+1}. {li.text.strip()}")
            
            # Extract tables
            if self.include_tables:
                for i, table in enumerate(soup.find_all('table')):
                    content_parts.append(f"\n[TABLE {i+1}]\n")
                    
                    for row in table.find_all('tr'):
                        cells = []
                        for cell in row.find_all(['td', 'th']):
                            cells.append(cell.text.strip())
                        
                        if cells:
                            content_parts.append(" | ".join(cells))
            
            # Extract image alt text if requested
            if self.extract_images:
                for i, img in enumerate(soup.find_all('img')):
                    alt_text = img.get('alt')
                    if alt_text:
                        content_parts.append(f"[IMAGE: {alt_text}]")
            
            # Combine everything into a single text
            content = "\n\n".join(content_parts)
            
            # Create metadata
            metadata = self._create_metadata(
                source_path=file_path,
                **html_metadata
            )
            
            # Create document
            doc = RagDocument(
                content=content,
                metadata=metadata,
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentType.HTML
            )
            
            return [doc]
        
        except Exception as e:
            log.error(f"Error loading HTML file {file_path}: {e}")
            raise


# Register the loader
document_loader_registry.register(HTMLDocumentLoader()) 