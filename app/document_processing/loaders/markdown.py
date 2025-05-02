"""
Markdown document loader.

This module provides a document loader for Markdown files.
"""
import os
from pathlib import Path
from typing import List, Optional
import uuid
import re

from app.core.interfaces import RagDocument, DocumentType
from app.document_processing.loaders.base import BaseDocumentLoader, document_loader_registry
from app.utils.logging import log


class MarkdownDocumentLoader(BaseDocumentLoader):
    """Document loader for Markdown files."""
    
    def __init__(self, extract_code_blocks: bool = True, extract_frontmatter: bool = True):
        """
        Initialize the Markdown document loader.
        
        Args:
            extract_code_blocks: Whether to include code blocks in the extraction.
            extract_frontmatter: Whether to extract YAML frontmatter as metadata.
        """
        super().__init__()
        self.supported_extensions = {"md", "markdown"}
        self.extract_code_blocks = extract_code_blocks
        self.extract_frontmatter = extract_frontmatter
    
    def _extract_frontmatter(self, content: str) -> tuple[str, dict]:
        """
        Extract YAML frontmatter from Markdown content.
        
        Args:
            content: Markdown content.
            
        Returns:
            Tuple of (content without frontmatter, frontmatter metadata dict).
        """
        frontmatter = {}
        content_without_frontmatter = content
        
        # Check for YAML frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if frontmatter_match:
            frontmatter_yaml = frontmatter_match.group(1)
            content_without_frontmatter = content[frontmatter_match.end():]
            
            # Parse YAML frontmatter
            try:
                import yaml
                frontmatter = yaml.safe_load(frontmatter_yaml)
                if frontmatter and isinstance(frontmatter, dict):
                    frontmatter = {f"md_{k}": v for k, v in frontmatter.items()}
                else:
                    frontmatter = {}
            except Exception as e:
                log.warning(f"Error parsing frontmatter: {e}")
        
        return content_without_frontmatter, frontmatter
    
    def _load_file(self, file_path: str) -> List[RagDocument]:
        """
        Load a Markdown file.
        
        Args:
            file_path: Path to the Markdown file.
            
        Returns:
            List containing a single RagDocument with the Markdown content.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        try:
            log.debug(f"Loading Markdown file: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract frontmatter if requested
            md_metadata = {}
            if self.extract_frontmatter:
                content, frontmatter = self._extract_frontmatter(content)
                md_metadata.update(frontmatter)
            
            # Extract title from the first heading if not in frontmatter
            if "md_title" not in md_metadata:
                title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                if title_match:
                    md_metadata["md_title"] = title_match.group(1).strip()
            
            # Handle code blocks
            if not self.extract_code_blocks:
                # Remove code blocks if not wanted
                content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            
            # Create metadata
            metadata = self._create_metadata(
                source_path=file_path,
                **md_metadata
            )
            
            # Create document
            doc = RagDocument(
                content=content,
                metadata=metadata,
                doc_id=str(uuid.uuid4()),
                doc_type=DocumentType.MARKDOWN
            )
            
            return [doc]
        
        except Exception as e:
            log.error(f"Error loading Markdown file {file_path}: {e}")
            raise


# Register the loader
document_loader_registry.register(MarkdownDocumentLoader()) 