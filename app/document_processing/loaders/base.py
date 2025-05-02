"""
Base document loader classes.

This module defines the base classes for document loaders, which are responsible
for loading documents from various sources and converting them to RagDocument objects.
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Set

from app.core.interfaces import DocumentLoader, RagDocument, DocumentType, DocumentMetadata
from app.utils.logging import log


class BaseDocumentLoader(DocumentLoader):
    """Base class for document loaders."""
    
    def __init__(self):
        """Initialize the document loader."""
        self.supported_extensions: Set[str] = set()
    
    def load(self, source: str) -> List[RagDocument]:
        """
        Load documents from a source.
        
        Args:
            source: Path to a file or directory.
            
        Returns:
            List of RagDocument objects.
        """
        if not self.supports(source):
            raise ValueError(f"Source not supported: {source}")
        
        path = Path(source)
        
        if path.is_file():
            return self._load_file(str(path))
        elif path.is_dir():
            return self._load_directory(str(path))
        else:
            raise ValueError(f"Source not found: {source}")
    
    def supports(self, source: str) -> bool:
        """
        Check if the loader supports the given source.
        
        Args:
            source: Path to a file or directory.
            
        Returns:
            True if the source is supported, False otherwise.
        """
        path = Path(source)
        
        # If it's a directory, we support it if we can load any file in it
        if path.is_dir():
            return True
        
        # If it's a file, we support it if the extension is in our list
        if path.is_file():
            return path.suffix.lower()[1:] in self.supported_extensions
        
        return False
    
    def _load_file(self, file_path: str) -> List[RagDocument]:
        """
        Load a single file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            List of RagDocument objects.
        """
        raise NotImplementedError("Subclasses must implement _load_file")
    
    def _load_directory(self, dir_path: str) -> List[RagDocument]:
        """
        Load all files in a directory.
        
        Args:
            dir_path: Path to the directory.
            
        Returns:
            List of RagDocument objects.
        """
        documents = []
        
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if self.supports(file_path):
                    try:
                        docs = self._load_file(file_path)
                        documents.extend(docs)
                        log.info(f"Loaded {len(docs)} documents from {file_path}")
                    except Exception as e:
                        log.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _create_metadata(
        self,
        source_path: str,
        page_number: Optional[int] = None,
        total_pages: Optional[int] = None,
        **kwargs
    ) -> DocumentMetadata:
        """
        Create metadata for a document.
        
        Args:
            source_path: Path to the source file.
            page_number: Page number of the document (if applicable).
            total_pages: Total number of pages in the source file (if applicable).
            **kwargs: Additional metadata fields.
            
        Returns:
            Document metadata dictionary.
        """
        path = Path(source_path)
        
        metadata = {
            "source": source_path,
            "filename": path.name,
            "extension": path.suffix.lower()[1:],
            "file_path": str(path.absolute()),
            "file_size": path.stat().st_size if path.exists() else None,
            "created_at": path.stat().st_ctime if path.exists() else None,
            "modified_at": path.stat().st_mtime if path.exists() else None,
        }
        
        # Add page information if provided
        if page_number is not None:
            metadata["page"] = page_number
        if total_pages is not None:
            metadata["total_pages"] = total_pages
        
        # Add additional metadata
        metadata.update(kwargs)
        
        return metadata


class DocumentLoaderRegistry:
    """Registry for document loaders."""
    
    def __init__(self):
        """Initialize the registry."""
        self.loaders: Dict[str, DocumentLoader] = {}
    
    def register(self, loader: DocumentLoader, name: Optional[str] = None) -> None:
        """
        Register a document loader.
        
        Args:
            loader: The document loader to register.
            name: Optional name for the loader. If not provided, the class name is used.
        """
        name = name or loader.__class__.__name__
        self.loaders[name] = loader
        log.debug(f"Registered document loader: {name}")
    
    def get(self, name: str) -> Optional[DocumentLoader]:
        """
        Get a document loader by name.
        
        Args:
            name: Name of the loader.
            
        Returns:
            The document loader, or None if not found.
        """
        return self.loaders.get(name)
    
    def get_for_source(self, source: str) -> Optional[DocumentLoader]:
        """
        Get a document loader that supports the given source.
        
        Args:
            source: Path to a file or directory.
            
        Returns:
            The first document loader that supports the source, or None if none found.
        """
        for loader in self.loaders.values():
            if loader.supports(source):
                return loader
        return None


# Global registry for document loaders
document_loader_registry = DocumentLoaderRegistry() 