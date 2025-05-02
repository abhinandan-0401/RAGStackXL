"""
Document processor.

This module provides a document processor that handles the entire
pipeline of loading, preprocessing, and chunking documents.
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from app.core.interfaces import RagDocument, Event, EventType, event_bus
from app.document_processing.loaders import document_loader_registry, BaseDocumentLoader
from app.document_processing.chunking import text_splitter_registry, BaseTextSplitter
from app.config.settings import settings
from app.utils.logging import log


class DocumentProcessor:
    """
    Document processor that handles loading, preprocessing, and chunking documents.
    """
    
    def __init__(
        self,
        loader_registry=None,
        splitter_registry=None,
        default_splitter="RecursiveSemanticSplitter",
        chunk_size=None,
        chunk_overlap=None,
    ):
        """
        Initialize the document processor.
        
        Args:
            loader_registry: Registry of document loaders. Defaults to global registry.
            splitter_registry: Registry of text splitters. Defaults to global registry.
            default_splitter: Name of the default text splitter to use.
            chunk_size: Default chunk size. If None, uses settings.
            chunk_overlap: Default chunk overlap. If None, uses settings.
        """
        self.loader_registry = loader_registry or document_loader_registry
        self.splitter_registry = splitter_registry or text_splitter_registry
        self.default_splitter = default_splitter
        self.chunk_size = chunk_size or settings.DOCUMENT.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DOCUMENT.CHUNK_OVERLAP
    
    def process_document(
        self,
        source: str,
        splitter_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        recursive: bool = False,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RagDocument]:
        """
        Process a document or directory of documents.
        
        Args:
            source: Path to a file or directory.
            splitter_name: Name of the text splitter to use.
            chunk_size: Chunk size to use. If None, uses instance default.
            chunk_overlap: Chunk overlap to use. If None, uses instance default.
            recursive: Whether to recursively process directories.
            metadata_filters: Filters to apply to document metadata.
            
        Returns:
            List of processed documents.
        """
        # Set up text splitter
        splitter_name = splitter_name or self.default_splitter
        splitter = self.splitter_registry.get(splitter_name)
        
        if not splitter:
            log.warning(f"Text splitter '{splitter_name}' not found. Using RecursiveSemanticSplitter.")
            splitter = self.splitter_registry.get("RecursiveSemanticSplitter")
        
        # Configure splitter with specified chunk size and overlap
        if chunk_size is not None or chunk_overlap is not None:
            # Clone the splitter with new parameters
            splitter = self._clone_splitter_with_params(
                splitter,
                chunk_size or self.chunk_size,
                chunk_overlap or self.chunk_overlap,
            )
        
        # Process source
        return self._process_source(source, splitter, recursive, metadata_filters)
    
    def _process_source(
        self,
        source: str,
        splitter: BaseTextSplitter,
        recursive: bool,
        metadata_filters: Optional[Dict[str, Any]],
    ) -> List[RagDocument]:
        """
        Process a single source.
        
        Args:
            source: Path to a file or directory.
            splitter: Text splitter to use.
            recursive: Whether to recursively process directories.
            metadata_filters: Filters to apply to document metadata.
            
        Returns:
            List of processed documents.
        """
        path = Path(source)
        
        if not path.exists():
            raise ValueError(f"Source not found: {source}")
        
        documents = []
        
        # Handle directories
        if path.is_dir():
            log.info(f"Processing directory: {source}")
            
            for root, dirs, files in os.walk(path):
                if not recursive and root != str(path):
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        docs = self._process_file(file_path, splitter, metadata_filters)
                        documents.extend(docs)
                    except Exception as e:
                        log.error(f"Error processing {file_path}: {e}")
        
        # Handle files
        elif path.is_file():
            log.info(f"Processing file: {source}")
            documents = self._process_file(source, splitter, metadata_filters)
        
        else:
            raise ValueError(f"Source is not a file or directory: {source}")
        
        log.info(f"Processed {len(documents)} documents from {source}")
        return documents
    
    def _process_file(
        self,
        file_path: str,
        splitter: BaseTextSplitter,
        metadata_filters: Optional[Dict[str, Any]],
    ) -> List[RagDocument]:
        """
        Process a single file.
        
        Args:
            file_path: Path to the file.
            splitter: Text splitter to use.
            metadata_filters: Filters to apply to document metadata.
            
        Returns:
            List of processed documents.
        """
        # Find a suitable loader
        loader = self.loader_registry.get_for_source(file_path)
        
        if not loader:
            log.warning(f"No loader found for {file_path}")
            return []
        
        try:
            # Load the document
            log.debug(f"Loading document: {file_path}")
            raw_docs = loader.load(file_path)
            
            # Apply metadata filters if specified
            if metadata_filters:
                raw_docs = self._filter_by_metadata(raw_docs, metadata_filters)
            
            # Split the documents
            log.debug(f"Splitting document: {file_path}")
            chunked_docs = splitter.split_documents(raw_docs)
            
            # Publish event for each document added
            for doc in chunked_docs:
                event_bus.publish(
                    Event(
                        event_type=EventType.DOCUMENT_ADDED,
                        payload={"document_id": doc.doc_id, "source": file_path},
                        metadata={"document_type": doc.doc_type}
                    )
                )
            
            return chunked_docs
        
        except Exception as e:
            log.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _filter_by_metadata(
        self,
        documents: List[RagDocument],
        filters: Dict[str, Any],
    ) -> List[RagDocument]:
        """
        Filter documents by metadata.
        
        Args:
            documents: List of documents to filter.
            filters: Dictionary of metadata field -> value pairs.
            
        Returns:
            Filtered list of documents.
        """
        filtered_docs = []
        
        for doc in documents:
            # Check if all filters match
            if all(doc.metadata.get(k) == v for k, v in filters.items()):
                filtered_docs.append(doc)
        
        log.debug(f"Filtered {len(documents)} documents to {len(filtered_docs)} based on metadata")
        return filtered_docs
    
    def _clone_splitter_with_params(
        self,
        splitter: BaseTextSplitter,
        chunk_size: int,
        chunk_overlap: int,
    ) -> BaseTextSplitter:
        """
        Create a new splitter with the specified parameters.
        
        Args:
            splitter: Original text splitter.
            chunk_size: New chunk size.
            chunk_overlap: New chunk overlap.
            
        Returns:
            New text splitter with updated parameters.
        """
        # Create a new instance of the same class
        new_splitter = splitter.__class__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Preserve other parameters if possible
            add_start_index=getattr(splitter, "add_start_index", True),
            strip_whitespace=getattr(splitter, "strip_whitespace", True),
        )
        
        return new_splitter


# Create a global document processor instance with default settings
document_processor = DocumentProcessor() 