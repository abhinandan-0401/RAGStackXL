"""
Base text splitter classes.

This module defines the base classes for text splitters, which are responsible
for splitting documents into smaller chunks for embedding and retrieval.
"""
import re
import uuid
from typing import List, Dict, Optional, Any, Set, Callable, Union

from app.core.interfaces import TextSplitter, RagDocument, DocumentMetadata
from app.utils.logging import log


class BaseTextSplitter(TextSplitter):
    """Base class for text splitters."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True,
        strip_whitespace: bool = True,
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return.
            chunk_overlap: Overlap in characters between chunks.
            add_start_index: If True, adds a "chunk_start_index" entry to chunk metadata.
            strip_whitespace: If True, strips whitespace from the start and end of chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.add_start_index = add_start_index
        self.strip_whitespace = strip_whitespace
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        raise NotImplementedError("Subclasses must implement split_text")
    
    def split_documents(self, documents: List[RagDocument]) -> List[RagDocument]:
        """
        Split documents into chunks.
        
        Args:
            documents: Documents to split.
            
        Returns:
            List of chunked documents.
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.content)
            log.debug(f"Split document into {len(chunks)} chunks")
            
            for i, chunk_text in enumerate(chunks):
                # Create metadata for the chunk
                chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                
                # Add chunk information to metadata
                chunk_metadata["chunk_index"] = i
                chunk_metadata["chunk_size"] = len(chunk_text)
                chunk_metadata["is_chunk"] = True
                chunk_metadata["total_chunks"] = len(chunks)
                
                if self.add_start_index:
                    chunk_metadata["chunk_start_index"] = doc.content.find(chunk_text)
                
                # Create the chunked document
                chunk_doc = RagDocument(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    doc_id=f"{doc.doc_id}_chunk_{i}" if doc.doc_id else str(uuid.uuid4()),
                    doc_type=doc.doc_type,
                )
                
                chunked_docs.append(chunk_doc)
        
        return chunked_docs


class TextSplitterRegistry:
    """Registry for text splitters."""
    
    def __init__(self):
        """Initialize the registry."""
        self.splitters: Dict[str, TextSplitter] = {}
    
    def register(self, splitter: TextSplitter, name: Optional[str] = None) -> None:
        """
        Register a text splitter.
        
        Args:
            splitter: The text splitter to register.
            name: Optional name for the splitter. If not provided, the class name is used.
        """
        name = name or splitter.__class__.__name__
        self.splitters[name] = splitter
        log.debug(f"Registered text splitter: {name}")
    
    def get(self, name: str) -> Optional[TextSplitter]:
        """
        Get a text splitter by name.
        
        Args:
            name: Name of the splitter.
            
        Returns:
            The text splitter, or None if not found.
        """
        return self.splitters.get(name)
    
    def list_splitters(self) -> List[str]:
        """
        List all registered splitters.
        
        Returns:
            List of splitter names.
        """
        return list(self.splitters.keys())


# Global registry for text splitters
text_splitter_registry = TextSplitterRegistry() 