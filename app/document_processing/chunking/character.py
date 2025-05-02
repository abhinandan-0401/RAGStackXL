"""
Character text splitter.

This module provides a text splitter that splits text based on character count.
"""
from typing import List, Optional, Set

from app.document_processing.chunking.base import BaseTextSplitter, text_splitter_registry
from app.utils.logging import log


class CharacterTextSplitter(BaseTextSplitter):
    """Text splitter that splits text based on character count."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        backup_separators: Optional[List[str]] = None,
        add_start_index: bool = True,
        strip_whitespace: bool = True,
    ):
        """
        Initialize the character text splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return.
            chunk_overlap: Overlap in characters between chunks.
            separator: Default separator to use for splitting.
            backup_separators: Additional separators to try if the default one
                              doesn't result in small enough chunks.
            add_start_index: If True, adds a "chunk_start_index" entry to chunk metadata.
            strip_whitespace: If True, strips whitespace from the start and end of chunks.
        """
        super().__init__(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )
        self.separator = separator
        self.backup_separators = backup_separators or ["\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        # Use the separator followed by backup separators if needed
        separators = [self.separator] + self.backup_separators
        
        # Start with the default separator, then try backups if needed
        for sep in separators:
            # Skip empty separator if we're not at the last option
            if not sep and sep != separators[-1]:
                continue
            
            chunks = self._split_with_separator(text, sep)
            
            if all(len(chunk) <= self.chunk_size for chunk in chunks) or sep == separators[-1]:
                # Either all chunks are small enough, or we've tried all separators
                log.debug(f"Split text into {len(chunks)} chunks using separator: '{sep}'")
                return chunks
            
            log.debug(f"Chunks too large with separator '{sep}', trying next separator")
        
        # We should never reach here due to the loop logic above
        return chunks
    
    def _split_with_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text using the specified separator.
        
        Args:
            text: Text to split.
            separator: Separator to use for splitting.
            
        Returns:
            List of text chunks.
        """
        # If no separator is specified, split by characters
        if not separator:
            return self._split_by_characters(text)
        
        # Split the text by separator
        splits = text.split(separator)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # If this is not the first split and the split itself
            # plus the separator would make the chunk too big,
            # finalize the current chunk and start a new one
            split_length = len(split)
            if current_chunk and current_length + split_length + len(separator) > self.chunk_size:
                # Finalize current chunk
                chunk_text = separator.join(current_chunk)
                if self.strip_whitespace:
                    chunk_text = chunk_text.strip()
                if chunk_text:  # Don't add empty chunks
                    chunks.append(chunk_text)
                
                # Start a new chunk with overlap
                if self.chunk_overlap > 0:
                    # Create overlap by keeping last pieces that fit within overlap
                    overlap_length = 0
                    overlap_chunks = []
                    
                    # Add pieces from the end of the current chunk to create overlap
                    for piece in reversed(current_chunk):
                        piece_full = piece + separator
                        if overlap_length + len(piece_full) > self.chunk_overlap:
                            break
                        overlap_length += len(piece_full)
                        overlap_chunks.insert(0, piece)
                    
                    current_chunk = overlap_chunks
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add the current split to the chunk
            current_chunk.append(split)
            current_length += split_length + len(separator)
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()
            if chunk_text:  # Don't add empty chunks
                chunks.append(chunk_text)
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[str]:
        """
        Split text by characters when no separator is available.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            start = i
            end = min(i + self.chunk_size, len(text))
            
            # Don't create chunks smaller than the overlap
            if end - start <= self.chunk_overlap and i > 0:
                break
            
            chunk = text[start:end]
            if self.strip_whitespace:
                chunk = chunk.strip()
            if chunk:  # Don't add empty chunks
                chunks.append(chunk)
        
        return chunks


# Register the splitter
text_splitter_registry.register(CharacterTextSplitter()) 