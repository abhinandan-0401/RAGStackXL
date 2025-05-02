"""
Semantic text splitter.

This module provides a text splitter that splits text based on semantic boundaries
such as paragraphs, sections, and sentences.
"""
import re
from typing import List, Optional, Dict, Any, Set, Pattern, Tuple

from app.document_processing.chunking.base import BaseTextSplitter, text_splitter_registry
from app.utils.logging import log


class SemanticTextSplitter(BaseTextSplitter):
    """Text splitter that respects semantic boundaries like paragraphs and sections."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True,
        strip_whitespace: bool = True,
        respect_markdown_headers: bool = True,
        respect_html_tags: bool = True,
        min_chunk_size: int = 50,
    ):
        """
        Initialize the semantic text splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return.
            chunk_overlap: Overlap in characters between chunks.
            add_start_index: If True, adds a "chunk_start_index" entry to chunk metadata.
            strip_whitespace: If True, strips whitespace from the start and end of chunks.
            respect_markdown_headers: If True, tries to keep text under the same header together.
            respect_html_tags: If True, tries to keep text within the same HTML tag together.
            min_chunk_size: Minimum chunk size to return.
        """
        super().__init__(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )
        self.respect_markdown_headers = respect_markdown_headers
        self.respect_html_tags = respect_html_tags
        self.min_chunk_size = min_chunk_size
        
        # Patterns for identifying different types of boundaries
        self.patterns = {
            "markdown_headers": re.compile(r"^#{1,6}\s+.+$", re.MULTILINE),
            "paragraph_breaks": re.compile(r"\n\s*\n"),
            "sentence_end": re.compile(r"(?<=[.!?])\s"),
            "html_block_tags": re.compile(r"<(div|section|article|header|footer|main|aside|p|h\d)[\s>]"),
        }
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on semantic boundaries.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        # Find all potential split points
        split_points = self._find_split_points(text)
        
        # Sort split points by position
        split_points.sort()
        
        # Create chunks based on split points
        chunks = []
        start = 0
        current_chunk_size = 0
        current_chunk_start = 0
        
        for pos, importance in split_points:
            # If adding this section would exceed chunk size, finalize the current chunk
            section_size = pos - start
            
            if current_chunk_size + section_size > self.chunk_size and start > current_chunk_start:
                # Don't create chunks smaller than the minimum size unless it's the last part
                chunk_text = text[current_chunk_start:start]
                if len(chunk_text) >= self.min_chunk_size or start >= len(text) - 1:
                    if self.strip_whitespace:
                        chunk_text = chunk_text.strip()
                    if chunk_text:  # Don't add empty chunks
                        chunks.append(chunk_text)
                    
                    # Start a new chunk with overlap
                    if self.chunk_overlap > 0:
                        # Find a good split point for the overlap
                        overlap_start = self._find_overlap_split_point(text, start, self.chunk_overlap)
                        current_chunk_start = overlap_start
                    else:
                        current_chunk_start = start
                    
                    current_chunk_size = start - current_chunk_start
            
            start = pos
            current_chunk_size = start - current_chunk_start
        
        # Add the final chunk
        if current_chunk_start < len(text):
            chunk_text = text[current_chunk_start:]
            if self.strip_whitespace:
                chunk_text = chunk_text.strip()
            if chunk_text:  # Don't add empty chunks
                chunks.append(chunk_text)
        
        # If we still have chunks that are too big, fall back to character splitting
        oversized_chunks = [chunk for chunk in chunks if len(chunk) > self.chunk_size]
        if oversized_chunks:
            log.warning(f"Found {len(oversized_chunks)} chunks larger than chunk_size. Falling back to character splitting for these.")
            final_chunks = []
            
            for chunk in chunks:
                if len(chunk) > self.chunk_size:
                    # Split this chunk by characters
                    char_chunks = self._split_by_characters(chunk)
                    final_chunks.extend(char_chunks)
                else:
                    final_chunks.append(chunk)
                    
            chunks = final_chunks
        
        log.debug(f"Split text into {len(chunks)} chunks using semantic boundaries")
        return chunks
    
    def _find_split_points(self, text: str) -> List[Tuple[int, int]]:
        """
        Find potential split points in the text based on semantic boundaries.
        
        Args:
            text: Text to analyze.
            
        Returns:
            List of tuples (position, importance) where position is the character index
            and importance is a value indicating how important this boundary is
            (higher values are more important).
        """
        split_points = []
        
        # Always include the end of the text
        split_points.append((len(text), 0))
        
        # Find paragraph breaks (highest priority)
        for match in self.patterns["paragraph_breaks"].finditer(text):
            split_points.append((match.start(), 100))
        
        # Find Markdown headers if requested
        if self.respect_markdown_headers:
            for match in self.patterns["markdown_headers"].finditer(text):
                # Split at the end of the line containing the header
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                
                # The fewer #s, the higher the importance (h1 > h2 > h3)
                header_level = len(match.group(0)) - len(match.group(0).lstrip("#"))
                importance = 90 - header_level  # h1=89, h2=88, etc.
                
                split_points.append((line_end, importance))
        
        # Find HTML block tags if requested
        if self.respect_html_tags:
            for match in self.patterns["html_block_tags"].finditer(text):
                split_points.append((match.start(), 80))
        
        # Find sentence boundaries (lowest priority)
        for match in self.patterns["sentence_end"].finditer(text):
            split_points.append((match.end(), 50))
        
        return split_points
    
    def _find_overlap_split_point(self, text: str, position: int, max_overlap: int) -> int:
        """
        Find a good point to start the overlap, working backwards from position.
        
        Args:
            text: The full text.
            position: Current position to work backwards from.
            max_overlap: Maximum overlap allowed.
            
        Returns:
            Position to start the overlap.
        """
        # Don't go beyond the start of the text
        start_limit = max(0, position - max_overlap)
        
        # Try to find a paragraph break
        paragraph_pos = text.rfind("\n\n", start_limit, position)
        if paragraph_pos >= start_limit:
            return paragraph_pos + 2  # Skip the double newline
        
        # Try to find a sentence break
        for match in reversed(list(self.patterns["sentence_end"].finditer(
                text[start_limit:position]))):
            return start_limit + match.end()
        
        # Fall back to a line break
        newline_pos = text.rfind("\n", start_limit, position)
        if newline_pos >= start_limit:
            return newline_pos + 1  # Skip the newline
        
        # If all else fails, just use the maximum overlap position
        return start_limit
    
    def _split_by_characters(self, text: str) -> List[str]:
        """
        Split text by characters when semantic boundaries don't work well.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            start = i
            end = min(i + self.chunk_size, len(text))
            
            # Don't create chunks smaller than the overlap unless it's the end
            if end - start <= self.chunk_overlap and i > 0 and end < len(text):
                continue
            
            chunk = text[start:end]
            if self.strip_whitespace:
                chunk = chunk.strip()
            if chunk:  # Don't add empty chunks
                chunks.append(chunk)
        
        return chunks


class RecursiveSemanticSplitter(BaseTextSplitter):
    """Text splitter that uses a recursive approach to split by semantic units."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True,
        strip_whitespace: bool = True,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the recursive splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return.
            chunk_overlap: Overlap in characters between chunks.
            add_start_index: If True, adds a "chunk_start_index" entry to chunk metadata.
            strip_whitespace: If True, strips whitespace from the start and end of chunks.
            separators: List of separators to use for splitting, in order of precedence.
                       Defaults to a comprehensive list of common separators.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )
        
        self.separators = separators or [
            # First try to split on double newlines (paragraphs)
            "\n\n",
            # Then try to split on single newlines
            "\n",
            # Then try to split on sentences
            ". ",
            "! ",
            "? ",
            # Then try to split on phrases
            "; ",
            ": ",
            # Then try to split on clauses
            ", ",
            # Then try to split on words
            " ",
            # Finally, split on characters if nothing else works
            ""
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using a recursive approach.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        # Check if we need to split at all
        if len(text) <= self.chunk_size:
            return [text] if text else []
        
        # Find the best separator to use at this level
        for separator in self.separators:
            # If we're at the last separator (empty string), we'll do character splitting
            if not separator:
                return self._split_by_characters(text)
            
            # Count how many splits this separator creates
            splits = text.split(separator)
            
            # Only use this separator if it actually splits the text
            if len(splits) > 1:
                # If all splits are small enough, we can do a simple join with overlap
                if all(len(split) < self.chunk_size for split in splits):
                    return self._merge_splits(splits, separator)
                
                # Otherwise, recursively split each piece that's too big
                chunks = []
                for split in splits:
                    # If this split is small enough, add it as is
                    if len(split) <= self.chunk_size:
                        if split.strip() if self.strip_whitespace else split:
                            chunks.append(split)
                    else:
                        # Recursively split this piece
                        sub_chunks = self.split_text(split)
                        chunks.extend(sub_chunks)
                
                # Merge small chunks back together to avoid too many tiny chunks
                return self._merge_splits(chunks, separator)
        
        # We should never reach here due to the empty string separator fallback
        return [text]
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge splits into chunks of appropriate size with overlap.
        
        Args:
            splits: List of text splits.
            separator: Separator used for joining.
            
        Returns:
            List of merged chunks.
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            separator_length = len(separator)
            
            # Check if adding this split would exceed the chunk size
            if current_chunk and current_length + split_length + separator_length > self.chunk_size:
                # Join the current chunk and add it to the list
                full_chunk = separator.join(current_chunk)
                if self.strip_whitespace:
                    full_chunk = full_chunk.strip()
                if full_chunk:
                    chunks.append(full_chunk)
                
                # Calculate overlap for the next chunk
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep as many of the last splits as needed for the overlap
                    overlap_length = 0
                    overlap_splits = []
                    
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) + separator_length > self.chunk_overlap:
                            break
                        
                        overlap_length += len(s) + separator_length
                        overlap_splits.insert(0, s)
                    
                    current_chunk = overlap_splits
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            # Add the current split to the chunk
            current_chunk.append(split)
            current_length += split_length + separator_length
        
        # Add the final chunk if there's anything left
        if current_chunk:
            full_chunk = separator.join(current_chunk)
            if self.strip_whitespace:
                full_chunk = full_chunk.strip()
            if full_chunk:
                chunks.append(full_chunk)
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[str]:
        """
        Split text by characters when no other separators work.
        
        Args:
            text: Text to split.
            
        Returns:
            List of text chunks.
        """
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            start = i
            end = min(i + self.chunk_size, len(text))
            
            # Skip small final chunks
            if end - start <= self.chunk_overlap and i > 0 and end < len(text):
                continue
            
            chunk = text[start:end]
            if self.strip_whitespace:
                chunk = chunk.strip()
            if chunk:  # Don't add empty chunks
                chunks.append(chunk)
        
        return chunks


# Register the splitters
text_splitter_registry.register(SemanticTextSplitter())
text_splitter_registry.register(RecursiveSemanticSplitter()) 