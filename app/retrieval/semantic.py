"""
Semantic retriever implementation for RAGStackXL.

This module provides a more advanced semantic retrieval mechanism.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from app.retrieval.interfaces import Retriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument
from app.utils.logging import log


class SemanticRetriever(Retriever):
    """
    Semantic retriever with advanced query understanding.
    
    This retriever enhances the basic retrieval process by:
    1. Analyzing the query for important keywords and concepts
    2. Potentially applying lightweight NLP techniques
    3. Using more sophisticated similarity calculations
    """
    
    def __init__(
        self,
        vectordb: VectorDB,
        embedding_model: EmbeddingModel,
        config: RetrieverConfig
    ):
        """
        Initialize the semantic retriever.
        
        Args:
            vectordb: Vector database to retrieve from
            embedding_model: Embedding model for queries
            config: Retriever configuration
        """
        super().__init__(vectordb, embedding_model, config)
        
        # Get retriever-specific configurations
        self.relevance_threshold = config.get("relevance_threshold", 0.0)
        self.use_keyword_boost = config.get("use_keyword_boost", True)
        self.keyword_boost_factor = config.get("keyword_boost_factor", 0.1)
        self.semantic_weight = config.get("semantic_weight", 0.85)
    
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using semantic search.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides config)
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        log.info(f"Performing semantic retrieval for query: '{query}'")
        
        # Use provided k or fall back to config with a margin for post-filtering
        top_k = k if k is not None else self.config.top_k
        search_k = int(top_k * 1.5)  # Retrieve more docs than needed for re-ranking
        
        # Prepare metadata filter
        prepared_filter = self._prepare_filter(filter)
        
        # Extract keywords for potential boosting
        keywords = self._extract_keywords(query) if self.use_keyword_boost else []
        
        # Embed the query
        query_embedding = await self.embedding_model.embed_query(query)
        
        # Perform similarity search with scores
        search_tuples = await self.vectordb.similarity_search_with_score(
            query_embedding=query_embedding,
            k=search_k,
            filter=prepared_filter
        )
        
        # Apply semantic adjustments
        search_results = self._apply_semantic_adjustments(
            query=query,
            search_tuples=search_tuples,
            keywords=keywords
        )
        
        # Filter by relevance threshold
        if self.relevance_threshold > 0:
            search_results = [
                result for result in search_results
                if result.score >= self.relevance_threshold
            ]
        
        # Apply post-retrieval filter if configured
        if self.config.filter_fn is not None:
            search_results = [
                result for result in search_results
                if self.config.filter_fn(result.document)
            ]
        
        # Truncate to requested k
        search_results = search_results[:top_k]
        
        # Publish retrieval event
        self._publish_retrieval_event(query, search_results)
        
        log.info(f"Retrieved {len(search_results)} documents for query: '{query}'")
        return search_results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from the query.
        
        Args:
            query: Query string
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - remove stopwords and get unique terms
        stopwords = {"a", "an", "the", "and", "or", "but", "is", "are", "in", 
                     "to", "of", "for", "with", "on", "at", "from", "by",
                     "how", "what", "why", "when", "who", "where", "which", "do", "does", "they"}
        
        # Strip punctuation and split
        words = query.lower().split()
        keywords = []
        
        for word in words:
            # Strip punctuation from the end of words
            cleaned_word = word.rstrip('?.,!:;')
            if cleaned_word not in stopwords and len(cleaned_word) > 2:
                keywords.append(cleaned_word)
        
        return keywords
    
    def _apply_semantic_adjustments(
        self,
        query: str,
        search_tuples: List[Tuple[RagDocument, float]],
        keywords: List[str]
    ) -> List[SearchResult]:
        """
        Apply semantic adjustments to search results.
        
        Args:
            query: Original query
            search_tuples: List of (document, score) tuples
            keywords: Extracted keywords
            
        Returns:
            List of adjusted search results
        """
        results = []
        
        for doc, score in search_tuples:
            adjusted_score = score * self.semantic_weight
            
            # Apply keyword boosting if enabled
            if self.use_keyword_boost and keywords:
                content_lower = doc.content.lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
                keyword_boost = (keyword_matches / len(keywords)) * self.keyword_boost_factor
                adjusted_score += keyword_boost
            
            # Normalize score to ensure it's between 0 and 1
            adjusted_score = min(max(adjusted_score, 0.0), 1.0)
            
            results.append(SearchResult(
                document=doc,
                score=adjusted_score,
                metadata=doc.metadata if self.config.include_metadata else None
            ))
        
        # Sort by adjusted score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results 