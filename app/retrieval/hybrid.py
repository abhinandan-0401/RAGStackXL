"""
Hybrid retriever implementation for RAGStackXL.

This module provides a hybrid retrieval mechanism that combines
vector-based semantic search with keyword-based search.
"""

from collections import defaultdict
import re
from typing import Dict, List, Any, Optional, Tuple, Set

from app.retrieval.interfaces import Retriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument
from app.utils.logging import log


class HybridRetriever(Retriever):
    """
    Hybrid retriever that combines vector search with keyword search.
    
    This retriever enhances retrieval by:
    1. Performing vector similarity search for semantic understanding
    2. Conducting keyword search based on important terms in the query
    3. Combining and reranking results from both approaches
    """
    
    def __init__(
        self,
        vectordb: VectorDB,
        embedding_model: EmbeddingModel,
        config: RetrieverConfig
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vectordb: Vector database to retrieve from
            embedding_model: Embedding model for queries
            config: Retriever configuration
        """
        super().__init__(vectordb, embedding_model, config)
        
        # Get retriever-specific configurations
        self.vector_weight = config.get("vector_weight", 0.7)
        self.keyword_weight = config.get("keyword_weight", 0.3)
        self.use_bm25 = config.get("use_bm25", True)
        self.min_keyword_length = config.get("min_keyword_length", 3)
        self.keyword_k = config.get("keyword_k", None)
        
        # Optional in-memory keyword index for testing/demo
        # In a real implementation, this would likely be replaced with a proper
        # search engine like Elasticsearch, Solr, or PyTerrier
        self._keyword_index = None
    
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides config)
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        log.info(f"Performing hybrid retrieval for query: '{query}'")
        
        # Use provided k or fall back to config
        top_k = k if k is not None else self.config.top_k
        
        # Prepare metadata filter
        prepared_filter = self._prepare_filter(filter)
        
        # Perform vector search
        vector_results = await self._perform_vector_search(query, top_k, prepared_filter)
        
        # Perform keyword search
        keyword_results = await self._perform_keyword_search(query, self.keyword_k or top_k, prepared_filter)
        
        # Combine and rerank results
        combined_results = self._combine_results(
            query=query,
            vector_results=vector_results,
            keyword_results=keyword_results,
            top_k=top_k
        )
        
        # Apply post-retrieval filter if configured
        if self.config.filter_fn is not None:
            combined_results = [
                result for result in combined_results
                if self.config.filter_fn(result.document)
            ]
        
        # Publish retrieval event
        self._publish_retrieval_event(query, combined_results)
        
        log.info(f"Retrieved {len(combined_results)} documents for query: '{query}'")
        return combined_results
    
    async def _perform_vector_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform vector-based similarity search.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        query_embedding = await self.embedding_model.embed_query(query)
        
        vector_results = await self.vectordb.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter=filter
        )
        
        return vector_results
    
    async def _perform_keyword_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform keyword-based search.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        # Build keyword search pattern
        pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        regex = re.compile(pattern, re.IGNORECASE)
        
        # Get all documents from the vector database
        # NOTE: In a production system, you would use a dedicated text search engine instead
        all_docs = []
        collection_stats = await self.vectordb.get_collection_stats()
        doc_count = collection_stats.get("count", 0)
        
        # Only do keyword search on reasonable number of documents
        if doc_count > 10000:
            log.warning(f"Too many documents ({doc_count}) for in-memory keyword search. Using vector search only.")
            return []
        
        # Create a naive keyword index if needed
        if self._keyword_index is None:
            log.info("Building in-memory keyword index for hybrid search")
            docs = []
            
            # This is inefficient but workable for demo purposes
            # In production, use a proper search engine
            async def fetch_documents():
                # This could be implemented by the VectorDB interface
                # For now, just return an empty list as a placeholder
                return []
            
            docs = await fetch_documents()
            
            self._keyword_index = defaultdict(list)
            for doc in docs:
                for word in re.findall(r'\b\w+\b', doc.content.lower()):
                    if len(word) >= self.min_keyword_length:
                        self._keyword_index[word].append(doc)
        
        # Search for keywords in documents
        doc_scores = defaultdict(float)
        
        for doc in all_docs:
            if filter:
                # Apply metadata filter if provided
                # This is a simplistic implementation for demo
                match = True
                for key, value in filter.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # Count keyword matches
            matches = len(regex.findall(doc.content.lower()))
            
            if matches > 0:
                if self.use_bm25:
                    # Simple BM25-inspired score
                    # Higher weight for multiple unique keywords, normalized by doc length
                    unique_matches = len(set(regex.findall(doc.content.lower())))
                    doc_len = len(doc.content.split())
                    doc_scores[doc.doc_id] = (unique_matches + 0.5 * matches) / (0.5 + 0.5 * doc_len / 500)
                else:
                    # Simple TF score
                    doc_scores[doc.doc_id] = matches
        
        # Create search results
        keyword_results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]:
            doc = next((d for d in all_docs if d.doc_id == doc_id), None)
            if doc:
                keyword_results.append(SearchResult(
                    document=doc,
                    score=min(score / 10, 1.0),  # Normalize score to 0-1 range
                    metadata=doc.metadata if self.config.include_metadata else None
                ))
        
        return keyword_results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from the query.
        
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
            if cleaned_word not in stopwords and len(cleaned_word) >= self.min_keyword_length:
                keywords.append(cleaned_word)
        
        return keywords
    
    def _combine_results(
        self,
        query: str,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Combine and rerank results from vector and keyword search.
        
        Args:
            query: Original query
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            top_k: Number of results to return
            
        Returns:
            Combined and reranked results
        """
        # Create a combined dictionary of document IDs to scores
        doc_scores = {}
        
        # Get unique document IDs
        all_doc_ids = set()
        
        # Process vector results
        for result in vector_results:
            doc_id = result.document.doc_id
            all_doc_ids.add(doc_id)
            doc_scores[doc_id] = {
                "document": result.document,
                "vector_score": result.score,
                "keyword_score": 0.0
            }
        
        # Process keyword results
        for result in keyword_results:
            doc_id = result.document.doc_id
            all_doc_ids.add(doc_id)
            if doc_id in doc_scores:
                doc_scores[doc_id]["keyword_score"] = result.score
            else:
                doc_scores[doc_id] = {
                    "document": result.document,
                    "vector_score": 0.0,
                    "keyword_score": result.score
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = (
                self.vector_weight * scores["vector_score"] +
                self.keyword_weight * scores["keyword_score"]
            )
            
            combined_results.append(SearchResult(
                document=scores["document"],
                score=combined_score,
                metadata=scores["document"].metadata if self.config.include_metadata else None
            ))
        
        # Sort by combined score and limit to top_k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k] 