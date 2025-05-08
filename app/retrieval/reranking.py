"""
Reranking retriever implementation for RAGStackXL.

This module provides a retriever that uses a two-stage process:
1. Initial retrieval of candidate documents
2. Reranking of candidates to improve relevance
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import asyncio
import numpy as np

from app.retrieval.interfaces import Retriever, RetrieverConfig, RetrieverType, Reranker
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument
from app.utils.logging import log


class SimpleScoreReranker:
    """
    Simple reranker that uses a scoring function to rerank documents.
    
    This reranker assigns new scores to documents based on a custom
    scoring function that considers the document content and query.
    """
    
    def __init__(self, scoring_fn: Optional[Callable] = None):
        """
        Initialize the simple score reranker.
        
        Args:
            scoring_fn: Optional scoring function to use for reranking
        """
        self.scoring_fn = scoring_fn or self._default_score_fn
    
    async def rerank(
        self, 
        query: str,
        documents: List[RagDocument],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[RagDocument, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Query string
            documents: List of retrieved documents
            scores: Optional original relevance scores
            
        Returns:
            Reranked list of (document, score) tuples
        """
        reranked = []
        
        for i, doc in enumerate(documents):
            original_score = scores[i] if scores is not None and i < len(scores) else 0.5
            new_score = self.scoring_fn(query, doc, original_score)
            reranked.append((doc, new_score))
        
        # Sort by new score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def _default_score_fn(
        self, 
        query: str, 
        document: RagDocument, 
        original_score: float
    ) -> float:
        """
        Default scoring function that combines:
        - Original score
        - Keyword matching (TF)
        - Position of matches
        
        Args:
            query: Query string
            document: Document to score
            original_score: Original relevance score
            
        Returns:
            New relevance score
        """
        content = document.content.lower()
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        
        if not query_terms:
            return original_score
        
        # Term frequency component
        term_counts = {}
        for term in query_terms:
            # Simple term frequency counting
            count = content.count(term)
            term_counts[term] = count
        
        # Average term frequency normalized by document length
        doc_len = len(content.split())
        avg_tf = sum(term_counts.values()) / len(query_terms) / max(1, doc_len / 500)
        
        # Position component - check if terms appear early in the document
        position_score = 0.0
        first_200_chars = content[:200].lower()
        for term in query_terms:
            if term in first_200_chars:
                position_score += 0.05
        
        # Coverage component - what percentage of terms are found
        terms_found = sum(1 for term in query_terms if term in content)
        coverage = terms_found / len(query_terms)
        
        # Combine components
        new_score = (
            0.6 * original_score +  # Original score weight
            0.2 * min(avg_tf, 1.0) +  # Term frequency weight
            0.1 * position_score +  # Position weight
            0.1 * coverage  # Coverage weight
        )
        
        return min(new_score, 1.0)  # Ensure score is at most 1.0


class CosineSimilarityReranker:
    """
    Reranker that uses cosine similarity between query and document embeddings.
    
    This reranker computes new similarity scores based on cosine similarity
    between query and document embeddings.
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize the cosine similarity reranker.
        
        Args:
            embedding_model: Embedding model to use for embeddings
        """
        self.embedding_model = embedding_model
    
    async def rerank(
        self, 
        query: str,
        documents: List[RagDocument],
        scores: Optional[List[float]] = None
    ) -> List[Tuple[RagDocument, float]]:
        """
        Rerank documents using cosine similarity.
        
        Args:
            query: Query string
            documents: List of retrieved documents
            scores: Optional original relevance scores
            
        Returns:
            Reranked list of (document, score) tuples
        """
        if not documents:
            return []
        
        # Get query embedding
        query_embedding = await self.embedding_model.embed_query(query)
        
        # Get document embeddings
        doc_contents = [doc.content for doc in documents]
        doc_embeddings = await self.embedding_model.embed_documents(doc_contents)
        
        # Compute cosine similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            # Convert to numpy arrays for efficient computation
            query_arr = np.array(query_embedding)
            doc_arr = np.array(doc_emb)
            
            # Compute cosine similarity
            sim = np.dot(query_arr, doc_arr) / (
                np.linalg.norm(query_arr) * np.linalg.norm(doc_arr)
            )
            
            # Blend with original score if available
            if scores is not None and i < len(scores):
                sim = 0.7 * sim + 0.3 * scores[i]
            
            similarities.append((documents[i], float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities


class RerankingRetriever(Retriever):
    """
    Reranking retriever that uses a two-stage retrieval process.
    
    This retriever:
    1. First retrieves candidate documents using vector search
    2. Then reranks these candidates using a more sophisticated reranker
    """
    
    def __init__(
        self,
        vectordb: VectorDB,
        embedding_model: EmbeddingModel,
        config: RetrieverConfig
    ):
        """
        Initialize the reranking retriever.
        
        Args:
            vectordb: Vector database to retrieve from
            embedding_model: Embedding model for queries
            config: Retriever configuration
        """
        super().__init__(vectordb, embedding_model, config)
        
        # Get retriever-specific configurations
        self.initial_k = config.get("initial_k", 20)  # Number of initial candidates to retrieve
        self.reranker_type = config.get("reranker_type", "simple")
        
        # Create appropriate reranker
        if self.reranker_type == "cosine":
            self.reranker = CosineSimilarityReranker(embedding_model)
        else:  # "simple" or any other value
            self.reranker = SimpleScoreReranker()
    
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using a two-stage process.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides config)
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        log.info(f"Performing reranking retrieval for query: '{query}'")
        
        # Use provided k or fall back to config
        top_k = k if k is not None else self.config.top_k
        
        # Get more candidates than needed for reranking
        initial_k = max(self.initial_k, top_k * 3)
        
        # Prepare metadata filter
        prepared_filter = self._prepare_filter(filter)
        
        # Step 1: Initial retrieval of candidates
        query_embedding = await self.embedding_model.embed_query(query)
        initial_results = await self.vectordb.similarity_search_with_score(
            query_embedding=query_embedding,
            k=initial_k,
            filter=prepared_filter
        )
        
        # Extract documents and scores
        candidate_docs = [doc for doc, _ in initial_results]
        candidate_scores = [score for _, score in initial_results]
        
        log.info(f"Retrieved {len(candidate_docs)} initial candidates for reranking")
        
        # Step 2: Rerank candidates
        reranked_results = await self.reranker.rerank(
            query=query,
            documents=candidate_docs,
            scores=candidate_scores
        )
        
        # Convert to search results
        reranked_search_results = []
        for doc, score in reranked_results[:top_k]:
            reranked_search_results.append(SearchResult(
                document=doc,
                score=score,
                metadata=doc.metadata if self.config.include_metadata else None
            ))
        
        # Apply post-retrieval filter if configured
        if self.config.filter_fn is not None:
            reranked_search_results = [
                result for result in reranked_search_results
                if self.config.filter_fn(result.document)
            ]
        
        # Publish retrieval event
        self._publish_retrieval_event(query, reranked_search_results)
        
        log.info(f"Retrieved and reranked {len(reranked_search_results)} documents for query: '{query}'")
        return reranked_search_results 