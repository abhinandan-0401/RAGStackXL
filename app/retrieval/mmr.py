"""
Maximum Marginal Relevance retriever implementation for RAGStackXL.

This module provides a retriever that uses Maximum Marginal Relevance (MMR)
to balance relevance with diversity in the retrieved documents.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from app.retrieval.interfaces import Retriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument
from app.utils.logging import log


class MMRRetriever(Retriever):
    """
    Maximum Marginal Relevance (MMR) retriever.
    
    This retriever balances relevance with diversity by:
    1. First retrieving a larger set of relevant documents
    2. Then selecting a subset that maximizes diversity while maintaining relevance
    """
    
    def __init__(
        self,
        vectordb: VectorDB,
        embedding_model: EmbeddingModel,
        config: RetrieverConfig
    ):
        """
        Initialize the MMR retriever.
        
        Args:
            vectordb: Vector database to retrieve from
            embedding_model: Embedding model for queries
            config: Retriever configuration
        """
        super().__init__(vectordb, embedding_model, config)
        
        # Get retriever-specific configurations
        self.lambda_param = config.get("lambda_param", 0.7)  # Balance between relevance and diversity
        self.fetch_k = config.get("fetch_k", 50)  # Number of docs to fetch before MMR selection
        
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using Maximum Marginal Relevance.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides config)
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        log.info(f"Performing MMR retrieval for query: '{query}'")
        
        # Use provided k or fall back to config
        top_k = k if k is not None else self.config.top_k
        fetch_k = min(self.fetch_k, max(top_k * 3, 30))  # Fetch more docs than needed
        
        # Prepare metadata filter
        prepared_filter = self._prepare_filter(filter)
        
        # Embed the query
        query_embedding = await self.embedding_model.embed_query(query)
        
        # Get initial set of candidate documents
        candidates = await self.vectordb.similarity_search_with_score(
            query_embedding=query_embedding,
            k=fetch_k,
            filter=prepared_filter
        )
        
        # Extract documents and scores
        docs = [doc for doc, _ in candidates]
        scores = [score for _, score in candidates]
        
        # Get document embeddings - ideally this would be part of the vectordb API
        # but for now we'll re-embed the documents
        doc_contents = [doc.content for doc in docs]
        doc_embeddings = await self.embedding_model.embed_documents(doc_contents)
        
        # Perform MMR selection
        mmr_indices = self._mmr_selection(
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
            doc_scores=scores,
            k=top_k,
            lambda_param=self.lambda_param
        )
        
        # Create search results from MMR selection
        mmr_results = []
        for idx in mmr_indices:
            doc = docs[idx]
            mmr_score = scores[idx] * self.lambda_param + (1 - self.lambda_param) * 0.9  # Approximate diversity score
            mmr_results.append(SearchResult(
                document=doc,
                score=mmr_score,
                metadata=doc.metadata if self.config.include_metadata else None
            ))
        
        # Apply post-retrieval filter if configured
        if self.config.filter_fn is not None:
            mmr_results = [
                result for result in mmr_results
                if self.config.filter_fn(result.document)
            ]
        
        # Publish retrieval event
        self._publish_retrieval_event(query, mmr_results)
        
        log.info(f"Retrieved {len(mmr_results)} documents for query: '{query}'")
        return mmr_results
    
    def _mmr_selection(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        doc_scores: List[float],
        k: int,
        lambda_param: float
    ) -> List[int]:
        """
        Apply Maximum Marginal Relevance selection algorithm.
        
        Args:
            query_embedding: Embedding of the query
            doc_embeddings: List of document embeddings
            doc_scores: List of initial relevance scores
            k: Number of documents to select
            lambda_param: Balance between relevance and diversity (0-1)
                Higher values focus more on relevance, lower values on diversity
            
        Returns:
            Indices of selected documents
        """
        # Convert to numpy for efficient computation
        query_embedding = np.array(query_embedding)
        doc_embeddings = np.array(doc_embeddings)
        doc_scores = np.array(doc_scores)
        
        # Normalize scores to 0-1 range if they aren't already
        if np.max(doc_scores) > 1.0:
            doc_scores = doc_scores / np.max(doc_scores)
        
        # Initialize selected indices and remaining indices
        selected_indices = []
        remaining_indices = list(range(len(doc_embeddings)))
        
        # Select the first document (most relevant)
        first_idx = np.argmax(doc_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select the rest using MMR
        for _ in range(min(k - 1, len(remaining_indices))):
            # Compute similarity to query for all remaining documents
            query_similarity = doc_scores[remaining_indices]
            
            # Compute similarity to already selected documents
            doc_similarity = np.zeros(len(remaining_indices))
            
            for idx, remaining_idx in enumerate(remaining_indices):
                # Compute max similarity to any already selected document
                similarities = []
                for selected_idx in selected_indices:
                    # Cosine similarity between document embeddings
                    similarity = np.dot(
                        doc_embeddings[remaining_idx], 
                        doc_embeddings[selected_idx]
                    ) / (
                        np.linalg.norm(doc_embeddings[remaining_idx]) * 
                        np.linalg.norm(doc_embeddings[selected_idx])
                    )
                    similarities.append(similarity)
                
                doc_similarity[idx] = max(similarities) if similarities else 0.0
            
            # Compute MMR score for each remaining document
            mmr_scores = lambda_param * query_similarity - (1 - lambda_param) * doc_similarity
            
            # Select document with highest MMR score
            next_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        return selected_indices 