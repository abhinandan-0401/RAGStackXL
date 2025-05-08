"""
Basic retriever implementation for RAGStackXL.

This module provides a simple vector similarity search retriever.
"""

from typing import Dict, List, Any, Optional

from app.retrieval.interfaces import Retriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.utils.logging import log


class BasicRetriever(Retriever):
    """
    Basic retriever that performs simple vector similarity search.
    
    This retriever embeds the query and uses vector similarity search
    to find relevant documents in the vector database.
    """
    
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using vector similarity search.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides config)
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        log.info(f"Retrieving documents for query: '{query}'")
        
        # Use provided k or fall back to config
        top_k = k if k is not None else self.config.top_k
        
        # Prepare metadata filter
        prepared_filter = self._prepare_filter(filter)
        
        # Embed the query
        query_embedding = await self.embedding_model.embed_query(query)
        
        # Perform similarity search
        search_results = await self.vectordb.similarity_search(
            query_embedding=query_embedding,
            k=top_k,
            filter=prepared_filter
        )
        
        # Apply post-retrieval filter if configured
        if self.config.filter_fn is not None:
            search_results = [
                result for result in search_results
                if self.config.filter_fn(result.document)
            ]
        
        # Publish retrieval event
        self._publish_retrieval_event(query, search_results)
        
        log.info(f"Retrieved {len(search_results)} documents for query: '{query}'")
        return search_results 