"""
Contextual retriever implementation for RAGStackXL.

This module provides a retriever that takes into account additional
context (like conversation history) to improve retrieval results.
"""

from typing import Dict, List, Any, Optional, Tuple, Sequence, Union
import asyncio

from app.retrieval.interfaces import Retriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument
from app.utils.logging import log


class ContextualRetriever(Retriever):
    """
    Contextual retriever that considers conversation history or user context.
    
    This retriever enhances results by:
    1. Taking into account previous interactions and context
    2. Combining current query with relevant context information
    3. Weighting document relevance based on contextual factors
    """
    
    def __init__(
        self,
        vectordb: VectorDB,
        embedding_model: EmbeddingModel,
        config: RetrieverConfig
    ):
        """
        Initialize the contextual retriever.
        
        Args:
            vectordb: Vector database to retrieve from
            embedding_model: Embedding model for queries
            config: Retriever configuration
        """
        super().__init__(vectordb, embedding_model, config)
        
        # Get retriever-specific configurations
        self.context_window_size = config.get("context_window_size", 3)
        self.current_context_weight = config.get("current_context_weight", 0.8)
        self.history_decay_factor = config.get("history_decay_factor", 0.7)
        self.context_strategy = config.get("context_strategy", "combine")
        
        # Initialize conversation history
        self.conversation_history = []
    
    def add_to_history(
        self, 
        text: str, 
        is_user: bool = True, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a message to the conversation history.
        
        Args:
            text: Message text
            is_user: Whether the message is from the user (vs. system)
            metadata: Optional message metadata
        """
        self.conversation_history.append({
            "text": text,
            "is_user": is_user,
            "metadata": metadata or {}
        })
        
        # Optionally trim history to keep only recent messages
        if len(self.conversation_history) > 20:  # Arbitrary limit
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        context: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using contextual information.
        
        Args:
            query: Query string
            k: Number of documents to retrieve (overrides config)
            filter: Optional metadata filter
            context: Optional additional context to consider
                (if provided, this overrides the conversation history)
            
        Returns:
            List of search results
        """
        log.info(f"Performing contextual retrieval for query: '{query}'")
        
        # Use provided k or fall back to config
        top_k = k if k is not None else self.config.top_k
        
        # Prepare metadata filter
        prepared_filter = self._prepare_filter(filter)
        
        # Process context
        if context is not None:
            prepared_context = self._prepare_external_context(context)
        else:
            prepared_context = self._prepare_history_context()
        
        # Choose retrieval strategy based on configuration
        if self.context_strategy == "separate":
            results = await self._retrieve_separate(
                query=query, 
                context=prepared_context,
                k=top_k,
                filter=prepared_filter
            )
        else:  # "combine" or any other value
            results = await self._retrieve_combined(
                query=query, 
                context=prepared_context,
                k=top_k,
                filter=prepared_filter
            )
        
        # Apply post-retrieval filter if configured
        if self.config.filter_fn is not None:
            results = [
                result for result in results
                if self.config.filter_fn(result.document)
            ]
        
        # Add query to history if using internal history
        if context is None:
            self.add_to_history(query, is_user=True)
        
        # Publish retrieval event
        self._publish_retrieval_event(query, results)
        
        log.info(f"Retrieved {len(results)} documents for query: '{query}'")
        return results
    
    def _prepare_external_context(
        self, 
        context: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    ) -> str:
        """
        Prepare external context for retrieval.
        
        Args:
            context: Context information (string or structured data)
            
        Returns:
            Prepared context string
        """
        if isinstance(context, str):
            return context
        
        if isinstance(context, dict):
            # Extract text from a dictionary
            if "text" in context:
                return context["text"]
            # Fallback: combine all string values
            return " ".join(str(v) for v in context.values() if isinstance(v, str))
        
        if isinstance(context, list):
            # Handle list of messages/dictionaries
            texts = []
            for item in context:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    # For chat-like structures
                    prefix = "User: " if item.get("is_user", True) else "System: "
                    texts.append(prefix + item["text"])
            return " ".join(texts)
        
        # Fallback for unexpected types
        return str(context)
    
    def _prepare_history_context(self) -> str:
        """
        Prepare context from conversation history.
        
        Returns:
            Context string from recent conversation history
        """
        if not self.conversation_history:
            return ""
        
        # Take the most recent messages
        recent_history = self.conversation_history[-self.context_window_size:]
        
        # Combine messages into a single context string
        context_parts = []
        for message in recent_history:
            prefix = "User: " if message["is_user"] else "System: "
            context_parts.append(prefix + message["text"])
        
        return " ".join(context_parts)
    
    async def _retrieve_combined(
        self,
        query: str,
        context: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve using combined query and context.
        
        Args:
            query: Query string
            context: Context string
            k: Number of results to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        # If no context, fall back to basic retrieval
        if not context:
            return await self._basic_retrieve(query, k, filter)
        
        # Combine query with context
        combined_query = f"{query} {context}"
        
        # Embed the combined query
        combined_embedding = await self.embedding_model.embed_query(combined_query)
        
        # Retrieve documents
        results = await self.vectordb.similarity_search(
            query_embedding=combined_embedding,
            k=k,
            filter=filter
        )
        
        return results
    
    async def _retrieve_separate(
        self,
        query: str,
        context: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve separately for query and context, then combine results.
        
        Args:
            query: Query string
            context: Context string
            k: Number of results to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        # If no context, fall back to basic retrieval
        if not context:
            return await self._basic_retrieve(query, k, filter)
        
        # Embed the query and context separately
        query_embedding = await self.embedding_model.embed_query(query)
        
        # Only embed context if it exists
        context_embedding = await self.embedding_model.embed_query(context)
        
        # Retrieve documents for query
        query_results = await self.vectordb.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter=filter
        )
        
        # Retrieve documents for context
        context_results = await self.vectordb.similarity_search(
            query_embedding=context_embedding,
            k=k,
            filter=filter
        )
        
        # Combine and deduplicate results
        combined_results = self._combine_separate_results(
            query_results=query_results,
            context_results=context_results,
            top_k=k
        )
        
        return combined_results
    
    async def _basic_retrieve(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform basic retrieval without context.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        # Embed the query
        query_embedding = await self.embedding_model.embed_query(query)
        
        # Retrieve documents
        results = await self.vectordb.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter=filter
        )
        
        return results
    
    def _combine_separate_results(
        self,
        query_results: List[SearchResult],
        context_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Combine results from query and context searches.
        
        Args:
            query_results: Results from query search
            context_results: Results from context search
            top_k: Number of results to return
            
        Returns:
            Combined and reranked results
        """
        # Create a dictionary of document IDs to combined scores
        doc_scores = {}
        
        # Process query results (with higher weight)
        for result in query_results:
            doc_id = result.document.doc_id
            doc_scores[doc_id] = {
                "document": result.document,
                "score": result.score * self.current_context_weight,
                "in_query": True,
                "in_context": False
            }
        
        # Process context results
        for result in context_results:
            doc_id = result.document.doc_id
            context_score = result.score * (1 - self.current_context_weight)
            
            if doc_id in doc_scores:
                # Document appears in both query and context results
                doc_scores[doc_id]["score"] += context_score
                doc_scores[doc_id]["in_context"] = True
            else:
                # Document appears only in context results
                doc_scores[doc_id] = {
                    "document": result.document,
                    "score": context_score,
                    "in_query": False,
                    "in_context": True
                }
        
        # Convert to search results
        combined_results = []
        for doc_id, info in doc_scores.items():
            # Boost documents that appear in both query and context
            if info["in_query"] and info["in_context"]:
                # Add a small boost (up to 10%) for documents in both
                boost = min(0.1, info["score"] * 0.2)
                info["score"] += boost
            
            combined_results.append(SearchResult(
                document=info["document"],
                score=info["score"],
                metadata=info["document"].metadata if self.config.include_metadata else None
            ))
        
        # Sort by score and limit to top_k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k] 