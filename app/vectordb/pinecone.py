"""
Pinecone integration for RAGStackXL.

Pinecone is a managed vector database service optimized for machine learning applications.
It provides high performance vector search with strong consistency guarantees.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from functools import partial

import pinecone
from pinecone import Pinecone, ServerlessSpec

from app.core.interfaces import RagDocument, DocumentMetadata, Event, EventType, event_bus
from app.vectordb.interfaces import VectorDB, VectorDBConfig, VectorDBProvider, VectorDBFactory, SearchResult
from app.utils.logging import log


class PineconeVectorDB(VectorDB):
    """Pinecone implementation of the VectorDB interface."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize the Pinecone vector database."""
        super().__init__(config)
        
        # Get Pinecone-specific configuration
        self.api_key = config.get("api_key")
        self.environment = config.get("environment")
        self.serverless = config.get("serverless", False)
        self.server_side_timeout = config.get("server_side_timeout", 20)
        self.dimension = config.embedding_dimension
        self.metric = self._translate_distance_metric(config.distance_metric)
        
        # Initialize Pinecone client
        self._init_pinecone()
        
        # Initialize or connect to index
        self._init_index()
        
        log.info(f"Pinecone index '{self.collection_name}' initialized with dimension {self.dimension}")
    
    def _translate_distance_metric(self, metric: str) -> str:
        """Translate our internal distance metric names to Pinecone's."""
        metric = metric.lower()
        
        if metric == "cosine":
            return "cosine"
        elif metric in ["euclidean", "l2"]:
            return "euclidean"
        elif metric in ["dot", "inner_product"]:
            return "dotproduct"
        else:
            log.warning(f"Unsupported distance metric for Pinecone: {metric}, defaulting to cosine")
            return "cosine"
    
    def _init_pinecone(self):
        """Initialize Pinecone client."""
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        log.info("Pinecone client initialized")
    
    def _init_index(self):
        """Initialize or connect to Pinecone index."""
        # List existing indexes
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        # Check if index exists
        if self.collection_name in existing_indexes:
            log.info(f"Connected to existing Pinecone index: {self.collection_name}")
        else:
            log.info(f"Creating new Pinecone index: {self.collection_name}")
            
            # Create index spec
            if self.serverless:
                # Serverless index
                spec = ServerlessSpec(cloud="aws", region="us-west-2")
                self.pc.create_index(
                    name=self.collection_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=spec
                )
            else:
                # Standard index
                self.pc.create_index(
                    name=self.collection_name,
                    dimension=self.dimension,
                    metric=self.metric
                )
        
        # Connect to index
        self.index = self.pc.Index(self.collection_name)
    
    async def add_documents(
        self,
        documents: List[RagDocument],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the Pinecone index."""
        if not documents:
            return []
        
        if not self._validate_embeddings(embeddings):
            raise ValueError("Invalid embeddings")
        
        if len(documents) != len(embeddings):
            raise ValueError(f"Number of documents ({len(documents)}) does not match number of embeddings ({len(embeddings)})")
        
        # Prepare document IDs
        doc_ids = []
        for i, doc in enumerate(documents):
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            elif doc.doc_id:
                doc_id = doc.doc_id
            else:
                doc_id = self._generate_doc_id()
            
            doc_ids.append(doc_id)
        
        # Prepare vectors for batch upsert
        vectors_to_upsert = []
        for i, (doc, doc_id, embedding) in enumerate(zip(documents, doc_ids, embeddings)):
            # Extract metadata and ensure it's serializable
            metadata = {
                "text": doc.content,
                "doc_id": doc_id
            }
            
            # Add all metadata fields if they exist
            if doc.metadata:
                for k, v in doc.metadata.items():
                    # Skip non-serializable values or convert them to strings
                    if isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)):
                        metadata[k] = v
                    else:
                        metadata[k] = str(v)
            
            # Create vector record
            vector = {
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            }
            
            vectors_to_upsert.append(vector)
        
        # Use asyncio to upload vectors in the background
        loop = asyncio.get_event_loop()
        
        # Define function for batch upsert
        async def _upsert_batch(vectors):
            def _inner_upsert():
                # Upsert in batches of 100 (Pinecone's recommended batch size)
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    # Handle errors during upsert
                    try:
                        self.index.upsert(vectors=batch, namespace="")
                    except Exception as e:
                        log.error(f"Error upserting batch to Pinecone: {e}")
                        raise e
                
                return True
            
            # Run the synchronous operation in a thread pool
            return await loop.run_in_executor(None, _inner_upsert)
        
        # Upsert vectors
        try:
            await _upsert_batch(vectors_to_upsert)
            
            # Notify that documents were added
            self._notify_documents_added(doc_ids)
            
            return doc_ids
        except Exception as e:
            log.error(f"Failed to add documents to Pinecone: {e}")
            return []
    
    async def get_document(self, doc_id: str) -> Optional[RagDocument]:
        """Get a document by ID from Pinecone."""
        loop = asyncio.get_event_loop()
        
        async def _fetch_document():
            def _inner_fetch():
                try:
                    # Fetch vector from Pinecone
                    response = self.index.fetch(ids=[doc_id])
                    
                    # Check if document exists
                    if not response.vectors or doc_id not in response.vectors:
                        return None
                    
                    # Extract vector data
                    vector_data = response.vectors[doc_id]
                    metadata = vector_data.metadata
                    
                    # Extract content from metadata
                    content = metadata.get("text", "")
                    
                    # Create metadata object
                    doc_metadata = {k: v for k, v in metadata.items() if k != "text" and k != "doc_id"}
                    
                    # Create and return document
                    return RagDocument(
                        doc_id=doc_id,
                        content=content,
                        metadata=doc_metadata
                    )
                except Exception as e:
                    log.error(f"Error fetching document {doc_id} from Pinecone: {e}")
                    return None
            
            return await loop.run_in_executor(None, _inner_fetch)
        
        return await _fetch_document()
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID from Pinecone."""
        loop = asyncio.get_event_loop()
        
        async def _delete_document():
            def _inner_delete():
                try:
                    # Delete vector from Pinecone
                    self.index.delete(ids=[doc_id])
                    return True
                except Exception as e:
                    log.error(f"Error deleting document {doc_id} from Pinecone: {e}")
                    return False
            
            return await loop.run_in_executor(None, _inner_delete)
        
        return await _delete_document()
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform a similarity search in Pinecone."""
        results = await self.similarity_search_with_score(
            query_embedding=query_embedding,
            k=k,
            filter=filter
        )
        
        return [
            SearchResult(document=doc, score=score)
            for doc, score in results
        ]
    
    async def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[RagDocument, float]]:
        """Perform a similarity search with scores in Pinecone."""
        loop = asyncio.get_event_loop()
        
        async def _search():
            def _inner_search():
                try:
                    # Convert filter to Pinecone filter format if provided
                    pinecone_filter = self._convert_to_pinecone_filter(filter) if filter else None
                    
                    # Query Pinecone
                    query_response = self.index.query(
                        vector=query_embedding,
                        top_k=k,
                        include_metadata=True,
                        filter=pinecone_filter,
                        namespace=""
                    )
                    
                    # Process results
                    results = []
                    for match in query_response.matches:
                        # Extract metadata
                        metadata = match.metadata
                        
                        # Extract content from metadata
                        content = metadata.get("text", "")
                        
                        # Create metadata object (excluding text field)
                        doc_metadata = {k: v for k, v in metadata.items() if k != "text" and k != "doc_id"}
                        
                        # Create document
                        doc = RagDocument(
                            doc_id=match.id,
                            content=content,
                            metadata=doc_metadata
                        )
                        
                        # Add to results
                        results.append((doc, match.score))
                    
                    return results
                except Exception as e:
                    log.error(f"Error performing similarity search in Pinecone: {e}")
                    return []
            
            return await loop.run_in_executor(None, _inner_search)
        
        return await _search()
    
    def _convert_to_pinecone_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Convert our filter format to Pinecone's filter format."""
        if not filter:
            return {}
        
        pinecone_filter = {}
        
        for key, value in filter.items():
            if isinstance(value, list):
                # For lists, we use $in operator
                pinecone_filter[key] = {"$in": value}
            else:
                # For single values, we use exact match
                pinecone_filter[key] = value
        
        return pinecone_filter
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone collection."""
        loop = asyncio.get_event_loop()
        
        async def _get_stats():
            def _inner_stats():
                try:
                    # Get index stats
                    stats = self.index.describe_index_stats()
                    
                    # Extract relevant information
                    return {
                        "count": stats.total_vector_count,
                        "dimension": self.dimension,
                        "distance_metric": self.config.distance_metric,
                        "namespaces": stats.namespaces
                    }
                except Exception as e:
                    log.error(f"Error getting collection stats from Pinecone: {e}")
                    return {
                        "count": 0,
                        "dimension": self.dimension,
                        "distance_metric": self.config.distance_metric
                    }
            
            return await loop.run_in_executor(None, _inner_stats)
        
        return await _get_stats()
    
    async def clear_collection(self) -> bool:
        """Clear the Pinecone collection."""
        loop = asyncio.get_event_loop()
        
        async def _clear_collection():
            def _inner_clear():
                try:
                    # Delete all vectors from the index
                    self.index.delete(delete_all=True, namespace="")
                    return True
                except Exception as e:
                    log.error(f"Error clearing Pinecone collection: {e}")
                    return False
            
            return await loop.run_in_executor(None, _inner_clear)
        
        return await _clear_collection()
    
    async def persist(self) -> bool:
        """Persist the database to disk (not applicable for Pinecone)."""
        # Pinecone is a managed service and data is automatically persisted
        return True
    
    def _notify_documents_added(self, doc_ids: List[str]) -> None:
        """Notify that documents were added to the database."""
        # Create event
        event = Event(
            event_type=EventType.DOCUMENT_ADDED,
            payload={"doc_ids": doc_ids, "collection_name": self.collection_name}
        )
        # Publish event
        event_bus.publish(event)


# Register Pinecone with the factory
VectorDBFactory.register(VectorDBProvider.PINECONE, PineconeVectorDB) 