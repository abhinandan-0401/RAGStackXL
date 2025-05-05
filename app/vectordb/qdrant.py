"""
Qdrant integration for RAGStackXL.

Qdrant is a vector database that provides a good balance between
performance and simplicity, making it suitable for standard applications.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.interfaces import RagDocument, DocumentMetadata
from app.vectordb.interfaces import VectorDB, VectorDBConfig, VectorDBProvider, VectorDBFactory, SearchResult
from app.utils.logging import log


class QdrantVectorDB(VectorDB):
    """Qdrant implementation of the VectorDB interface."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize the Qdrant vector database."""
        super().__init__(config)
        
        # Configure Qdrant client
        location = config.get("location")
        url = config.get("url")
        api_key = config.get("api_key")
        
        self.client = QdrantClient(
            location=location,
            url=url,
            api_key=api_key,
            prefer_grpc=config.get("prefer_grpc", True)
        )
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
        log.info(f"Qdrant collection '{self.collection_name}' ready")
    
    def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            log.info(f"Collection '{self.collection_name}' already exists")
        except (UnexpectedResponse, ValueError):
            # Collection doesn't exist, create it
            log.info(f"Creating collection '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.embedding_dimension,
                    distance=self._get_qdrant_distance_metric()
                )
            )
    
    def _get_qdrant_distance_metric(self) -> models.Distance:
        """Convert config distance metric to Qdrant distance metric."""
        metric = self.config.distance_metric.lower()
        if metric == "cosine":
            return models.Distance.COSINE
        elif metric == "euclidean":
            return models.Distance.EUCLID
        elif metric == "dot":
            return models.Distance.DOT
        else:
            log.warning(f"Unsupported distance metric: {metric}, defaulting to cosine")
            return models.Distance.COSINE
    
    async def add_documents(
        self,
        documents: List[RagDocument],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the Qdrant collection."""
        if not documents:
            return []
        
        if not self._validate_embeddings(embeddings):
            raise ValueError("Invalid embeddings")
        
        if len(documents) != len(embeddings):
            raise ValueError(f"Number of documents ({len(documents)}) does not match number of embeddings ({len(embeddings)})")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_doc_id() for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError(f"Number of IDs ({len(ids)}) does not match number of documents ({len(documents)})")
        
        # Prepare points for Qdrant
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Convert string IDs to UUIDs for Qdrant
            doc_id = ids[i]
            
            # Create point with document content and metadata
            point = models.PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "content": doc.content,
                    **(doc.metadata or {})
                }
            )
            points.append(point)
        
        # Add points to Qdrant
        loop = asyncio.get_event_loop()
        
        def _add_points():
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                return ids
            except Exception as e:
                log.error(f"Error adding documents to Qdrant: {e}")
                return []
        
        doc_ids = await loop.run_in_executor(None, _add_points)
        
        if doc_ids:
            # Notify that documents were added
            self._notify_documents_added(doc_ids)
        
        return doc_ids
    
    async def get_document(self, doc_id: str) -> Optional[RagDocument]:
        """Get a document by ID from Qdrant."""
        loop = asyncio.get_event_loop()
        
        def _get_point():
            try:
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[doc_id]
                )
                
                if not points:
                    return None
                
                point = points[0]
                payload = point.payload
                
                # Extract content and metadata
                content = payload.pop("content", "")
                metadata = payload
                
                return RagDocument(
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id
                )
            except Exception as e:
                log.error(f"Error getting document {doc_id} from Qdrant: {e}")
                return None
        
        return await loop.run_in_executor(None, _get_point)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID from Qdrant."""
        loop = asyncio.get_event_loop()
        
        def _delete_point():
            try:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=[doc_id]
                    )
                )
                return True
            except Exception as e:
                log.error(f"Error deleting document {doc_id} from Qdrant: {e}")
                return False
        
        return await loop.run_in_executor(None, _delete_point)
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform a similarity search in Qdrant."""
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
        """Perform a similarity search with scores in Qdrant."""
        loop = asyncio.get_event_loop()
        
        def _search():
            try:
                # Convert filter to Qdrant filter format if provided
                qdrant_filter = None
                if filter:
                    qdrant_filter = self._convert_to_qdrant_filter(filter)
                
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=k,
                    query_filter=qdrant_filter,
                    with_payload=True
                )
                
                results = []
                for scored_point in search_result:
                    point_id = str(scored_point.id)
                    score = scored_point.score
                    payload = scored_point.payload
                    
                    # Extract content and metadata
                    content = payload.pop("content", "")
                    metadata = payload
                    
                    doc = RagDocument(
                        content=content,
                        metadata=metadata,
                        doc_id=point_id
                    )
                    
                    results.append((doc, score))
                
                return results
            except Exception as e:
                log.error(f"Error in similarity search with Qdrant: {e}")
                return []
        
        return await loop.run_in_executor(None, _search)
    
    def _convert_to_qdrant_filter(self, filter: Dict[str, Any]) -> models.Filter:
        """Convert a simple filter dict to Qdrant filter format."""
        conditions = []
        
        for key, value in filter.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                )
            elif isinstance(value, dict):
                # Handle range conditions
                if "gt" in value or "gte" in value or "lt" in value or "lte" in value:
                    range_params = {}
                    if "gt" in value:
                        range_params["gt"] = value["gt"]
                    if "gte" in value:
                        range_params["gte"] = value["gte"]
                    if "lt" in value:
                        range_params["lt"] = value["lt"]
                    if "lte" in value:
                        range_params["lte"] = value["lte"]
                    
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            range=models.Range(**range_params)
                        )
                    )
                else:
                    # Nested filter not supported in this simple implementation
                    log.warning(f"Nested filter for {key} not supported, ignoring")
            else:
                # Match exact value
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(
            must=conditions
        )
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection."""
        loop = asyncio.get_event_loop()
        
        def _get_stats():
            try:
                collection_info = self.client.get_collection(self.collection_name)
                
                return {
                    "name": self.collection_name,
                    "vector_count": collection_info.vectors_count,
                    "dimension": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance,
                    "status": collection_info.status
                }
            except Exception as e:
                log.error(f"Error getting collection stats from Qdrant: {e}")
                return {"error": str(e)}
        
        return await loop.run_in_executor(None, _get_stats)
    
    async def clear_collection(self) -> bool:
        """Clear the Qdrant collection."""
        loop = asyncio.get_event_loop()
        
        def _clear():
            try:
                # Delete all points in the collection
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter()
                    )
                )
                return True
            except Exception as e:
                log.error(f"Error clearing Qdrant collection: {e}")
                return False
        
        return await loop.run_in_executor(None, _clear)
    
    async def persist(self) -> bool:
        """Persist the Qdrant database to disk."""
        # Qdrant automatically persists data, this is a no-op
        return True


# Register Qdrant with the factory
VectorDBFactory.register(VectorDBProvider.QDRANT, QdrantVectorDB) 