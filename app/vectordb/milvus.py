"""
Milvus integration for RAGStackXL.

Milvus is an open-source vector database designed for scalable similarity search
and AI applications. It's optimized for billion-scale vector operations.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

from app.core.interfaces import RagDocument, DocumentMetadata, Event, EventType, event_bus
from app.vectordb.interfaces import VectorDB, VectorDBConfig, VectorDBProvider, VectorDBFactory, SearchResult
from app.utils.logging import log


class MilvusVectorDB(VectorDB):
    """Milvus implementation of the VectorDB interface."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize the Milvus vector database."""
        super().__init__(config)
        
        # Get Milvus-specific configuration
        self.uri = config.get("uri", "localhost:19530")
        self.user = config.get("user", "")
        self.password = config.get("password", "")
        self.token = config.get("token", "")
        self.dimension = config.embedding_dimension
        self.distance_metric = self._translate_distance_metric(config.distance_metric)
        
        # Initialize connection
        self._connect_to_milvus()
        
        # Initialize or create collection
        self._init_collection()
        
        log.info(f"Milvus collection '{self.collection_name}' initialized with dimension {self.dimension}")
    
    def _translate_distance_metric(self, metric: str) -> str:
        """Translate our internal distance metric names to Milvus metric type."""
        metric = metric.lower()
        
        if metric == "cosine":
            return "COSINE"
        elif metric in ["euclidean", "l2"]:
            return "L2"
        elif metric in ["dot", "inner_product"]:
            return "IP"  # Inner Product
        else:
            log.warning(f"Unsupported distance metric for Milvus: {metric}, defaulting to COSINE")
            return "COSINE"
    
    def _connect_to_milvus(self):
        """Connect to Milvus server."""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=self.uri,
                user=self.user,
                password=self.password,
                token=self.token
            )
            log.info(f"Connected to Milvus server at {self.uri}")
        except Exception as e:
            log.error(f"Failed to connect to Milvus: {e}")
            raise ValueError(f"Failed to connect to Milvus: {e}")
    
    def _init_collection(self):
        """Initialize or create Milvus collection."""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                log.info(f"Using existing collection: {self.collection_name}")
                self.collection = Collection(self.collection_name)
                self.collection.load()
            else:
                log.info(f"Creating new collection: {self.collection_name}")
                self._create_collection()
                
        except Exception as e:
            log.error(f"Error initializing Milvus collection: {e}")
            raise ValueError(f"Error initializing Milvus collection: {e}")
    
    def _create_collection(self):
        """Create a new Milvus collection with necessary fields."""
        # Define fields
        fields = [
            # Primary key field
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            
            # Document content field
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            
            # Vector field
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dimension
            ),
            
            # Metadata fields
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="created_at", dtype=DataType.INT64)
        ]
        
        # Create schema
        schema = CollectionSchema(fields)
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
        )
        
        # Create index on the vector field
        index_params = {
            "metric_type": self.distance_metric,
            "index_type": "HNSW",  # Using HNSW for better performance
            "params": {"M": 8, "efConstruction": 64}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        # Load collection into memory
        self.collection.load()
    
    async def add_documents(
        self,
        documents: List[RagDocument],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the Milvus collection."""
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
        
        # Prepare data for insertion
        entity_ids = []
        entity_texts = []
        entity_embeddings = []
        entity_sources = []
        entity_categories = []
        entity_created_ats = []
        
        # Current timestamp for created_at
        current_time = int(datetime.now().timestamp())
        
        for i, (doc, doc_id, embedding) in enumerate(zip(documents, doc_ids, embeddings)):
            entity_ids.append(doc_id)
            entity_texts.append(doc.content)
            entity_embeddings.append(embedding)
            
            # Extract metadata
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "general")
            
            entity_sources.append(source)
            entity_categories.append(category)
            entity_created_ats.append(current_time)
        
        # Prepare entities
        entities = [
            entity_ids,
            entity_texts,
            entity_embeddings,
            entity_sources,
            entity_categories,
            entity_created_ats
        ]
        
        loop = asyncio.get_event_loop()
        
        async def _insert_entities():
            def _inner_insert():
                try:
                    # Insert entities
                    self.collection.insert(entities)
                    return True
                except Exception as e:
                    log.error(f"Error inserting entities into Milvus: {e}")
                    raise e
            
            return await loop.run_in_executor(None, _inner_insert)
        
        # Insert entities
        try:
            await _insert_entities()
            
            # Flush to ensure data is written
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: utility.index_building_progress(self.collection_name))
            
            # Notify that documents were added
            self._notify_documents_added(doc_ids)
            
            return doc_ids
        
        except Exception as e:
            log.error(f"Failed to add documents to Milvus: {e}")
            return []
    
    async def get_document(self, doc_id: str) -> Optional[RagDocument]:
        """Get a document by ID from Milvus."""
        loop = asyncio.get_event_loop()
        
        async def _fetch_document():
            def _inner_fetch():
                try:
                    # Query for the document
                    results = self.collection.query(
                        expr=f'id == "{doc_id}"',
                        output_fields=["id", "text", "source", "category"]
                    )
                    
                    # Check if document exists
                    if not results:
                        return None
                    
                    # Extract data
                    result = results[0]
                    content = result.get("text", "")
                    
                    # Create metadata
                    metadata = {
                        "source": result.get("source", "unknown"),
                        "category": result.get("category", "general")
                    }
                    
                    # Create and return document
                    return RagDocument(
                        doc_id=doc_id,
                        content=content,
                        metadata=metadata
                    )
                
                except Exception as e:
                    log.error(f"Error fetching document {doc_id} from Milvus: {e}")
                    return None
            
            return await loop.run_in_executor(None, _inner_fetch)
        
        return await _fetch_document()
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID from Milvus."""
        loop = asyncio.get_event_loop()
        
        async def _delete_document():
            def _inner_delete():
                try:
                    # Delete the document
                    expr = f'id == "{doc_id}"'
                    self.collection.delete(expr)
                    return True
                except Exception as e:
                    log.error(f"Error deleting document {doc_id} from Milvus: {e}")
                    return False
            
            return await loop.run_in_executor(None, _inner_delete)
        
        return await _delete_document()
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform a similarity search in Milvus."""
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
        """Perform a similarity search with scores in Milvus."""
        loop = asyncio.get_event_loop()
        
        async def _search():
            def _inner_search():
                try:
                    # Prepare filter expression if provided
                    expr = self._convert_to_milvus_expr(filter) if filter else ""
                    
                    # Prepare search parameters based on the distance metric
                    search_params = {
                        "metric_type": self.distance_metric,
                        "params": {"nprobe": 10}  # Number of clusters to search
                    }
                    
                    # Search
                    results = self.collection.search(
                        data=[query_embedding],
                        anns_field="embedding",
                        param=search_params,
                        limit=k,
                        expr=expr,
                        output_fields=["id", "text", "source", "category"]
                    )
                    
                    # Process results
                    processed_results = []
                    for hits in results:
                        for hit in hits:
                            # Extract data
                            doc_id = hit.id
                            content = hit.entity.get("text", "")
                            
                            # Create metadata
                            metadata = {
                                "source": hit.entity.get("source", "unknown"),
                                "category": hit.entity.get("category", "general")
                            }
                            
                            # Create document
                            doc = RagDocument(
                                doc_id=doc_id,
                                content=content,
                                metadata=metadata
                            )
                            
                            # Calculate score
                            # Note: Milvus returns distance, lower is better, so we convert to similarity
                            distance = hit.distance
                            
                            # Convert distance to similarity score (0-1 range)
                            if self.distance_metric in ["L2", "COSINE"]:
                                # For L2 and COSINE, distance is lower is better, so convert to similarity
                                max_dist = 2.0 if self.distance_metric == "COSINE" else float(self.dimension)
                                similarity = max(0.0, 1.0 - (distance / max_dist))
                            else:  # IP (Inner Product)
                                # For IP, larger value is better and can be > 1.0, so normalize
                                similarity = max(0.0, distance)
                                if similarity > 1.0:
                                    similarity = 1.0
                            
                            processed_results.append((doc, similarity))
                    
                    return processed_results
                
                except Exception as e:
                    log.error(f"Error performing similarity search in Milvus: {e}")
                    return []
            
            return await loop.run_in_executor(None, _inner_search)
        
        return await _search()
    
    def _convert_to_milvus_expr(self, filter: Dict[str, Any]) -> str:
        """Convert our filter format to Milvus expression string."""
        if not filter:
            return ""
        
        conditions = []
        
        for key, value in filter.items():
            if isinstance(value, list):
                # For lists, we use IN operator
                values_str = ", ".join([f'"{v}"' if isinstance(v, str) else str(v) for v in value])
                conditions.append(f'{key} in [{values_str}]')
            elif isinstance(value, str):
                # For strings, we use exact match with quotes
                conditions.append(f'{key} == "{value}"')
            else:
                # For other types, we use exact match without quotes
                conditions.append(f"{key} == {value}")
        
        # Join conditions with AND
        return " and ".join(conditions)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Milvus collection."""
        loop = asyncio.get_event_loop()
        
        async def _get_stats():
            def _inner_stats():
                try:
                    # Get collection statistics
                    stats = utility.get_collection_stats(self.collection_name)
                    
                    return {
                        "count": stats.get("row_count", 0),
                        "dimension": self.dimension,
                        "distance_metric": self.config.distance_metric,
                        "index_type": "HNSW"  # Hardcoded for now
                    }
                except Exception as e:
                    log.error(f"Error getting collection stats from Milvus: {e}")
                    return {
                        "count": 0,
                        "dimension": self.dimension,
                        "distance_metric": self.config.distance_metric
                    }
            
            return await loop.run_in_executor(None, _inner_stats)
        
        return await _get_stats()
    
    async def clear_collection(self) -> bool:
        """Clear the Milvus collection."""
        loop = asyncio.get_event_loop()
        
        async def _clear_collection():
            def _inner_clear():
                try:
                    # Delete all entities
                    self.collection.delete("id != \"\"")
                    return True
                except Exception as e:
                    log.error(f"Error clearing Milvus collection: {e}")
                    return False
            
            return await loop.run_in_executor(None, _inner_clear)
        
        return await _clear_collection()
    
    async def persist(self) -> bool:
        """Persist the database (not needed for Milvus as it's persistent by default)."""
        # Milvus is persistent by default
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


# Register Milvus with the factory
VectorDBFactory.register(VectorDBProvider.MILVUS, MilvusVectorDB) 