"""
Weaviate integration for RAGStackXL.

Weaviate is a vector database with GraphQL support and hybrid search capabilities,
combining vector similarity with structured data filtering.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
import uuid

import weaviate
from weaviate.exceptions import WeaviateQueryError

from app.core.interfaces import RagDocument, DocumentMetadata
from app.vectordb.interfaces import VectorDB, VectorDBConfig, VectorDBProvider, VectorDBFactory, SearchResult
from app.utils.logging import log


class WeaviateVectorDB(VectorDB):
    """Weaviate implementation of the VectorDB interface."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize the Weaviate vector database."""
        super().__init__(config)
        
        # Convert collection name to class name (Weaviate uses CamelCase for classes)
        self.class_name = self._to_class_name(self.collection_name)
        
        # Configure Weaviate client
        url = config.get("url", "http://localhost:8080")
        api_key = config.get("api_key")
        
        auth_config = None
        if api_key:
            auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
        
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config,
            additional_headers=config.get("additional_headers")
        )
        
        # Create class if it doesn't exist
        self._create_class_if_not_exists()
        log.info(f"Weaviate class '{self.class_name}' ready")
    
    def _to_class_name(self, name: str) -> str:
        """Convert collection name to Weaviate class name (CamelCase)."""
        # Replace non-alphanumeric characters with underscores
        clean_name = ''.join(c if c.isalnum() else '_' for c in name)
        
        # Split by underscores and capitalize each part
        parts = clean_name.split('_')
        camel_case = ''.join(part.capitalize() for part in parts if part)
        
        # Ensure name starts with uppercase letter (Weaviate requirement)
        if not camel_case:
            return "Document"
        
        return camel_case
    
    def _create_class_if_not_exists(self):
        """Create Weaviate class if it doesn't exist."""
        try:
            # Check if class exists
            schema = self.client.schema.get()
            classes = [c["class"] for c in schema["classes"]] if "classes" in schema else []
            
            if self.class_name in classes:
                log.info(f"Weaviate class '{self.class_name}' already exists")
                return
            
            # Class doesn't exist, create it
            log.info(f"Creating Weaviate class '{self.class_name}'")
            
            class_schema = {
                "class": self.class_name,
                "description": f"RAG documents for {self.collection_name}",
                "vectorizer": "none",  # We provide vectors directly
                "vectorIndexConfig": {
                    "distance": self._get_weaviate_distance_metric()
                },
                "properties": [
                    {
                        "name": "content",
                        "description": "Document content",
                        "dataType": ["text"]
                    },
                    # Add optional properties for common metadata
                    {
                        "name": "source",
                        "description": "Document source",
                        "dataType": ["text"]
                    },
                    {
                        "name": "title",
                        "description": "Document title",
                        "dataType": ["text"]
                    },
                    {
                        "name": "metadata",
                        "description": "Additional document metadata as JSON",
                        "dataType": ["text"]
                    }
                ]
            }
            
            self.client.schema.create_class(class_schema)
            
        except Exception as e:
            log.error(f"Error creating Weaviate class: {e}")
            raise
    
    def _get_weaviate_distance_metric(self) -> str:
        """Convert config distance metric to Weaviate distance metric."""
        metric = self.config.distance_metric.lower()
        if metric == "cosine":
            return "cosine"
        elif metric in ["euclidean", "l2"]:
            return "l2-squared"
        elif metric == "dot":
            return "dot"
        else:
            log.warning(f"Unsupported distance metric: {metric}, defaulting to cosine")
            return "cosine"
    
    async def add_documents(
        self,
        documents: List[RagDocument],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the Weaviate database."""
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
        
        # Add documents to Weaviate in batch
        loop = asyncio.get_event_loop()
        
        def _add_docs():
            try:
                with self.client.batch as batch:
                    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                        doc_id = ids[i]
                        
                        # Extract common metadata fields
                        properties = {
                            "content": doc.content
                        }
                        
                        if doc.metadata:
                            # Extract specific fields
                            if "source" in doc.metadata:
                                properties["source"] = doc.metadata["source"]
                            
                            if "title" in doc.metadata:
                                properties["title"] = doc.metadata["title"]
                            
                            # Store remaining metadata as JSON
                            metadata_json = {}
                            for k, v in doc.metadata.items():
                                if k not in ["source", "title"]:
                                    metadata_json[k] = v
                            
                            if metadata_json:
                                properties["metadata"] = json.dumps(metadata_json)
                        
                        # Add object to batch
                        batch.add_data_object(
                            data_object=properties,
                            class_name=self.class_name,
                            uuid=doc_id,
                            vector=embedding
                        )
                
                return ids
            
            except Exception as e:
                log.error(f"Error adding documents to Weaviate: {e}")
                return []
        
        doc_ids = await loop.run_in_executor(None, _add_docs)
        
        if doc_ids:
            # Notify that documents were added
            self._notify_documents_added(doc_ids)
        
        return doc_ids
    
    async def get_document(self, doc_id: str) -> Optional[RagDocument]:
        """Get a document by ID from Weaviate."""
        loop = asyncio.get_event_loop()
        
        def _get_doc():
            try:
                # Construct GraphQL query
                result = self.client.data_object.get_by_id(
                    uuid=doc_id,
                    class_name=self.class_name,
                    with_vector=False
                )
                
                if not result:
                    return None
                
                # Extract content and properties
                content = result.get("properties", {}).get("content", "")
                
                # Extract and merge metadata
                metadata = {}
                
                # Add specific fields
                if "source" in result.get("properties", {}):
                    metadata["source"] = result["properties"]["source"]
                
                if "title" in result.get("properties", {}):
                    metadata["title"] = result["properties"]["title"]
                
                # Add JSON metadata if available
                if "metadata" in result.get("properties", {}):
                    try:
                        json_metadata = json.loads(result["properties"]["metadata"])
                        metadata.update(json_metadata)
                    except json.JSONDecodeError:
                        log.warning(f"Failed to parse metadata JSON for document {doc_id}")
                
                return RagDocument(
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id
                )
            
            except Exception as e:
                log.error(f"Error getting document {doc_id} from Weaviate: {e}")
                return None
        
        return await loop.run_in_executor(None, _get_doc)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID from Weaviate."""
        loop = asyncio.get_event_loop()
        
        def _delete_doc():
            try:
                self.client.data_object.delete(
                    uuid=doc_id,
                    class_name=self.class_name
                )
                return True
            except Exception as e:
                log.error(f"Error deleting document {doc_id} from Weaviate: {e}")
                return False
        
        return await loop.run_in_executor(None, _delete_doc)
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform a similarity search in Weaviate."""
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
        """Perform a similarity search with scores in Weaviate."""
        loop = asyncio.get_event_loop()
        
        def _search():
            try:
                # Prepare nearVector query
                query_builder = self.client.query.get(
                    class_name=self.class_name,
                    properties=["content", "source", "title", "metadata"]
                ).with_limit(k)
                
                # Add vector search
                query_builder = query_builder.with_near_vector({
                    "vector": query_embedding
                })
                
                # Add filter if provided
                if filter:
                    where_filter = self._convert_to_weaviate_filter(filter)
                    query_builder = query_builder.with_where(where_filter)
                
                # Add additional search options
                query_builder = query_builder.with_additional(["id", "certainty", "distance"])
                
                # Execute query
                result = query_builder.do()
                
                # Process results
                objects = result.get("data", {}).get("Get", {}).get(self.class_name, [])
                
                search_results = []
                for obj in objects:
                    # Extract document content
                    content = obj.get("content", "")
                    
                    # Extract metadata
                    metadata = {}
                    if "source" in obj:
                        metadata["source"] = obj["source"]
                    
                    if "title" in obj:
                        metadata["title"] = obj["title"]
                    
                    # Add JSON metadata if available
                    if "metadata" in obj and obj["metadata"]:
                        try:
                            json_metadata = json.loads(obj["metadata"])
                            metadata.update(json_metadata)
                        except json.JSONDecodeError:
                            log.warning("Failed to parse metadata JSON")
                    
                    # Get document ID
                    doc_id = obj.get("_additional", {}).get("id")
                    
                    # Create document
                    doc = RagDocument(
                        content=content,
                        metadata=metadata,
                        doc_id=doc_id
                    )
                    
                    # Get score
                    certainty = obj.get("_additional", {}).get("certainty")
                    score = certainty if certainty is not None else 0.0
                    
                    search_results.append((doc, score))
                
                return search_results
            
            except Exception as e:
                log.error(f"Error in similarity search with Weaviate: {e}")
                return []
        
        return await loop.run_in_executor(None, _search)
    
    def _convert_to_weaviate_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a simple filter dict to Weaviate filter format."""
        # Simple implementation - could be extended for more complex filters
        weaviate_filter = {}
        
        for key, value in filter.items():
            if key in ["source", "title"]:
                # Direct field filter
                if isinstance(value, list):
                    weaviate_filter["operator"] = "Or"
                    weaviate_filter["operands"] = [
                        {"path": [key], "operator": "Equal", "valueText": v}
                        for v in value
                    ]
                else:
                    weaviate_filter["path"] = [key]
                    weaviate_filter["operator"] = "Equal"
                    weaviate_filter["valueText"] = value
            else:
                # For other fields, we'd need to search in the metadata JSON
                # This is a limitation of the current approach
                log.warning(f"Filter on {key} not supported directly, complex filters on JSON metadata not implemented")
        
        return weaviate_filter
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Weaviate collection."""
        loop = asyncio.get_event_loop()
        
        def _get_stats():
            try:
                # Get schema information
                schema = self.client.schema.get(self.class_name)
                
                # Get object count
                result = self.client.query.aggregate(self.class_name).with_meta_count().do()
                count = result.get("data", {}).get("Aggregate", {}).get(self.class_name, [{}])[0].get("meta", {}).get("count", 0)
                
                return {
                    "name": self.class_name,
                    "count": count,
                    "dimension": self.config.embedding_dimension,
                    "distance": self._get_weaviate_distance_metric()
                }
            
            except Exception as e:
                log.error(f"Error getting collection stats from Weaviate: {e}")
                return {"error": str(e)}
        
        return await loop.run_in_executor(None, _get_stats)
    
    async def clear_collection(self) -> bool:
        """Clear the Weaviate collection."""
        loop = asyncio.get_event_loop()
        
        def _clear():
            try:
                self.client.schema.delete_all()
                self._create_class_if_not_exists()
                return True
            except Exception as e:
                log.error(f"Error clearing Weaviate collection: {e}")
                return False
        
        return await loop.run_in_executor(None, _clear)
    
    async def persist(self) -> bool:
        """Persist the Weaviate database to disk."""
        # Weaviate automatically persists data, this is a no-op
        return True


# Register Weaviate with the factory
VectorDBFactory.register(VectorDBProvider.WEAVIATE, WeaviateVectorDB) 