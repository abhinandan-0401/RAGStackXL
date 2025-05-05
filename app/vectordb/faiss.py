"""
FAISS integration for RAGStackXL.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
and clustering of dense vectors. It's ideal for local vector search applications.
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import uuid
import numpy as np

import faiss

from app.core.interfaces import RagDocument, DocumentMetadata, Event, EventType, event_bus
from app.vectordb.interfaces import VectorDB, VectorDBConfig, VectorDBProvider, VectorDBFactory, SearchResult
from app.utils.logging import log


class FaissVectorDB(VectorDB):
    """FAISS implementation of the VectorDB interface."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize the FAISS vector database."""
        super().__init__(config)
        
        # Set up storage directory
        self.persist_directory = config.get("persist_directory")
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            
        # FAISS index file path
        self.index_file = os.path.join(self.persist_directory, f"{self.collection_name}_index.faiss") if self.persist_directory else None
        self.docstore_file = os.path.join(self.persist_directory, f"{self.collection_name}_docstore.pickle") if self.persist_directory else None
        
        # Initialize FAISS index based on distance metric
        self.dimension = config.embedding_dimension
        self.index = self._create_index()
        
        # Dictionary to store documents (docstore)
        self.docstore = {}
        self.doc_ids_map = {}  # Maps FAISS index to document ID
        
        # Load existing index and docstore if they exist
        if self.persist_directory:
            self._load_index_and_docstore()
        
        log.info(f"FAISS index initialized with dimension {self.dimension}")
    
    def _create_index(self):
        """Create a FAISS index based on configuration."""
        metric = self.config.distance_metric.lower()
        
        if metric == "cosine":
            # For cosine similarity, we use L2 after normalizing vectors
            index = faiss.IndexFlatL2(self.dimension)
        elif metric == "euclidean" or metric == "l2":
            # For Euclidean distance
            index = faiss.IndexFlatL2(self.dimension)
        elif metric == "dot" or metric == "inner_product":
            # For dot product
            index = faiss.IndexFlatIP(self.dimension)
        else:
            log.warning(f"Unsupported distance metric: {metric}, defaulting to L2")
            index = faiss.IndexFlatL2(self.dimension)
            
        return index
    
    def _load_index_and_docstore(self):
        """Load existing index and docstore if they exist."""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.docstore_file):
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                
                # Load docstore
                with open(self.docstore_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                    self.docstore = loaded_data.get('docstore', {})
                    self.doc_ids_map = loaded_data.get('doc_ids_map', {})
                
                log.info(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.docstore)} documents")
            else:
                log.info("No existing FAISS index found, starting with empty index")
        except Exception as e:
            log.error(f"Error loading FAISS index: {e}")
            log.info("Starting with empty index")
    
    def _normalize_vector(self, vector):
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    async def add_documents(
        self,
        documents: List[RagDocument],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the FAISS index."""
        if not documents:
            return []
        
        if not self._validate_embeddings(embeddings):
            raise ValueError("Invalid embeddings")
        
        if len(documents) != len(embeddings):
            raise ValueError(f"Number of documents ({len(documents)}) does not match number of embeddings ({len(embeddings)})")
        
        # Use document IDs if provided, otherwise generate new ones
        doc_ids = []
        for i, doc in enumerate(documents):
            if ids is not None and i < len(ids):
                # Use provided ID
                doc_id = ids[i]
            elif doc.doc_id:
                # Use document's own ID if available
                doc_id = doc.doc_id
            else:
                # Generate a new ID
                doc_id = self._generate_doc_id()
            
            doc_ids.append(doc_id)
        
        loop = asyncio.get_event_loop()
        
        def _add_docs():
            # Convert to numpy array
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity if needed
            if self.config.distance_metric.lower() == "cosine":
                vectors = np.array([self._normalize_vector(vec) for vec in vectors], dtype=np.float32)
            
            # Get current number of vectors for index mapping
            start_index = self.index.ntotal
            
            # Add vectors to FAISS index
            self.index.add(vectors)
            
            # Store documents and mapping
            for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
                self.docstore[doc_id] = doc
                self.doc_ids_map[start_index + i] = doc_id
            
            # Persist immediately if directory is set
            if self.persist_directory:
                self._persist_index_and_docstore()
            
            return doc_ids
        
        doc_ids = await loop.run_in_executor(None, _add_docs)
        
        # Notify that documents were added
        self._notify_documents_added(doc_ids)
        
        return doc_ids
    
    async def get_document(self, doc_id: str) -> Optional[RagDocument]:
        """Get a document by ID from FAISS."""
        loop = asyncio.get_event_loop()
        
        def _get_doc():
            return self.docstore.get(doc_id)
        
        return await loop.run_in_executor(None, _get_doc)
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID from FAISS.
        
        Note: FAISS doesn't support direct deletion. We need to rebuild the index
        without the deleted document. This is inefficient for large indices.
        """
        loop = asyncio.get_event_loop()
        
        def _delete_doc():
            try:
                # Check if document exists
                if doc_id not in self.docstore:
                    return False
                
                # Remove from docstore
                self.docstore.pop(doc_id)
                
                # Find and remove index mapping
                index_to_remove = None
                for idx, id_val in self.doc_ids_map.items():
                    if id_val == doc_id:
                        index_to_remove = idx
                        break
                
                if index_to_remove is not None:
                    self.doc_ids_map.pop(index_to_remove)
                    
                    # For truly removing the vector, we would need to rebuild the index
                    # This is a limitation of FAISS, but for now, we'll just mark it as removed
                    # in our docstore and doc_ids_map
                    
                    # For production use, periodic index rebuilding would be recommended
                
                # Persist changes
                if self.persist_directory:
                    self._persist_index_and_docstore()
                
                return True
            except Exception as e:
                log.error(f"Error deleting document {doc_id}: {e}")
                return False
        
        return await loop.run_in_executor(None, _delete_doc)
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform a similarity search in FAISS."""
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
        """Perform a similarity search with scores in FAISS."""
        loop = asyncio.get_event_loop()
        
        def _search():
            try:
                # Convert query to numpy array
                query_vector = np.array([query_embedding], dtype=np.float32)
                
                # Normalize for cosine similarity if needed
                if self.config.distance_metric.lower() == "cosine":
                    query_vector = np.array([self._normalize_vector(query_vector[0])], dtype=np.float32)
                
                # Perform search
                # FAISS returns (distances, indices)
                k_adjusted = min(k * 10, self.index.ntotal)  # Get more results for filtering
                if k_adjusted == 0:
                    return []
                
                distances, indices = self.index.search(query_vector, k_adjusted)
                
                # Process results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx == -1:  # FAISS may return -1 for not enough results
                        continue
                    
                    doc_id = self.doc_ids_map.get(int(idx))
                    if not doc_id:
                        continue
                    
                    doc = self.docstore.get(doc_id)
                    if not doc:
                        continue
                    
                    # Filter results if filter is provided
                    if filter and not self._document_matches_filter(doc, filter):
                        continue
                    
                    # Convert distance to similarity score
                    # For L2 distance, smaller is better, so we convert to similarity
                    if self.config.distance_metric.lower() in ["euclidean", "l2"]:
                        # Convert L2 distance to similarity score (max distance is hard to determine,
                        # but we can use a heuristic based on dimension)
                        max_distance = np.sqrt(self.dimension)  # Theoretical max for normalized vectors
                        score = max(0, 1 - (distances[0][i] / max_distance))
                    elif self.config.distance_metric.lower() == "cosine":
                        # Convert L2 squared distance to cosine similarity
                        # For normalized vectors, d^2 = 2(1-cos(α))
                        # So cos(α) = 1 - d^2/2
                        score = max(0, 1 - (distances[0][i] / 2))
                    else:  # Inner product
                        # Inner product is already a similarity metric (higher is better)
                        score = float(distances[0][i])
                    
                    results.append((doc, score))
                    
                    # Stop once we have enough results after filtering
                    if len(results) >= k:
                        break
                
                return results
            
            except Exception as e:
                log.error(f"Error in similarity search: {e}")
                return []
        
        return await loop.run_in_executor(None, _search)
    
    def _document_matches_filter(self, document: RagDocument, filter: Dict[str, Any]) -> bool:
        """Check if document matches the filter criteria."""
        if not document.metadata or not filter:
            return False
        
        for key, value in filter.items():
            if key not in document.metadata:
                return False
            
            if isinstance(value, list):
                if document.metadata[key] not in value:
                    return False
            elif document.metadata[key] != value:
                return False
                
        return True
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS collection."""
        loop = asyncio.get_event_loop()
        
        def _get_stats():
            return {
                "count": self.index.ntotal,
                "dimension": self.dimension,
                "distance_metric": self.config.distance_metric,
                "docstore_size": len(self.docstore)
            }
        
        return await loop.run_in_executor(None, _get_stats)
    
    async def clear_collection(self) -> bool:
        """Clear the FAISS collection."""
        loop = asyncio.get_event_loop()
        
        def _clear():
            try:
                # Create new empty index
                self.index = self._create_index()
                
                # Clear docstore and mapping
                self.docstore = {}
                self.doc_ids_map = {}
                
                # Persist empty index
                if self.persist_directory:
                    self._persist_index_and_docstore()
                
                return True
            except Exception as e:
                log.error(f"Error clearing FAISS collection: {e}")
                return False
        
        return await loop.run_in_executor(None, _clear)
    
    async def persist(self) -> bool:
        """Persist the FAISS index and docstore to disk."""
        if not self.persist_directory:
            log.warning("No persist directory set, cannot persist FAISS index")
            return False
        
        loop = asyncio.get_event_loop()
        
        def _persist():
            return self._persist_index_and_docstore()
        
        return await loop.run_in_executor(None, _persist)
    
    def _persist_index_and_docstore(self) -> bool:
        """Internal method to persist index and docstore."""
        if not self.persist_directory:
            return False
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save docstore
            with open(self.docstore_file, 'wb') as f:
                pickle.dump({
                    'docstore': self.docstore,
                    'doc_ids_map': self.doc_ids_map
                }, f)
            
            return True
        except Exception as e:
            log.error(f"Error persisting FAISS index: {e}")
            return False
    
    def _notify_documents_added(self, doc_ids: List[str]) -> None:
        """Notify that documents were added to the database."""
        # Create event
        event = Event(
            event_type=EventType.DOCUMENT_ADDED,
            payload={"doc_ids": doc_ids, "collection_name": self.collection_name}
        )
        # Publish event
        event_bus.publish(event)


# Register FAISS with the factory
VectorDBFactory.register(VectorDBProvider.FAISS, FaissVectorDB) 