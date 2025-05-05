"""
Advanced FAISS implementations for RAGStackXL.

This module provides enhanced FAISS vector database implementations with:
1. HNSW indices for faster approximate search
2. IVF indices with product quantization for memory efficiency
3. Automatic index selection based on collection size
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Literal
import asyncio
import numpy as np

import faiss

from app.core.interfaces import RagDocument, DocumentMetadata, Event, EventType, event_bus
from app.vectordb.interfaces import VectorDB, VectorDBConfig, VectorDBProvider, VectorDBFactory, SearchResult
from app.vectordb.faiss import FaissVectorDB
from app.utils.logging import log


class FaissIndexType:
    """FAISS index types."""
    FLAT = "flat"  # Exact search
    IVF = "ivf"    # Inverted file index
    HNSW = "hnsw"  # Hierarchical Navigable Small World
    PQ = "pq"      # Product Quantization
    IVFPQ = "ivfpq"  # IVF with PQ compression


class FaissAdvancedVectorDB(FaissVectorDB):
    """Advanced FAISS implementation with optimized indices."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize the advanced FAISS vector database."""
        # Don't call parent constructor directly
        VectorDB.__init__(self, config)
        
        # Get advanced config settings
        self.index_type = config.get("index_type", FaissIndexType.FLAT)
        self.auto_index = config.get("auto_index", True)
        self.nlist = config.get("nlist", 100)  # Number of clusters for IVF
        self.nprobe = config.get("nprobe", 10)  # Number of clusters to search
        self.m = config.get("m", 16)  # HNSW parameter
        self.ef_construction = config.get("ef_construction", 100)  # HNSW parameter
        self.ef_search = config.get("ef_search", 50)  # HNSW parameter
        self.pq_m = config.get("pq_m", 8)  # Number of PQ subquantizers
        
        # Set up storage directory
        self.persist_directory = config.get("persist_directory")
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            
        # FAISS index file path
        self.index_file = os.path.join(self.persist_directory, f"{self.collection_name}_index.faiss") if self.persist_directory else None
        self.docstore_file = os.path.join(self.persist_directory, f"{self.collection_name}_docstore.pickle") if self.persist_directory else None
        
        # Initialize FAISS index based on distance metric
        self.dimension = config.embedding_dimension
        self.metric = self._get_faiss_metric(config.distance_metric)
        self.index = self._create_index()
        
        # Dictionary to store documents (docstore)
        self.docstore = {}
        self.doc_ids_map = {}  # Maps FAISS index to document ID
        
        # Load existing index and docstore if they exist
        if self.persist_directory:
            self._load_index_and_docstore()
        
        log.info(f"Advanced FAISS index ({self.index_type}) initialized with dimension {self.dimension}")
    
    def _get_faiss_metric(self, metric: str) -> int:
        """Convert our metric string to FAISS metric type."""
        metric = metric.lower()
        if metric in ["euclidean", "l2"]:
            return faiss.METRIC_L2
        elif metric in ["dot", "inner_product"]:
            return faiss.METRIC_INNER_PRODUCT
        elif metric == "cosine":
            # Cosine similarity is handled by normalizing vectors before indexing
            return faiss.METRIC_L2
        else:
            log.warning(f"Unsupported metric: {metric}, defaulting to L2")
            return faiss.METRIC_L2
    
    def _select_optimal_index_type(self, num_vectors: int) -> str:
        """
        Select the optimal index type based on collection size.
        
        This is a heuristic approach based on FAISS recommendations:
        - <10K vectors: FLAT index (exact search)
        - 10K-1M vectors: IVF index
        - >1M vectors: IVFPQ (quantized for memory efficiency)
        - All sizes benefit from HNSW but has higher memory requirements
        """
        if not self.auto_index:
            return self.index_type
            
        if num_vectors < 10_000:
            return FaissIndexType.FLAT
        elif num_vectors < 1_000_000:
            return FaissIndexType.IVF
        else:
            return FaissIndexType.IVFPQ
    
    def _rebuild_index_if_needed(self):
        """Rebuild index if collection size warrants a different index type."""
        if not self.auto_index or not self.docstore:
            return
            
        current_size = len(self.docstore)
        optimal_index_type = self._select_optimal_index_type(current_size)
        
        if optimal_index_type != self.index_type and current_size > 0:
            log.info(f"Rebuilding index: changing from {self.index_type} to {optimal_index_type} based on size ({current_size} vectors)")
            
            # Extract all vectors
            all_ids = list(self.doc_ids_map.values())
            vectors = []
            
            # Get vectors from current index (not efficient but necessary for rebuilding)
            for idx in range(self.index.ntotal):
                faiss_id = idx
                doc_id = self.doc_ids_map.get(faiss_id)
                if doc_id:
                    # We need to extract the vector - this is expensive but necessary
                    vector = np.zeros((1, self.dimension), dtype=np.float32)
                    self.index.reconstruct(faiss_id, vector[0])
                    vectors.append(vector[0])
            
            if vectors:
                # Build new index
                self.index_type = optimal_index_type
                new_index = self._create_index(force_index_type=optimal_index_type)
                
                # Add vectors to new index
                vectors_array = np.array(vectors, dtype=np.float32)
                new_index.add(vectors_array)
                
                # Replace index
                self.index = new_index
                
                # Rebuild doc_ids_map (the map must start from 0 for the new index)
                self.doc_ids_map = {i: doc_id for i, doc_id in enumerate(all_ids)}
                
                # Persist new index
                if self.persist_directory:
                    self._persist_index_and_docstore()
    
    def _create_index(self, force_index_type: Optional[str] = None) -> faiss.Index:
        """Create a FAISS index based on configuration and collection size."""
        index_type = force_index_type or self.index_type
        metric = self.metric
        dim = self.dimension
        
        # For cosine similarity, we use L2 after normalizing vectors
        need_vector_normalization = self.config.distance_metric.lower() == "cosine"
        
        # Choose index type
        if index_type == FaissIndexType.FLAT:
            if metric == faiss.METRIC_L2:
                index = faiss.IndexFlatL2(dim)
            else:  # Inner product
                index = faiss.IndexFlatIP(dim)
        
        elif index_type == FaissIndexType.IVF:
            # IVF requires a quantizer (using FLAT)
            if metric == faiss.METRIC_L2:
                quantizer = faiss.IndexFlatL2(dim)
            else:  # Inner product
                quantizer = faiss.IndexFlatIP(dim)
                
            # Create IVF index
            index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, metric)
            # IVF indices need training, but we defer this until we have vectors
            index.nprobe = self.nprobe
        
        elif index_type == FaissIndexType.HNSW:
            # HNSW parameters
            M = self.m  # Number of connections per layer
            
            # Create HNSW index
            if metric == faiss.METRIC_L2:
                index = faiss.IndexHNSWFlat(dim, M, metric)
            else:  # Inner product
                index = faiss.IndexHNSWFlat(dim, M, metric)
                
            # Set construction parameters
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
        
        elif index_type in [FaissIndexType.PQ, FaissIndexType.IVFPQ]:
            # Product Quantization parameters
            M = self.pq_m  # Number of subquantizers
            nbits = 8  # Bits per subquantizer (usually 8)
            
            if index_type == FaissIndexType.PQ:
                index = faiss.IndexPQ(dim, M, nbits, metric)
            else:  # IVFPQ
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, M, nbits, metric)
                index.nprobe = self.nprobe
        
        else:
            log.warning(f"Unknown index type: {index_type}, defaulting to FLAT")
            if metric == faiss.METRIC_L2:
                index = faiss.IndexFlatL2(dim)
            else:  # Inner product
                index = faiss.IndexFlatIP(dim)
        
        return index
    
    async def add_documents(
        self,
        documents: List[RagDocument],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the FAISS index with optimizations for batch processing."""
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
                doc_id = ids[i]
            elif doc.doc_id:
                doc_id = doc.doc_id
            else:
                doc_id = self._generate_doc_id()
            
            doc_ids.append(doc_id)
        
        loop = asyncio.get_event_loop()
        
        # Check if index needs training before adding documents
        def _add_docs():
            # Convert to numpy array
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity if needed
            if self.config.distance_metric.lower() == "cosine":
                vectors = np.array([self._normalize_vector(vec) for vec in vectors], dtype=np.float32)
            
            # Handle IVF indices which require training
            if isinstance(self.index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ)) and not self.index.is_trained:
                # Train index if we have enough vectors
                if vectors.shape[0] >= self.nlist:
                    log.info(f"Training IVF index with {vectors.shape[0]} vectors")
                    self.index.train(vectors)
                else:
                    # Not enough vectors for training, fall back to flat index
                    log.warning(f"Not enough vectors for IVF training ({vectors.shape[0]} < {self.nlist}), falling back to FLAT index")
                    self.index = faiss.IndexFlatL2(self.dimension) if self.metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dimension)
            
            # Get current number of vectors for index mapping
            start_index = self.index.ntotal
            
            # Add vectors to FAISS index
            self.index.add(vectors)
            
            # Store documents and mapping
            for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
                self.docstore[doc_id] = doc
                self.doc_ids_map[start_index + i] = doc_id
            
            # Check if index should be rebuilt for collection size
            self._rebuild_index_if_needed()
            
            # Persist immediately if directory is set
            if self.persist_directory:
                self._persist_index_and_docstore()
            
            return doc_ids
        
        doc_ids = await loop.run_in_executor(None, _add_docs)
        
        # Notify that documents were added
        self._notify_documents_added(doc_ids)
        
        return doc_ids
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform a similarity search in FAISS with optimized parameters.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter
            ef_search: Optional HNSW ef_search parameter override
        """
        results = await self.similarity_search_with_score(
            query_embedding=query_embedding,
            k=k,
            filter=filter,
            ef_search=ef_search
        )
        
        return [
            SearchResult(document=doc, score=score)
            for doc, score in results
        ]
    
    async def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None
    ) -> List[Tuple[RagDocument, float]]:
        """
        Perform a similarity search with scores in FAISS with optimized parameters.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter
            ef_search: Optional HNSW ef_search parameter override
        """
        loop = asyncio.get_event_loop()
        
        def _search():
            try:
                # Convert query to numpy array
                query_vector = np.array([query_embedding], dtype=np.float32)
                
                # Normalize for cosine similarity if needed
                if self.config.distance_metric.lower() == "cosine":
                    query_vector = np.array([self._normalize_vector(query_vector[0])], dtype=np.float32)
                
                # Set HNSW search params if provided and applicable
                if ef_search is not None and hasattr(self.index, 'hnsw'):
                    original_ef_search = self.index.hnsw.efSearch
                    self.index.hnsw.efSearch = ef_search
                
                # Perform search
                # Adjust k based on filters to ensure we get enough results
                k_adjusted = min(k * 10, self.index.ntotal) if filter else k
                if k_adjusted == 0:
                    return []
                
                distances, indices = self.index.search(query_vector, k_adjusted)
                
                # Reset HNSW search params if we changed them
                if ef_search is not None and hasattr(self.index, 'hnsw'):
                    self.index.hnsw.efSearch = original_ef_search
                
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
                    score = self._convert_distance_to_score(distances[0][i])
                    
                    results.append((doc, score))
                    
                    # Stop once we have enough results after filtering
                    if len(results) >= k:
                        break
                
                return results
            
            except Exception as e:
                log.error(f"Error in similarity search: {e}")
                return []
        
        return await loop.run_in_executor(None, _search)
    
    def _convert_distance_to_score(self, distance: float) -> float:
        """Convert FAISS distance to a similarity score (0-1 range)."""
        if self.metric == faiss.METRIC_L2:
            # For L2 distance, smaller is better, so we convert to similarity
            # Use a heuristic based on dimension to determine max distance
            max_distance = np.sqrt(self.dimension) * 2.0  # Theoretical max for normalized vectors
            return max(0, 1 - (distance / max_distance))
        elif self.config.distance_metric.lower() == "cosine":
            # Convert L2 squared distance to cosine similarity
            # For normalized vectors, d^2 = 2(1-cos(α))
            # So cos(α) = 1 - d^2/2
            return max(0, 1 - (distance / 2))
        else:  # Inner product
            # Inner product is already a similarity metric (higher is better)
            # Normalize to 0-1 range (assuming vectors are normalized)
            return min(1.0, max(0.0, distance))
    
    def _document_matches_filter(self, document: RagDocument, filter: Dict[str, Any]) -> bool:
        """Enhanced filter handling with support for complex operators."""
        from app.vectordb.utils import FilterOperator
        
        if not document.metadata or not filter:
            return False
        
        # Handle complex operators
        def match_condition(condition, metadata):
            if FilterOperator.AND in condition:
                return all(match_condition(subcond, metadata) for subcond in condition[FilterOperator.AND])
            elif FilterOperator.OR in condition:
                return any(match_condition(subcond, metadata) for subcond in condition[FilterOperator.OR])
            elif FilterOperator.NOT in condition:
                return not match_condition(condition[FilterOperator.NOT], metadata)
            
            # Simple key-value pairs
            for key, value in condition.items():
                if key.startswith("$"):
                    continue  # Skip operators
                
                if key not in metadata:
                    return False
                
                meta_value = metadata[key]
                
                # Handle complex expressions
                if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                    # This is a complex expression with operators
                    for op, op_value in value.items():
                        if op == FilterOperator.EQ:
                            if meta_value != op_value:
                                return False
                        elif op == FilterOperator.NE:
                            if meta_value == op_value:
                                return False
                        elif op == FilterOperator.GT:
                            if not (isinstance(meta_value, (int, float)) and meta_value > op_value):
                                return False
                        elif op == FilterOperator.GTE:
                            if not (isinstance(meta_value, (int, float)) and meta_value >= op_value):
                                return False
                        elif op == FilterOperator.LT:
                            if not (isinstance(meta_value, (int, float)) and meta_value < op_value):
                                return False
                        elif op == FilterOperator.LTE:
                            if not (isinstance(meta_value, (int, float)) and meta_value <= op_value):
                                return False
                        elif op == FilterOperator.IN:
                            if meta_value not in op_value:
                                return False
                        elif op == FilterOperator.NIN:
                            if meta_value in op_value:
                                return False
                        elif op == FilterOperator.CONTAINS:
                            if not (isinstance(meta_value, str) and isinstance(op_value, str) and op_value in meta_value):
                                return False
                        elif op == FilterOperator.STARTSWITH:
                            if not (isinstance(meta_value, str) and isinstance(op_value, str) and meta_value.startswith(op_value)):
                                return False
                        elif op == FilterOperator.ENDSWITH:
                            if not (isinstance(meta_value, str) and isinstance(op_value, str) and meta_value.endswith(op_value)):
                                return False
                        # REGEX not implemented for in-memory filtering
                else:
                    # Simple equality check
                    if isinstance(value, list):
                        if meta_value not in value:
                            return False
                    elif meta_value != value:
                        return False
            
            return True
        
        return match_condition(filter, document.metadata)


# Register advanced FAISS with a new provider
VectorDBProvider.FAISS_ADVANCED = "faiss_advanced"  # Add to enum dynamically
VectorDBFactory.register(VectorDBProvider.FAISS_ADVANCED, FaissAdvancedVectorDB) 