"""
Tests for vector database implementations.
"""
import os
import pytest
import asyncio
from typing import List

from app.core.interfaces import RagDocument
from app.vectordb.interfaces import (
    VectorDBConfig,
    VectorDBProvider,
    VectorDBFactory
)
from app.vectordb.faiss import FaissVectorDB
from app.vectordb.factory import create_vectordb


@pytest.fixture
def sample_documents() -> List[RagDocument]:
    """Sample documents for testing."""
    return [
        RagDocument(
            content="This is a test document about AI",
            metadata={"source": "test", "category": "ai"},
            doc_id="doc1"
        ),
        RagDocument(
            content="This document is about machine learning",
            metadata={"source": "test", "category": "ml"},
            doc_id="doc2"
        ),
        RagDocument(
            content="Python is a programming language",
            metadata={"source": "test", "category": "programming"},
            doc_id="doc3"
        ),
    ]


@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """Sample embeddings for testing."""
    # Simple 4D embeddings for testing
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]


@pytest.fixture
def faiss_db():
    """Create a FAISS database instance for testing."""
    # Use a temporary directory for testing
    temp_dir = os.path.join("tests", "temp", "vectordb_test")
    os.makedirs(temp_dir, exist_ok=True)
    
    config = VectorDBConfig(
        collection_name="test_collection",
        embedding_dimension=4,  # Use small dimension for testing
        persist_directory=temp_dir
    )
    
    db = FaissVectorDB(config)
    
    yield db
    
    # Clean up
    loop = asyncio.get_event_loop()
    loop.run_until_complete(db.clear_collection())


def test_vectordb_factory():
    """Test VectorDBFactory registration and creation."""
    assert VectorDBProvider.FAISS in VectorDBFactory._registry
    assert VectorDBProvider.QDRANT in VectorDBFactory._registry
    assert VectorDBProvider.WEAVIATE in VectorDBFactory._registry
    
    config = VectorDBConfig(
        collection_name="test_collection",
        embedding_dimension=4
    )
    
    db = VectorDBFactory.create(VectorDBProvider.FAISS, config)
    assert isinstance(db, FaissVectorDB)


def test_create_vectordb():
    """Test create_vectordb factory function."""
    db = create_vectordb(
        provider="faiss",
        collection_name="test_collection",
        embedding_dimension=4
    )
    
    assert isinstance(db, FaissVectorDB)
    assert db.collection_name == "test_collection"
    assert db.config.embedding_dimension == 4


@pytest.mark.asyncio
async def test_faiss_add_documents(faiss_db, sample_documents, sample_embeddings):
    """Test adding documents to FAISS."""
    doc_ids = await faiss_db.add_documents(sample_documents, sample_embeddings)
    
    assert len(doc_ids) == 3
    assert doc_ids[0] == "doc1"
    assert doc_ids[1] == "doc2"
    assert doc_ids[2] == "doc3"
    
    # Get document
    doc = await faiss_db.get_document("doc1")
    assert doc is not None
    assert doc.content == "This is a test document about AI"
    assert doc.metadata["category"] == "ai"


@pytest.mark.asyncio
async def test_faiss_similarity_search(faiss_db, sample_documents, sample_embeddings):
    """Test similarity search with FAISS."""
    # Add documents
    await faiss_db.add_documents(sample_documents, sample_embeddings)
    
    # Search with first embedding - should return first document with highest score
    results = await faiss_db.similarity_search(
        query_embedding=[1.0, 0.1, 0.1, 0.1],
        k=3
    )
    
    assert len(results) == 3
    assert results[0].document.doc_id == "doc1"
    assert results[0].score > 0.9  # High similarity score
    
    # Search with filter
    results = await faiss_db.similarity_search(
        query_embedding=[0.5, 0.5, 0.5, 0.5],
        k=3,
        filter={"category": "ml"}
    )
    
    assert len(results) == 1
    assert results[0].document.doc_id == "doc2"
    assert results[0].document.metadata["category"] == "ml"


@pytest.mark.asyncio
async def test_faiss_delete_document(faiss_db, sample_documents, sample_embeddings):
    """Test deleting a document from FAISS."""
    # Add documents
    await faiss_db.add_documents(sample_documents, sample_embeddings)
    
    # Delete a document
    success = await faiss_db.delete_document("doc1")
    assert success is True
    
    # Try to get deleted document
    doc = await faiss_db.get_document("doc1")
    assert doc is None
    
    # Get collection stats
    stats = await faiss_db.get_collection_stats()
    assert stats["docstore_size"] == 2 