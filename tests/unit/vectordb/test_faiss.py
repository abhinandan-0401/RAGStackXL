"""
Tests for FAISS vector database implementation.
"""
import os
import pytest
import asyncio
import shutil
import numpy as np
from typing import List
import tempfile
from unittest.mock import patch

from app.vectordb.interfaces import VectorDBConfig, SearchResult
from app.vectordb.faiss import FaissVectorDB
from app.core.interfaces import RagDocument, event_bus, Event


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_documents() -> List[RagDocument]:
    """Sample documents for testing."""
    return [
        RagDocument(
            content="This is a test document about artificial intelligence",
            metadata={"source": "test", "category": "ai"},
            doc_id="doc1"
        ),
        RagDocument(
            content="This document is about machine learning algorithms",
            metadata={"source": "test", "category": "ml"},
            doc_id="doc2"
        ),
        RagDocument(
            content="Python is a popular programming language for data science",
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
def faiss_db(temp_dir):
    """Create a FAISS database instance for testing."""
    config = VectorDBConfig(
        collection_name="test_collection",
        embedding_dimension=4,  # Use small dimension for testing
        persist_directory=temp_dir
    )
    
    db = FaissVectorDB(config)
    return db


# Mock the event bus publish method to avoid errors
@pytest.fixture(autouse=True)
def mock_event_bus():
    """Mock the event bus publish method."""
    with patch.object(event_bus, 'publish') as mock_publish:
        yield mock_publish


@pytest.mark.asyncio
async def test_faiss_empty_index(faiss_db):
    """Test an empty FAISS index."""
    stats = await faiss_db.get_collection_stats()
    assert stats["count"] == 0
    assert stats["dimension"] == 4
    assert stats["docstore_size"] == 0


@pytest.mark.asyncio
async def test_faiss_add_documents(faiss_db, sample_documents, sample_embeddings, mock_event_bus):
    """Test adding documents to FAISS."""
    doc_ids = await faiss_db.add_documents(sample_documents, sample_embeddings)
    
    # Check that we got 3 document IDs
    assert len(doc_ids) == 3
    
    # Find document by content (regardless of ID)
    doc_with_ai_content = None
    for doc_id in doc_ids:
        doc = await faiss_db.get_document(doc_id)
        if doc and "artificial intelligence" in doc.content:
            doc_with_ai_content = doc
            break
    
    # Verify document was found and has correct metadata
    assert doc_with_ai_content is not None
    assert doc_with_ai_content.metadata["category"] == "ai"
    
    # Check stats
    stats = await faiss_db.get_collection_stats()
    assert stats["count"] == 3
    assert stats["docstore_size"] == 3
    
    # Verify event bus was called
    mock_event_bus.assert_called_once()


@pytest.mark.asyncio
async def test_faiss_similarity_search(faiss_db, sample_documents, sample_embeddings, mock_event_bus):
    """Test similarity search with FAISS."""
    # Add documents
    await faiss_db.add_documents(sample_documents, sample_embeddings)
    
    # Test search with embedding similar to first document
    query_embedding = [0.9, 0.1, 0.0, 0.0]  # More similar to first embedding
    results = await faiss_db.similarity_search(
        query_embedding=query_embedding,
        k=3
    )
    
    assert len(results) == 3
    # First result should have content about AI since query is closest to first embedding
    assert "artificial intelligence" in results[0].document.content
    
    # Test filter
    results = await faiss_db.similarity_search(
        query_embedding=query_embedding,
        k=3,
        filter={"category": "ml"}
    )
    
    assert len(results) == 1
    assert "machine learning" in results[0].document.content
    assert results[0].document.metadata["category"] == "ml"


@pytest.mark.asyncio
async def test_faiss_persistence(temp_dir, sample_documents, sample_embeddings, mock_event_bus):
    """Test FAISS persistence."""
    # Create and populate first instance
    config = VectorDBConfig(
        collection_name="persist_test",
        embedding_dimension=4,
        persist_directory=temp_dir
    )
    
    db1 = FaissVectorDB(config)
    doc_ids = await db1.add_documents(sample_documents, sample_embeddings)
    
    # Create second instance with same config (should load from disk)
    db2 = FaissVectorDB(config)
    
    # Verify documents were loaded by checking for content
    found_ai_doc = False
    found_ml_doc = False
    found_python_doc = False
    
    # Get all docs and check their content
    stats = await db2.get_collection_stats()
    assert stats["count"] == 3
    
    for doc_id in doc_ids:
        doc = await db2.get_document(doc_id)
        if doc:
            if "artificial intelligence" in doc.content:
                found_ai_doc = True
            elif "machine learning" in doc.content:
                found_ml_doc = True
            elif "programming language" in doc.content:
                found_python_doc = True
    
    assert found_ai_doc, "AI document not found after persistence"
    assert found_ml_doc, "ML document not found after persistence"
    assert found_python_doc, "Python document not found after persistence" 