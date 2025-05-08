"""
Unit tests for the BasicRetriever.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval import BasicRetriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument, DocumentMetadata


@pytest.fixture
def mock_vectordb():
    """Create a mock vector database."""
    vectordb = AsyncMock(spec=VectorDB)
    
    # Mock search results
    doc1 = RagDocument(
        content="This is document 1",
        doc_id="doc1",
        metadata={"source": "test"}
    )
    doc2 = RagDocument(
        content="This is document 2",
        doc_id="doc2",
        metadata={"source": "test"}
    )
    
    search_results = [
        SearchResult(document=doc1, score=0.9),
        SearchResult(document=doc2, score=0.7)
    ]
    
    vectordb.similarity_search.return_value = search_results
    return vectordb


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock(spec=EmbeddingModel)
    model.embed_query.return_value = [0.1, 0.2, 0.3]
    return model


def test_retriever_init():
    """Test BasicRetriever initialization."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(retriever_type=RetrieverType.BASIC, top_k=4)
    
    retriever = BasicRetriever(vectordb, embedding_model, config)
    
    assert retriever.vectordb == vectordb
    assert retriever.embedding_model == embedding_model
    assert retriever.config == config
    assert retriever.config.top_k == 4


@pytest.mark.asyncio
async def test_retrieve(mock_vectordb, mock_embedding_model):
    """Test basic retrieval functionality."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.BASIC,
        top_k=4
    )
    
    # Create a retriever with mocked dependencies
    retriever = BasicRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve
        results = await retriever.retrieve("test query")
        
        # Verify the embedding model was called with the query
        mock_embedding_model.embed_query.assert_called_once_with("test query")
        
        # Verify vector search was called with the embedding
        mock_vectordb.similarity_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            k=4,
            filter=None
        )
        
        # Verify the results
        assert len(results) == 2
        assert results[0].document.doc_id == "doc1"
        assert results[0].score == 0.9
        assert results[1].document.doc_id == "doc2"
        assert results[1].score == 0.7
        
        # Verify event was published
        retriever._publish_retrieval_event.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_with_custom_k(mock_vectordb, mock_embedding_model):
    """Test retrieval with custom k override."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.BASIC,
        top_k=4
    )
    
    # Create a retriever with mocked dependencies
    retriever = BasicRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve with custom k
        await retriever.retrieve("test query", k=10)
        
        # Verify vector search was called with the custom k
        mock_vectordb.similarity_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            k=10,
            filter=None
        )


@pytest.mark.asyncio
async def test_retrieve_with_filter(mock_vectordb, mock_embedding_model):
    """Test retrieval with metadata filter."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.BASIC,
        top_k=4
    )
    
    # Create a retriever with mocked dependencies
    retriever = BasicRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve with filter
        test_filter = {"source": "test"}
        await retriever.retrieve("test query", filter=test_filter)
        
        # Verify vector search was called with the filter
        mock_vectordb.similarity_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            k=4,
            filter=test_filter
        )


@pytest.mark.asyncio
async def test_retrieve_with_custom_filter_function(mock_vectordb, mock_embedding_model):
    """Test retrieval with custom filter function."""
    # Define a filter function
    def custom_filter(doc):
        return doc.doc_id == "doc1"
    
    config = RetrieverConfig(
        retriever_type=RetrieverType.BASIC,
        top_k=4,
        filter_fn=custom_filter
    )
    
    # Create a retriever with mocked dependencies
    retriever = BasicRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve
        results = await retriever.retrieve("test query")
        
        # Verify only doc1 passes the filter
        assert len(results) == 1
        assert results[0].document.doc_id == "doc1" 