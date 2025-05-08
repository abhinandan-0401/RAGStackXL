"""
Unit tests for the HybridRetriever.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval import HybridRetriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument, DocumentMetadata


@pytest.fixture
def mock_vectordb():
    """Create a mock vector database."""
    vectordb = AsyncMock(spec=VectorDB)
    
    # Mock search results for vector search
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model.",
        doc_id="doc1",
        metadata={"source": "test"}
    )
    doc2 = RagDocument(
        content="Transformers are a neural network architecture.",
        doc_id="doc2",
        metadata={"source": "test"}
    )
    doc3 = RagDocument(
        content="Python is a popular programming language.",
        doc_id="doc3",
        metadata={"source": "test"}
    )
    
    vector_results = [
        SearchResult(document=doc2, score=0.9),  # Vector search prioritizes doc2
        SearchResult(document=doc1, score=0.7),
        SearchResult(document=doc3, score=0.5)
    ]
    
    # Mock collection stats
    collection_stats = {
        "count": 3,
        "dimension": 384,
        "name": "test_collection"
    }
    
    vectordb.similarity_search.return_value = vector_results
    vectordb.get_collection_stats.return_value = collection_stats
    return vectordb


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock(spec=EmbeddingModel)
    model.embed_query.return_value = [0.1, 0.2, 0.3]
    return model


def test_retriever_init():
    """Test HybridRetriever initialization."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.HYBRID,
        top_k=4,
        vector_weight=0.6,
        keyword_weight=0.4,
        use_bm25=True,
        min_keyword_length=3
    )
    
    retriever = HybridRetriever(vectordb, embedding_model, config)
    
    assert retriever.vectordb == vectordb
    assert retriever.embedding_model == embedding_model
    assert retriever.config == config
    assert retriever.config.top_k == 4
    assert retriever.vector_weight == 0.6
    assert retriever.keyword_weight == 0.4
    assert retriever.use_bm25 is True
    assert retriever.min_keyword_length == 3


@pytest.mark.asyncio
async def test_extract_keywords():
    """Test keyword extraction."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.HYBRID,
        min_keyword_length=3
    )
    
    retriever = HybridRetriever(vectordb, embedding_model, config)
    
    # Test keyword extraction
    keywords = retriever._extract_keywords("What are neural networks and machine learning?")
    
    # Stopwords should be removed
    assert "are" not in keywords
    assert "and" not in keywords
    
    # Short words should be removed
    assert "is" not in keywords
    
    # Important terms should be included
    assert "neural" in keywords
    assert "networks" in keywords
    assert "machine" in keywords
    assert "learning" in keywords


@pytest.mark.asyncio
async def test_perform_vector_search(mock_vectordb, mock_embedding_model):
    """Test vector search component."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.HYBRID,
        top_k=3
    )
    
    retriever = HybridRetriever(mock_vectordb, mock_embedding_model, config)
    
    query = "neural networks"
    filter = {"source": "test"}
    
    results = await retriever._perform_vector_search(query, 3, filter)
    
    # Check that embedding model was called
    mock_embedding_model.embed_query.assert_called_once_with(query)
    
    # Check that vector search was called with correct parameters
    mock_vectordb.similarity_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        k=3,
        filter=filter
    )
    
    # Check results
    assert len(results) == 3
    assert results[0].document.doc_id == "doc2"
    assert results[1].document.doc_id == "doc1"
    assert results[2].document.doc_id == "doc3"


@pytest.mark.asyncio
async def test_combine_results(mock_vectordb, mock_embedding_model):
    """Test result combination logic."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.HYBRID,
        top_k=3,
        vector_weight=0.6,
        keyword_weight=0.4
    )
    
    retriever = HybridRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model.",
        doc_id="doc1",
        metadata={"source": "test"}
    )
    doc2 = RagDocument(
        content="Transformers are a neural network architecture.",
        doc_id="doc2",
        metadata={"source": "test"}
    )
    doc3 = RagDocument(
        content="Python is a popular programming language.",
        doc_id="doc3",
        metadata={"source": "test"}
    )
    
    # Create vector results (doc2 has higher vector similarity)
    vector_results = [
        SearchResult(document=doc2, score=0.9),
        SearchResult(document=doc1, score=0.7),
        SearchResult(document=doc3, score=0.5)
    ]
    
    # Create keyword results (doc1 has higher keyword match)
    keyword_results = [
        SearchResult(document=doc1, score=0.8),
        SearchResult(document=doc2, score=0.3),
        SearchResult(document=doc3, score=0.1)
    ]
    
    # Combine results
    combined = retriever._combine_results("test query", vector_results, keyword_results, 3)
    
    # Doc1 should be ranked highest after combination
    # doc1 combined score: 0.6*0.7 + 0.4*0.8 = 0.74
    # doc2 combined score: 0.6*0.9 + 0.4*0.3 = 0.66
    assert combined[0].document.doc_id == "doc1"
    assert combined[1].document.doc_id == "doc2"


@pytest.mark.asyncio
async def test_retrieve_with_different_weights(mock_vectordb, mock_embedding_model):
    """Test retrieval with different vector and keyword weights."""
    # Create patches for the submethods to control their outputs
    
    # Create test documents
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model.",
        doc_id="doc1",
        metadata={"source": "test"}
    )
    doc2 = RagDocument(
        content="Transformers are a neural network architecture.",
        doc_id="doc2",
        metadata={"source": "test"}
    )
    
    # Vector results (doc2 has higher vector similarity)
    vector_results = [
        SearchResult(document=doc2, score=0.9),
        SearchResult(document=doc1, score=0.7)
    ]
    
    # Keyword results (doc1 has higher keyword match)
    keyword_results = [
        SearchResult(document=doc1, score=0.8),
        SearchResult(document=doc2, score=0.3)
    ]
    
    # Test with vector_weight > keyword_weight
    config1 = RetrieverConfig(
        retriever_type=RetrieverType.HYBRID,
        top_k=2,
        vector_weight=0.8,
        keyword_weight=0.2
    )
    
    retriever1 = HybridRetriever(mock_vectordb, mock_embedding_model, config1)
    
    with patch.object(retriever1, '_perform_vector_search', return_value=vector_results):
        with patch.object(retriever1, '_perform_keyword_search', return_value=keyword_results):
            with patch.object(retriever1, '_publish_retrieval_event'):
                results1 = await retriever1.retrieve("test query")
                
                # With high vector weight, doc2 should be ranked first
                # doc2: 0.8*0.9 + 0.2*0.3 = 0.78
                # doc1: 0.8*0.7 + 0.2*0.8 = 0.72
                assert results1[0].document.doc_id == "doc2"
    
    # Test with keyword_weight > vector_weight
    config2 = RetrieverConfig(
        retriever_type=RetrieverType.HYBRID,
        top_k=2,
        vector_weight=0.3,
        keyword_weight=0.7
    )
    
    retriever2 = HybridRetriever(mock_vectordb, mock_embedding_model, config2)
    
    with patch.object(retriever2, '_perform_vector_search', return_value=vector_results):
        with patch.object(retriever2, '_perform_keyword_search', return_value=keyword_results):
            with patch.object(retriever2, '_publish_retrieval_event'):
                results2 = await retriever2.retrieve("test query")
                
                # With high keyword weight, doc1 should be ranked first
                # doc2: 0.3*0.9 + 0.7*0.3 = 0.48
                # doc1: 0.3*0.7 + 0.7*0.8 = 0.77
                assert results2[0].document.doc_id == "doc1"


@pytest.mark.asyncio
async def test_retrieve_end_to_end(mock_vectordb, mock_embedding_model):
    """Test the entire retrieve method."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.HYBRID,
        top_k=3,
        vector_weight=0.5,
        keyword_weight=0.5
    )
    
    retriever = HybridRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Create mocks for the internal methods to isolate the test
    with patch.object(retriever, '_perform_vector_search') as mock_vector_search:
        with patch.object(retriever, '_perform_keyword_search') as mock_keyword_search:
            with patch.object(retriever, '_publish_retrieval_event'):
                # Set up return values for the mocked methods
                doc1 = RagDocument(content="Test document 1", doc_id="doc1")
                doc2 = RagDocument(content="Test document 2", doc_id="doc2")
                
                mock_vector_search.return_value = [
                    SearchResult(document=doc1, score=0.8),
                    SearchResult(document=doc2, score=0.6)
                ]
                
                mock_keyword_search.return_value = [
                    SearchResult(document=doc2, score=0.9),
                    SearchResult(document=doc1, score=0.4)
                ]
                
                # Call retrieve
                results = await retriever.retrieve("test query", filter={"source": "test"})
                
                # Check that vector search was called with the correct arguments
                mock_vector_search.assert_called_once_with(
                    "test query", 
                    3, 
                    {"source": "test"}
                )
                
                # Check that keyword search was called with the correct arguments
                mock_keyword_search.assert_called_once_with(
                    "test query", 
                    3, 
                    {"source": "test"}
                )
                
                # Verify the correct number of results
                assert len(results) == 2
                
                # With equal weights, doc2 should be ranked first due to higher combined score
                # doc1: 0.5*0.8 + 0.5*0.4 = 0.6
                # doc2: 0.5*0.6 + 0.5*0.9 = 0.75
                assert results[0].document.doc_id == "doc2"
                assert results[1].document.doc_id == "doc1" 