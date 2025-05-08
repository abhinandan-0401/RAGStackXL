"""
Unit tests for the SemanticRetriever.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval import SemanticRetriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument, DocumentMetadata


@pytest.fixture
def mock_vectordb():
    """Create a mock vector database."""
    vectordb = AsyncMock(spec=VectorDB)
    
    # Mock search results
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model inspired by the human brain.",
        doc_id="doc1",
        metadata={"source": "test"}
    )
    doc2 = RagDocument(
        content="Transformers are a type of neural network architecture that uses self-attention mechanisms.",
        doc_id="doc2",
        metadata={"source": "test"}
    )
    doc3 = RagDocument(
        content="Python is a popular programming language for data science and machine learning.",
        doc_id="doc3",
        metadata={"source": "test"}
    )
    
    search_tuples = [
        (doc2, 0.9),  # Doc2 has higher vector similarity
        (doc1, 0.8),  # Doc1 has both keywords "neural" and "networks"
        (doc3, 0.5)   # Doc3 has lower relevance
    ]
    
    vectordb.similarity_search_with_score.return_value = search_tuples
    return vectordb


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock(spec=EmbeddingModel)
    model.embed_query.return_value = [0.1, 0.2, 0.3]
    return model


def test_retriever_init():
    """Test SemanticRetriever initialization."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.SEMANTIC,
        top_k=4,
        use_keyword_boost=True,
        keyword_boost_factor=0.2,
        relevance_threshold=0.5,
        semantic_weight=0.8
    )
    
    retriever = SemanticRetriever(vectordb, embedding_model, config)
    
    assert retriever.vectordb == vectordb
    assert retriever.embedding_model == embedding_model
    assert retriever.config == config
    assert retriever.config.top_k == 4
    assert retriever.use_keyword_boost is True
    assert retriever.keyword_boost_factor == 0.2
    assert retriever.relevance_threshold == 0.5
    assert retriever.semantic_weight == 0.8


@pytest.mark.asyncio
async def test_retrieve_with_keyword_boost(mock_vectordb, mock_embedding_model):
    """Test retrieval with keyword boosting."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.SEMANTIC,
        top_k=3,
        use_keyword_boost=True,
        keyword_boost_factor=0.3,
        semantic_weight=0.7
    )
    
    # Create a retriever with mocked dependencies
    retriever = SemanticRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the keyword extraction method to return known keywords
    with patch.object(retriever, '_extract_keywords', return_value=["neural", "networks"]):
        # Patch the event publishing method
        with patch.object(retriever, '_publish_retrieval_event'):
            # Call retrieve
            query = "What are neural networks?"
            results = await retriever.retrieve(query)
            
            # Verify the embedding model was called with the query
            mock_embedding_model.embed_query.assert_called_once_with(query)
            
            # Verify search was called
            mock_vectordb.similarity_search_with_score.assert_called_once()
            
            # Doc1 should now be ranked highest due to keyword boosting
            # Doc1 has both "neural" and "networks" keywords, so it gets full boost
            assert results[0].document.doc_id == "doc1"
            assert results[1].document.doc_id == "doc2"


@pytest.mark.asyncio
async def test_retrieve_without_keyword_boost(mock_vectordb, mock_embedding_model):
    """Test retrieval without keyword boosting."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.SEMANTIC,
        top_k=3,
        use_keyword_boost=False,
        semantic_weight=1.0
    )
    
    # Create a retriever with mocked dependencies
    retriever = SemanticRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve
        results = await retriever.retrieve("What are neural networks?")
        
        # Without keyword boosting, doc2 should remain highest ranked
        # as it has the highest vector similarity
        assert results[0].document.doc_id == "doc2"
        assert results[1].document.doc_id == "doc1"


@pytest.mark.asyncio
async def test_retrieve_with_relevance_threshold(mock_vectordb, mock_embedding_model):
    """Test retrieval with relevance threshold."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.SEMANTIC,
        top_k=3,
        relevance_threshold=0.6,
        use_keyword_boost=False
    )
    
    # Create a retriever with mocked dependencies
    retriever = SemanticRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve
        results = await retriever.retrieve("What are neural networks?")
        
        # Doc3 should be filtered out as its score is below threshold
        assert len(results) == 2
        assert all(result.document.doc_id != "doc3" for result in results)


@pytest.mark.asyncio
async def test_extract_keywords():
    """Test keyword extraction."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(retriever_type=RetrieverType.SEMANTIC)
    
    retriever = SemanticRetriever(vectordb, embedding_model, config)
    
    # Test keyword extraction
    keywords = retriever._extract_keywords("What are neural networks and how do they work?")
    
    # Stopwords should be removed
    assert "are" not in keywords
    assert "and" not in keywords
    assert "how" not in keywords
    assert "do" not in keywords
    
    # Important terms should be included
    assert "neural" in keywords
    assert "networks" in keywords
    assert "work" in keywords


@pytest.mark.asyncio
async def test_semantic_adjustments():
    """Test the semantic adjustment logic."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.SEMANTIC,
        keyword_boost_factor=0.3,
        semantic_weight=0.7
    )
    
    retriever = SemanticRetriever(vectordb, embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model.",
        doc_id="doc1"
    )
    doc2 = RagDocument(
        content="Transformers are a neural network architecture.",
        doc_id="doc2"
    )
    
    # Original scores (doc2 has higher vector similarity)
    search_tuples = [
        (doc2, 0.9),
        (doc1, 0.8)
    ]
    
    # Apply adjustments with two keywords, both present in doc1
    keywords = ["neural", "networks"]
    results = retriever._apply_semantic_adjustments("test query", search_tuples, keywords)
    
    # Doc1 contains both keywords, so should get full boost: 0.8*0.7 + (2/2)*0.3 = 0.86
    # Doc2 contains one keyword, so gets partial boost: 0.9*0.7 + (1/2)*0.3 = 0.78
    
    # Doc1 should now be ranked higher
    assert results[0].document.doc_id == "doc1"
    assert results[1].document.doc_id == "doc2" 