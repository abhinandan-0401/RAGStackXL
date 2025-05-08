"""
Unit tests for the QueryExpansionRetriever.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval import QueryExpansionRetriever, RetrieverConfig, RetrieverType
from app.retrieval.query_expansion import SimpleQueryReformulation, LLMQueryReformulation
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument, DocumentMetadata


@pytest.fixture
def mock_vectordb():
    """Create a mock vector database."""
    vectordb = AsyncMock(spec=VectorDB)
    
    # Mock search results for different queries
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model.",
        doc_id="doc1",
        metadata={"source": "test"}
    )
    doc2 = RagDocument(
        content="Deep learning models have multiple hidden layers.",
        doc_id="doc2",
        metadata={"source": "test"}
    )
    doc3 = RagDocument(
        content="Transformers are a neural network architecture for NLP.",
        doc_id="doc3",
        metadata={"source": "test"}
    )
    doc4 = RagDocument(
        content="Python is a popular programming language for AI.",
        doc_id="doc4",
        metadata={"source": "test"}
    )
    
    # Different results for different query variations
    original_results = [
        SearchResult(document=doc1, score=0.9),
        SearchResult(document=doc3, score=0.7)
    ]
    
    # Results for a different query variation
    variation1_results = [
        SearchResult(document=doc2, score=0.85),
        SearchResult(document=doc1, score=0.75),
        SearchResult(document=doc4, score=0.6)
    ]
    
    # Results for another query variation
    variation2_results = [
        SearchResult(document=doc3, score=0.95),
        SearchResult(document=doc4, score=0.8),
        SearchResult(document=doc2, score=0.7)
    ]
    
    # Define side effects for different query embeddings
    def mock_similarity_search(query_embedding, **kwargs):
        # Return different results based on the embedding
        # Note: This is simplified - in a real implementation, we'd match on actual embeddings
        embedding_str = str(query_embedding)
        if "0.1, 0.2" in embedding_str:  # Original query
            return original_results
        elif "0.3, 0.4" in embedding_str:  # First variation
            return variation1_results
        elif "0.5, 0.6" in embedding_str:  # Second variation
            return variation2_results
        else:
            return []  # Default empty response
    
    vectordb.similarity_search.side_effect = mock_similarity_search
    return vectordb


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock(spec=EmbeddingModel)
    
    # Return different embeddings for different queries
    async def mock_embed_query(query):
        if "neural networks" == query:
            return [0.1, 0.2, 0.3]
        elif "neural" == query:
            return [0.3, 0.4, 0.5]
        elif "information about neural networks" == query:
            return [0.5, 0.6, 0.7]
        else:
            return [0.9, 0.9, 0.9]  # Default embedding
    
    model.embed_query.side_effect = mock_embed_query
    return model


@pytest.mark.asyncio
async def test_simple_query_reformulation():
    """Test SimpleQueryReformulation functionality."""
    reformulator = SimpleQueryReformulation()
    
    # Test with a simple query
    query = "What are neural networks?"
    variations = await reformulator.reformulate(query)
    
    # Should include original query
    assert query in variations
    
    # Should include a stopword-free version
    assert "neural networks" in variations
    
    # Test with synonyms
    query = "What is a quick algorithm?"
    variations = await reformulator.reformulate(query)
    
    # Should include variations with synonyms for "quick"
    assert any("fast" in var for var in variations)
    assert any("rapid" in var for var in variations)
    assert any("speedy" in var for var in variations)


@pytest.mark.asyncio
async def test_llm_query_reformulation():
    """Test LLMQueryReformulation functionality."""
    reformulator = LLMQueryReformulation()
    
    query = "neural networks"
    variations = await reformulator.reformulate(query)
    
    # Should include original query
    assert query in variations
    
    # Should include some predefined variations
    assert "information about neural networks" in variations
    assert "explain neural networks" in variations
    assert "tell me about neural networks" in variations


def test_retriever_init():
    """Test QueryExpansionRetriever initialization."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        top_k=3,
        max_queries=4,
        use_llm=True,
        aggregate_method="weighted"
    )
    
    retriever = QueryExpansionRetriever(vectordb, embedding_model, config)
    
    assert retriever.vectordb == vectordb
    assert retriever.embedding_model == embedding_model
    assert retriever.config == config
    assert retriever.config.top_k == 3
    assert retriever.max_queries == 4
    assert retriever.use_llm is True
    assert retriever.aggregate_method == "weighted"
    assert isinstance(retriever.reformulator, LLMQueryReformulation)
    
    # Test with different config
    config2 = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        use_llm=False
    )
    
    retriever2 = QueryExpansionRetriever(vectordb, embedding_model, config2)
    assert isinstance(retriever2.reformulator, SimpleQueryReformulation)


@pytest.mark.asyncio
async def test_retrieve_from_variations(mock_vectordb, mock_embedding_model):
    """Test retrieving documents from multiple query variations."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        top_k=3
    )
    
    retriever = QueryExpansionRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Test with multiple query variations
    query_variations = ["neural networks", "neural", "information about neural networks"]
    results = await retriever._retrieve_from_variations(
        query_variations=query_variations,
        k=3,
        filter=None
    )
    
    # Should have results for each query variation
    assert len(results) == 3
    assert "neural networks" in results
    assert "neural" in results
    assert "information about neural networks" in results
    
    # Embeddings should have been requested for each query
    assert mock_embedding_model.embed_query.call_count == 3
    
    # Similarity search should have been called for each query
    assert mock_vectordb.similarity_search.call_count == 3


@pytest.mark.asyncio
async def test_combine_results_max_strategy():
    """Test combining results with 'max' aggregation strategy."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        aggregate_method="max"
    )
    
    retriever = QueryExpansionRetriever(vectordb, embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(content="Test document 1", doc_id="doc1")
    doc2 = RagDocument(content="Test document 2", doc_id="doc2")
    
    # Create results for different query variations
    all_results = {
        "original query": [
            SearchResult(document=doc1, score=0.8),
            SearchResult(document=doc2, score=0.6)
        ],
        "variation 1": [
            SearchResult(document=doc1, score=0.7),
            SearchResult(document=doc2, score=0.9)  # Higher score for doc2
        ]
    }
    
    # Combine results
    combined = retriever._combine_results(
        original_query="original query",
        query_variations=["original query", "variation 1"],
        all_results=all_results,
        top_k=2
    )
    
    # With max strategy, each document should get its highest score across variations
    # doc1: max(0.8*1.0, 0.7*0.7) = 0.8
    # doc2: max(0.6*1.0, 0.9*0.7) = 0.63
    # So doc1 should be ranked first
    assert combined[0].document.doc_id == "doc1"
    assert combined[1].document.doc_id == "doc2"


@pytest.mark.asyncio
async def test_combine_results_weighted_strategy():
    """Test combining results with 'weighted' aggregation strategy."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        aggregate_method="weighted"
    )
    
    retriever = QueryExpansionRetriever(vectordb, embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(content="Test document 1", doc_id="doc1")
    doc2 = RagDocument(content="Test document 2", doc_id="doc2")
    
    # Create results for different query variations
    all_results = {
        "original query": [
            SearchResult(document=doc1, score=0.6),
            SearchResult(document=doc2, score=0.4)
        ],
        "variation 1": [
            SearchResult(document=doc2, score=0.9),  # Higher score for doc2
        ]
    }
    
    # Combine results
    combined = retriever._combine_results(
        original_query="original query",
        query_variations=["original query", "variation 1"],
        all_results=all_results,
        top_k=2
    )
    
    # With weighted strategy, scores are summed with weights
    # doc1: 0.6*1.0 = 0.6
    # doc2: 0.4*1.0 + 0.9*0.7 = 1.03
    # After normalization by max (1.03):
    # doc1: 0.6/1.03 = ~0.58
    # doc2: 1.03/1.03 = 1.0
    # So doc2 should be ranked first
    assert combined[0].document.doc_id == "doc2"
    assert combined[1].document.doc_id == "doc1"


@pytest.mark.asyncio
async def test_combine_results_mean_strategy():
    """Test combining results with 'mean' aggregation strategy."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        aggregate_method="mean"
    )
    
    retriever = QueryExpansionRetriever(vectordb, embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(content="Test document 1", doc_id="doc1")
    doc2 = RagDocument(content="Test document 2", doc_id="doc2")
    
    # Create results for different query variations
    all_results = {
        "original query": [
            SearchResult(document=doc1, score=1.0),
            SearchResult(document=doc2, score=0.8)
        ],
        "variation 1": [
            SearchResult(document=doc1, score=0.6),
            SearchResult(document=doc2, score=0.8)  # Same score for doc2
        ]
    }
    
    # Combine results
    combined = retriever._combine_results(
        original_query="original query",
        query_variations=["original query", "variation 1"],
        all_results=all_results,
        top_k=2
    )
    
    # With mean strategy, scores are averaged
    # doc1: (1.0 + 0.6)/2 = 0.8
    # doc2: (0.8 + 0.8)/2 = 0.8
    # They should have the same score, but doc1 appeared first in the original results
    assert combined[0].document.doc_id == "doc1"
    assert combined[1].document.doc_id == "doc2"
    assert combined[0].score == combined[1].score


@pytest.mark.asyncio
async def test_retrieve_end_to_end():
    """Test the entire retrieve method with mocked components."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        top_k=3,
        max_queries=2
    )
    
    retriever = QueryExpansionRetriever(vectordb, embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(content="Test document 1", doc_id="doc1")
    doc2 = RagDocument(content="Test document 2", doc_id="doc2")
    
    # Mock the reformulator to return known variations
    mock_reformulator = AsyncMock()
    mock_reformulator.reformulate.return_value = ["original query", "variation 1"]
    retriever.reformulator = mock_reformulator
    
    # Mock internal methods
    with patch.object(retriever, '_retrieve_from_variations') as mock_retrieve:
        with patch.object(retriever, '_combine_results') as mock_combine:
            with patch.object(retriever, '_publish_retrieval_event'):
                # Set up return values
                mock_retrieve.return_value = {
                    "original query": [SearchResult(document=doc1, score=0.8)]
                }
                mock_combine.return_value = [SearchResult(document=doc1, score=0.8)]
                
                # Call retrieve
                results = await retriever.retrieve("original query", filter={"source": "test"})
                
                # Verify the reformulator was called
                mock_reformulator.reformulate.assert_called_once_with("original query")
                
                # Verify _retrieve_from_variations was called with correct arguments
                mock_retrieve.assert_called_once_with(
                    query_variations=["original query", "variation 1"],
                    k=3,
                    filter={"source": "test"}
                )
                
                # Verify _combine_results was called
                mock_combine.assert_called_once()
                
                # Verify the results
                assert len(results) == 1
                assert results[0].document.doc_id == "doc1"


@pytest.mark.asyncio
async def test_retrieve_with_filter_function():
    """Test retrieval with a custom filter function."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    
    # Define a custom filter function
    def filter_fn(doc):
        return "neural" in doc.content.lower()
    
    config = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        top_k=3,
        filter_fn=filter_fn
    )
    
    retriever = QueryExpansionRetriever(vectordb, embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(content="Neural networks are a type of model.", doc_id="doc1")
    doc2 = RagDocument(content="This document has no relevant terms.", doc_id="doc2")
    
    # Mock the reformulator to return known variations
    mock_reformulator = AsyncMock()
    mock_reformulator.reformulate.return_value = ["test query"]
    retriever.reformulator = mock_reformulator
    
    # Mock internal methods
    with patch.object(retriever, '_retrieve_from_variations') as mock_retrieve:
        with patch.object(retriever, '_publish_retrieval_event'):
            # Set up return values (both docs)
            mock_retrieve.return_value = {
                "test query": [
                    SearchResult(document=doc1, score=0.8),
                    SearchResult(document=doc2, score=0.7)
                ]
            }
            
            # Call retrieve
            results = await retriever.retrieve("test query")
            
            # Verify the filter was applied - only doc1 should be in results
            assert len(results) == 1
            assert results[0].document.doc_id == "doc1" 