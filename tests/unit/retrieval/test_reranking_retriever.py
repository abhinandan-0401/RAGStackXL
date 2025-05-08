"""
Unit tests for the RerankingRetriever.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval import RerankingRetriever, RetrieverConfig, RetrieverType
from app.retrieval.reranking import SimpleScoreReranker, CosineSimilarityReranker
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument, DocumentMetadata


@pytest.fixture
def mock_vectordb():
    """Create a mock vector database."""
    vectordb = AsyncMock(spec=VectorDB)
    
    # Create test documents
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
        content="Python is a popular programming language for data science and AI.",
        doc_id="doc3",
        metadata={"source": "test"}
    )
    doc4 = RagDocument(
        content="The term neural networks refers to systems of neurons, either artificial or biological.",
        doc_id="doc4",
        metadata={"source": "test"}
    )
    doc5 = RagDocument(
        content="Machine learning algorithms can be trained on data to make predictions.",
        doc_id="doc5",
        metadata={"source": "test"}
    )
    
    # Mock search results
    search_tuples = [
        (doc2, 0.92),  # Vector similarity might rank transformers highest
        (doc1, 0.90),
        (doc4, 0.85),
        (doc3, 0.70),
        (doc5, 0.65),
    ]
    
    vectordb.similarity_search_with_score.return_value = search_tuples
    return vectordb


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock(spec=EmbeddingModel)
    
    # Return query embedding
    model.embed_query.return_value = [0.1, 0.2, 0.8, 0.1]
    
    # Return document embeddings
    # Designed so doc4 has highest cosine similarity with query, followed by doc1
    model.embed_documents.return_value = [
        [0.2, 0.3, 0.7, 0.2],  # doc2
        [0.1, 0.3, 0.8, 0.1],  # doc1
        [0.3, 0.5, 0.6, 0.1],  # doc4
        [0.8, 0.1, 0.2, 0.7],  # doc3
        [0.5, 0.4, 0.5, 0.3],  # doc5
    ]
    
    return model


def test_retriever_init_simple_reranker():
    """Test RerankingRetriever initialization with simple reranker."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.RERANKING,
        top_k=5,
        initial_k=20,
        reranker_type="simple"
    )
    
    retriever = RerankingRetriever(vectordb, embedding_model, config)
    
    assert retriever.vectordb == vectordb
    assert retriever.embedding_model == embedding_model
    assert retriever.config == config
    assert retriever.config.top_k == 5
    assert retriever.initial_k == 20
    assert retriever.reranker_type == "simple"
    assert isinstance(retriever.reranker, SimpleScoreReranker)


def test_retriever_init_cosine_reranker():
    """Test RerankingRetriever initialization with cosine reranker."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.RERANKING,
        top_k=4,
        initial_k=15,
        reranker_type="cosine"
    )
    
    retriever = RerankingRetriever(vectordb, embedding_model, config)
    
    assert retriever.vectordb == vectordb
    assert retriever.embedding_model == embedding_model
    assert retriever.config == config
    assert retriever.config.top_k == 4
    assert retriever.initial_k == 15
    assert retriever.reranker_type == "cosine"
    assert isinstance(retriever.reranker, CosineSimilarityReranker)


@pytest.mark.asyncio
async def test_simple_score_reranker():
    """Test SimpleScoreReranker functionality."""
    reranker = SimpleScoreReranker()
    
    # Create test documents with different properties
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model.",
        doc_id="doc1"
    )
    doc2 = RagDocument(
        content="This document has nothing to do with the query.",
        doc_id="doc2"
    )
    doc3 = RagDocument(
        content="Neural networks and deep learning are used for AI tasks.",
        doc_id="doc3"
    )
    
    # Original scores favor doc2
    original_scores = [0.7, 0.9, 0.6]
    
    # Rerank with query about neural networks
    reranked = await reranker.rerank(
        query="neural networks",
        documents=[doc1, doc2, doc3],
        scores=original_scores
    )
    
    # Doc1 and doc3 should be reranked higher than doc2, since they contain the query terms
    doc_ids = [doc.doc_id for doc, _ in reranked]
    
    # Check that doc2 is not first (despite having highest original score)
    assert doc_ids[0] != "doc2"
    
    # Check that docs with query terms are ranked higher
    assert doc_ids.index("doc1") < doc_ids.index("doc2")
    assert doc_ids.index("doc3") < doc_ids.index("doc2")


@pytest.mark.asyncio
async def test_cosine_similarity_reranker(mock_embedding_model):
    """Test CosineSimilarityReranker functionality."""
    reranker = CosineSimilarityReranker(mock_embedding_model)
    
    # Create test documents
    doc1 = RagDocument(content="Document 1", doc_id="doc1")
    doc2 = RagDocument(content="Document 2", doc_id="doc2")
    doc3 = RagDocument(content="Document 3", doc_id="doc3")
    
    # Set up mock embeddings to favor doc2
    query_embedding = [0.1, 0.2, 0.3]
    doc_embeddings = [
        [0.2, 0.3, 0.4],  # doc1 (medium similarity)
        [0.1, 0.2, 0.3],  # doc2 (highest similarity, identical to query)
        [0.9, 0.8, 0.7]   # doc3 (lowest similarity)
    ]
    
    mock_embedding_model.embed_query.return_value = query_embedding
    mock_embedding_model.embed_documents.return_value = doc_embeddings
    
    # Rerank documents
    reranked = await reranker.rerank(
        query="test query",
        documents=[doc1, doc2, doc3]
    )
    
    # Check ranking - doc2 should be first due to highest cosine similarity
    assert reranked[0][0].doc_id == "doc2"
    assert reranked[1][0].doc_id == "doc1"
    assert reranked[2][0].doc_id == "doc3"
    
    # Check that scores are in range [0, 1]
    for _, score in reranked:
        assert 0 <= score <= 1


@pytest.mark.asyncio
async def test_retrieve_with_simple_reranker(mock_vectordb, mock_embedding_model):
    """Test retrieval with SimpleScoreReranker."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.RERANKING,
        top_k=3,
        initial_k=5,
        reranker_type="simple"
    )
    
    retriever = RerankingRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Create a custom simple reranker that will rerank doc1 highest
    custom_reranker = AsyncMock()
    custom_reranker.rerank.return_value = [
        # Rerank doc1 highest (originally ranked 2nd by vector search)
        (mock_vectordb.similarity_search_with_score.return_value[1][0], 0.95),
        (mock_vectordb.similarity_search_with_score.return_value[0][0], 0.85),
        (mock_vectordb.similarity_search_with_score.return_value[2][0], 0.75),
        (mock_vectordb.similarity_search_with_score.return_value[3][0], 0.65),
        (mock_vectordb.similarity_search_with_score.return_value[4][0], 0.55)
    ]
    retriever.reranker = custom_reranker
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve
        query = "What are neural networks?"
        results = await retriever.retrieve(query)
        
        # Verify embedding model was called
        mock_embedding_model.embed_query.assert_called_once_with(query)
        
        # Verify vector search was called with initial_k
        mock_vectordb.similarity_search_with_score.assert_called_once_with(
            query_embedding=mock_embedding_model.embed_query.return_value,
            k=9,  # Implementation uses max(initial_k, top_k * 3) = max(5, 9) = 9
            filter=None
        )
        
        # Verify reranker was called
        custom_reranker.rerank.assert_called_once()
        
        # Verify results are limited to top_k
        assert len(results) == 3
        
        # Verify reranking changed the order
        assert results[0].document.doc_id == "doc1"  # Originally ranked 2nd
        assert results[1].document.doc_id == "doc2"  # Originally ranked 1st
        assert results[2].document.doc_id == "doc4"  # Originally ranked 3rd


@pytest.mark.asyncio
async def test_retrieve_with_cosine_reranker(mock_vectordb, mock_embedding_model):
    """Test retrieval with CosineSimilarityReranker."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.RERANKING,
        top_k=3,
        initial_k=5,
        reranker_type="cosine"
    )
    
    retriever = RerankingRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Create a custom cosine reranker that will rerank doc4 highest
    custom_reranker = AsyncMock()
    custom_reranker.rerank.return_value = [
        # Rerank doc4 highest (originally ranked 3rd by vector search)
        (mock_vectordb.similarity_search_with_score.return_value[2][0], 0.95),
        (mock_vectordb.similarity_search_with_score.return_value[1][0], 0.90),
        (mock_vectordb.similarity_search_with_score.return_value[0][0], 0.85),
        (mock_vectordb.similarity_search_with_score.return_value[3][0], 0.70),
        (mock_vectordb.similarity_search_with_score.return_value[4][0], 0.65)
    ]
    retriever.reranker = custom_reranker
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve
        query = "What are neural networks?"
        results = await retriever.retrieve(query)
        
        # Verify results are limited to top_k
        assert len(results) == 3
        
        # Verify reranking changed the order
        assert results[0].document.doc_id == "doc4"  # Originally ranked 3rd
        assert results[1].document.doc_id == "doc1"  # Originally ranked 2nd
        assert results[2].document.doc_id == "doc2"  # Originally ranked 1st


@pytest.mark.asyncio
async def test_retrieve_with_filter(mock_vectordb, mock_embedding_model):
    """Test retrieval with metadata filter."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.RERANKING,
        top_k=3
    )
    
    retriever = RerankingRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Mock the reranker to return unchanged order
    with patch.object(retriever.reranker, 'rerank', return_value=[
        (doc, score) for doc, score in mock_vectordb.similarity_search_with_score.return_value
    ]):
        # Patch the event publishing method
        with patch.object(retriever, '_publish_retrieval_event'):
            # Call retrieve with filter
            test_filter = {"source": "test"}
            await retriever.retrieve("test query", filter=test_filter)
            
            # Verify vector search was called with the filter
            mock_vectordb.similarity_search_with_score.assert_called_once_with(
                query_embedding=mock_embedding_model.embed_query.return_value,
                k=retriever.initial_k,
                filter=test_filter
            )


@pytest.mark.asyncio
async def test_default_score_fn():
    """Test the default scoring function of SimpleScoreReranker."""
    reranker = SimpleScoreReranker()
    
    # Create test documents
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model inspired by the human brain.",
        doc_id="doc1"
    )
    
    doc2 = RagDocument(
        content="This document does not contain relevant query terms but has some other content.",
        doc_id="doc2"
    )
    
    doc3 = RagDocument(
        content="Neural networks and neural language models are foundational to modern AI. Neural networks process information similar to how neurons work in the brain.",
        doc_id="doc3"
    )
    
    # Test scoring function
    query = "neural networks"
    
    # Score each document
    score1 = reranker._default_score_fn(query, doc1, 0.8)
    score2 = reranker._default_score_fn(query, doc2, 0.8)
    score3 = reranker._default_score_fn(query, doc3, 0.8)
    
    # Doc with query terms should score higher than doc without
    assert score1 > score2
    
    # Doc with multiple occurrences of query terms should score higher or equal
    assert score3 >= score1
    
    # All scores should be in range [0, 1]
    assert 0 <= score1 <= 1
    assert 0 <= score2 <= 1
    assert 0 <= score3 <= 1


@pytest.mark.asyncio
async def test_reranker_with_custom_scoring_fn():
    """Test SimpleScoreReranker with a custom scoring function."""
    # Define custom scoring function that prioritizes document length
    def custom_score_fn(query, doc, original_score):
        # Simply score based on document length - longer is better
        return min(len(doc.content) / 200, 1.0)
    
    reranker = SimpleScoreReranker(scoring_fn=custom_score_fn)
    
    # Create test documents of different lengths
    short_doc = RagDocument(content="Short document", doc_id="short")
    medium_doc = RagDocument(content="Medium length document with more content than the short one", doc_id="medium")
    long_doc = RagDocument(
        content="Long document with lots of content. This document should be scored highest by our custom scoring function that prioritizes document length. The longer the document, the higher the score it should receive.",
        doc_id="long"
    )
    
    # Rerank documents
    reranked = await reranker.rerank(
        query="test query",
        documents=[short_doc, medium_doc, long_doc]
    )
    
    # Long document should be ranked first
    assert reranked[0][0].doc_id == "long"
    assert reranked[1][0].doc_id == "medium"
    assert reranked[2][0].doc_id == "short" 