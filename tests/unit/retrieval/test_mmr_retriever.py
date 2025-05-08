"""
Unit tests for the MMRRetriever.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval import MMRRetriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument, DocumentMetadata


@pytest.fixture
def mock_vectordb():
    """Create a mock vector database."""
    vectordb = AsyncMock(spec=VectorDB)
    
    # Create test documents that are similar to each other
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model.",
        doc_id="doc1",
        metadata={"source": "test"}
    )
    doc2 = RagDocument(
        content="Deep neural networks have multiple hidden layers.",
        doc_id="doc2",
        metadata={"source": "test"}
    )  # Similar to doc1
    doc3 = RagDocument(
        content="Transformers are a neural network architecture for NLP.",
        doc_id="doc3",
        metadata={"source": "test"}
    )  # Similar to doc1 and doc2
    doc4 = RagDocument(
        content="Python is a popular programming language for AI.",
        doc_id="doc4",
        metadata={"source": "test"}
    )  # Different topic
    doc5 = RagDocument(
        content="Clustering is an unsupervised machine learning technique.",
        doc_id="doc5",
        metadata={"source": "test"}
    )  # Different from neural networks but related to ML
    
    # Mock search results for vector search
    search_tuples = [
        (doc1, 0.95),
        (doc2, 0.92),
        (doc3, 0.90),
        (doc5, 0.85),
        (doc4, 0.70)
    ]
    
    vectordb.similarity_search_with_score.return_value = search_tuples
    return vectordb


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model with realistic embeddings."""
    model = AsyncMock(spec=EmbeddingModel)
    
    # Create embeddings where doc1, doc2, doc3 are similar to each other
    # and doc4, doc5 are different
    query_embedding = [0.1, 0.2, 0.8, 0.1]
    
    doc1_embedding = [0.2, 0.3, 0.9, 0.1]  # Neural networks (similar to query)
    doc2_embedding = [0.1, 0.3, 0.9, 0.2]  # Deep neural networks (similar to doc1)
    doc3_embedding = [0.2, 0.4, 0.8, 0.1]  # Transformers (similar to doc1 and doc2)
    doc4_embedding = [0.8, 0.1, 0.2, 0.7]  # Python (different)
    doc5_embedding = [0.5, 0.4, 0.5, 0.3]  # Clustering (somewhat different)
    
    model.embed_query.return_value = query_embedding
    model.embed_documents.return_value = [
        doc1_embedding, doc2_embedding, doc3_embedding, doc5_embedding, doc4_embedding
    ]
    
    return model


def test_retriever_init():
    """Test MMRRetriever initialization."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.MMR,
        top_k=4,
        lambda_param=0.6,
        fetch_k=20
    )
    
    retriever = MMRRetriever(vectordb, embedding_model, config)
    
    assert retriever.vectordb == vectordb
    assert retriever.embedding_model == embedding_model
    assert retriever.config == config
    assert retriever.config.top_k == 4
    assert retriever.lambda_param == 0.6
    assert retriever.fetch_k == 20


@pytest.mark.asyncio
async def test_mmr_selection_high_lambda():
    """Test MMR selection algorithm with high lambda (focus on relevance)."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.MMR,
        top_k=3,
        lambda_param=0.9  # High lambda values prioritize relevance
    )
    
    retriever = MMRRetriever(vectordb, embedding_model, config)
    
    # Create query and document embeddings
    query_embedding = [0.1, 0.2, 0.8, 0.1]
    
    doc_embeddings = [
        [0.2, 0.3, 0.9, 0.1],  # doc1: Most similar to query
        [0.1, 0.3, 0.9, 0.2],  # doc2: Second most similar to query, very similar to doc1
        [0.2, 0.4, 0.8, 0.1],  # doc3: Third most similar to query, similar to doc1 and doc2
        [0.5, 0.4, 0.5, 0.3],  # doc4: Less similar to query
        [0.8, 0.1, 0.2, 0.7]   # doc5: Least similar to query
    ]
    
    doc_scores = [0.95, 0.92, 0.90, 0.85, 0.70]
    
    # Perform MMR selection
    selected_indices = retriever._mmr_selection(
        query_embedding=query_embedding,
        doc_embeddings=doc_embeddings,
        doc_scores=doc_scores,
        k=3,
        lambda_param=0.9
    )
    
    # With high lambda, should mostly follow relevance order
    # First selection should always be the most relevant document
    assert selected_indices[0] == 0  # doc1 has highest relevance
    
    # The rest should mostly follow relevance too
    assert set(selected_indices[:3]) == {0, 1, 2}  # Top 3 relevant docs


@pytest.mark.asyncio
async def test_mmr_selection_low_lambda():
    """Test MMR selection algorithm with low lambda (focus on diversity)."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.MMR,
        top_k=3,
        lambda_param=0.3  # Low lambda values prioritize diversity
    )
    
    retriever = MMRRetriever(vectordb, embedding_model, config)
    
    # Create query and document embeddings
    query_embedding = [0.1, 0.2, 0.8, 0.1]
    
    doc_embeddings = [
        [0.2, 0.3, 0.9, 0.1],  # doc1: Most similar to query
        [0.1, 0.3, 0.9, 0.2],  # doc2: Very similar to doc1 (should be deprioritized by MMR)
        [0.2, 0.4, 0.8, 0.1],  # doc3: Similar to doc1 (should be deprioritized by MMR)
        [0.5, 0.4, 0.5, 0.3],  # doc4: Different topic
        [0.8, 0.1, 0.2, 0.7]   # doc5: Very different topic
    ]
    
    doc_scores = [0.95, 0.92, 0.90, 0.85, 0.70]
    
    # Perform MMR selection
    selected_indices = retriever._mmr_selection(
        query_embedding=query_embedding,
        doc_embeddings=doc_embeddings,
        doc_scores=doc_scores,
        k=3,
        lambda_param=0.3
    )
    
    # With low lambda, should prioritize diversity over relevance
    # First selection should still be the most relevant document
    assert selected_indices[0] == 0  # doc1 has highest relevance
    
    # The next selections should prioritize diversity
    assert 4 in selected_indices  # doc5 should be included due to being different
    # Don't include both doc2 and doc3 since they're similar to each other
    assert not (1 in selected_indices and 2 in selected_indices)


@pytest.mark.asyncio
async def test_retrieve(mock_vectordb, mock_embedding_model):
    """Test the retrieve method with MMR selection."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.MMR,
        top_k=3,
        lambda_param=0.5,  # Balance between relevance and diversity
        fetch_k=10
    )
    
    retriever = MMRRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the MMR selection method to control its output
    with patch.object(retriever, '_mmr_selection', return_value=[0, 4, 3]):
        # Patch the event publishing method
        with patch.object(retriever, '_publish_retrieval_event'):
            # Call retrieve
            query = "neural networks"
            results = await retriever.retrieve(query)
            
            # Verify the embedding model was called
            mock_embedding_model.embed_query.assert_called_once_with(query)
            mock_embedding_model.embed_documents.assert_called_once()
            
            # Verify vector search was called
            mock_vectordb.similarity_search_with_score.assert_called_once()
            
            # Verify MMR selection was called
            retriever._mmr_selection.assert_called_once()
            
            # Verify the results (based on the mocked MMR selection)
            assert len(results) == 3
            assert results[0].document.doc_id == "doc1"
            assert results[1].document.doc_id == "doc4"
            assert results[2].document.doc_id == "doc5"


@pytest.mark.asyncio
async def test_real_mmr_selection(mock_vectordb, mock_embedding_model):
    """Test actual MMR selection implementation."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.MMR,
        top_k=3,
        lambda_param=0.5  # Equal balance between relevance and diversity
    )
    
    retriever = MMRRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch only the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve
        results = await retriever.retrieve("neural networks")
        
        # Verify the results with actual MMR selection
        assert len(results) == 3
        
        # First document should be most relevant
        assert results[0].document.doc_id == "doc1"
        
        # Results should include at least one document that's different
        # from the neural network documents
        different_docs = [r for r in results if r.document.doc_id in ["doc4", "doc5"]]
        assert len(different_docs) > 0


@pytest.mark.asyncio
async def test_mmr_with_filter(mock_vectordb, mock_embedding_model):
    """Test MMR retrieval with metadata filter."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.MMR,
        top_k=3,
        lambda_param=0.7
    )
    
    retriever = MMRRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve with filter
        test_filter = {"source": "test"}
        await retriever.retrieve("neural networks", filter=test_filter)
        
        # Verify vector search was called with the filter
        mock_vectordb.similarity_search_with_score.assert_called_once_with(
            query_embedding=mock_embedding_model.embed_query.return_value,
            k=30,  # This is the default fetch_k in the implementation
            filter=test_filter
        ) 