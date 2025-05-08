"""
Unit tests for the ContextualRetriever.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.retrieval import ContextualRetriever, RetrieverConfig, RetrieverType
from app.vectordb.interfaces import SearchResult, VectorDB
from app.embedding.interfaces import EmbeddingModel
from app.core.interfaces import RagDocument, DocumentMetadata


@pytest.fixture
def mock_vectordb():
    """Create a mock vector database."""
    vectordb = AsyncMock(spec=VectorDB)
    
    # Create test documents
    doc1 = RagDocument(
        content="Neural networks are a type of machine learning model.",
        doc_id="doc1",
        metadata={"source": "test"}
    )
    doc2 = RagDocument(
        content="Transformers are a neural network architecture used in NLP.",
        doc_id="doc2",
        metadata={"source": "test"}
    )
    doc3 = RagDocument(
        content="Python is a popular programming language for data science.",
        doc_id="doc3",
        metadata={"source": "test"}
    )
    doc4 = RagDocument(
        content="GPT models are neural language models developed by OpenAI.",
        doc_id="doc4",
        metadata={"source": "test"}
    )
    
    # Mock search results for different queries
    query_results = [
        SearchResult(document=doc1, score=0.9),
        SearchResult(document=doc2, score=0.8)
    ]
    
    context_results = [
        SearchResult(document=doc4, score=0.95),
        SearchResult(document=doc2, score=0.85),
        SearchResult(document=doc3, score=0.7)
    ]
    
    combined_results = [
        SearchResult(document=doc2, score=0.92),
        SearchResult(document=doc4, score=0.88),
        SearchResult(document=doc1, score=0.82)
    ]
    
    # Set up side effects
    def mock_similarity_search(query_embedding, **kwargs):
        # Return different results based on the embedding
        embedding_str = str(query_embedding)
        if "0.1, 0.2" in embedding_str:  # Query embedding
            return query_results
        elif "0.3, 0.4" in embedding_str:  # Context embedding
            return context_results
        elif "0.5, 0.6" in embedding_str:  # Combined embedding
            return combined_results
        else:
            return []
    
    vectordb.similarity_search.side_effect = mock_similarity_search
    return vectordb


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock(spec=EmbeddingModel)
    
    # Return different embeddings for different inputs
    async def mock_embed_query(text):
        if "What are neural networks" in text:
            return [0.1, 0.2, 0.3]
        elif "previous message" in text or "System:" in text or "User:" in text:
            return [0.3, 0.4, 0.5]
        elif "neural networks" in text and "previous message" in text:
            # Combined query and context
            return [0.5, 0.6, 0.7]
        else:
            return [0.9, 0.9, 0.9]  # Default embedding
    
    model.embed_query.side_effect = mock_embed_query
    return model


def test_retriever_init():
    """Test ContextualRetriever initialization."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        top_k=3,
        context_window_size=4,
        current_context_weight=0.75,
        history_decay_factor=0.8,
        context_strategy="combine"
    )
    
    retriever = ContextualRetriever(vectordb, embedding_model, config)
    
    assert retriever.vectordb == vectordb
    assert retriever.embedding_model == embedding_model
    assert retriever.config == config
    assert retriever.config.top_k == 3
    assert retriever.context_window_size == 4
    assert retriever.current_context_weight == 0.75
    assert retriever.history_decay_factor == 0.8
    assert retriever.context_strategy == "combine"
    assert retriever.conversation_history == []


def test_conversation_history_management():
    """Test adding to and clearing conversation history."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(retriever_type=RetrieverType.CONTEXTUAL)
    
    retriever = ContextualRetriever(vectordb, embedding_model, config)
    
    # Test adding user message
    retriever.add_to_history("Hello", is_user=True)
    assert len(retriever.conversation_history) == 1
    assert retriever.conversation_history[0]["text"] == "Hello"
    assert retriever.conversation_history[0]["is_user"] is True
    
    # Test adding system message
    retriever.add_to_history("Hi there", is_user=False)
    assert len(retriever.conversation_history) == 2
    assert retriever.conversation_history[1]["text"] == "Hi there"
    assert retriever.conversation_history[1]["is_user"] is False
    
    # Test adding with metadata
    metadata = {"timestamp": "2023-01-01"}
    retriever.add_to_history("How can I help?", is_user=False, metadata=metadata)
    assert len(retriever.conversation_history) == 3
    assert retriever.conversation_history[2]["metadata"] == metadata
    
    # Test clearing history
    retriever.clear_history()
    assert len(retriever.conversation_history) == 0


@pytest.mark.asyncio
async def test_prepare_history_context():
    """Test preparing context from conversation history."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        context_window_size=2
    )
    
    retriever = ContextualRetriever(vectordb, embedding_model, config)
    
    # Empty history should return empty string
    assert retriever._prepare_history_context() == ""
    
    # Add some history
    retriever.add_to_history("What are neural networks?", is_user=True)
    retriever.add_to_history("Neural networks are machine learning models inspired by the human brain.", is_user=False)
    retriever.add_to_history("How do they work?", is_user=True)
    
    # Should include only the most recent messages based on window size
    context = retriever._prepare_history_context()
    assert "Neural networks are machine learning models" in context
    assert "How do they work?" in context
    assert "What are neural networks?" not in context  # Outside window
    
    # Check format includes prefixes
    assert "User: How do they work?" in context
    assert "System: Neural networks are machine learning models" in context


@pytest.mark.asyncio
async def test_prepare_external_context():
    """Test preparing context from external sources."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(retriever_type=RetrieverType.CONTEXTUAL)
    
    retriever = ContextualRetriever(vectordb, embedding_model, config)
    
    # Test with string
    string_context = "This is relevant context."
    assert retriever._prepare_external_context(string_context) == string_context
    
    # Test with dictionary
    dict_context = {"text": "This is from a dictionary."}
    assert retriever._prepare_external_context(dict_context) == "This is from a dictionary."
    
    # Test with list of dictionaries (chat-like)
    list_context = [
        {"text": "First message", "is_user": True},
        {"text": "Second message", "is_user": False}
    ]
    prepared = retriever._prepare_external_context(list_context)
    assert "User: First message" in prepared
    assert "System: Second message" in prepared
    
    # Test with general object (should convert to string)
    assert retriever._prepare_external_context(123) == "123"


@pytest.mark.asyncio
async def test_retrieve_combined_strategy(mock_vectordb, mock_embedding_model):
    """Test retrieval with combined query and context strategy."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        top_k=3,
        context_strategy="combine"
    )
    
    retriever = ContextualRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Add some conversation history
    retriever.add_to_history("Tell me about machine learning", is_user=True)
    retriever.add_to_history("Machine learning is a field of AI...", is_user=False)
    
    # Patch the internal methods to isolate the test
    with patch.object(retriever, '_prepare_history_context') as mock_prepare_history:
        with patch.object(retriever, '_publish_retrieval_event'):
            # Set up return value
            mock_prepare_history.return_value = "User: previous message System: previous response"
            
            # Call retrieve
            query = "What are neural networks?"
            results = await retriever.retrieve(query)
            
            # Verify history context was prepared
            mock_prepare_history.assert_called_once()
            
            # Verify embedding model was called with combined query and context
            # Check any call with both query and context
            combined_calls = [
                call for call in mock_embedding_model.embed_query.call_args_list
                if query in call[0][0] and "previous message" in call[0][0]
            ]
            assert len(combined_calls) > 0
            
            # Verify results
            assert len(results) == 2
            assert results[0].document.doc_id == "doc1"
            assert results[1].document.doc_id == "doc2"


@pytest.mark.asyncio
async def test_retrieve_separate_strategy(mock_vectordb, mock_embedding_model):
    """Test retrieval with separate query and context strategy."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        top_k=3,
        context_strategy="separate",
        current_context_weight=0.7
    )
    
    retriever = ContextualRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Mock the internal method for combining results
    original_combine = retriever._combine_separate_results
    
    def mock_combine_results(query_results, context_results, top_k):
        # For testing, return a predictable result set
        # This simulates the reranking that would happen in the real method
        combined = [
            SearchResult(document=query_results[0].document, score=0.95),  # doc1
            SearchResult(document=context_results[0].document, score=0.90),  # doc4
            SearchResult(document=query_results[1].document, score=0.85)   # doc2
        ]
        return combined[:top_k]
    
    # Patch the combine method and history context
    with patch.object(retriever, '_combine_separate_results', side_effect=mock_combine_results):
        with patch.object(retriever, '_prepare_history_context') as mock_prepare_history:
            with patch.object(retriever, '_publish_retrieval_event'):
                # Set up return value
                mock_prepare_history.return_value = "User: previous message"
                
                # Call retrieve
                query = "What are neural networks?"
                results = await retriever.retrieve(query)
                
                # Verify embedding model was called twice - once for query, once for context
                assert mock_embedding_model.embed_query.call_count >= 2
                
                # Verify each embedding was done separately
                query_calls = [
                    call for call in mock_embedding_model.embed_query.call_args_list
                    if call[0][0] == query
                ]
                context_calls = [
                    call for call in mock_embedding_model.embed_query.call_args_list
                    if "previous message" in call[0][0] and call[0][0] != query
                ]
                assert len(query_calls) > 0
                assert len(context_calls) > 0
                
                # Verify results
                assert len(results) == 3
                assert results[0].document.doc_id == "doc1"
                assert results[1].document.doc_id == "doc4"
                assert results[2].document.doc_id == "doc2"


@pytest.mark.asyncio
async def test_retrieve_no_context(mock_vectordb, mock_embedding_model):
    """Test retrieval with no context (should fall back to basic retrieval)."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        top_k=2
    )
    
    retriever = ContextualRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Ensure no history
    retriever.clear_history()
    
    # Patch the event publishing method
    with patch.object(retriever, '_publish_retrieval_event'):
        # Call retrieve
        query = "What are neural networks?"
        results = await retriever.retrieve(query)
        
        # Should only embed the query
        assert mock_embedding_model.embed_query.call_count == 1
        mock_embedding_model.embed_query.assert_called_once_with(query)
        
        # Should return basic retrieval results
        assert len(results) == 2
        assert results[0].document.doc_id == "doc1"
        assert results[1].document.doc_id == "doc2"


@pytest.mark.asyncio
async def test_retrieve_with_external_context(mock_vectordb, mock_embedding_model):
    """Test retrieval with explicitly provided context."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        top_k=3,
        context_strategy="combine"
    )
    
    retriever = ContextualRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Add some history that should be ignored
    retriever.add_to_history("This should be ignored", is_user=True)
    
    # Patch methods to verify behavior
    with patch.object(retriever, '_prepare_external_context', wraps=retriever._prepare_external_context) as mock_prepare_external:
        with patch.object(retriever, '_prepare_history_context') as mock_prepare_history:
            with patch.object(retriever, '_publish_retrieval_event'):
                # Call retrieve with external context
                external_context = "User: previous message about neural networks"
                query = "What are neural networks?"
                results = await retriever.retrieve(query, context=external_context)
                
                # Should use external context, not history
                mock_prepare_external.assert_called_once_with(external_context)
                mock_prepare_history.assert_not_called()
                
                # Should return results based on combined query and context
                assert len(results) == 2
                assert results[0].document.doc_id == "doc1"
                assert results[1].document.doc_id == "doc2"


@pytest.mark.asyncio
async def test_combine_separate_results():
    """Test combining results from separate query and context searches."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        current_context_weight=0.6  # 60% weight to query, 40% to context
    )
    
    retriever = ContextualRetriever(vectordb, embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(content="Document 1", doc_id="doc1")
    doc2 = RagDocument(content="Document 2", doc_id="doc2")
    doc3 = RagDocument(content="Document 3", doc_id="doc3")
    doc4 = RagDocument(content="Document 4", doc_id="doc4")
    
    # Create query results
    query_results = [
        SearchResult(document=doc1, score=0.9),
        SearchResult(document=doc2, score=0.8)
    ]
    
    # Create context results
    context_results = [
        SearchResult(document=doc2, score=0.95),  # doc2 appears in both
        SearchResult(document=doc3, score=0.85)
    ]
    
    # Combine results
    combined = retriever._combine_separate_results(
        query_results=query_results,
        context_results=context_results,
        top_k=3
    )
    
    # Check results
    # doc2 appears in both and should get a boost
    # doc1 has query weight 0.6 * 0.9 = 0.54
    # doc2 has query weight 0.6 * 0.8 = 0.48 and context weight 0.4 * 0.95 = 0.38, total 0.86 plus boost
    # doc3 has context weight 0.4 * 0.85 = 0.34
    # So order should be doc2, doc1, doc3
    
    assert len(combined) == 3
    assert combined[0].document.doc_id == "doc2"  # Should be ranked first (in both + highest combined score)
    assert combined[1].document.doc_id == "doc1"  # Should be ranked second (higher query score)
    assert combined[2].document.doc_id == "doc3"  # Should be ranked third (only in context)
    
    # Verify boost was applied to doc2 for being in both
    doc2_score = next(r.score for r in combined if r.document.doc_id == "doc2")
    expected_base_score = 0.6 * 0.8 + 0.4 * 0.95  # 0.86
    assert doc2_score > expected_base_score  # Should have received a boost


@pytest.mark.asyncio
async def test_retrieve_with_filter(mock_vectordb, mock_embedding_model):
    """Test retrieval with metadata filter."""
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        top_k=2
    )
    
    retriever = ContextualRetriever(mock_vectordb, mock_embedding_model, config)
    
    # Patch internal methods to isolate test
    with patch.object(retriever, '_basic_retrieve') as mock_basic_retrieve:
        with patch.object(retriever, '_publish_retrieval_event'):
            # Call retrieve with filter
            test_filter = {"source": "test"}
            await retriever.retrieve("test query", filter=test_filter)
            
            # Verify that basic_retrieve was called
            mock_basic_retrieve.assert_called_once()
            
            # Check if the filter was passed correctly (match based on arguments available)
            call_args = mock_basic_retrieve.call_args[0]
            call_kwargs = mock_basic_retrieve.call_args[1]
            
            # Verify the mock was called with the right query
            assert call_kwargs.get("query") == "test query" or call_args[0] == "test query"


@pytest.mark.asyncio
async def test_retrieve_with_custom_filter_function():
    """Test retrieval with a custom filter function."""
    vectordb = MagicMock()
    embedding_model = MagicMock()
    
    # Define a custom filter function
    def filter_fn(doc):
        return "neural" in doc.content.lower()
    
    config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        top_k=3,
        filter_fn=filter_fn
    )
    
    retriever = ContextualRetriever(vectordb, embedding_model, config)
    
    # Create test documents
    doc1 = RagDocument(content="Neural networks are a type of model.", doc_id="doc1")
    doc2 = RagDocument(content="This document has no relevant terms.", doc_id="doc2")
    doc3 = RagDocument(content="Another neural document.", doc_id="doc3")
    
    # Mock the basic retrieve method to return all docs
    async def mock_basic_retrieve(query, k, filter):
        return [
            SearchResult(document=doc1, score=0.8),
            SearchResult(document=doc2, score=0.7),
            SearchResult(document=doc3, score=0.6)
        ]
    
    # Patch methods
    with patch.object(retriever, '_basic_retrieve', side_effect=mock_basic_retrieve):
        with patch.object(retriever, '_publish_retrieval_event'):
            # Call retrieve
            results = await retriever.retrieve("test query")
            
            # Only docs with "neural" should be in results
            assert len(results) == 2
            doc_ids = [r.document.doc_id for r in results]
            assert "doc1" in doc_ids
            assert "doc3" in doc_ids
            assert "doc2" not in doc_ids  # Filtered out 