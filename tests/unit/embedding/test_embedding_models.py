"""
Tests for embedding models.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from app.embedding.interfaces import EmbeddingModel, EmbeddingConfig, EmbeddingModelType
from app.embedding.factory import create_embedding_model


class TestEmbeddingModels:
    """Tests for embedding models."""
    
    @pytest.mark.asyncio
    async def test_sbert_embedding_model(self):
        """Test that the SBERT embedding model works correctly."""
        try:
            # Skip test if sentence-transformers not installed
            import sentence_transformers
        except ImportError:
            pytest.skip("sentence-transformers not installed")
        
        from app.embedding.sbert import SentenceTransformerEmbedding
        
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",  # Small, fast model for testing
            model_type=EmbeddingModelType.SBERT,
            dimension=384,  # Dimension of all-MiniLM-L6-v2
            normalize=True,
            device="cpu",
            show_progress_bar=False
        )
        
        model = SentenceTransformerEmbedding(config)
        
        # Test embedding a query
        query_embedding = await model.embed_query("This is a test query")
        assert isinstance(query_embedding, list)
        assert len(query_embedding) == 384  # MiniLM model has 384 dimensions
        
        # Test embedding documents
        docs = ["First document", "Second document", "Third document"]
        doc_embeddings = await model.embed_documents(docs)
        assert isinstance(doc_embeddings, list)
        assert len(doc_embeddings) == 3
        assert all(len(emb) == 384 for emb in doc_embeddings)
    
    @pytest.mark.asyncio
    async def test_fastembed_embedding_model(self):
        """Test that the FastEmbed embedding model works correctly."""
        try:
            # Skip test if fastembed not installed
            import fastembed
        except ImportError:
            pytest.skip("fastembed not installed")
        
        from app.embedding.fastembed import FastEmbedModel
        
        config = EmbeddingConfig(
            model_name="default",  # Use default model
            model_type=EmbeddingModelType.FASTEMBED,
            dimension=384,  # Should be updated to actual dimension by the model
            normalize=True,
            threads=1,
            batch_size=8
        )
        
        model = FastEmbedModel(config)
        
        # Test embedding a query
        query_embedding = await model.embed_query("This is a test query")
        assert isinstance(query_embedding, list)
        assert len(query_embedding) > 0
        
        # Test embedding documents
        docs = ["First document", "Second document", "Third document"]
        doc_embeddings = await model.embed_documents(docs)
        assert isinstance(doc_embeddings, list)
        assert len(doc_embeddings) == 3
        assert all(len(emb) > 0 for emb in doc_embeddings)
    
    @pytest.mark.asyncio
    async def test_openai_embedding_model_mocked(self):
        """Test the OpenAI embedding model with mocked responses."""
        from app.embedding.openai import OpenAIEmbedding
        
        # Create a mock embedding result
        mock_embedding = [0.1] * 1536  # OpenAI uses 1536 dimensions for text-embedding-ada-002
        
        # Mock for new OpenAI client (v1.0.0+)
        mock_response_new = MagicMock()
        mock_response_new.data = [MagicMock(embedding=mock_embedding, index=0)]
        
        # Mock for old OpenAI client (before v1.0.0)
        mock_response_old = {
            "data": [{"embedding": mock_embedding, "index": 0}]
        }
        
        config = EmbeddingConfig(
            model_name="text-embedding-ada-002",
            model_type=EmbeddingModelType.OPENAI,
            dimension=1536,
            normalize=True,
            api_key="fake-api-key"
        )
        
        # Test with new client
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response_new
            mock_openai.return_value = mock_client
            
            with patch.dict("sys.modules", {"openai.OpenAI": True}):
                model = OpenAIEmbedding(config)
                model.client = mock_client
                
                query_embedding = await model.embed_query("This is a test query")
                assert isinstance(query_embedding, list)
                assert len(query_embedding) == 1536
        
        # Test with old client
        with patch("openai.Embedding.create") as mock_create:
            mock_create.return_value = mock_response_old
            
            with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=None)}):
                model = OpenAIEmbedding(config)
                model.client = MagicMock(Embedding=MagicMock(create=mock_create))
                
                query_embedding = await model.embed_query("This is a test query")
                assert isinstance(query_embedding, list)
                assert len(query_embedding) == 1536
    
    @pytest.mark.asyncio
    async def test_huggingface_embedding_model_mocked(self):
        """Test the HuggingFace embedding model with mocked responses."""
        from app.embedding.huggingface import HuggingFaceEmbedding
        
        # Mock the HuggingFace tokenizer and model
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Set up mocks
        import torch
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.zeros((1, 1, 768))
        mock_model.return_value = mock_outputs
        
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("transformers.AutoModel.from_pretrained") as mock_model_cls:
            
            mock_tokenizer_cls.return_value = mock_tokenizer
            mock_model_cls.return_value = mock_model
            
            config = EmbeddingConfig(
                model_name="bert-base-uncased",
                model_type=EmbeddingModelType.HUGGINGFACE,
                dimension=768,
                normalize=True,
                device="cpu"
            )
            
            model = HuggingFaceEmbedding(config)
            
            # Manually set the tokenizer and model
            model.tokenizer = mock_tokenizer
            model.model = mock_model
            
            # Replace _get_embedding with a simpler version for testing
            def mock_get_embedding(text):
                return [0.01] * 768
            
            model._get_embedding = mock_get_embedding
            
            # Test embedding a query
            query_embedding = await model.embed_query("This is a test query")
            assert isinstance(query_embedding, list)
            assert len(query_embedding) == 768
    
    @pytest.mark.asyncio
    async def test_cohere_embedding_model_mocked(self):
        """Test the Cohere embedding model with mocked responses."""
        from app.embedding.cohere import CohereEmbedding
        
        # Create a mock embedding result
        mock_embedding = [0.1] * 1024  # Cohere typically uses 1024 dimensions
        
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        
        mock_client = MagicMock()
        mock_client.embed.return_value = mock_response
        
        with patch("cohere.Client") as mock_cohere:
            mock_cohere.return_value = mock_client
            
            config = EmbeddingConfig(
                model_name="embed-english-v2.0",
                model_type=EmbeddingModelType.COHERE,
                dimension=1024,
                normalize=True,
                api_key="fake-api-key"
            )
            
            model = CohereEmbedding(config)
            
            # Test embedding a query
            query_embedding = await model.embed_query("This is a test query")
            assert isinstance(query_embedding, list)
            assert len(query_embedding) == 1024
            
            # Verify correct input type was used
            call_args = mock_client.embed.call_args[1]
            assert call_args["input_type"] == "search_query"
            
            # Reset mock for document embedding test
            mock_client.reset_mock()
            
            # Test embedding documents
            mock_response.embeddings = [mock_embedding, mock_embedding, mock_embedding]
            mock_client.embed.return_value = mock_response
            
            docs = ["First document", "Second document", "Third document"]
            doc_embeddings = await model.embed_documents(docs)
            
            assert isinstance(doc_embeddings, list)
            assert len(doc_embeddings) == 3
            assert all(len(emb) == 1024 for emb in doc_embeddings)
            
            # Verify correct input type was used
            call_args = mock_client.embed.call_args[1]
            assert call_args["input_type"] == "search_document"
    
    @pytest.mark.asyncio
    async def test_embedding_factory(self):
        """Test the embedding factory creates models correctly."""
        
        # Mock the create method of EmbeddingModelFactory
        with patch("app.embedding.interfaces.EmbeddingModelFactory.create") as mock_create:
            # Set up a mock model
            mock_model = MagicMock(spec=EmbeddingModel)
            mock_create.return_value = mock_model
            
            # Test creating a model via factory
            model = create_embedding_model(
                model_name="test-model",
                model_type="fastembed",
                dimension=384,
                normalize=True
            )
            
            assert model == mock_model
            assert mock_create.called 