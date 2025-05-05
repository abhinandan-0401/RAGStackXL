# Embedding Models for RAGStackXL

This module provides a collection of embedding models for converting text into vector representations for use in the RAGStackXL system. These embeddings are essential for semantic search and retrieval in the vector databases.

## Available Embedding Models

The following embedding models are supported:

1. **OpenAI** - Industry-standard embeddings from OpenAI's API
2. **SentenceTransformers (SBERT)** - Powerful open-source models for text embeddings
3. **HuggingFace** - Custom embeddings using any Transformer model from HuggingFace
4. **Cohere** - High-quality embeddings optimized for search from Cohere
5. **FastEmbed** - Lightweight, efficient embedding models

## Configuration

Embedding models can be configured in two ways:

1. Through the application settings (see `app/config/settings.py`)
2. Programmatically when creating the model

### Default Configuration

The default configuration in `settings.py` uses FastEmbed as it requires no API keys:

```python
class EmbeddingSettings(BaseSettings):
    MODEL_TYPE: str = "fastembed"
    MODEL_NAME: str = "default"
    DIMENSION: int = 768
    NORMALIZE: bool = True
```

### Provider-Specific Settings

Each provider has its own settings:

- **OpenAI**: API key, base URL, version, timeout
- **SBERT**: Device (CPU/GPU), batch size, progress bar
- **HuggingFace**: Token (API key), revision, device
- **Cohere**: API key, timeout
- **FastEmbed**: Cache directory, threads, batch size

## Usage Examples

### Basic Usage

```python
from app.embedding import create_embedding_model

# Create embedding model using default settings
embedding_model = create_embedding_model()

# Embed a query
query_embedding = await embedding_model.embed_query("What is the capital of France?")

# Embed documents
documents = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
document_embeddings = await embedding_model.embed_documents(documents)
```

### Custom Configuration

```python
from app.embedding import create_embedding_model, EmbeddingModelType

# Create embedding model with custom settings
embedding_model = create_embedding_model(
    model_name="text-embedding-ada-002",
    model_type=EmbeddingModelType.OPENAI,
    api_key="your-openai-api-key",
    dimension=1536,
    normalize=True
)
```

## Advanced Usage

### Using with Vector Databases

Embedding models can be used directly with vector databases:

```python
from app.embedding import create_embedding_model
from app.vectordb import create_vectordb

# Create embedding model
embedding_model = create_embedding_model()

# Create vector database with this embedding model
vectordb = create_vectordb(
    provider="faiss",
    embedding_model=embedding_model
)

# Add documents
documents = ["Document 1 content", "Document 2 content"]
await vectordb.add_documents(documents)

# Query
results = await vectordb.query("What is the capital of France?", top_k=5)
```

## Performance Considerations

- **OpenAI** and **Cohere** require API calls, which may introduce latency but provide high-quality embeddings
- **SBERT** and **HuggingFace** run locally and can use GPU acceleration for better performance
- **FastEmbed** is optimized for efficiency and works well on CPU-only environments

Choose the embedding model that best fits your requirements for quality, speed, and cost. 