# Vector Database Layer

The vector database layer in RAGStackXL provides a flexible, high-performance system for storing and retrieving vector embeddings and their associated documents. This documentation covers the architecture, available providers, configuration options, and usage examples.

## Architecture

The vector database layer uses an abstract base class (`VectorDB`) that defines a common interface for all vector database implementations. This allows you to easily swap between different vector database providers without changing your application code.

Key components:

1. **Abstract Base Class**: The `VectorDB` class defines standard methods for adding, retrieving, and searching vector embeddings.
2. **Factory Pattern**: The `VectorDBFactory` creates instances of specific vector database implementations.
3. **Configuration**: The `VectorDBConfig` class manages provider-specific settings.
4. **Search Results**: The `SearchResult` class provides a uniform interface for search results from different providers.

## Supported Providers

RAGStackXL supports the following vector database providers:

### 1. FAISS

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It's ideal for local vector search applications.

**Key features:**
- Local deployment
- High performance for in-memory search
- Multiple similarity metrics (cosine, euclidean, dot product)
- Persistent storage to disk

### 2. FAISS Advanced

An enhanced version of FAISS with additional features:

**Key features:**
- Automatic index selection based on collection size
- HNSW indices for faster search
- IVF and PQ for memory-efficient storage
- Specialized for different workload sizes

### 3. Qdrant

[Qdrant](https://qdrant.tech/) is a vector similarity search engine with extended filtering support.

**Key features:**
- Complex filtering operations
- Horizontal scaling
- Rich payload storage
- Cloud-native architecture

### 4. Weaviate

[Weaviate](https://weaviate.io/) is an open-source vector search engine with multi-modal capabilities.

**Key features:**
- GraphQL API
- Hybrid search (vector + keyword)
- Multi-tenancy support
- Cloud deployment options

### 5. Pinecone

[Pinecone](https://www.pinecone.io/) is a managed vector database service optimized for machine learning applications.

**Key features:**
- Fully managed service
- Serverless deployment option
- Automatic scaling
- High availability and consistency

### 6. Milvus

[Milvus](https://milvus.io/) is an open-source vector database built for scalable similarity search.

**Key features:**
- Horizontal scaling
- Optimized for high dimensions
- Dynamic schema
- Support for partitioning

## Configuration Options

### Common Configuration

All vector databases support these common configuration options:

```python
config = VectorDBConfig(
    collection_name="documents",         # Name of the collection/index
    embedding_dimension=1536,            # Dimension of embedding vectors
    distance_metric="cosine",            # Similarity metric (cosine, euclidean, dot)
    persist_directory="/path/to/data"    # Directory for persistent storage (if applicable)
)
```

### Provider-Specific Configuration

#### FAISS Advanced

```python
config = VectorDBConfig(
    # Common configuration
    collection_name="documents",
    embedding_dimension=1536,
    distance_metric="cosine",
    persist_directory="/path/to/data",
    
    # FAISS Advanced specific
    index_type="hnsw",                   # Index type (flat, ivf, hnsw, pq, ivfpq)
    auto_index=True,                     # Automatically select best index based on size
    nlist=100,                           # Number of clusters for IVF
    nprobe=10,                           # Number of clusters to search
    m=16,                                # HNSW parameter (connections per layer)
    ef_construction=100,                 # HNSW construction parameter
    ef_search=50                         # HNSW search parameter
)
```

#### Qdrant

```python
config = VectorDBConfig(
    # Common configuration
    collection_name="documents",
    embedding_dimension=1536,
    distance_metric="cosine",
    
    # Qdrant specific
    url="http://localhost:6333",         # Qdrant server URL
    api_key="your-api-key",              # API key for authentication
    prefer_grpc=True                     # Use gRPC instead of REST
)
```

#### Weaviate

```python
config = VectorDBConfig(
    # Common configuration
    collection_name="documents",
    embedding_dimension=1536,
    distance_metric="cosine",
    
    # Weaviate specific
    url="http://localhost:8080",         # Weaviate server URL
    api_key="your-api-key"               # API key for authentication
)
```

#### Pinecone

```python
config = VectorDBConfig(
    # Common configuration
    collection_name="documents",
    embedding_dimension=1536,
    distance_metric="cosine",
    
    # Pinecone specific
    api_key="your-api-key",              # Pinecone API key
    environment="us-west1-gcp",          # Pinecone environment
    serverless=False                     # Use serverless deployment
)
```

#### Milvus

```python
config = VectorDBConfig(
    # Common configuration
    collection_name="documents",
    embedding_dimension=1536,
    distance_metric="cosine",
    
    # Milvus specific
    uri="localhost:19530",               # Milvus server URI
    user="username",                     # Username for authentication
    password="password",                 # Password for authentication
    token="token"                        # Token for authentication
)
```

## Creating a Vector Database

You can create a vector database instance using the factory function:

```python
from app.vectordb import create_vectordb

# Create a FAISS vector database
db = create_vectordb(
    provider="faiss",
    collection_name="documents",
    embedding_dimension=1536,
    distance_metric="cosine",
    persist_directory="./data/vectordb"
)

# Create a Pinecone vector database
db = create_vectordb(
    provider="pinecone",
    collection_name="documents",
    embedding_dimension=1536,
    distance_metric="cosine",
    api_key="your-api-key",
    environment="us-west1-gcp"
)
```

## API Reference

### Adding Documents

```python
doc_ids = await db.add_documents(
    documents=[doc1, doc2, doc3],        # List of RagDocument objects
    embeddings=[emb1, emb2, emb3],       # List of embedding vectors
    ids=["id1", "id2", "id3"]            # Optional list of IDs
)
```

### Retrieving Documents

```python
# Get a document by ID
doc = await db.get_document("doc_id")

# Delete a document by ID
success = await db.delete_document("doc_id")
```

### Searching

```python
# Basic similarity search
results = await db.similarity_search(
    query_embedding=[0.1, 0.2, ...],     # Query embedding vector
    k=4,                                 # Number of results to return
    filter={"category": "science"}       # Optional metadata filter
)

# Search with score
results = await db.similarity_search_with_score(
    query_embedding=[0.1, 0.2, ...],
    k=4,
    filter={"category": "science"}
)
```

### Advanced Filtering

The vector database layer supports complex filtering operations:

```python
from app.vectordb.utils import FilterOperator, create_filter, combine_filters

# Simple filter
filter1 = create_filter("category", "science")

# Range filter
filter2 = create_filter("year", 2020, FilterOperator.GT)

# List membership filter
filter3 = create_filter("tags", ["research", "technology"], FilterOperator.IN)

# Combined filters (AND)
combined_filter = combine_filters([filter1, filter2], FilterOperator.AND)

# Complex filter
complex_filter = create_filter(
    conditions=[
        create_filter("category", "science"),
        create_filter(
            conditions=[
                create_filter("year", 2020, FilterOperator.GT),
                create_filter("year", 2023, FilterOperator.LTE)
            ],
            operator=FilterOperator.AND
        )
    ],
    operator=FilterOperator.AND
)

# Use filter in search
results = await db.similarity_search(
    query_embedding=[0.1, 0.2, ...],
    k=4,
    filter=complex_filter
)
```

## Collection Management

```python
# Get collection statistics
stats = await db.get_collection_stats()

# Clear collection
success = await db.clear_collection()

# Persist collection to disk (for providers that support it)
success = await db.persist()
```

## Best Practices

1. **Provider Selection**:
   - Use FAISS for local development or small collections
   - Use FAISS Advanced for larger collections with performance requirements
   - Use Qdrant or Weaviate for complex filtering and hybrid search
   - Use Pinecone for fully managed, production deployments
   - Use Milvus for high-scale, distributed deployments

2. **Performance Optimization**:
   - Use appropriate batch sizes when adding documents (50-100 documents per batch)
   - Select embedding dimension based on your model (OpenAI: 1536, BERT: 768, etc.)
   - For FAISS Advanced, use HNSW for better recall/latency tradeoff
   - Use filtering to narrow search space before vector similarity

3. **Memory Management**:
   - For large collections, consider IVF or PQ compression with FAISS
   - Monitor memory usage during ingestion and search

## Benchmarking

The `tests/benchmark/vectordb_benchmark.py` script can be used to benchmark different vector database providers:

```bash
python -m tests.benchmark.vectordb_benchmark --providers faiss faiss_advanced --dimensions 384 768 --collection-sizes 1000 10000
```

This will generate performance metrics and plots to help you select the most appropriate vector database for your use case. 