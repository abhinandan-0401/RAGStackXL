# Retrieval Mechanisms Guide

## Overview

RAGStackXL provides a comprehensive set of retrieval mechanisms that go beyond basic vector similarity search. These advanced retrieval strategies enable more effective and contextually relevant document retrieval, improving the quality of generated responses in RAG applications.

## Key Features

- **Multiple retrieval strategies** with different strengths and use cases
- **Flexible configuration** for fine-tuning retrieval behavior
- **Async-first design** for high-performance retrieval
- **Extensible interface** for custom retrieval implementations
- **Integration with vector databases** and embedding models
- **Event-driven architecture** for system monitoring

## Retriever Types

RAGStackXL supports the following retrieval mechanisms:

### 1. Basic Retriever

The simplest retrieval mechanism that performs standard vector similarity search.

```python
from app.retrieval import RetrieverFactory, RetrieverConfig, RetrieverType

config = RetrieverConfig(
    retriever_type=RetrieverType.BASIC,
    top_k=5
)

retriever = RetrieverFactory.create(vectordb, embedding_model, config)
results = await retriever.retrieve("How do neural networks work?")
```

**Use cases**: Simple retrieval tasks, baseline for comparisons, when simplicity and speed are priorities.

**Configuration options**:
- `top_k`: Number of documents to retrieve (default: 4)

### 2. Semantic Retriever

Enhances basic retrieval with keyword boosting and improved semantic understanding.

```python
config = RetrieverConfig(
    retriever_type=RetrieverType.SEMANTIC,
    top_k=4,
    use_keyword_boost=True,
    keyword_boost_factor=0.2,
    relevance_threshold=0.6,
    semantic_weight=0.8
)
```

**Use cases**: When a balance of semantic similarity and keyword matching is desired, for improved precision.

**Configuration options**:
- `use_keyword_boost`: Whether to boost scores based on keyword matching (default: True)
- `keyword_boost_factor`: Factor for boosting keyword matches (default: 0.1)
- `relevance_threshold`: Minimum relevance score to include in results (default: 0.0)
- `semantic_weight`: Weight given to vector similarity vs keywords (default: 0.85)

### 3. Hybrid Retriever

Combines vector-based search with keyword-based search for more robust retrieval.

```python
config = RetrieverConfig(
    retriever_type=RetrieverType.HYBRID,
    top_k=5,
    vector_weight=0.7,
    keyword_weight=0.3,
    use_bm25=True,
    min_keyword_length=3
)
```

**Use cases**: When robust retrieval across both semantic meaning and exact keyword matches is needed.

**Configuration options**:
- `vector_weight`: Weight for vector similarity (default: 0.7)
- `keyword_weight`: Weight for keyword matching (default: 0.3)
- `use_bm25`: Whether to use BM25-inspired scoring for keywords (default: True)
- `min_keyword_length`: Minimum length for keywords to consider (default: 3)

### 4. MMR Retriever (Maximum Marginal Relevance)

Balances relevance and diversity in search results to reduce redundancy.

```python
config = RetrieverConfig(
    retriever_type=RetrieverType.MMR,
    top_k=5,
    lambda_param=0.6,
    fetch_k=30
)
```

**Use cases**: When diversity in results is important, or when you want to avoid redundant information.

**Configuration options**:
- `lambda_param`: Balance between relevance (1.0) and diversity (0.0) (default: 0.7)
- `fetch_k`: Number of initial candidates to fetch before MMR reranking (default: 30)

### 5. Query Expansion Retriever

Generates multiple variations of the input query to improve recall.

```python
config = RetrieverConfig(
    retriever_type=RetrieverType.QUERY_EXPANSION,
    top_k=5,
    max_queries=3,
    use_llm=False,
    aggregate_method="weighted"
)
```

**Use cases**: When recall is important, or when the initial query might miss relevant information.

**Configuration options**:
- `max_queries`: Maximum number of query variations to use (default: 3)
- `use_llm`: Whether to use LLM-based reformulation (default: False)
- `aggregate_method`: Method to combine results ("weighted", "max", "mean") (default: "weighted")

### 6. Reranking Retriever

Two-stage retrieval with advanced reranking of initial candidates.

```python
config = RetrieverConfig(
    retriever_type=RetrieverType.RERANKING,
    top_k=5,
    initial_k=20,
    reranker_type="simple"  # or "cosine"
)
```

**Use cases**: When fine-grained ranking of candidates is needed for precision.

**Configuration options**:
- `initial_k`: Number of initial candidates to retrieve (default: 20)
- `reranker_type`: Type of reranker to use ("simple" or "cosine") (default: "simple")

### 7. Contextual Retriever

Considers conversation history or external context for retrieval.

```python
config = RetrieverConfig(
    retriever_type=RetrieverType.CONTEXTUAL,
    top_k=4,
    context_window_size=5,
    context_strategy="combine",  # or "separate"
    current_context_weight=0.7
)

retriever = RetrieverFactory.create(vectordb, embedding_model, config)

# Add conversation history
retriever.add_to_history("What is machine learning?", is_user=True)
retriever.add_to_history("Machine learning is a branch of AI...", is_user=False)

# Now the next query will consider the conversation history
results = await retriever.retrieve("How does it differ from deep learning?")
```

**Use cases**: Conversational applications where context is important for understanding queries.

**Configuration options**:
- `context_window_size`: Number of recent messages to consider (default: 5)
- `context_strategy`: How to use context ("combine" or "separate") (default: "combine")
- `current_context_weight`: Weight for current query vs history (default: 0.75)
- `history_decay_factor`: Factor for weighing older messages less (default: 0.8)

## Common Configuration Options

All retrievers support these common options:

- `top_k`: Number of documents to retrieve (default: 4)
- `filter_fn`: Optional function to filter documents (default: None)
- `use_metadata_filter`: Whether to use metadata for filtering (default: True)
- `include_metadata`: Whether to include metadata in results (default: True)

## Examples

### Basic Retrieval Example

```python
from app.retrieval import RetrieverFactory, RetrieverConfig, RetrieverType
from app.vectordb import VectorDBFactory, VectorDBConfig
from app.embedding import EmbeddingModelFactory, EmbeddingModelConfig

# Create vector database
vectordb_config = VectorDBConfig(
    provider="faiss",
    collection_name="documents",
    embedding_dimension=384
)
vectordb = VectorDBFactory.create(vectordb_config)

# Create embedding model
embedding_config = EmbeddingModelConfig(
    provider="sbert",
    model_name="all-MiniLM-L6-v2"
)
embedding_model = EmbeddingModelFactory.create(embedding_config)

# Create retriever
retriever_config = RetrieverConfig(
    retriever_type=RetrieverType.SEMANTIC,
    top_k=5,
    use_keyword_boost=True,
    keyword_boost_factor=0.2
)
retriever = RetrieverFactory.create(vectordb, embedding_model, retriever_config)

# Retrieve documents
results = await retriever.retrieve(
    query="How do neural networks learn?",
    filter={"category": "machine_learning"}
)

# Process results
for result in results:
    print(f"Document: {result.document.doc_id}")
    print(f"Score: {result.score}")
    print(f"Content: {result.document.content[:100]}...")
    print("---")
```

### Combined Retrieval Strategies

You can also combine multiple retrievers for more advanced use cases:

```python
# First retriever with MMR for diversity
mmr_config = RetrieverConfig(
    retriever_type=RetrieverType.MMR,
    top_k=10,
    lambda_param=0.6
)
mmr_retriever = RetrieverFactory.create(vectordb, embedding_model, mmr_config)

# Second retriever with reranking
reranking_config = RetrieverConfig(
    retriever_type=RetrieverType.RERANKING,
    top_k=5,
    reranker_type="cosine"
)
reranking_retriever = RetrieverFactory.create(vectordb, embedding_model, reranking_config)

# Use MMR first to get diverse results
diverse_results = await mmr_retriever.retrieve(query)

# Extract documents from results
diverse_docs = [r.document for r in diverse_results]

# Use reranker to get final ranking
final_results = await reranking_retriever.reranker.rerank(query, diverse_docs)
```

## Best Practices

1. **Choose the right retriever for your use case**:
   - Use Basic or Semantic Retriever for simple tasks
   - Use MMR when diversity is important
   - Use Query Expansion when you need high recall
   - Use Contextual Retriever for conversational applications

2. **Tune parameters**:
   - Start with defaults and adjust based on results
   - Experiment with top_k values depending on your application
   - For hybrid retrieval, adjust vector_weight vs keyword_weight
   - For MMR, adjust lambda_param depending on diversity needs

3. **Consider performance implications**:
   - Query Expansion performs multiple searches
   - Reranking Retriever requires processing more initial candidates
   - Contextual Retriever may need more complex embedding operations

4. **Testing and evaluation**:
   - Compare different retrievers on your specific dataset
   - Use metrics like precision@k, recall@k, and MRR
   - Monitor performance in production with the event system

## Extending the System

To create a custom retriever:

1. Implement the `Retriever` abstract base class
2. Register with `RetrieverFactory`:

```python
from app.retrieval import Retriever, RetrieverFactory, RetrieverType

class MyCustomRetriever(Retriever):
    async def retrieve(self, query, k=None, filter=None):
        # Custom implementation
        pass

# Register with factory
RetrieverFactory.register(RetrieverType.CUSTOM, MyCustomRetriever)
```

## Troubleshooting

- **Poor retrieval quality**: Try different retriever types or tune parameters
- **Slow performance**: Reduce top_k, use simpler retrievers, or optimize vector DB
- **Missing relevant documents**: Try Query Expansion or reduce relevance thresholds
- **Too many similar results**: Use MMR Retriever to increase diversity
- **Context not considered**: Ensure Contextual Retriever has conversation history 