# RAGStackXL Examples

This directory contains example scripts that demonstrate how to use various components of the RAGStackXL framework.

## Embedding Examples

### Basic Embedding Usage

The `embedding_examples.py` script demonstrates how to use the different embedding models:

```bash
python examples/embedding_examples.py
```

This script:
- Initializes different embedding models (FastEmbed, SBERT, HuggingFace)
- Embeds queries and documents
- Shows basic timing metrics

### RAG with Embeddings

The `rag_with_embeddings.py` script demonstrates a complete RAG workflow:

```bash
python examples/rag_with_embeddings.py
```

This script:
- Creates an embedding model
- Sets up a vector database
- Adds documents to the vector database
- Performs semantic search queries

### Benchmarking Embedding Models

The `benchmark_embeddings.py` script provides a comprehensive benchmark of different embedding models:

```bash
python examples/benchmark_embeddings.py
```

This script:
- Compares query embedding speed
- Measures document embedding throughput
- Tracks memory usage
- Saves benchmark results to the `benchmark_results` directory

## API Keys for External Models

To test API-based embedding models (OpenAI, Cohere), you'll need to set your API keys as environment variables:

```bash
# For OpenAI
$env:OPENAI_API_KEY="your-openai-api-key"

# For Cohere
$env:COHERE_API_KEY="your-cohere-api-key"
```

## Dependencies

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

Some examples require specific packages like `sentence-transformers`, `transformers`, `openai`, and `cohere` which are listed in the main requirements.txt file. 