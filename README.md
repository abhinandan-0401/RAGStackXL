# RAGStackXL

An advanced RAG (Retrieval-Augmented Generation) system with agentic capabilities.

## Overview

RAGStackXL is a modular and extensible Retrieval-Augmented Generation system designed to showcase advanced RAG techniques, integration with LLMs, and agentic workflows. This project aims to demonstrate the depth of engineering involved in building state-of-the-art AI systems.

Key features:

- Advanced document processing with multiple chunking strategies
- Flexible embedding and vectorization pipeline
- Sophisticated retrieval mechanisms
- Agentic components for complex tasks
- Comprehensive evaluation framework
- Clean architecture with dependency injection

## Project Status

### Completed Phases:

#### Phase 1: Project Setup ✓
- Established project structure with modular components
- Set up configuration system using Pydantic
- Implemented logging with Loguru
- Created core interfaces and abstract classes
- Set up basic CLI application entry point

#### Phase 2: Document Processing ✓
- Implemented document loaders for multiple formats:
  - Plain text (TXT)
  - PDF documents
  - Microsoft Word (DOCX)
  - HTML files
  - Markdown content
- Created a loader registry system for format discovery
- Implemented multiple text chunking strategies:
  - Character-based chunking
  - Semantic chunking (preserving document structure)
  - Recursive chunking (hierarchical approach)
- Built a unified document processor combining loading and chunking
- Added comprehensive unit tests with good coverage

#### Phase 3: Vector Database ✓
- Designed a flexible abstraction layer for vector databases
- Implemented multiple vector database providers:
  - FAISS (local, fast vector search)
  - FAISS Advanced (with optimized indices: HNSW, IVF, PQ)
  - Qdrant (balanced performance with rich filtering)
  - Weaviate (hybrid search capabilities)
  - Pinecone (managed service)
  - Milvus (high-scale distributed)
- Created advanced filtering capabilities with complex operators
- Implemented efficient batch operations and persistence
- Added comprehensive benchmarking tools for provider comparisons
- Integrated with the event bus system for notifications

#### Phase 4: Embedding Models ✓
- Designed a clean embedding model abstraction and interface
- Implemented multiple embedding model providers:
  - SentenceTransformers (SBERT)
  - FastEmbed (lightweight, efficient embeddings)
  - OpenAI (API-based embeddings)
  - HuggingFace (transformer-based models)
  - Cohere (specialized for retrieval)
- Created factory pattern for embedding model instantiation
- Added async support for non-blocking operations
- Built comprehensive benchmarking tools for model comparison
- Integrated with vector database layer for complete RAG pipeline
- Developed example scripts for usage and testing

### In Progress:
- Phase 5: Retrieval Mechanisms
- Building advanced retrieval strategies for optimal document retrieval

## Project Structure

```
app/
├── core/              # Core interfaces and abstractions
│   ├── loaders/       # Document loaders for different formats
│   └── chunking/      # Text chunking strategies
├── vectordb/          # Vector database implementations
│   ├── interfaces.py  # Vector database abstract interfaces
│   ├── factory.py     # Factory for creating vector database instances
│   ├── faiss.py       # FAISS implementation
│   ├── faiss_advanced.py # Advanced FAISS with optimized indices
│   ├── qdrant.py      # Qdrant implementation
│   ├── weaviate.py    # Weaviate implementation
│   ├── pinecone.py    # Pinecone implementation
│   └── milvus.py      # Milvus implementation
├── embedding/         # Embedding model implementations
│   ├── interfaces.py  # Embedding model abstract interfaces
│   ├── factory.py     # Factory for creating embedding models
│   ├── sbert.py       # SentenceTransformers implementation
│   ├── fastembed.py   # FastEmbed implementation
│   ├── openai.py      # OpenAI API implementation
│   ├── huggingface.py # HuggingFace implementation
│   └── cohere.py      # Cohere implementation
├── retrieval/         # Retrieval mechanisms (coming soon)
├── llm/               # LLM integration (coming soon)
├── agents/            # Agent system (coming soon)
├── api/               # API endpoints (coming soon)
├── ui/                # User interface (coming soon)
├── utils/             # Utility functions
├── evaluation/        # Evaluation metrics (coming soon)
└── config/            # Configuration management
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/RAGStackXL.git
   cd RAGStackXL
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. For development installation:
   ```
   pip install -e .
   ```

## Usage

### Ingesting Documents

To ingest documents into the system:

```
python main.py ingest --source /path/to/documents --recursive
```

Available options:
- `--recursive`: Process subdirectories recursively
- `--chunk-size`: Specify custom chunk size (default: 1000)
- `--chunk-overlap`: Specify custom chunk overlap (default: 200)
- `--splitter`: Choose chunking strategy (RecursiveSemanticSplitter, SemanticTextSplitter, CharacterTextSplitter)
- `--db-provider`: Choose vector database provider (faiss, faiss_advanced, qdrant, weaviate, pinecone, milvus)
- `--db-collection`: Specify collection name
- `--db-metric`: Choose distance metric (cosine, euclidean, dot)
- `--embedding-model`: Choose embedding model (sbert, fastembed, openai, huggingface, cohere)
- `--model-name`: Specify model name for the chosen embedding model

### Querying the System

To query the system:

```
python main.py query --query "Your question here" --k 4
```

Available options:
- `--k`: Number of results to return (default: 4)
- `--filter`: Filter query in JSON format (e.g., '{"category":"science"}')
- `--db-provider`: Choose vector database provider for retrieval
- `--embedding-model`: Choose embedding model for query embedding

### Running the Server

To start the API server (coming soon):

```
python main.py server --host 0.0.0.0 --port 8000
```

### Database Management

To manage the vector database:

```
# Clear the database
python main.py db clear --db-provider faiss

# Get database statistics
python main.py db stats --db-provider faiss
```

## Examples

RAGStackXL includes example scripts in the `examples/` directory to demonstrate various components:

```
# Test embedding models
python examples/embedding_examples.py

# Run a complete RAG workflow example
python examples/rag_with_embeddings.py

# Benchmark embedding models
python examples/benchmark_embeddings.py
```

## Vector Database Benchmarking

To benchmark different vector database providers:

```
python -m tests.benchmark.vectordb_benchmark --providers faiss faiss_advanced --dimensions 384 768 --collection-sizes 1000 10000
```

This will generate performance metrics and plots in the `benchmark_results` directory.

## Development

### Running Tests

```
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app

# Run specific test files
pytest tests/unit/vectordb/test_faiss.py
pytest tests/unit/embedding/test_embedding_models.py
```

### Adding New Components

1. Implement the appropriate interface from `app/core/interfaces.py`
2. Register the component in the appropriate registry
3. Add configuration options to `app/config/settings.py` if needed

## Documentation

For more detailed documentation, see:

- [Vector Database Guide](docs/vector_databases.md)
- [Embedding Models Guide](docs/embedding_models.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for GPT models
- Anthropic for Claude models
- The Langchain community for inspiration 