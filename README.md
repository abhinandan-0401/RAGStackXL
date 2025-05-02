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

### In Progress:
- Phase 3: Embedding and Vectorization
- Implementing embedding models and vector databases

## Project Structure

```
app/
├── core/              # Core interfaces and abstractions
│   ├── loaders/       # Document loaders for different formats
│   └── chunking/      # Text chunking strategies
├── embedding/         # Embedding models (coming soon)
├── vectordb/         # Vector database integration (coming soon)
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

### Querying the System

To query the system (coming soon):

```
python main.py query --query "Your question here"
```

### Running the Server

To start the API server (coming soon):

```
python main.py server --host 0.0.0.0 --port 8000
```

## Development

### Running Tests

```
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app

# Run specific test files
pytest tests/unit/test_document_processor.py
```

### Adding New Components

1. Implement the appropriate interface from `app/core/interfaces.py`
2. Register the component in the appropriate registry
3. Add configuration options to `app/config/settings.py` if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for GPT models
- Anthropic for Claude models
- The Langchain community for inspiration 