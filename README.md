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

## Project Structure

```
app/
├── core/              # Core interfaces and abstractions
├── document_processing/ # Document loading and chunking
├── embedding/         # Embedding models
├── vectordb/         # Vector database integration
├── retrieval/         # Retrieval mechanisms
├── llm/               # LLM integration
├── agents/            # Agent system
├── api/               # API endpoints
├── ui/                # User interface
├── utils/             # Utility functions
├── evaluation/        # Evaluation metrics
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

4. Create a `.env` file by copying `.env.sample` and filling in your API keys:
   ```
   cp .env.sample .env
   # Edit the .env file with your API keys
   ```

## Usage

### Ingesting Documents

To ingest documents into the system:

```
python main.py ingest --source /path/to/documents --recursive
```

### Querying the System

To query the system:

```
python main.py query --query "Your question here"
```

### Running the Server

To start the API server:

```
python main.py server --host 0.0.0.0 --port 8000
```

## Development

### Running Tests

```
pytest tests/
```

### Adding New Components

1. Implement the appropriate interface from `app/core/interfaces.py`
2. Register the component in the dependency injection system
3. Add configuration options to `app/config/settings.py` if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAI for GPT models
- Anthropic for Claude models
- The Langchain community for inspiration 