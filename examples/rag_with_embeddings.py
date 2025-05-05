"""
Example script demonstrating a complete RAG workflow using embedding models and vector databases.

This script shows how to:
1. Create and configure an embedding model
2. Set up a vector database
3. Add documents to the vector database
4. Perform semantic search queries
"""

import os
import sys
import asyncio
import random
from typing import List, Dict, Any

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embedding import create_embedding_model
from app.vectordb import create_vectordb
from app.utils.logging import log
from app.core.interfaces import RagDocument


# Sample documents for demonstration
DOCUMENTS = [
    "Paris is the capital of France and is known as the City of Light.",
    "Berlin is the capital of Germany and is famous for its history and architecture.",
    "London is the capital of the United Kingdom and is home to the royal family.",
    "Rome is the capital of Italy and was the center of the ancient Roman Empire.",
    "Madrid is the capital of Spain and is known for its art and culture.",
    "Tokyo is the capital of Japan and is one of the most populous cities in the world.",
    "Beijing is the capital of China and has a rich history spanning over 3,000 years.",
    "Moscow is the capital of Russia and is known for its iconic architecture.",
    "Cairo is the capital of Egypt and is located near the ancient pyramids.",
    "New Delhi is the capital of India and is one of the oldest continuously inhabited cities in the world."
]


async def add_documents_to_vectordb(vectordb, embedding_model, documents):
    """Add documents to a vector database."""
    log.info(f"Adding {len(documents)} documents to vector database")
    
    # Create document objects
    doc_objects = []
    for i, text in enumerate(documents):
        doc = RagDocument(
            id=f"doc_{i}",
            content=text,
            metadata={"source": "example", "document_id": i}
        )
        doc_objects.append(doc)
    
    # Add documents to vector database
    await vectordb.add_documents(doc_objects)
    
    log.info(f"Added {len(documents)} documents to vector database")


async def search_vectordb(vectordb, embedding_model, query, top_k=3):
    """Search the vector database using semantic search."""
    log.info(f"Searching for: '{query}'")
    
    # Search the vector database
    results = await vectordb.query(query, top_k=top_k)
    
    log.info(f"Found {len(results)} results")
    for i, result in enumerate(results):
        log.info(f"Result {i+1} (score: {result.score:.4f}): {result.content}")
    
    return results


async def main():
    """Main entry point for the example script."""
    log.info("Starting RAG example with embedding models and vector databases")
    
    # Create an embedding model
    # Using FastEmbed as it doesn't require API keys
    log.info("Creating embedding model")
    embedding_model = create_embedding_model(
        model_type="fastembed",
        model_name="default"
    )
    
    # Create a vector database
    log.info("Creating vector database")
    vectordb = create_vectordb(
        provider="faiss",  # Using FAISS as it's in-memory and doesn't require setup
        collection_name="example_collection",
        embedding_model=embedding_model,
        embedding_dimension=embedding_model.config.dimension
    )
    
    # Add documents to the vector database
    await add_documents_to_vectordb(vectordb, embedding_model, DOCUMENTS)
    
    # Perform some example queries
    await search_vectordb(vectordb, embedding_model, "What is the capital of France?")
    await search_vectordb(vectordb, embedding_model, "Tell me about ancient cities")
    await search_vectordb(vectordb, embedding_model, "Which city has historical architecture?")
    
    log.info("RAG example completed successfully")


if __name__ == "__main__":
    asyncio.run(main()) 