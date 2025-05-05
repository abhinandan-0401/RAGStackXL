"""
Example script to demonstrate and test embedding models in RAGStackXL.

This script initializes different embedding models and shows how to use them
for basic embedding operations.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any
import time

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embedding import (
    create_embedding_model,
    EmbeddingModelType,
    EmbeddingConfig
)
from app.utils.logging import log


async def test_embedding_model(model_type: str, model_name: str, **kwargs):
    """Test an embedding model with sample inputs."""
    log.info(f"Testing {model_type} embedding model: {model_name}")
    
    try:
        # Create the embedding model
        model = create_embedding_model(
            model_type=model_type,
            model_name=model_name,
            **kwargs
        )
        
        # Sample texts for embedding
        query = "What is the capital of France?"
        documents = [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "London is the capital of the United Kingdom.",
            "Rome is the capital of Italy."
        ]
        
        # Time query embedding
        start_time = time.time()
        query_embedding = await model.embed_query(query)
        query_time = time.time() - start_time
        
        log.info(f"Query embedding completed in {query_time:.4f} seconds")
        log.info(f"Query embedding dimension: {len(query_embedding)}")
        
        # Time document embeddings
        start_time = time.time()
        doc_embeddings = await model.embed_documents(documents)
        doc_time = time.time() - start_time
        
        log.info(f"Document embeddings completed in {doc_time:.4f} seconds for {len(documents)} documents")
        log.info(f"Document embedding dimensions: {[len(emb) for emb in doc_embeddings]}")
        
        return {
            "model_type": model_type,
            "model_name": model_name,
            "query_time": query_time,
            "query_dimension": len(query_embedding),
            "docs_time": doc_time,
            "docs_per_second": len(documents) / doc_time,
            "docs_dimensions": [len(emb) for emb in doc_embeddings]
        }
    
    except Exception as e:
        log.error(f"Error testing {model_type} model: {str(e)}")
        return {
            "model_type": model_type,
            "model_name": model_name,
            "error": str(e)
        }


async def test_all_available_models():
    """Test all available embedding models that can be used without API keys."""
    results = []
    
    # Test FastEmbed (default model)
    results.append(await test_embedding_model(
        model_type="fastembed",
        model_name="default"
    ))
    
    # Test SBERT with a small model
    try:
        results.append(await test_embedding_model(
            model_type="sbert",
            model_name="all-MiniLM-L6-v2"  # Small, fast model
        ))
    except Exception as e:
        log.error(f"Error testing SBERT model: {str(e)}")
    
    # Test HuggingFace with a small BERT model
    # Note: This requires transformers to be installed
    try:
        results.append(await test_embedding_model(
            model_type="huggingface",
            model_name="prajjwal1/bert-tiny"  # Very small BERT model for testing
        ))
    except Exception as e:
        log.error(f"Error testing HuggingFace model: {str(e)}")
    
    # Print summary of results
    log.info("\n--- Embedding Model Test Results ---")
    for result in results:
        if "error" in result:
            log.info(f"{result['model_type']} ({result['model_name']}): Error - {result['error']}")
        else:
            log.info(
                f"{result['model_type']} ({result['model_name']}): "
                f"Query time: {result['query_time']:.4f}s, "
                f"Docs per second: {result['docs_per_second']:.2f}, "
                f"Dimension: {result['query_dimension']}"
            )
    
    return results


async def test_api_models():
    """Test API-based models if API keys are available."""
    results = []
    
    # Test OpenAI if API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        log.info("Found OpenAI API key, testing OpenAI embedding")
        results.append(await test_embedding_model(
            model_type="openai",
            model_name="text-embedding-ada-002",
            api_key=openai_api_key
        ))
    else:
        log.info("Skipping OpenAI test (no API key)")
    
    # Test Cohere if API key is available
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if cohere_api_key:
        log.info("Found Cohere API key, testing Cohere embedding")
        results.append(await test_embedding_model(
            model_type="cohere",
            model_name="embed-english-v2.0",
            api_key=cohere_api_key
        ))
    else:
        log.info("Skipping Cohere test (no API key)")
    
    # Print summary of results
    if results:
        log.info("\n--- API-based Model Test Results ---")
        for result in results:
            if "error" in result:
                log.info(f"{result['model_type']} ({result['model_name']}): Error - {result['error']}")
            else:
                log.info(
                    f"{result['model_type']} ({result['model_name']}): "
                    f"Query time: {result['query_time']:.4f}s, "
                    f"Docs per second: {result['docs_per_second']:.2f}, "
                    f"Dimension: {result['query_dimension']}"
                )
    
    return results


async def main():
    """Main entry point for the example script."""
    log.info("Beginning embedding model tests")
    
    # Test models that don't require API keys
    await test_all_available_models()
    
    # Test API-based models if API keys are available
    await test_api_models()
    
    log.info("Embedding model tests completed")


if __name__ == "__main__":
    asyncio.run(main()) 