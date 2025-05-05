"""
Benchmarking script for embedding models in RAGStackXL.

This script compares the performance of different embedding models on:
1. Query embedding speed
2. Document embedding speed
3. Memory usage
4. Embedding quality (if possible)
"""

import os
import sys
import asyncio
import time
import json
import psutil
from datetime import datetime
from typing import List, Dict, Any

# Add the parent directory to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embedding import (
    create_embedding_model,
    EmbeddingModelType,
    EmbeddingConfig
)
from app.utils.logging import log


# Test data
SHORT_QUERY = "What is the capital of France?"
MEDIUM_QUERY = "Can you explain the basic principles of quantum mechanics and how they differ from classical physics?"
LONG_QUERY = "I'm interested in understanding the historical development of artificial intelligence, from its early days with figures like Alan Turing, through the AI winters, to today's deep learning revolution. Could you outline the major milestones and paradigm shifts in the field?"

SHORT_DOCUMENT = "Paris is the capital of France."
MEDIUM_DOCUMENT = """
Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales of energy levels of atoms and subatomic particles. 
Classical physics, on the other hand, is an older theory that describes many aspects of nature at an ordinary scale.
The main difference between the two is that quantum mechanics introduces concepts like wave-particle duality, quantum entanglement, and uncertainty principle.
"""
LONG_DOCUMENT = """
The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. 
The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. 
This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.
This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.

Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism of Sir James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter", a period when obtaining funding for AI projects was difficult.

During the 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began.

The development of metal–oxide–semiconductor (MOS) very-large-scale integration (VLSI), in the form of complementary MOS (CMOS) technology, enabled the development of practical artificial neural network (ANN) technology in the 1980s. Interest in neural networks and "connectionism" was revived by Geoffrey Hinton, David Rumelhart and others in the middle of the 1980s. Neural networks would become an important part of AI in the 2010s.

The field of AI research has experienced multiple cycles of optimism and disappointment, massive funding followed by cutbacks. There was constant progress in various specialized domains such as machine vision, speech recognition, board games, and robotics, but general human-level AI remained elusive. As of the early 21st century, the field has seen renewed commitment from governments and corporations, notably due to the successes in machine learning, and AI has become the linchpin of the so-called Fourth Industrial Revolution.
"""

# Medium corpus: 10 documents
MEDIUM_CORPUS = [
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


async def benchmark_query_embedding(model, query: str, num_runs: int = 3):
    """Benchmark query embedding performance."""
    times = []
    memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    for i in range(num_runs):
        start_time = time.time()
        embedding = await model.embed_query(query)
        elapsed = time.time() - start_time
        times.append(elapsed)
        
    memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    memory_usage = memory_after - memory_before
    
    avg_time = sum(times) / len(times)
    
    return {
        "avg_time": avg_time,
        "min_time": min(times),
        "max_time": max(times),
        "embedding_dimension": len(embedding),
        "memory_usage_mb": memory_usage
    }


async def benchmark_document_embedding(model, documents: List[str], num_runs: int = 3):
    """Benchmark document embedding performance."""
    times = []
    memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    for i in range(num_runs):
        start_time = time.time()
        embeddings = await model.embed_documents(documents)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    memory_usage = memory_after - memory_before
    
    avg_time = sum(times) / len(times)
    docs_per_second = len(documents) / avg_time
    
    return {
        "avg_time": avg_time,
        "min_time": min(times),
        "max_time": max(times),
        "docs_per_second": docs_per_second,
        "embedding_dimension": len(embeddings[0]),
        "memory_usage_mb": memory_usage
    }


async def run_benchmark(model_type: str, model_name: str, **kwargs):
    """Run a full benchmark for a single model."""
    log.info(f"Benchmarking {model_type} model: {model_name}")
    
    try:
        # Create the model
        model = create_embedding_model(
            model_type=model_type,
            model_name=model_name,
            **kwargs
        )
        
        # Benchmark query embedding (short, medium, long)
        short_query_results = await benchmark_query_embedding(model, SHORT_QUERY)
        medium_query_results = await benchmark_query_embedding(model, MEDIUM_QUERY)
        long_query_results = await benchmark_query_embedding(model, LONG_QUERY)
        
        # Benchmark document embedding (short, medium, long)
        short_doc_results = await benchmark_query_embedding(model, SHORT_DOCUMENT)
        medium_doc_results = await benchmark_query_embedding(model, MEDIUM_DOCUMENT)
        long_doc_results = await benchmark_query_embedding(model, LONG_DOCUMENT)
        
        # Benchmark corpus embedding
        corpus_results = await benchmark_document_embedding(model, MEDIUM_CORPUS)
        
        # Compile results
        results = {
            "model_type": model_type,
            "model_name": model_name,
            "embedding_dimension": model.config.dimension,
            "query_embedding": {
                "short": short_query_results,
                "medium": medium_query_results,
                "long": long_query_results
            },
            "document_embedding": {
                "short": short_doc_results,
                "medium": medium_doc_results,
                "long": long_doc_results
            },
            "corpus_embedding": corpus_results,
            "timestamp": datetime.now().isoformat(),
            "machine_info": {
                "cpu": psutil.cpu_count(),
                "memory": psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
            }
        }
        
        return results
    
    except Exception as e:
        log.error(f"Error benchmarking {model_type} model: {str(e)}")
        return {
            "model_type": model_type,
            "model_name": model_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def main():
    """Main entry point for the benchmark script."""
    log.info("Starting embedding model benchmarks")
    
    results = []
    
    # Benchmark FastEmbed
    results.append(await run_benchmark("fastembed", "default"))
    
    # Benchmark SBERT (if available)
    try:
        import sentence_transformers
        results.append(await run_benchmark("sbert", "all-MiniLM-L6-v2"))
    except ImportError:
        log.info("Skipping SBERT benchmark (not installed)")
    
    # Benchmark HuggingFace (if available)
    try:
        import transformers
        results.append(await run_benchmark("huggingface", "prajjwal1/bert-tiny"))
    except ImportError:
        log.info("Skipping HuggingFace benchmark (not installed)")
    
    # Benchmark OpenAI (if API key available)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        results.append(await run_benchmark(
            "openai", 
            "text-embedding-ada-002", 
            api_key=openai_api_key
        ))
    
    # Benchmark Cohere (if API key available)
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if cohere_api_key:
        results.append(await run_benchmark(
            "cohere", 
            "embed-english-v2.0", 
            api_key=cohere_api_key
        ))
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results/embedding_benchmark_{timestamp}.json"
    
    # Create directory if it doesn't exist
    os.makedirs("benchmark_results", exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Benchmark results saved to {output_file}")
    
    # Print summary
    log.info("\n--- Benchmark Summary ---")
    for result in results:
        if "error" in result:
            log.info(f"{result['model_type']} ({result['model_name']}): Error - {result['error']}")
            continue
        
        query_time = result["query_embedding"]["medium"]["avg_time"]
        corpus_docs_per_sec = result["corpus_embedding"]["docs_per_second"]
        dimension = result["embedding_dimension"]
        
        log.info(
            f"{result['model_type']} ({result['model_name']}): "
            f"Query time: {query_time:.4f}s, "
            f"Docs/sec: {corpus_docs_per_sec:.2f}, "
            f"Dimension: {dimension}"
        )


if __name__ == "__main__":
    asyncio.run(main()) 