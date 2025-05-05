"""
Benchmarking utility for RAGStackXL vector databases.

This module provides tools to benchmark different vector database implementations
with varying collection sizes, indices, and query patterns.
"""

import os
import sys
import time
import argparse
import asyncio
import random
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Add the project root to path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_dir))

from app.core.interfaces import RagDocument
from app.vectordb.interfaces import VectorDBProvider, VectorDBConfig, VectorDB
from app.vectordb.factory import create_vectordb
from app.utils.logging import log


# Configuration
DEFAULT_OUTPUT_DIR = "benchmark_results"
DEFAULT_DIMENSIONS = [96, 384, 768, 1536]
DEFAULT_COLLECTION_SIZES = [100, 1000, 10000, 100000]
DEFAULT_PROVIDERS = ["faiss", "faiss_advanced", "qdrant", "weaviate", "pinecone", "milvus"]
DEFAULT_METRICS = ["cosine", "euclidean", "dot"]
DEFAULT_QUERY_COUNTS = [10, 100]


class BenchmarkConfig:
    """Configuration for vector database benchmarks."""
    
    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        dimensions: List[int] = None,
        collection_sizes: List[int] = None,
        providers: List[str] = None,
        metrics: List[str] = None,
        query_counts: List[int] = None,
        k_values: List[int] = None,
        num_runs: int = 3,
        random_seed: int = 42,
        plot_results: bool = True,
        run_all: bool = False
    ):
        """
        Initialize benchmark configuration.
        
        Args:
            output_dir: Directory to store benchmark results
            dimensions: List of embedding dimensions to test
            collection_sizes: List of collection sizes to test
            providers: List of vector database providers to test
            metrics: List of distance metrics to test
            query_counts: List of query counts to test
            k_values: List of k values for similarity search
            num_runs: Number of runs for each configuration
            random_seed: Random seed for reproducibility
            plot_results: Whether to plot results
            run_all: Whether to run all combinations (can be very time-consuming)
        """
        self.output_dir = output_dir
        self.dimensions = dimensions or DEFAULT_DIMENSIONS
        self.collection_sizes = collection_sizes or DEFAULT_COLLECTION_SIZES
        self.providers = providers or DEFAULT_PROVIDERS
        self.metrics = metrics or DEFAULT_METRICS
        self.query_counts = query_counts or DEFAULT_QUERY_COUNTS
        self.k_values = k_values or [1, 10, 100]
        self.num_runs = num_runs
        self.random_seed = random_seed
        self.plot_results = plot_results
        self.run_all = run_all
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)


class BenchmarkMetrics:
    """Metrics for benchmarking vector databases."""
    
    def __init__(self):
        """Initialize benchmark metrics."""
        self.init_time = 0.0
        self.add_times = []
        self.search_times = []
        self.memory_usage = 0.0
        self.index_size = 0.0


class BenchmarkResult:
    """Result of a vector database benchmark."""
    
    def __init__(
        self,
        provider: str,
        dimension: int,
        collection_size: int,
        metric: str,
        query_count: int,
        k: int
    ):
        """Initialize benchmark result."""
        self.provider = provider
        self.dimension = dimension
        self.collection_size = collection_size
        self.metric = metric
        self.query_count = query_count
        self.k = k
        self.metrics = BenchmarkMetrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "provider": self.provider,
            "dimension": self.dimension,
            "collection_size": self.collection_size,
            "metric": self.metric,
            "query_count": self.query_count,
            "k": self.k,
            "init_time": self.metrics.init_time,
            "add_time_mean": np.mean(self.metrics.add_times) if self.metrics.add_times else 0.0,
            "add_time_std": np.std(self.metrics.add_times) if self.metrics.add_times else 0.0,
            "search_time_mean": np.mean(self.metrics.search_times) if self.metrics.search_times else 0.0,
            "search_time_std": np.std(self.metrics.search_times) if self.metrics.search_times else 0.0,
            "memory_usage": self.metrics.memory_usage,
            "index_size": self.metrics.index_size
        }


def generate_random_documents(
    count: int,
    dimension: int,
    seed: Optional[int] = None
) -> Tuple[List[RagDocument], List[List[float]]]:
    """
    Generate random documents and embeddings for benchmarking.
    
    Args:
        count: Number of documents to generate
        dimension: Dimension of embeddings
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (documents, embeddings)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    documents = []
    embeddings = []
    
    categories = ["ai", "ml", "dl", "nlp", "cv", "rl", "robotics", "ethics"]
    tags = ["research", "tutorial", "review", "application", "theory", "code", "paper"]
    years = list(range(2018, 2024))
    
    for i in range(count):
        # Generate random metadata
        category = random.choice(categories)
        doc_tags = random.sample(tags, random.randint(1, 3))
        year = random.choice(years)
        
        # Create document
        doc = RagDocument(
            content=f"Document {i} about {category}",
            metadata={
                "category": category,
                "tags": doc_tags,
                "year": year,
                "score": random.random()
            },
            doc_id=f"doc_{i}"
        )
        
        # Generate random embedding
        embedding = list(np.random.rand(dimension).astype(np.float32))
        
        documents.append(doc)
        embeddings.append(embedding)
    
    return documents, embeddings


def generate_random_queries(
    count: int,
    dimension: int,
    seed: Optional[int] = None
) -> List[List[float]]:
    """
    Generate random query embeddings for benchmarking.
    
    Args:
        count: Number of queries to generate
        dimension: Dimension of embeddings
        seed: Random seed for reproducibility
        
    Returns:
        List of query embeddings
    """
    if seed is not None:
        np.random.seed(seed)
    
    return [list(np.random.rand(dimension).astype(np.float32)) for _ in range(count)]


async def benchmark_provider(
    provider: str,
    dimension: int,
    collection_size: int,
    metric: str,
    query_count: int,
    k: int,
    random_seed: int,
    temp_dir: str
) -> BenchmarkResult:
    """
    Benchmark a vector database provider.
    
    Args:
        provider: Vector database provider name
        dimension: Embedding dimension
        collection_size: Collection size
        metric: Distance metric
        query_count: Number of queries to run
        k: Number of results to return
        random_seed: Random seed for reproducibility
        temp_dir: Temporary directory for persistence
        
    Returns:
        Benchmark result
    """
    # Create result object
    result = BenchmarkResult(
        provider=provider,
        dimension=dimension,
        collection_size=collection_size,
        metric=metric,
        query_count=query_count,
        k=k
    )
    
    # Ensure temp directory exists
    persist_dir = os.path.join(temp_dir, f"{provider}_{dimension}_{collection_size}_{metric}")
    os.makedirs(persist_dir, exist_ok=True)
    
    # Generate documents and queries
    documents, embeddings = generate_random_documents(collection_size, dimension, random_seed)
    queries = generate_random_queries(query_count, dimension, random_seed + 1)
    
    # Create vector database
    try:
        start_time = time.time()
        db = create_vectordb(
            provider=provider,
            collection_name=f"benchmark_{collection_size}_{dimension}",
            embedding_dimension=dimension,
            distance_metric=metric,
            persist_directory=persist_dir
        )
        result.metrics.init_time = time.time() - start_time
    except Exception as e:
        log.error(f"Error creating vector database {provider}: {e}")
        return result
    
    # Add documents
    try:
        # Add documents in batches
        batch_size = 100
        for i in range(0, collection_size, batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            start_time = time.time()
            await db.add_documents(batch_docs, batch_embeddings)
            result.metrics.add_times.append(time.time() - start_time)
    except Exception as e:
        log.error(f"Error adding documents to {provider}: {e}")
        return result
    
    # Run queries
    try:
        for query in queries:
            start_time = time.time()
            await db.similarity_search(query, k=k)
            result.metrics.search_times.append(time.time() - start_time)
    except Exception as e:
        log.error(f"Error searching with {provider}: {e}")
    
    # Get index size
    try:
        if os.path.exists(persist_dir):
            # Calculate directory size in MB
            total_size = 0
            for path, dirs, files in os.walk(persist_dir):
                for f in files:
                    fp = os.path.join(path, f)
                    total_size += os.path.getsize(fp)
            
            result.metrics.index_size = total_size / (1024 * 1024)  # MB
    except Exception as e:
        log.error(f"Error getting index size for {provider}: {e}")
    
    return result


async def run_benchmarks(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """
    Run vector database benchmarks.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        List of benchmark results
    """
    results = []
    temp_dir = os.path.join(config.output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create list of configurations to benchmark
    benchmark_configs = []
    
    if config.run_all:
        # Run all combinations (warning: this can be very time-consuming)
        for provider in config.providers:
            for dimension in config.dimensions:
                for size in config.collection_sizes:
                    for metric in config.metrics:
                        for query_count in config.query_counts:
                            for k in config.k_values:
                                benchmark_configs.append((provider, dimension, size, metric, query_count, k))
    else:
        # Run a more reasonable subset of configurations
        default_metric = "cosine"
        default_query_count = 10
        default_k = 10
        default_dimension = 384
        default_size = 1000
        
        # Test all providers with default settings
        for provider in config.providers:
            benchmark_configs.append((provider, default_dimension, default_size, default_metric, default_query_count, default_k))
        
        # Test different dimensions with default provider
        default_provider = config.providers[0]
        for dimension in config.dimensions:
            benchmark_configs.append((default_provider, dimension, default_size, default_metric, default_query_count, default_k))
        
        # Test different collection sizes with default provider
        for size in config.collection_sizes:
            benchmark_configs.append((default_provider, default_dimension, size, default_metric, default_query_count, default_k))
        
        # Test different metrics with default provider
        for metric in config.metrics:
            benchmark_configs.append((default_provider, default_dimension, default_size, metric, default_query_count, default_k))
        
        # Test different query counts with default provider
        for query_count in config.query_counts:
            benchmark_configs.append((default_provider, default_dimension, default_size, default_metric, query_count, default_k))
        
        # Test different k values with default provider
        for k in config.k_values:
            benchmark_configs.append((default_provider, default_dimension, default_size, default_metric, default_query_count, k))
    
    # Run benchmarks
    for provider, dimension, size, metric, query_count, k in benchmark_configs:
        log.info(f"Benchmarking {provider} (dim={dimension}, size={size}, metric={metric}, queries={query_count}, k={k})")
        
        # Run multiple times and average results
        for run in range(config.num_runs):
            log.info(f"  Run {run+1}/{config.num_runs}")
            
            try:
                result = await benchmark_provider(
                    provider=provider,
                    dimension=dimension,
                    collection_size=size,
                    metric=metric,
                    query_count=query_count,
                    k=k,
                    random_seed=config.random_seed + run,
                    temp_dir=temp_dir
                )
                
                results.append(result)
            except Exception as e:
                log.error(f"Error in benchmark run: {e}")
    
    return results


def save_results(results: List[BenchmarkResult], output_dir: str):
    """
    Save benchmark results to files.
    
    Args:
        results: List of benchmark results
        output_dir: Output directory
    """
    # Convert results to list of dictionaries
    results_dicts = [result.to_dict() for result in results]
    
    # Save as JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results_dicts, f, indent=2)
    
    log.info(f"Results saved to {json_path}")


def plot_results(results: List[BenchmarkResult], output_dir: str):
    """
    Plot benchmark results.
    
    Args:
        results: List of benchmark results
        output_dir: Output directory
    """
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier manipulation
    import pandas as pd
    df = pd.DataFrame([r.to_dict() for r in results])
    
    # Plot 1: Provider comparison (init time)
    plt.figure(figsize=(12, 8))
    provider_times = df.groupby("provider")["init_time"].mean().sort_values()
    provider_times.plot(kind="bar")
    plt.title("Initialization Time by Provider")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "init_time_by_provider.png"))
    
    # Plot 2: Provider comparison (add time)
    plt.figure(figsize=(12, 8))
    add_times = df.groupby("provider")["add_time_mean"].mean().sort_values()
    add_times.plot(kind="bar")
    plt.title("Document Addition Time by Provider")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "add_time_by_provider.png"))
    
    # Plot 3: Provider comparison (search time)
    plt.figure(figsize=(12, 8))
    search_times = df.groupby("provider")["search_time_mean"].mean().sort_values()
    search_times.plot(kind="bar")
    plt.title("Search Time by Provider")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "search_time_by_provider.png"))
    
    # Plot 4: Dimension impact on search time
    plt.figure(figsize=(12, 8))
    for provider in df["provider"].unique():
        provider_df = df[df["provider"] == provider]
        if len(provider_df) > 0 and len(provider_df["dimension"].unique()) > 1:
            dim_times = provider_df.groupby("dimension")["search_time_mean"].mean()
            plt.plot(dim_times.index, dim_times.values, marker="o", label=provider)
    
    plt.title("Impact of Dimension on Search Time")
    plt.xlabel("Dimension")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "search_time_by_dimension.png"))
    
    # Plot 5: Collection size impact on search time
    plt.figure(figsize=(12, 8))
    for provider in df["provider"].unique():
        provider_df = df[df["provider"] == provider]
        if len(provider_df) > 0 and len(provider_df["collection_size"].unique()) > 1:
            size_times = provider_df.groupby("collection_size")["search_time_mean"].mean()
            plt.plot(size_times.index, size_times.values, marker="o", label=provider)
    
    plt.title("Impact of Collection Size on Search Time")
    plt.xlabel("Collection Size")
    plt.ylabel("Time (seconds)")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "search_time_by_collection_size.png"))
    
    # Plot 6: Index size by provider and collection size
    if any(r.metrics.index_size > 0 for r in results):
        plt.figure(figsize=(12, 8))
        for provider in df["provider"].unique():
            provider_df = df[df["provider"] == provider]
            if len(provider_df) > 0 and len(provider_df["collection_size"].unique()) > 1:
                size_index = provider_df.groupby("collection_size")["index_size"].mean()
                plt.plot(size_index.index, size_index.values, marker="o", label=provider)
        
        plt.title("Index Size by Collection Size")
        plt.xlabel("Collection Size")
        plt.ylabel("Size (MB)")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "index_size_by_collection_size.png"))
    
    log.info(f"Plots saved to {plots_dir}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Vector Database Benchmark")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--providers", nargs="+", choices=[p.value for p in VectorDBProvider], help="Providers to benchmark")
    parser.add_argument("--dimensions", type=int, nargs="+", help="Dimensions to benchmark")
    parser.add_argument("--collection-sizes", type=int, nargs="+", help="Collection sizes to benchmark")
    parser.add_argument("--metrics", nargs="+", choices=["cosine", "euclidean", "dot"], help="Distance metrics to benchmark")
    parser.add_argument("--query-counts", type=int, nargs="+", help="Query counts to benchmark")
    parser.add_argument("--k-values", type=int, nargs="+", help="K values to benchmark")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-plot", action="store_true", help="Don't plot results")
    parser.add_argument("--run-all", action="store_true", help="Run all combinations (warning: very time-consuming)")
    
    args = parser.parse_args()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        output_dir=args.output_dir,
        dimensions=args.dimensions,
        collection_sizes=args.collection_sizes,
        providers=args.providers,
        metrics=args.metrics,
        query_counts=args.query_counts,
        k_values=args.k_values,
        num_runs=args.num_runs,
        random_seed=args.random_seed,
        plot_results=not args.no_plot,
        run_all=args.run_all
    )
    
    # Run benchmarks
    log.info("Starting vector database benchmarks")
    results = await run_benchmarks(config)
    
    # Save results
    save_results(results, config.output_dir)
    
    # Plot results
    if config.plot_results:
        try:
            plot_results(results, config.output_dir)
        except Exception as e:
            log.error(f"Error plotting results: {e}")
            log.info("To plot results, install matplotlib and pandas")
    
    log.info("Benchmark complete")


if __name__ == "__main__":
    asyncio.run(main()) 