"""
Example usage of RAGStackXL retrieval mechanisms.

This script demonstrates how to use the different retrieval mechanisms
implemented in RAGStackXL.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval import (
    RetrieverType,
    RetrieverConfig,
    RetrieverFactory,
    BasicRetriever,
    SemanticRetriever,
    HybridRetriever,
    MMRRetriever,
    QueryExpansionRetriever,
    RerankingRetriever,
    ContextualRetriever
)
from app.vectordb import VectorDBFactory, VectorDBProvider, VectorDBConfig
from app.embedding import EmbeddingModelFactory, EmbeddingModelType, EmbeddingConfig
from app.core.interfaces import RagDocument, DocumentMetadata
from app.utils.logging import log


async def setup_vector_database():
    """Set up a sample vector database with test documents."""
    # Configure the vector database
    db_config = VectorDBConfig(
        collection_name="test_collection",
        embedding_dimension=384,  # Using FASTEMBED_SMALL dimension
        distance_metric="cosine"
    )
    
    # Create vector database
    vectordb = VectorDBFactory.create(
        provider=VectorDBProvider.FAISS,
        config=db_config
    )
    
    # Create sample documents
    documents = [
        RagDocument(
            content="Neural networks are a type of machine learning model inspired by the human brain.",
            doc_id="doc1",
            metadata={"category": "ML", "difficulty": "beginner"}
        ),
        RagDocument(
            content="Transformers are a type of neural network architecture that uses self-attention mechanisms.",
            doc_id="doc2",
            metadata={"category": "ML", "difficulty": "intermediate"}
        ),
        RagDocument(
            content="Python is a popular programming language for data science and machine learning.",
            doc_id="doc3",
            metadata={"category": "Programming", "difficulty": "beginner"}
        ),
        RagDocument(
            content="GPT (Generative Pre-trained Transformer) is a language model developed by OpenAI.",
            doc_id="doc4",
            metadata={"category": "AI", "difficulty": "advanced"}
        ),
        RagDocument(
            content="TensorFlow and PyTorch are popular deep learning frameworks.",
            doc_id="doc5",
            metadata={"category": "ML", "difficulty": "intermediate"}
        ),
        RagDocument(
            content="The field of artificial intelligence aims to create systems that can perform tasks that typically require human intelligence.",
            doc_id="doc6",
            metadata={"category": "AI", "difficulty": "beginner"}
        ),
        RagDocument(
            content="Retrieval-Augmented Generation (RAG) combines retrieval systems with generative models to enhance response quality.",
            doc_id="doc7",
            metadata={"category": "AI", "difficulty": "advanced"}
        ),
        RagDocument(
            content="Transfer learning allows models trained on one task to be adapted for another related task.",
            doc_id="doc8",
            metadata={"category": "ML", "difficulty": "intermediate"}
        ),
        RagDocument(
            content="Embedding models convert text into dense vector representations that capture semantic meaning.",
            doc_id="doc9",
            metadata={"category": "ML", "difficulty": "intermediate"}
        ),
        RagDocument(
            content="Vector databases store and efficiently search through vector embeddings using approximate nearest neighbor algorithms.",
            doc_id="doc10",
            metadata={"category": "Databases", "difficulty": "advanced"}
        )
    ]
    
    # Configure embedding model
    embedding_config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_type=EmbeddingModelType.FASTEMBED,
        dimension=384
    )
    
    # Create embedding model
    embedding_model = EmbeddingModelFactory.create(embedding_config)
    
    # Embed documents
    doc_contents = [doc.content for doc in documents]
    embeddings = await embedding_model.embed_documents(doc_contents)
    
    # Add documents to vector database
    await vectordb.add_documents(documents, embeddings)
    
    log.info(f"Added {len(documents)} documents to the vector database")
    
    return vectordb, embedding_model


async def compare_retrievers(
    query: str,
    vectordb: Any,
    embedding_model: Any,
    k: int = 3,
    filter: Dict[str, Any] = None
):
    """Compare different retrieval mechanisms for the same query."""
    log.info(f"Query: '{query}'")
    log.info(f"Top-k: {k}")
    log.info(f"Filter: {filter}")
    log.info("-" * 50)
    
    # Create retriever configurations
    basic_config = RetrieverConfig(
        retriever_type=RetrieverType.BASIC,
        top_k=k
    )
    
    semantic_config = RetrieverConfig(
        retriever_type=RetrieverType.SEMANTIC,
        top_k=k,
        use_keyword_boost=True,
        keyword_boost_factor=0.2
    )
    
    hybrid_config = RetrieverConfig(
        retriever_type=RetrieverType.HYBRID,
        top_k=k,
        vector_weight=0.7,
        keyword_weight=0.3
    )
    
    mmr_config = RetrieverConfig(
        retriever_type=RetrieverType.MMR,
        top_k=k,
        lambda_param=0.7,
        fetch_k=10
    )
    
    query_expansion_config = RetrieverConfig(
        retriever_type=RetrieverType.QUERY_EXPANSION,
        top_k=k,
        max_queries=3,
        use_llm=False
    )
    
    reranking_config = RetrieverConfig(
        retriever_type=RetrieverType.RERANKING,
        top_k=k,
        initial_k=10,
        reranker_type="simple"
    )
    
    # Create retrievers
    retrievers = {
        "Basic": RetrieverFactory.create(vectordb, embedding_model, basic_config),
        "Semantic": RetrieverFactory.create(vectordb, embedding_model, semantic_config),
        "Hybrid": RetrieverFactory.create(vectordb, embedding_model, hybrid_config),
        "MMR": RetrieverFactory.create(vectordb, embedding_model, mmr_config),
        "QueryExpansion": RetrieverFactory.create(vectordb, embedding_model, query_expansion_config),
        "Reranking": RetrieverFactory.create(vectordb, embedding_model, reranking_config)
    }
    
    # Retrieve documents using each retriever
    for name, retriever in retrievers.items():
        log.info(f"\n{name} Retriever:")
        
        results = await retriever.retrieve(query, filter=filter)
        
        for i, result in enumerate(results, 1):
            log.info(f"{i}. [Score: {result.score:.4f}] {result.document.content}")
            log.info(f"   Metadata: {result.document.metadata}")
        
        log.info("-" * 50)


async def demonstrate_contextual_retriever(
    vectordb: Any,
    embedding_model: Any,
    k: int = 3
):
    """Demonstrate the contextual retriever with conversation history."""
    log.info("\nDemonstrating Contextual Retriever:")
    log.info("-" * 50)
    
    # Create contextual retriever
    contextual_config = RetrieverConfig(
        retriever_type=RetrieverType.CONTEXTUAL,
        top_k=k,
        context_window_size=2,
        context_strategy="combine"
    )
    
    contextual_retriever = ContextualRetriever(vectordb, embedding_model, contextual_config)
    
    # Simulate conversation history
    log.info("1. First query with no history:")
    results = await contextual_retriever.retrieve("What are neural networks?")
    
    for i, result in enumerate(results, 1):
        log.info(f"{i}. [Score: {result.score:.4f}] {result.document.content}")
    
    # Add system response to history
    contextual_retriever.add_to_history(
        "Neural networks are computational models inspired by the human brain that can learn patterns from data.",
        is_user=False
    )
    
    log.info("\n2. Follow-up query with history:")
    results = await contextual_retriever.retrieve("How do they compare to transformers?")
    
    for i, result in enumerate(results, 1):
        log.info(f"{i}. [Score: {result.score:.4f}] {result.document.content}")
    
    # Add system response to history
    contextual_retriever.add_to_history(
        "Transformers are a specific type of neural network architecture that uses self-attention mechanisms.",
        is_user=False
    )
    
    log.info("\n3. Another follow-up:")
    results = await contextual_retriever.retrieve("What are practical applications of these models?")
    
    for i, result in enumerate(results, 1):
        log.info(f"{i}. [Score: {result.score:.4f}] {result.document.content}")


async def main():
    """Run retrieval examples."""
    log.info("Setting up sample vector database...")
    vectordb, embedding_model = await setup_vector_database()
    
    log.info("\nComparing different retrievers:")
    
    # Compare retrievers with a simple query
    await compare_retrievers(
        query="What are neural networks?",
        vectordb=vectordb,
        embedding_model=embedding_model,
        k=3
    )
    
    # Compare retrievers with a different query
    await compare_retrievers(
        query="Tell me about machine learning models",
        vectordb=vectordb,
        embedding_model=embedding_model,
        k=3
    )
    
    # Compare retrievers with a filter
    await compare_retrievers(
        query="How do embedding models work?",
        vectordb=vectordb,
        embedding_model=embedding_model,
        k=3,
        filter={"category": "ML"}
    )
    
    # Demonstrate contextual retriever
    await demonstrate_contextual_retriever(vectordb, embedding_model)


if __name__ == "__main__":
    asyncio.run(main()) 