"""
Retrieval mechanisms package for RAGStackXL.

This package provides various retrieval mechanisms for the RAG system.
"""

from app.retrieval.interfaces import (
    RetrieverType,
    QueryReformulation,
    Reranker,
    RetrieverConfig,
    Retriever,
    RetrieverFactory
)

from app.retrieval.basic import BasicRetriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.mmr import MMRRetriever
from app.retrieval.query_expansion import (
    QueryExpansionRetriever,
    SimpleQueryReformulation,
    LLMQueryReformulation
)
from app.retrieval.reranking import (
    RerankingRetriever,
    SimpleScoreReranker,
    CosineSimilarityReranker
)
from app.retrieval.contextual import ContextualRetriever

# Register retrievers with the factory
RetrieverFactory.register(RetrieverType.BASIC, BasicRetriever)
RetrieverFactory.register(RetrieverType.SEMANTIC, SemanticRetriever)
RetrieverFactory.register(RetrieverType.HYBRID, HybridRetriever)
RetrieverFactory.register(RetrieverType.MMR, MMRRetriever)
RetrieverFactory.register(RetrieverType.QUERY_EXPANSION, QueryExpansionRetriever)
RetrieverFactory.register(RetrieverType.RERANKING, RerankingRetriever)
RetrieverFactory.register(RetrieverType.CONTEXTUAL, ContextualRetriever)

__all__ = [
    # Interfaces and base classes
    "RetrieverType",
    "QueryReformulation",
    "Reranker",
    "RetrieverConfig",
    "Retriever",
    "RetrieverFactory",
    
    # Retriever implementations
    "BasicRetriever",
    "SemanticRetriever",
    "HybridRetriever",
    "MMRRetriever",
    "QueryExpansionRetriever",
    "RerankingRetriever",
    "ContextualRetriever",
    
    # Helper classes
    "SimpleQueryReformulation",
    "LLMQueryReformulation",
    "SimpleScoreReranker",
    "CosineSimilarityReranker"
] 