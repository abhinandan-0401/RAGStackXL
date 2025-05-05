"""
Vector database utilities.

This module contains utility functions for working with vector databases.
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional, TypeVar, Union, Callable

from app.config.settings import settings
from app.vectordb.interfaces import VectorDB, VectorDBConfig, VectorDBProvider, SearchResult
from app.utils.logging import log

# Import the factory function from factory.py
from app.vectordb.factory import create_vectordb


T = TypeVar('T')

# Define filter operation types
class FilterOperator:
    """Filter operators for complex queries."""
    AND = "$and"
    OR = "$or"
    NOT = "$not"
    EQ = "$eq"
    NE = "$ne"
    GT = "$gt"
    GTE = "$gte"
    LT = "$lt"
    LTE = "$lte"
    IN = "$in"
    NIN = "$nin"
    CONTAINS = "$contains"
    STARTSWITH = "$startswith"
    ENDSWITH = "$endswith"
    REGEX = "$regex"


def add_vectordb_args(parser: argparse.ArgumentParser):
    """Add vector database arguments to a command line parser."""
    db_group = parser.add_argument_group("Vector Database")
    db_group.add_argument(
        "--db-provider",
        choices=[p.value for p in VectorDBProvider],
        help=f"Vector database provider (default: {settings.VECTORDB.PROVIDER})"
    )
    db_group.add_argument(
        "--db-collection",
        help=f"Vector database collection name (default: {settings.VECTORDB.COLLECTION_NAME})"
    )
    db_group.add_argument(
        "--db-dimension",
        type=int,
        help=f"Vector embedding dimension (default: {settings.VECTORDB.EMBEDDING_DIMENSION})"
    )
    db_group.add_argument(
        "--db-metric",
        choices=["cosine", "euclidean", "dot"],
        help=f"Distance metric to use (default: {settings.VECTORDB.DISTANCE_METRIC})"
    )
    db_group.add_argument(
        "--db-path",
        help=f"Vector database persist directory (default: {settings.VECTORDB.PERSIST_DIRECTORY})"
    )


def create_vectordb_from_args(args: argparse.Namespace) -> VectorDB:
    """Create a vector database from command line arguments."""
    # Extract arguments
    provider = getattr(args, "db_provider", None)
    collection = getattr(args, "db_collection", None)
    dimension = getattr(args, "db_dimension", None)
    metric = getattr(args, "db_metric", None)
    persist_dir = getattr(args, "db_path", None)
    
    return create_vectordb(
        provider=provider,
        collection_name=collection,
        embedding_dimension=dimension,
        distance_metric=metric,
        persist_directory=persist_dir
    )


def format_search_results(results: List[SearchResult], max_length: int = 100) -> str:
    """
    Format search results for display.
    
    Args:
        results: Search results to format
        max_length: Maximum length of content to display
        
    Returns:
        Formatted string
    """
    if not results:
        return "No results found."
    
    formatted = []
    for i, result in enumerate(results):
        doc = result.document
        content = doc.content[:max_length] + "..." if len(doc.content) > max_length else doc.content
        formatted.append(
            f"[{i+1}] Score: {result.score:.4f} | ID: {doc.doc_id}\n"
            f"    {content}\n"
            f"    Metadata: {doc.metadata}"
        )
    
    return "\n\n".join(formatted)


def create_filter(
    field: Optional[str] = None,
    value: Optional[Any] = None,
    operator: Optional[str] = None,
    conditions: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create a filter for vector database queries.
    
    This function allows creating both simple filters and complex nested filters.
    
    Examples:
        Simple equality filter:
            create_filter("source", "wikipedia")
        
        Range filter:
            create_filter("score", 0.5, FilterOperator.GT)
        
        List membership:
            create_filter("category", ["science", "technology"], FilterOperator.IN)
        
        Complex AND condition:
            create_filter(
                conditions=[
                    create_filter("source", "wikipedia"),
                    create_filter("score", 0.5, FilterOperator.GT)
                ],
                operator=FilterOperator.AND
            )
    
    Args:
        field: Field name to filter on (for simple filters)
        value: Value to filter by (for simple filters)
        operator: Operator to use (for simple filters or complex conditions)
        conditions: List of conditions for complex filters
        
    Returns:
        Filter dictionary
    """
    # Handle complex filters with conditions
    if conditions and operator:
        if operator in [FilterOperator.AND, FilterOperator.OR]:
            return {operator: conditions}
        elif operator == FilterOperator.NOT:
            if len(conditions) != 1:
                raise ValueError("NOT operator requires exactly one condition")
            return {operator: conditions[0]}
        else:
            raise ValueError(f"Invalid operator for conditions: {operator}")
    
    # Handle simple filters
    if field is not None:
        if operator is None:
            # Simple equality
            return {field: value}
        elif operator == FilterOperator.EQ:
            return {field: value}
        elif operator == FilterOperator.NE:
            return {field: {FilterOperator.NE: value}}
        elif operator in [FilterOperator.GT, FilterOperator.GTE, FilterOperator.LT, FilterOperator.LTE]:
            return {field: {operator: value}}
        elif operator == FilterOperator.IN:
            if not isinstance(value, list):
                value = [value]
            return {field: {FilterOperator.IN: value}}
        elif operator == FilterOperator.NIN:
            if not isinstance(value, list):
                value = [value]
            return {field: {FilterOperator.NIN: value}}
        elif operator in [FilterOperator.CONTAINS, FilterOperator.STARTSWITH, FilterOperator.ENDSWITH, FilterOperator.REGEX]:
            # Text operators (provider compatibility may vary)
            return {field: {operator: value}}
        else:
            raise ValueError(f"Invalid operator: {operator}")
    
    raise ValueError("Either field or conditions must be provided")


def combine_filters(
    filters: List[Dict[str, Any]], 
    operator: str = FilterOperator.AND
) -> Dict[str, Any]:
    """
    Combine multiple filters with the specified operator.
    
    Args:
        filters: List of filters to combine
        operator: Operator to use (AND or OR)
        
    Returns:
        Combined filter
    """
    if not filters:
        return {}
    
    if len(filters) == 1:
        return filters[0]
    
    if operator not in [FilterOperator.AND, FilterOperator.OR]:
        raise ValueError(f"Invalid operator for combining filters: {operator}")
    
    return {operator: filters}


class HybridSearchParams:
    """Parameters for hybrid search (vector + keyword)."""
    
    def __init__(
        self,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        keyword_fields: Optional[List[str]] = None,
        min_keyword_score: float = 0.1,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75
    ):
        """
        Initialize hybrid search parameters.
        
        Args:
            vector_weight: Weight for vector search scores (0.0-1.0)
            keyword_weight: Weight for keyword search scores (0.0-1.0)
            keyword_fields: Fields to search for keywords (if None, searches all text fields)
            min_keyword_score: Minimum keyword score to consider (0.0-1.0)
            bm25_k1: BM25 k1 parameter (controls term frequency saturation)
            bm25_b: BM25 b parameter (controls field length normalization)
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.keyword_fields = keyword_fields or ["content", "text"]
        self.min_keyword_score = min_keyword_score
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        if total_weight != 1.0:
            self.vector_weight /= total_weight
            self.keyword_weight /= total_weight


# Function to validate non-empty texts and convert to list if needed
def validate_text_query(text_query: Union[str, List[str]]) -> List[str]:
    """Validate and normalize text query input."""
    if isinstance(text_query, str):
        text_query = [text_query]
    
    if not text_query or not all(t.strip() for t in text_query):
        raise ValueError("Text query cannot be empty")
    
    return [t.strip() for t in text_query]


def apply_value_transform(value: T, transform: Optional[Callable[[T], Any]] = None) -> Any:
    """Apply transformation to a value if transform function is provided."""
    if transform is None:
        return value
    return transform(value) 