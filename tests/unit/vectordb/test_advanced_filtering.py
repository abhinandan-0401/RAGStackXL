"""
Tests for advanced filtering capabilities in vector databases.
"""
import pytest
from typing import List, Dict, Any

from app.vectordb.utils import FilterOperator, create_filter, combine_filters
from app.core.interfaces import RagDocument


@pytest.fixture
def sample_documents() -> List[RagDocument]:
    """Sample documents with various metadata for testing filters."""
    return [
        RagDocument(
            content="Document about artificial intelligence",
            metadata={"category": "ai", "year": 2023, "tags": ["research", "technology"]},
            doc_id="doc1"
        ),
        RagDocument(
            content="Document about machine learning algorithms",
            metadata={"category": "ml", "year": 2022, "tags": ["research", "algorithms"]},
            doc_id="doc2"
        ),
        RagDocument(
            content="Introduction to deep learning",
            metadata={"category": "dl", "year": 2021, "tags": ["neural networks", "tutorial"]},
            doc_id="doc3"
        ),
        RagDocument(
            content="Python programming for data science",
            metadata={"category": "programming", "year": 2023, "tags": ["python", "tutorial"]},
            doc_id="doc4"
        ),
        RagDocument(
            content="Guide to natural language processing",
            metadata={"category": "nlp", "year": 2022, "tags": ["language", "ai"]},
            doc_id="doc5"
        ),
    ]


def test_simple_filter_creation():
    """Test creating simple filters."""
    # Test equality filter
    filter1 = create_filter("category", "ai")
    assert filter1 == {"category": "ai"}
    
    # Test list filter
    filter2 = create_filter("tags", ["research", "technology"])
    assert filter2 == {"tags": ["research", "technology"]}


def test_operator_filter_creation():
    """Test creating filters with operators."""
    # Test greater than
    filter1 = create_filter("year", 2022, FilterOperator.GT)
    assert filter1 == {"year": {FilterOperator.GT: 2022}}
    
    # Test in operator
    filter2 = create_filter("category", ["ai", "ml"], FilterOperator.IN)
    assert filter2 == {"category": {FilterOperator.IN: ["ai", "ml"]}}
    
    # Test contains operator
    filter3 = create_filter("content", "intelligence", FilterOperator.CONTAINS)
    assert filter3 == {"content": {FilterOperator.CONTAINS: "intelligence"}}


def test_complex_filter_creation():
    """Test creating complex filters with multiple conditions."""
    # AND condition
    filter1 = create_filter(
        conditions=[
            create_filter("category", "ai"),
            create_filter("year", 2022, FilterOperator.GT)
        ],
        operator=FilterOperator.AND
    )
    
    assert filter1 == {
        FilterOperator.AND: [
            {"category": "ai"},
            {"year": {FilterOperator.GT: 2022}}
        ]
    }
    
    # OR condition
    filter2 = create_filter(
        conditions=[
            create_filter("category", "ai"),
            create_filter("category", "ml")
        ],
        operator=FilterOperator.OR
    )
    
    assert filter2 == {
        FilterOperator.OR: [
            {"category": "ai"},
            {"category": "ml"}
        ]
    }
    
    # NOT condition
    filter3 = create_filter(
        conditions=[create_filter("category", "programming")],
        operator=FilterOperator.NOT
    )
    
    assert filter3 == {
        FilterOperator.NOT: {"category": "programming"}
    }


def test_combining_filters():
    """Test combining multiple filters."""
    filter1 = create_filter("category", "ai")
    filter2 = create_filter("year", 2022, FilterOperator.GT)
    
    # Combine with AND
    combined1 = combine_filters([filter1, filter2], FilterOperator.AND)
    assert combined1 == {
        FilterOperator.AND: [
            {"category": "ai"},
            {"year": {FilterOperator.GT: 2022}}
        ]
    }
    
    # Combine with OR
    combined2 = combine_filters([filter1, filter2], FilterOperator.OR)
    assert combined2 == {
        FilterOperator.OR: [
            {"category": "ai"},
            {"year": {FilterOperator.GT: 2022}}
        ]
    }


def test_filter_application(sample_documents):
    """
    Test applying filters to a document collection.
    
    This is a simple implementation to verify filter logic.
    In real usage, filtering will be performed by the vector database.
    """
    def apply_filter(docs: List[RagDocument], filter_dict: Dict[str, Any]) -> List[RagDocument]:
        """Simple filter application for testing purposes."""
        # Import the document matching logic from FaissAdvancedVectorDB
        # This is a simplified version for testing
        def match_document(doc: RagDocument, condition: Dict[str, Any]) -> bool:
            if not doc.metadata:
                return False
            
            metadata = doc.metadata
            
            # Handle complex operators
            if FilterOperator.AND in condition:
                return all(match_document(doc, subcond) for subcond in condition[FilterOperator.AND])
            elif FilterOperator.OR in condition:
                return any(match_document(doc, subcond) for subcond in condition[FilterOperator.OR])
            elif FilterOperator.NOT in condition:
                return not match_document(doc, condition[FilterOperator.NOT])
            
            # Simple key-value pairs
            for key, value in condition.items():
                if key.startswith("$"):
                    continue  # Skip operators
                
                if key not in metadata:
                    return False
                
                meta_value = metadata[key]
                
                # Handle complex expressions
                if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                    # This is a complex expression with operators
                    for op, op_value in value.items():
                        if op == FilterOperator.EQ:
                            if meta_value != op_value:
                                return False
                        elif op == FilterOperator.NE:
                            if meta_value == op_value:
                                return False
                        elif op == FilterOperator.GT:
                            if not (isinstance(meta_value, (int, float)) and meta_value > op_value):
                                return False
                        elif op == FilterOperator.GTE:
                            if not (isinstance(meta_value, (int, float)) and meta_value >= op_value):
                                return False
                        elif op == FilterOperator.LT:
                            if not (isinstance(meta_value, (int, float)) and meta_value < op_value):
                                return False
                        elif op == FilterOperator.LTE:
                            if not (isinstance(meta_value, (int, float)) and meta_value <= op_value):
                                return False
                        elif op == FilterOperator.IN:
                            # Handle different cases for IN operator
                            if isinstance(meta_value, list):
                                # If metadata value is a list, check if there's any overlap
                                if not any(mv == op_value or mv in op_value for mv in meta_value):
                                    return False
                            else:
                                # If metadata value is not a list, check if it's in the operator value
                                if meta_value not in op_value:
                                    return False
                        elif op == FilterOperator.NIN:
                            if meta_value in op_value:
                                return False
                        elif op == FilterOperator.CONTAINS:
                            if not (isinstance(meta_value, str) and isinstance(op_value, str) and op_value in meta_value):
                                return False
                        elif op == FilterOperator.STARTSWITH:
                            if not (isinstance(meta_value, str) and isinstance(op_value, str) and meta_value.startswith(op_value)):
                                return False
                        elif op == FilterOperator.ENDSWITH:
                            if not (isinstance(meta_value, str) and isinstance(op_value, str) and meta_value.endswith(op_value)):
                                return False
                else:
                    # Simple equality check
                    if isinstance(value, list):
                        # If filter value is a list
                        if isinstance(meta_value, list):
                            # If both are lists, check for any common elements
                            if not any(mv in value for mv in meta_value):
                                return False
                        elif meta_value not in value:
                            # If metadata value is not a list, check if it's in the filter list
                            return False
                    elif isinstance(meta_value, list):
                        # If metadata value is a list but filter value is not,
                        # check if filter value is in the metadata list
                        if value not in meta_value:
                            return False
                    elif meta_value != value:
                        # Simple equality check for non-list values
                        return False
            
            return True
        
        return [doc for doc in docs if match_document(doc, filter_dict)]
    
    # Test simple equality filter
    category_filter = create_filter("category", "ai")
    ai_docs = apply_filter(sample_documents, category_filter)
    assert len(ai_docs) == 1
    assert ai_docs[0].doc_id == "doc1"
    
    # Test greater than or equal filter
    year_filter = create_filter("year", 2022, FilterOperator.GTE)
    recent_docs = apply_filter(sample_documents, year_filter)
    assert len(recent_docs) == 4
    assert {doc.doc_id for doc in recent_docs} == {"doc1", "doc2", "doc4", "doc5"}
    
    # Test complex AND filter
    complex_filter = create_filter(
        conditions=[
            create_filter("year", 2022, FilterOperator.GTE),
            create_filter("tags", "research", FilterOperator.IN)
        ],
        operator=FilterOperator.AND
    )
    
    research_recent_docs = apply_filter(sample_documents, complex_filter)
    assert len(research_recent_docs) == 2
    assert {doc.doc_id for doc in research_recent_docs} == {"doc1", "doc2"}
    
    # Test OR filter
    or_filter = create_filter(
        conditions=[
            create_filter("category", "ai"),
            create_filter("category", "nlp"),
        ],
        operator=FilterOperator.OR
    )
    
    ai_nlp_docs = apply_filter(sample_documents, or_filter)
    assert len(ai_nlp_docs) == 2
    assert {doc.doc_id for doc in ai_nlp_docs} == {"doc1", "doc5"}
    
    # Test NOT filter
    not_filter = create_filter(
        conditions=[create_filter("year", 2023)],
        operator=FilterOperator.NOT
    )
    
    older_docs = apply_filter(sample_documents, not_filter)
    assert len(older_docs) == 3
    assert {doc.doc_id for doc in older_docs} == {"doc2", "doc3", "doc5"} 