"""
Tests for the document processor.
"""
import os
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from typing import List

from app.core.interfaces import RagDocument, DocumentType
from app.document_processing.processor import DocumentProcessor
from app.document_processing.loaders import BaseDocumentLoader
from app.document_processing.chunking import BaseTextSplitter


class MockDocumentLoader(BaseDocumentLoader):
    """Mock document loader for testing."""
    
    def __init__(self, supported_extensions=None, mock_documents=None):
        """Initialize with mock documents to return."""
        super().__init__()
        self.supported_extensions = supported_extensions or {"txt", "md"}
        self.mock_documents = mock_documents or []
        self.loaded_sources = []
    
    def load(self, source: str) -> List[RagDocument]:
        """Mock loading documents."""
        self.loaded_sources.append(source)
        return self.mock_documents
    
    def supports(self, source: str) -> bool:
        """Check if the source is supported."""
        path = Path(source)
        return path.suffix.lower()[1:] in self.supported_extensions


class MockTextSplitter(BaseTextSplitter):
    """Mock text splitter for testing."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Initialize with chunk size and overlap."""
        super().__init__(chunk_size, chunk_overlap)
        self.split_calls = []
    
    def split_text(self, text: str) -> List[str]:
        """Mock splitting text."""
        self.split_calls.append(text)
        # Just return the text as a single chunk for simplicity
        return [text]
    
    def split_documents(self, documents: List[RagDocument]) -> List[RagDocument]:
        """Mock splitting documents."""
        # Just return the documents as is for simplicity
        return documents


class MockLoaderRegistry:
    """Mock document loader registry."""
    
    def __init__(self, loader):
        """Initialize with a mock loader."""
        self.loader = loader
    
    def get_for_source(self, source: str):
        """Return the mock loader if it supports the source."""
        if self.loader.supports(source):
            return self.loader
        return None


class MockSplitterRegistry:
    """Mock text splitter registry."""
    
    def __init__(self, splitters):
        """Initialize with mock splitters."""
        self.splitters = splitters
    
    def get(self, name: str):
        """Return the mock splitter with the given name."""
        return self.splitters.get(name)


class TestDocumentProcessor:
    """Test suite for DocumentProcessor."""
    
    @pytest.fixture
    def mock_loader(self):
        """Create a mock document loader."""
        return MockDocumentLoader()
    
    @pytest.fixture
    def mock_splitter(self):
        """Create a mock text splitter."""
        return MockTextSplitter()
    
    @pytest.fixture
    def mock_loader_registry(self, mock_loader):
        """Create a mock loader registry."""
        return MockLoaderRegistry(mock_loader)
    
    @pytest.fixture
    def mock_splitter_registry(self, mock_splitter):
        """Create a mock splitter registry."""
        return MockSplitterRegistry({"MockSplitter": mock_splitter})
    
    @pytest.fixture
    def processor(self, mock_loader_registry, mock_splitter_registry):
        """Create a document processor with mock components."""
        return DocumentProcessor(
            loader_registry=mock_loader_registry,
            splitter_registry=mock_splitter_registry,
            default_splitter="MockSplitter"
        )
    
    def test_processor_initialization(self, processor, mock_loader_registry, mock_splitter_registry):
        """Test that the processor initializes correctly."""
        assert processor.loader_registry == mock_loader_registry
        assert processor.splitter_registry == mock_splitter_registry
        assert processor.default_splitter == "MockSplitter"
    
    def test_process_file(self, processor, mock_loader, mock_splitter, tmp_path):
        """Test processing a single file."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")
        
        # Create a test document to return from the mock loader
        test_doc = RagDocument(
            content="This is a test document.",
            doc_id="test1",
            doc_type=DocumentType.TEXT,
            metadata={"source": str(test_file)}
        )
        mock_loader.mock_documents = [test_doc]
        
        # Process the document
        with patch("app.document_processing.processor.event_bus"):  # Mock the event bus
            result = processor._process_file(str(test_file), mock_splitter, None)
        
        # Check that the loader was called with the correct source
        assert str(test_file) in mock_loader.loaded_sources
        
        # Check that the result contains the test document
        assert len(result) == 1
        assert result[0].content == "This is a test document."
        assert result[0].doc_type == DocumentType.TEXT
    
    def test_process_document(self, processor, mock_loader, mock_splitter, tmp_path):
        """Test processing a document with the main method."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")
        
        # Create a test document to return from the mock loader
        test_doc = RagDocument(
            content="This is a test document.",
            doc_id="test1",
            doc_type=DocumentType.TEXT,
            metadata={"source": str(test_file)}
        )
        mock_loader.mock_documents = [test_doc]
        
        # Process the document
        with patch("app.document_processing.processor.event_bus"):  # Mock the event bus
            result = processor.process_document(
                source=str(test_file),
                splitter_name="MockSplitter"
            )
        
        # Check that the result contains the test document
        assert len(result) == 1
        assert result[0].content == "This is a test document."
        assert result[0].doc_type == DocumentType.TEXT
    
    def test_process_directory(self, processor, mock_loader, mock_splitter, tmp_path):
        """Test processing a directory."""
        # Create test files
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        test_file1 = test_dir / "test1.txt"
        test_file1.write_text("This is test document 1.")
        
        test_file2 = test_dir / "test2.txt"
        test_file2.write_text("This is test document 2.")
        
        # Create test documents to return from the mock loader
        test_doc1 = RagDocument(
            content="This is test document 1.",
            doc_id="test1",
            doc_type=DocumentType.TEXT,
            metadata={"source": str(test_file1)}
        )
        
        test_doc2 = RagDocument(
            content="This is test document 2.",
            doc_id="test2",
            doc_type=DocumentType.TEXT,
            metadata={"source": str(test_file2)}
        )
        
        # Set the mock documents to return
        mock_loader.mock_documents = [test_doc1, test_doc2]
        
        # Process the directory
        with patch("app.document_processing.processor.event_bus"):  # Mock the event bus
            result = processor.process_document(
                source=str(test_dir),
                splitter_name="MockSplitter",
                recursive=True
            )
        
        # Check that both files were processed
        assert len(result) == 2 * len(mock_loader.mock_documents)  # 2 files * 2 docs per file
    
    def test_filter_by_metadata(self, processor):
        """Test filtering documents by metadata."""
        # Create test documents
        docs = [
            RagDocument(
                content="Document 1",
                metadata={"category": "A", "language": "en"}
            ),
            RagDocument(
                content="Document 2",
                metadata={"category": "B", "language": "en"}
            ),
            RagDocument(
                content="Document 3",
                metadata={"category": "A", "language": "fr"}
            )
        ]
        
        # Filter by category A
        result = processor._filter_by_metadata(docs, {"category": "A"})
        assert len(result) == 2
        assert result[0].content == "Document 1"
        assert result[1].content == "Document 3"
        
        # Filter by category A and language en
        result = processor._filter_by_metadata(docs, {"category": "A", "language": "en"})
        assert len(result) == 1
        assert result[0].content == "Document 1"
        
        # Filter by non-existent category
        result = processor._filter_by_metadata(docs, {"category": "C"})
        assert len(result) == 0 