"""
Core interfaces for RAGStackXL components.

This module defines the abstract base classes and protocols that all
components in the system must implement. Following a clean architecture approach,
these interfaces define the contracts between components without implementation details.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union, TypeVar, Generic

# Custom types
Document = TypeVar('Document')
Embedding = TypeVar('Embedding')
Query = TypeVar('Query')
Response = TypeVar('Response')
AgentState = TypeVar('AgentState')
ToolResult = TypeVar('ToolResult')


class DocumentType(str, Enum):
    """Enumeration of document types."""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"
    UNKNOWN = "unknown"


class DocumentMetadata(Dict[str, Any]):
    """Type for document metadata."""
    pass


class RagDocument:
    """Base document class with content and metadata."""
    
    def __init__(
        self,
        content: str,
        metadata: Optional[DocumentMetadata] = None,
        doc_id: Optional[str] = None,
        doc_type: DocumentType = DocumentType.TEXT,
        embedding: Optional[List[float]] = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id
        self.doc_type = doc_type
        self.embedding = embedding
    
    def __str__(self) -> str:
        return f"Document(id={self.doc_id}, type={self.doc_type}, content_length={len(self.content)})"


class DocumentLoader(ABC):
    """Interface for document loaders."""
    
    @abstractmethod
    def load(self, source: str) -> List[RagDocument]:
        """Load documents from a source."""
        pass
    
    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if the loader supports the given source."""
        pass


class TextSplitter(ABC):
    """Interface for text splitters."""
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass
    
    @abstractmethod
    def split_documents(self, documents: List[RagDocument]) -> List[RagDocument]:
        """Split documents into chunks."""
        pass


class EmbeddingModel(ABC):
    """Interface for embedding models."""
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query."""
        pass
    
    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for documents."""
        pass


class VectorStore(ABC):
    """Interface for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[RagDocument]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RagDocument]:
        """Search for documents similar to the query."""
        pass
    
    @abstractmethod
    def search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RagDocument]:
        """Search for documents similar to the embedding vector."""
        pass


class LLMProvider(ABC):
    """Interface for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text based on a prompt."""
        pass
    
    @abstractmethod
    def generate_with_context(
        self,
        prompt: str,
        context: List[RagDocument],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text based on a prompt and context documents."""
        pass


class Tool(ABC):
    """Interface for agent tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the tool."""
        pass
    
    @abstractmethod
    def run(self, input_text: str) -> str:
        """Run the tool with the given input text."""
        pass


class Agent(ABC):
    """Interface for agents."""
    
    @abstractmethod
    def run(self, query: str) -> Response:
        """Run the agent with the given query."""
        pass
    
    @abstractmethod
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        pass


class EventType(str, Enum):
    """Enumeration of event types."""
    DOCUMENT_ADDED = "document_added"
    DOCUMENT_DELETED = "document_deleted"
    QUERY_RECEIVED = "query_received"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    GENERATION_COMPLETED = "generation_completed"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT_RECEIVED = "tool_result_received"
    ERROR = "error"


class Event:
    """Base event class."""
    
    def __init__(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_type = event_type
        self.payload = payload
        self.metadata = metadata or {}


class EventHandler(Protocol):
    """Protocol for event handlers."""
    
    def handle(self, event: Event) -> None:
        """Handle an event."""
        pass


class EventBus:
    """Simple event bus implementation."""
    
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
    
    def register(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def publish(self, event: Event) -> None:
        """Publish an event to all registered handlers."""
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                handler.handle(event)


# Create a global event bus instance
event_bus = EventBus() 