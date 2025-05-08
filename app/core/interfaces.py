"""
Core interfaces for the RAG system.
"""
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, Union, Protocol
from dataclasses import dataclass


class DocumentType(Enum):
    """Type of document."""
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MARKDOWN = "markdown"


class EventType(Enum):
    """Type of event."""
    DOCUMENT_ADDED = "document_added"
    DOCUMENT_PROCESSED = "document_processed"
    EMBEDDING_CREATED = "embedding_created"
    QUERY_EXECUTED = "query_executed"
    QUERY_RECEIVED = "query_received"
    GENERATION_COMPLETED = "generation_completed"
    LLM_GENERATED = "llm_generated"
    ERROR = "error"


@dataclass
class DocumentMetadata(Dict[str, Any]):
    """Document metadata."""
    pass


@dataclass
class RagDocument:
    """RAG document."""
    content: str
    metadata: DocumentMetadata = None
    doc_id: Optional[str] = None
    doc_type: Optional[DocumentType] = None


class DocumentLoader:
    """Interface for document loaders."""
    
    def load(self, source: str) -> List[RagDocument]:
        """Load documents from a source."""
        raise NotImplementedError("Subclasses must implement load")
    
    def supports(self, source: str) -> bool:
        """Check if the loader supports the given source."""
        raise NotImplementedError("Subclasses must implement supports")


class TextSplitter:
    """Interface for text splitters."""
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        raise NotImplementedError("Subclasses must implement split_text")
    
    def split_documents(self, documents: List[RagDocument]) -> List[RagDocument]:
        """Split documents into chunks."""
        raise NotImplementedError("Subclasses must implement split_documents")


class Event:
    """Event class for the event bus."""
    
    def __init__(self, event_type: EventType, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Initialize the event."""
        self.event_type = event_type
        self.payload = payload
        self.metadata = metadata or {}


class EventHandler(Protocol):
    """Protocol for event handlers."""
    
    def handle(self, event: Event) -> None:
        """Handle an event."""
        pass


class EventBus:
    """Simple event bus for the RAG system."""
    
    def __init__(self):
        """Initialize the event bus."""
        self._handlers = {}
    
    def register(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to an event type (alias for register)."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(callback)
    
    def publish(self, event: Event) -> None:
        """Publish an event."""
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                if hasattr(handler, 'handle'):
                    handler.handle(event)
                else:
                    handler(event)


# Global event bus
event_bus = EventBus() 