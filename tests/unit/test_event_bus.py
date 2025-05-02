"""
Unit tests for the event bus system.
"""
import pytest
from unittest.mock import Mock

from app.core.interfaces import Event, EventBus, EventType, EventHandler


class TestEventBus:
    """Test suite for the EventBus class."""
    
    def test_event_creation(self):
        """Test that events can be created properly."""
        payload = {"key": "value"}
        metadata = {"meta": "data"}
        
        event = Event(
            event_type=EventType.QUERY_RECEIVED,
            payload=payload,
            metadata=metadata
        )
        
        assert event.event_type == EventType.QUERY_RECEIVED
        assert event.payload == payload
        assert event.metadata == metadata
    
    def test_event_handler_registration(self):
        """Test that handlers can be registered for event types."""
        event_bus = EventBus()
        handler = Mock(spec=EventHandler)
        
        # Initially, no handlers should be registered
        assert not hasattr(event_bus, "_handlers") or not event_bus._handlers.get(EventType.QUERY_RECEIVED)
        
        # Register a handler
        event_bus.register(EventType.QUERY_RECEIVED, handler)
        
        # Check that the handler was registered
        assert EventType.QUERY_RECEIVED in event_bus._handlers
        assert handler in event_bus._handlers[EventType.QUERY_RECEIVED]
    
    def test_event_publishing(self):
        """Test that events are published to registered handlers."""
        event_bus = EventBus()
        handler1 = Mock(spec=EventHandler)
        handler2 = Mock(spec=EventHandler)
        
        # Register handlers
        event_bus.register(EventType.QUERY_RECEIVED, handler1)
        event_bus.register(EventType.QUERY_RECEIVED, handler2)
        event_bus.register(EventType.GENERATION_COMPLETED, handler1)
        
        # Create an event
        event = Event(
            event_type=EventType.QUERY_RECEIVED,
            payload={"query": "What is RAG?"}
        )
        
        # Publish the event
        event_bus.publish(event)
        
        # Check that both handlers were called with the event
        handler1.handle.assert_called_once_with(event)
        handler2.handle.assert_called_once_with(event)
        
        # Reset the mocks
        handler1.reset_mock()
        handler2.reset_mock()
        
        # Create a different event
        event2 = Event(
            event_type=EventType.GENERATION_COMPLETED,
            payload={"response": "RAG is Retrieval-Augmented Generation"}
        )
        
        # Publish the second event
        event_bus.publish(event2)
        
        # Check that only handler1 was called
        handler1.handle.assert_called_once_with(event2)
        handler2.handle.assert_not_called()
    
    def test_multiple_handler_registration(self):
        """Test that multiple handlers can be registered for the same event type."""
        event_bus = EventBus()
        handler1 = Mock(spec=EventHandler)
        handler2 = Mock(spec=EventHandler)
        handler3 = Mock(spec=EventHandler)
        
        # Register multiple handlers for the same event type
        event_bus.register(EventType.QUERY_RECEIVED, handler1)
        event_bus.register(EventType.QUERY_RECEIVED, handler2)
        event_bus.register(EventType.QUERY_RECEIVED, handler3)
        
        # Check that all handlers were registered
        assert len(event_bus._handlers[EventType.QUERY_RECEIVED]) == 3
        
        # Create and publish an event
        event = Event(
            event_type=EventType.QUERY_RECEIVED,
            payload={"query": "Test"}
        )
        event_bus.publish(event)
        
        # Check that all handlers were called
        handler1.handle.assert_called_once_with(event)
        handler2.handle.assert_called_once_with(event)
        handler3.handle.assert_called_once_with(event)
    
    def test_event_with_no_handlers(self):
        """Test publishing an event with no registered handlers."""
        event_bus = EventBus()
        event = Event(
            event_type=EventType.ERROR,
            payload={"error": "Test error"}
        )
        
        # This should not raise an exception
        event_bus.publish(event) 