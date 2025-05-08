"""
Unit tests for the LLM interfaces.
"""

import unittest
import pytest
from unittest.mock import MagicMock, patch
import asyncio
from typing import List, Dict, Any

from app.llm.interfaces import (
    LLMProvider,
    ChatRole,
    ChatMessage,
    LLMConfig,
    LLMResponse,
    LLM,
    LLMFactory
)
from app.core.interfaces import EventType


class TestLLMInterfaces(unittest.TestCase):
    """Test cases for LLM interfaces."""
    
    def test_llm_provider_enum(self):
        """Test the LLMProvider enum."""
        self.assertEqual(LLMProvider.OPENAI.value, "openai")
        self.assertEqual(LLMProvider.ANTHROPIC.value, "anthropic")
        self.assertEqual(LLMProvider.HUGGINGFACE.value, "huggingface")
        self.assertEqual(LLMProvider.COHERE.value, "cohere")
        self.assertEqual(LLMProvider.OLLAMA.value, "ollama")
        self.assertEqual(LLMProvider.CUSTOM.value, "custom")
        self.assertEqual(LLMProvider.TEST.value, "test")
        
        # Test string conversion
        self.assertEqual(LLMProvider("openai"), LLMProvider.OPENAI)
        self.assertEqual(LLMProvider("test"), LLMProvider.TEST)
    
    def test_chat_role_enum(self):
        """Test the ChatRole enum."""
        self.assertEqual(ChatRole.SYSTEM.value, "system")
        self.assertEqual(ChatRole.USER.value, "user")
        self.assertEqual(ChatRole.ASSISTANT.value, "assistant")
        self.assertEqual(ChatRole.FUNCTION.value, "function")
        
        # Test string conversion
        self.assertEqual(ChatRole("system"), ChatRole.SYSTEM)
    
    def test_chat_message(self):
        """Test the ChatMessage class."""
        # Test basic construction
        msg = ChatMessage(role=ChatRole.USER, content="Hello")
        self.assertEqual(msg.role, ChatRole.USER)
        self.assertEqual(msg.content, "Hello")
        self.assertIsNone(msg.name)
        self.assertIsNone(msg.function_call)
        self.assertIsNone(msg.tool_calls)
        
        # Test string role
        msg = ChatMessage(role="system", content="System message")
        self.assertEqual(msg.role, ChatRole.SYSTEM)
        
        # Test to_dict method
        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["role"], "system")
        self.assertEqual(msg_dict["content"], "System message")
        
        # Test with function call
        func_call = {"name": "test_func", "arguments": "{\"arg1\": \"value1\"}"}
        msg = ChatMessage(
            role=ChatRole.ASSISTANT, 
            content="",
            function_call=func_call
        )
        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["function_call"], func_call)
        
        # Test from_dict method
        orig_msg = {
            "role": "user",
            "content": "Test message",
            "name": "test_user"
        }
        msg = ChatMessage.from_dict(orig_msg)
        self.assertEqual(msg.role, ChatRole.USER)
        self.assertEqual(msg.content, "Test message")
        self.assertEqual(msg.name, "test_user")
        
        # Test class factory methods
        sys_msg = ChatMessage.system("System instruction")
        self.assertEqual(sys_msg.role, ChatRole.SYSTEM)
        self.assertEqual(sys_msg.content, "System instruction")
        
        user_msg = ChatMessage.user("User message")
        self.assertEqual(user_msg.role, ChatRole.USER)
        self.assertEqual(user_msg.content, "User message")
        
        asst_msg = ChatMessage.assistant("Assistant response")
        self.assertEqual(asst_msg.role, ChatRole.ASSISTANT)
        self.assertEqual(asst_msg.content, "Assistant response")
    
    def test_llm_config(self):
        """Test the LLMConfig class."""
        # Test basic construction
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=100
        )
        self.assertEqual(config.provider, LLMProvider.OPENAI)
        self.assertEqual(config.model_name, "gpt-4")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 100)
        self.assertEqual(config.top_p, 1.0)  # Default value
        
        # Test string provider
        config = LLMConfig(provider="anthropic", model_name="claude-3-opus-20240229")
        self.assertEqual(config.provider, LLMProvider.ANTHROPIC)
        
        # Test additional config
        config = LLMConfig(
            provider="openai",
            model_name="gpt-4",
            custom_param="custom_value"
        )
        self.assertEqual(config.get("custom_param"), "custom_value")
        self.assertIsNone(config.get("non_existent_param"))
        self.assertEqual(config.get("non_existent_param", "default"), "default")
    
    def test_llm_response(self):
        """Test the LLMResponse class."""
        # Test basic construction
        response = LLMResponse(
            content="This is a response",
            model="gpt-4"
        )
        self.assertEqual(response.content, "This is a response")
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.usage, {})
        self.assertIsNone(response.finish_reason)
        self.assertEqual(response.metadata, {})
        
        # Test with usage stats
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        response = LLMResponse(
            content="Response with usage",
            model="gpt-4",
            usage=usage,
            finish_reason="stop"
        )
        self.assertEqual(response.usage, usage)
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 20)
        self.assertEqual(response.total_tokens, 30)
        self.assertEqual(response.finish_reason, "stop")
    
    @patch("app.llm.interfaces.event_bus")
    def test_llm_publish_event(self, mock_event_bus):
        """Test the LLM _publish_generation_event method."""
        # Create a mock LLM
        class MockLLM(LLM):
            async def generate(self, prompt, **kwargs):
                pass
            
            async def generate_stream(self, prompt, **kwargs):
                yield "test"
                
            async def chat(self, messages, **kwargs):
                pass
                
            async def chat_stream(self, messages, **kwargs):
                yield "test"
        
        # Initialize with a config
        config = LLMConfig(provider="openai", model_name="gpt-4")
        llm = MockLLM(config)
        
        # Call the method
        llm._publish_generation_event(10, 20, 30)
        
        # Check that event_bus.publish was called correctly
        mock_event_bus.publish.assert_called_once()
        event_arg = mock_event_bus.publish.call_args[0][0]
        self.assertEqual(event_arg.event_type, EventType.LLM_GENERATED)
        payload = event_arg.payload
        self.assertEqual(payload["model"], "gpt-4")
        self.assertEqual(payload["provider"], "openai")
        self.assertEqual(payload["prompt_tokens"], 10)
        self.assertEqual(payload["completion_tokens"], 20)
        self.assertEqual(payload["total_tokens"], 30)


@pytest.mark.asyncio
class TestLLMFactory:
    """Test cases for LLMFactory."""
    
    def test_register_provider(self):
        """Test registering a provider with the factory."""
        # Create a mock LLM class
        mock_llm_class = MagicMock()
        
        # Register the provider
        LLMFactory.register(LLMProvider.CUSTOM, mock_llm_class)
        
        # Check if it was registered
        assert LLMProvider.CUSTOM in LLMFactory._registry
        assert LLMFactory._registry[LLMProvider.CUSTOM] == mock_llm_class
    
    def test_get_providers(self):
        """Test getting registered providers."""
        # Register a test provider
        mock_llm_class = MagicMock()
        LLMFactory.register(LLMProvider.CUSTOM, mock_llm_class)
        
        # Get providers
        providers = LLMFactory.get_providers()
        
        # Check if our provider is in the list
        assert LLMProvider.CUSTOM in providers
    
    def test_create_provider(self):
        """Test creating an LLM instance from the factory."""
        # Create a mock LLM class
        mock_llm_class = MagicMock()
        
        # Register the provider
        LLMFactory.register(LLMProvider.CUSTOM, mock_llm_class)
        
        # Create a config
        config = LLMConfig(provider=LLMProvider.CUSTOM, model_name="test-model")
        
        # Create an instance
        LLMFactory.create(config)
        
        # Check if the class was initialized with the config
        mock_llm_class.assert_called_once_with(config)
    
    def test_create_invalid_provider(self):
        """Test creating with an invalid provider."""
        # Create a valid config but with a provider that's not registered in the factory
        # First clear registry for TEST provider if it exists
        if LLMProvider.TEST in LLMFactory._registry:
            del LLMFactory._registry[LLMProvider.TEST]
        
        # Create a config with the TEST provider which is valid in enum but not registered
        config = LLMConfig(provider=LLMProvider.TEST, model_name="test-model")
        
        # Check that creating raises an error
        with pytest.raises(ValueError):
            LLMFactory.create(config) 