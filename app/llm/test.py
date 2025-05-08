"""
Test LLM implementation for RAGStackXL.

This module provides a mock implementation of the LLM interface for testing purposes.
"""

from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio

from app.llm.interfaces import LLM, LLMConfig, LLMResponse, ChatMessage, LLMProvider
from app.utils.logging import log


class TestLLM(LLM):
    """
    Test LLM implementation for unit testing.
    
    This is a mock implementation that returns predefined responses
    for testing purposes only.
    """
    
    def __init__(self, config):
        """Initialize with a config."""
        super().__init__(config)
        self.responses = []
        self.generate_calls = []
        self.chat_calls = []
    
    async def generate(self, prompt, **kwargs):
        """Mock generate method."""
        self.generate_calls.append((prompt, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content="Test response",
            model=self.config.model_name,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
    
    async def generate_stream(self, prompt, **kwargs):
        """Mock generate_stream method."""
        yield "Test"
        yield " streamed"
        yield " response"
    
    async def chat(self, messages, **kwargs):
        """Mock chat method."""
        self.chat_calls.append((messages, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content="Test chat response",
            model=self.config.model_name,
            usage={"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
        )
    
    async def chat_stream(self, messages, **kwargs):
        """Mock chat_stream method."""
        yield "Test"
        yield " streamed"
        yield " chat"
        yield " response" 