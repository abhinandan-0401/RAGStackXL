"""
Language Model module for RAGStackXL.

This module provides a unified interface to various LLM providers.
"""

# Import public interfaces
from app.llm.interfaces import (
    LLM,
    LLMConfig,
    LLMResponse,
    LLMProvider,
    LLMFactory,
    ChatMessage,
    ChatRole
)

# Import specific implementation modules
from app.llm.openai import OpenAILLM
from app.llm.anthropic import AnthropicLLM
from app.llm.huggingface import HuggingFaceLLM
from app.llm.test import TestLLM

# Try importing optional providers
try:
    from app.llm.cohere import CohereLLM
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

try:
    from app.llm.ollama import OllamaLLM
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# Register providers with the factory
LLMFactory.register(LLMProvider.OPENAI, OpenAILLM)
LLMFactory.register(LLMProvider.ANTHROPIC, AnthropicLLM)
LLMFactory.register(LLMProvider.HUGGINGFACE, HuggingFaceLLM)
LLMFactory.register(LLMProvider.TEST, TestLLM)

if HAS_COHERE:
    LLMFactory.register(LLMProvider.COHERE, CohereLLM)

if HAS_OLLAMA:
    LLMFactory.register(LLMProvider.OLLAMA, OllamaLLM)

# Add public exports
__all__ = [
    "LLM",
    "LLMConfig",
    "LLMResponse",
    "LLMProvider",
    "LLMFactory",
    "ChatMessage",
    "ChatRole",
    "OpenAILLM",
    "AnthropicLLM",
    "HuggingFaceLLM",
    "TestLLM"
] 