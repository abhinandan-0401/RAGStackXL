"""
LLM interfaces for RAGStackXL.

This module defines the interfaces and abstractions for language models.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
import asyncio

from app.core.interfaces import EventType, event_bus, Event
from app.utils.logging import log


class LLMProvider(str, Enum):
    """Types of LLM providers supported by the system."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    OLLAMA = "ollama"
    CUSTOM = "custom"
    TEST = "test"  # For testing purposes only


class ChatRole(str, Enum):
    """Roles in a chat conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ChatMessage:
    """Representation of a message in a chat conversation."""
    
    def __init__(
        self,
        role: Union[ChatRole, str],
        content: str,
        name: Optional[str] = None,
        function_call: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize a chat message.
        
        Args:
            role: Role of the message sender
            content: Content of the message
            name: Optional name (for function calls)
            function_call: Optional function call details
            tool_calls: Optional tool call details
        """
        self.role = role if isinstance(role, ChatRole) else ChatRole(role)
        self.content = content
        self.name = name
        self.function_call = function_call
        self.tool_calls = tool_calls
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary format."""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        
        if self.name:
            result["name"] = self.name
            
        if self.function_call:
            result["function_call"] = self.function_call
            
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create a ChatMessage from a dictionary."""
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            function_call=data.get("function_call"),
            tool_calls=data.get("tool_calls")
        )
    
    @classmethod
    def system(cls, content: str) -> 'ChatMessage':
        """Create a system message."""
        return cls(role=ChatRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> 'ChatMessage':
        """Create a user message."""
        return cls(role=ChatRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> 'ChatMessage':
        """Create an assistant message."""
        return cls(role=ChatRole.ASSISTANT, content=content)


class LLMConfig:
    """Configuration for language models."""
    
    def __init__(
        self,
        provider: Union[LLMProvider, str],
        model_name: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM configuration.
        
        Args:
            provider: LLM provider
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            presence_penalty: Penalty for token presence
            frequency_penalty: Penalty for token frequency
            timeout: Request timeout in seconds
            api_key: Optional API key
            api_base: Optional API base URL
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider if isinstance(provider, LLMProvider) else LLMProvider(provider)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.timeout = timeout
        self.api_key = api_key
        self.api_base = api_base
        self.additional_config = kwargs
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_config.get(key, default)


class LLMResponse:
    """
    Response from a language model.
    
    This class wraps the response from a language model and provides
    access to the generated content and metadata.
    """
    
    def __init__(
        self,
        content: str,
        model: str,
        usage: Optional[Dict[str, int]] = None,
        finish_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an LLM response.
        
        Args:
            content: Generated text content
            model: Name of the model that generated the content
            usage: Token usage statistics
            finish_reason: Reason the generation finished
            metadata: Additional metadata
        """
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.finish_reason = finish_reason
        self.metadata = metadata or {}
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used for this response."""
        return self.usage.get("total_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        """Tokens used for the completion."""
        return self.usage.get("completion_tokens", 0)
    
    @property
    def prompt_tokens(self) -> int:
        """Tokens used for the prompt."""
        return self.usage.get("prompt_tokens", 0)


class LLM(ABC):
    """Abstract base class for language models."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the language model.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        log.info(f"Initializing {self.__class__.__name__} with model: {config.model_name}")
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming text based on a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Text chunks as they are generated
        """
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response in a chat context.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        pass
    
    @abstractmethod
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response in a chat context.
        
        Args:
            messages: List of chat messages
            **kwargs: Additional generation parameters
            
        Yields:
            Text chunks as they are generated
        """
        pass
    
    def _publish_generation_event(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int
    ) -> None:
        """
        Publish event for LLM generation.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total number of tokens
        """
        event_bus.publish(
            Event(
                event_type=EventType.LLM_GENERATED,
                payload={
                    "model": self.config.model_name,
                    "provider": self.config.provider.value,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )
        )


class LLMFactory:
    """Factory for creating LLM instances."""
    
    _registry = {}
    
    @classmethod
    def register(cls, provider: LLMProvider, llm_class):
        """
        Register an LLM class for a provider.
        
        Args:
            provider: LLM provider
            llm_class: Class implementing the LLM
        """
        cls._registry[provider] = llm_class
    
    @classmethod
    def create(cls, config: LLMConfig) -> LLM:
        """
        Create an LLM instance.
        
        Args:
            config: LLM configuration
            
        Returns:
            LLM instance
        """
        provider = config.provider
        if provider not in cls._registry:
            raise ValueError(f"LLM provider '{provider.value}' is not registered")
            
        llm_class = cls._registry[provider]
        return llm_class(config)
    
    @classmethod
    def get_providers(cls) -> List[LLMProvider]:
        """
        Get list of registered LLM providers.
        
        Returns:
            List of registered LLM providers
        """
        return list(cls._registry.keys()) 