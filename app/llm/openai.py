"""
OpenAI LLM implementation for RAGStackXL.

This module provides an implementation of the LLM interface for OpenAI models,
supporting both text completion and chat completion APIs.
"""

import os
import json
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
import asyncio

try:
    import openai
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.completion import Completion, CompletionChoice
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from app.llm.interfaces import LLM, LLMConfig, LLMResponse, ChatMessage, LLMProvider
from app.utils.logging import log


class OpenAILLM(LLM):
    """
    OpenAI LLM implementation.
    
    Supports:
    - GPT-3.5-turbo
    - GPT-4
    - GPT-4-turbo
    - Other OpenAI-compatible models
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the OpenAI LLM.
        
        Args:
            config: LLM configuration
        """
        super().__init__(config)
        
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI package is not installed. "
                "Please install it with 'pip install openai'"
            )
        
        # Get API key from config or environment
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Please provide it in the config or set OPENAI_API_KEY environment variable."
            )
        
        # Get API base URL from config or use default
        api_base = config.api_base or os.environ.get("OPENAI_API_BASE")
        
        # Initialize client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        # Store common parameters
        self.model = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.presence_penalty = config.presence_penalty
        self.frequency_penalty = config.frequency_penalty
        self.timeout = config.timeout
    
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
        try:
            # Prepare parameters
            params = {
                "model": kwargs.get("model", self.model),
                "prompt": prompt,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
                "timeout": kwargs.get("timeout", self.timeout)
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Call API
            response = await self.client.completions.create(**params)
            
            # Extract content
            content = response.choices[0].text.strip()
            
            # Extract usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Publish event
            self._publish_generation_event(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"]
            )
            
            return LLMResponse(
                content=content,
                model=params["model"],
                usage=usage,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            log.error(f"Error generating text with OpenAI: {e}")
            raise
    
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
        try:
            # Prepare parameters
            params = {
                "model": kwargs.get("model", self.model),
                "prompt": prompt,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
                "timeout": kwargs.get("timeout", self.timeout),
                "stream": True
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Call API with streaming
            stream = await self.client.completions.create(**params)
            
            # Process stream
            async for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].text or ""
                    if content:
                        yield content
            
        except Exception as e:
            log.error(f"Error streaming text with OpenAI: {e}")
            raise
    
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
        try:
            # Convert messages to OpenAI format
            openai_messages = [message.to_dict() for message in messages]
            
            # Prepare parameters
            params = {
                "model": kwargs.get("model", self.model),
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
                "timeout": kwargs.get("timeout", self.timeout)
            }
            
            # Handle function calling if provided
            if "functions" in kwargs:
                params["functions"] = kwargs["functions"]
            
            if "function_call" in kwargs:
                params["function_call"] = kwargs["function_call"]
            
            # Handle tools if provided
            if "tools" in kwargs:
                params["tools"] = kwargs["tools"]
            
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Call API
            response = await self.client.chat.completions.create(**params)
            
            # Extract content
            content = response.choices[0].message.content or ""
            
            # Extract usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Extract function call if any
            function_call = None
            if hasattr(response.choices[0].message, "function_call") and response.choices[0].message.function_call:
                function_call = {
                    "name": response.choices[0].message.function_call.name,
                    "arguments": response.choices[0].message.function_call.arguments
                }
            
            # Extract tool calls if any
            tool_calls = None
            if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                tool_calls = [
                    {
                        "id": tool.id,
                        "type": tool.type,
                        "function": {
                            "name": tool.function.name,
                            "arguments": tool.function.arguments
                        }
                    }
                    for tool in response.choices[0].message.tool_calls
                ]
            
            # Publish event
            self._publish_generation_event(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"]
            )
            
            # Create metadata
            metadata = {}
            if function_call:
                metadata["function_call"] = function_call
            if tool_calls:
                metadata["tool_calls"] = tool_calls
            
            return LLMResponse(
                content=content,
                model=params["model"],
                usage=usage,
                finish_reason=response.choices[0].finish_reason,
                metadata=metadata
            )
            
        except Exception as e:
            log.error(f"Error generating chat response with OpenAI: {e}")
            raise
    
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
        try:
            # Convert messages to OpenAI format
            openai_messages = [message.to_dict() for message in messages]
            
            # Prepare parameters
            params = {
                "model": kwargs.get("model", self.model),
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
                "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
                "timeout": kwargs.get("timeout", self.timeout),
                "stream": True
            }
            
            # Handle function calling if provided
            if "functions" in kwargs:
                params["functions"] = kwargs["functions"]
            
            if "function_call" in kwargs:
                params["function_call"] = kwargs["function_call"]
            
            # Handle tools if provided
            if "tools" in kwargs:
                params["tools"] = kwargs["tools"]
            
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Call API with streaming
            stream = await self.client.chat.completions.create(**params)
            
            # Process stream
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    if content:
                        yield content
            
        except Exception as e:
            log.error(f"Error streaming chat response with OpenAI: {e}")
            raise 