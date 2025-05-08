"""
Anthropic LLM implementation for RAGStackXL.

This module provides an implementation of the LLM interface for Anthropic Claude models.
"""

import os
import json
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
import asyncio

try:
    import anthropic
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from app.llm.interfaces import LLM, LLMConfig, LLMResponse, ChatMessage, ChatRole, LLMProvider
from app.utils.logging import log


class AnthropicLLM(LLM):
    """
    Anthropic Claude LLM implementation.
    
    Supports:
    - Claude 3 Opus
    - Claude 3 Sonnet
    - Claude 3 Haiku
    - Claude 2.1
    - Claude 2.0
    - Claude Instant
    """
    
    # Mapping from our ChatRole to Anthropic roles
    ROLE_MAPPING = {
        ChatRole.SYSTEM: "system",
        ChatRole.USER: "user",
        ChatRole.ASSISTANT: "assistant",
        ChatRole.FUNCTION: "user"  # Anthropic doesn't support function role, convert to user
    }
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the Anthropic LLM.
        
        Args:
            config: LLM configuration
        """
        super().__init__(config)
        
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic package is not installed. "
                "Please install it with 'pip install anthropic'"
            )
        
        # Get API key from config or environment
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Please provide it in the config or set ANTHROPIC_API_KEY environment variable."
            )
        
        # Get API base URL from config or use default
        api_base = config.api_base or os.environ.get("ANTHROPIC_API_BASE")
        
        # Initialize client
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
            
        self.client = AsyncAnthropic(**client_kwargs)
        
        # Store common parameters
        self.model = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens or 4096  # Anthropic requires max_tokens
        self.top_p = config.top_p
        self.timeout = config.timeout
        
        # Specific to Anthropic
        self.system_prompt = config.get("system_prompt", "")
    
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
        # For Anthropic, we use chat API for everything
        # We'll create a simple user message from the prompt
        messages = [ChatMessage.user(prompt)]
        
        # Add system message if provided
        system_prompt = kwargs.get("system_prompt", self.system_prompt)
        if system_prompt:
            messages.insert(0, ChatMessage.system(system_prompt))
        
        return await self.chat(messages, **kwargs)
    
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
        # For Anthropic, we use chat API for everything
        # We'll create a simple user message from the prompt
        messages = [ChatMessage.user(prompt)]
        
        # Add system message if provided
        system_prompt = kwargs.get("system_prompt", self.system_prompt)
        if system_prompt:
            messages.insert(0, ChatMessage.system(system_prompt))
        
        async for chunk in self.chat_stream(messages, **kwargs):
            yield chunk
    
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
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages(messages)
            
            # Get system prompt if any
            system_prompt = kwargs.get("system_prompt", self.system_prompt)
            if not system_prompt and messages and messages[0].role == ChatRole.SYSTEM:
                system_prompt = messages[0].content
                anthropic_messages = anthropic_messages[1:]  # Remove system message from messages
            
            # Prepare parameters
            params = {
                "model": kwargs.get("model", self.model),
                "messages": anthropic_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
            }
            
            # Add system prompt if available
            if system_prompt:
                params["system"] = system_prompt
            
            # Add tools if available (Claude 3 supports tools)
            if "tools" in kwargs:
                params["tools"] = kwargs["tools"]
            
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Call API
            response = await self.client.messages.create(**params)
            
            # Extract content
            content = response.content[0].text
            
            # Extract usage
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            # Extract tool calls if any
            tool_calls = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
            
            # Publish event
            self._publish_generation_event(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"]
            )
            
            # Create metadata
            metadata = {}
            if tool_calls:
                metadata["tool_calls"] = tool_calls
            
            return LLMResponse(
                content=content,
                model=params["model"],
                usage=usage,
                finish_reason=response.stop_reason,
                metadata=metadata
            )
            
        except Exception as e:
            log.error(f"Error generating chat response with Anthropic: {e}")
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
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages(messages)
            
            # Get system prompt if any
            system_prompt = kwargs.get("system_prompt", self.system_prompt)
            if not system_prompt and messages and messages[0].role == ChatRole.SYSTEM:
                system_prompt = messages[0].content
                anthropic_messages = anthropic_messages[1:]  # Remove system message from messages
            
            # Prepare parameters
            params = {
                "model": kwargs.get("model", self.model),
                "messages": anthropic_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "top_p": kwargs.get("top_p", self.top_p),
                "stream": True
            }
            
            # Add system prompt if available
            if system_prompt:
                params["system"] = system_prompt
            
            # Add tools if available (Claude 3 supports tools)
            if "tools" in kwargs:
                params["tools"] = kwargs["tools"]
            
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Call API with streaming
            stream = await self.client.messages.create(**params)
            
            # Process stream
            async for chunk in stream:
                if hasattr(chunk, "delta") and chunk.delta.text:
                    yield chunk.delta.text
                elif (
                    hasattr(chunk, "content") and 
                    chunk.content and 
                    hasattr(chunk.content[0], "text") and
                    chunk.content[0].text
                ):
                    yield chunk.content[0].text
            
        except Exception as e:
            log.error(f"Error streaming chat response with Anthropic: {e}")
            raise
    
    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert our message format to Anthropic's format.
        
        Args:
            messages: List of chat messages
            
        Returns:
            List of Anthropic-formatted messages
        """
        result = []
        
        for message in messages:
            # Skip system messages as they are handled separately
            if message.role == ChatRole.SYSTEM:
                continue
                
            # Convert role
            role = self.ROLE_MAPPING.get(message.role, "user")
            
            # Create message
            anthropic_message = {
                "role": role,
                "content": message.content
            }
            
            # Handle function/tool call content for user role
            if message.role == ChatRole.FUNCTION:
                # For function messages, prefix with function name
                if message.name:
                    anthropic_message["content"] = f"Function {message.name} returned: {message.content}"
            
            result.append(anthropic_message)
        
        return result 