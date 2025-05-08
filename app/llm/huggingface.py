"""
HuggingFace LLM implementation for RAGStackXL.

This module provides an implementation of the LLM interface for models hosted on 
HuggingFace's Inference API or run locally through transformers.
"""

import os
import json
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
import asyncio
from threading import Thread

try:
    import requests
    import aiohttp
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from app.llm.interfaces import LLM, LLMConfig, LLMResponse, ChatMessage, ChatRole, LLMProvider
from app.utils.logging import log


class HuggingFaceLLM(LLM):
    """
    HuggingFace LLM implementation.
    
    This class supports two ways of accessing HuggingFace models:
    1. Via Inference API (online, using API token)
    2. Via local Transformers installation (offline, using downloaded models)
    
    The mode is determined by the config parameter "use_local_model".
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the HuggingFace LLM.
        
        Args:
            config: LLM configuration
        """
        super().__init__(config)
        
        # Check if we're using local models or the HF Inference API
        self.use_local_model = config.get("use_local_model", False)
        
        # For local models, we need transformers
        if self.use_local_model and not HAS_TRANSFORMERS:
            raise ImportError(
                "Transformers package is not installed. "
                "Please install it with 'pip install transformers torch'"
            )
        
        # For API calls, we need requests/aiohttp
        if not self.use_local_model and not HAS_REQUESTS:
            raise ImportError(
                "Requests and aiohttp packages are not installed. "
                "Please install them with 'pip install requests aiohttp'"
            )
        
        # Set up model
        self.model_name = config.model_name
        
        # Store common parameters
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.timeout = config.timeout
        
        # Initialize based on mode
        if self.use_local_model:
            self._init_local_model()
        else:
            self._init_inference_api()
    
    def _init_local_model(self):
        """Initialize a local HuggingFace model."""
        try:
            log.info(f"Loading local model: {self.model_name}")
            
            # Get local device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Using device: {device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            # Set up generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto"
            )
            
            # Add chat template if not already present
            if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
                # Use a simple default chat template
                self.has_chat_template = False
                log.warning(f"Model {self.model_name} has no chat template, using simple message concatenation")
            else:
                self.has_chat_template = True
            
            # Store max model length
            if hasattr(self.model.config, "max_position_embeddings"):
                self.model_max_length = self.model.config.max_position_embeddings
            else:
                self.model_max_length = 2048  # Default assumption
                log.warning(f"Could not determine max length for model {self.model_name}, using {self.model_max_length}")
            
        except Exception as e:
            log.error(f"Error loading local HuggingFace model: {e}")
            raise
    
    def _init_inference_api(self):
        """Initialize connection to HuggingFace Inference API."""
        # Get API token from config or environment
        self.api_token = self.config.api_key or os.environ.get("HUGGINGFACE_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "HuggingFace API token is required for Inference API. "
                "Please provide it in the config or set HUGGINGFACE_API_TOKEN environment variable."
            )
        
        # Set API URL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        # Set headers
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        
        log.info(f"Initialized HuggingFace Inference API for model: {self.model_name}")
    
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
        if self.use_local_model:
            return await self._generate_local(prompt, **kwargs)
        else:
            return await self._generate_api(prompt, **kwargs)
    
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
        if self.use_local_model:
            async for chunk in self._generate_local_stream(prompt, **kwargs):
                yield chunk
        else:
            async for chunk in self._generate_api_stream(prompt, **kwargs):
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
        # Convert chat messages to a single prompt
        prompt = self._format_chat_messages(messages)
        
        return await self.generate(prompt, **kwargs)
    
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
        # Convert chat messages to a single prompt
        prompt = self._format_chat_messages(messages)
        
        async for chunk in self.generate_stream(prompt, **kwargs):
            yield chunk
    
    async def _generate_local(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using a local model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        try:
            # Get parameters
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            
            # Count input tokens (approximate)
            input_tokens = len(self.tokenizer.encode(prompt))
            
            # Run generation in a thread to not block the event loop
            def _generate():
                return self.pipe(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    return_full_text=False
                )
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _generate)
            
            # Extract text from result
            generated_text = result[0]["generated_text"]
            
            # Count output tokens (approximate)
            output_tokens = len(self.tokenizer.encode(generated_text))
            
            # Create usage stats
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
            # Publish event
            self._publish_generation_event(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"]
            )
            
            return LLMResponse(
                content=generated_text,
                model=self.model_name,
                usage=usage,
                finish_reason="stop"
            )
            
        except Exception as e:
            log.error(f"Error generating text with local HuggingFace model: {e}")
            raise
    
    async def _generate_local_stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming text using a local model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Text chunks as they are generated
        """
        try:
            # Get parameters
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            
            # Prepare inputs
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate with streaming
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "streamer": streamer,
            }
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield from streamer
            for text in streamer:
                yield text
                
            # Wait for thread to finish
            thread.join()
            
        except Exception as e:
            log.error(f"Error streaming text with local HuggingFace model: {e}")
            raise
    
    async def _generate_api(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using the HuggingFace Inference API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        try:
            # Prepare parameters
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "temperature": kwargs.get("temperature", self.temperature),
                    "top_p": kwargs.get("top_p", self.top_p),
                    "return_full_text": False
                }
            }
            
            # Call API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            
            # Extract text from response
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = ""
                log.warning(f"Unexpected response from HuggingFace Inference API: {result}")
            
            # Make a simple estimation of token counts
            # Note: This is approximate since we don't have actual token counts from the API
            input_words = len(prompt.split())
            output_words = len(generated_text.split())
            
            # Rough estimate: 1.33 tokens per word
            input_tokens = int(input_words * 1.33)
            output_tokens = int(output_words * 1.33)
            
            # Create usage stats
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
            # Publish event
            self._publish_generation_event(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"]
            )
            
            return LLMResponse(
                content=generated_text,
                model=self.model_name,
                usage=usage,
                finish_reason="stop"
            )
            
        except Exception as e:
            log.error(f"Error generating text with HuggingFace Inference API: {e}")
            raise
    
    async def _generate_api_stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming text using the HuggingFace Inference API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Text chunks as they are generated
        """
        # Note: HuggingFace Inference API doesn't support proper streaming
        # We'll simulate it by generating the full response and then chunking it
        response = await self._generate_api(prompt, **kwargs)
        
        # Chunk the response into smaller parts (by words)
        words = response.content.split()
        chunk_size = 3  # Number of words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            yield chunk + " "
            await asyncio.sleep(0.05)  # Small delay to simulate streaming
    
    def _format_chat_messages(self, messages: List[ChatMessage]) -> str:
        """
        Format chat messages into a prompt string.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Formatted prompt string
        """
        if self.use_local_model and self.has_chat_template:
            # Use the model's chat template
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            
            return self.tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Simple formatting
            result = []
            for msg in messages:
                role = msg.role.value.capitalize()
                result.append(f"{role}: {msg.content}")
            
            # Add assistant prefix to prompt the model to generate
            result.append("Assistant: ")
            
            return "\n".join(result)


# Add this if we're using local models
class TextIteratorStreamer:
    """
    Stream text from a tokenizer.
    
    This is a simple implementation for completeness, but in a real-world
    scenario, you'd want to use the transformers.TextIteratorStreamer.
    """
    
    def __init__(self, tokenizer, skip_prompt=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.generated_text = ""
        self.previous_text = ""
        self.tokens = []
        self.text_queue = asyncio.Queue()
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # This is called in the thread where generation happens
        if self.tokens:
            token = self.tokens.pop(0)
            text = self.tokenizer.decode(token)
            if text:
                return text
        return ""
    
    def put(self, token_ids):
        # Called by model generation
        self.tokens.extend(token_ids.tolist()[0]) 