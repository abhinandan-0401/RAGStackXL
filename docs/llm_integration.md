# LLM Integration Guide

## Overview

The LLM (Large Language Model) integration module is a key component of RAGStackXL, providing a unified interface to various LLM providers including OpenAI, Anthropic, HuggingFace, and others. This module enables seamless interaction with different language models while maintaining a consistent API.

## Key Features

- **Multiple provider support**: Integrates with various LLM providers (OpenAI, Anthropic, HuggingFace, etc.)
- **Unified interface**: Common API for all LLM providers
- **Async-first design**: Built with asyncio for high performance
- **Streaming support**: Stream LLM responses for real-time applications
- **Chat and completion modes**: Support for both chat and completion APIs
- **Prompt templating**: Structured prompt engineering with variable substitution
- **LLM chaining**: Chain multiple LLM operations together
- **Event system**: Track token usage and other metrics

## Architecture

The LLM module follows a modular, extensible architecture:

1. **Interfaces**: Abstract classes and protocols defining the common API
2. **Implementations**: Provider-specific implementations of the interfaces
3. **Factory**: Factory pattern for creating LLM instances
4. **Prompt System**: Templating system for structured prompt engineering
5. **Chains**: Systems for combining LLMs and prompts into reusable pipelines

## Usage

### Basic Usage

```python
from app.llm import LLMConfig, LLMFactory, LLMProvider, ChatMessage

# Create an LLM configuration
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Create an LLM instance
llm = LLMFactory.create(config)

# Text generation
response = await llm.generate("What is retrieval-augmented generation?")
print(response.content)

# Chat completion
messages = [
    ChatMessage.system("You are a helpful assistant."),
    ChatMessage.user("What is RAG?")
]
response = await llm.chat(messages)
print(response.content)
```

### Using Prompt Templates

```python
from app.llm.prompts import PromptTemplate, ChatPromptTemplate

# Simple prompt template
template = PromptTemplate("Explain {concept} to {audience}.")
prompt = template.format(concept="neural networks", audience="a beginner")

# Chat prompt template
chat_template = ChatPromptTemplate([
    {"role": "system", "content": "You are a helpful teacher."},
    {"role": "user", "content": "Explain {concept} to {audience}."}
])
messages = chat_template.format_messages(concept="AI", audience="a beginner")
```

### Using LLM Chains

```python
from app.llm.chain import LLMChain, SequentialChain

# Create a simple chain
template = PromptTemplate("Summarize this text: {text}")
chain = LLMChain(llm=llm, prompt=template, output_key="summary")
result = await chain.execute(text="Long text to summarize...")
print(result["summary"])

# Create a sequential chain
template1 = PromptTemplate("Translate to French: {input_text}")
chain1 = LLMChain(llm=llm, prompt=template1, output_key="french_text")

template2 = PromptTemplate("Translate from French to German: {french_text}")
chain2 = LLMChain(llm=llm, prompt=template2, output_key="german_text")

seq_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["input_text"],
    output_variables=["french_text", "german_text"]
)
result = await seq_chain.execute(input_text="Hello, world!")
```

### Streaming Responses

```python
# Stream text generation
async for chunk in llm.generate_stream("Tell me a story about..."):
    print(chunk, end="", flush=True)

# Stream from a chain
template = PromptTemplate("Write a story about {topic}.")
chain = LLMChain(llm=llm, prompt=template)
async for chunk in chain.stream(topic="space exploration"):
    print(chunk, end="", flush=True)
```

## Provider-Specific Configuration

### OpenAI

```python
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model_name="gpt-4",  # or gpt-3.5-turbo, etc.
    temperature=0.7,
    max_tokens=500,
    api_key="your-api-key"  # Optional, can also use OPENAI_API_KEY env var
)
```

### Anthropic

```python
config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model_name="claude-3-opus-20240229",  # or any other Claude model
    temperature=0.7,
    max_tokens=1000,
    api_key="your-api-key",  # Optional, can also use ANTHROPIC_API_KEY env var
    system_prompt="You are a helpful assistant."  # Optional system prompt
)
```

### HuggingFace

```python
# Using Inference API
config = LLMConfig(
    provider=LLMProvider.HUGGINGFACE,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    api_key="your-huggingface-token"  # Optional, can also use HUGGINGFACE_API_TOKEN env var
)

# Using local models
config = LLMConfig(
    provider=LLMProvider.HUGGINGFACE,
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    use_local_model=True  # Run the model locally
)
```

## Error Handling

The LLM module is designed to handle errors gracefully:

```python
try:
    response = await llm.generate("Prompt that might cause an error")
    print(response.content)
except Exception as e:
    print(f"Error: {e}")
```

## Extending

To add a new LLM provider:

1. Create a new implementation class extending the LLM abstract class
2. Implement all required methods (generate, generate_stream, chat, chat_stream)
3. Register the new provider with the LLMFactory

```python
from app.llm.interfaces import LLM, LLMConfig, LLMProvider

class MyCustomLLM(LLM):
    # Implement required methods
    
# Register with factory
LLMFactory.register(LLMProvider.CUSTOM, MyCustomLLM)
```

## Best Practices

1. **Use environment variables** for API keys rather than hardcoding them
2. **Implement proper error handling** to gracefully handle API errors
3. **Use prompt templates** for maintainable prompt engineering
4. **Consider token limits** when designing prompts and chains
5. **Use streaming** for better user experience with longer responses
6. **Monitor token usage** via the event system