"""
Examples of using the RAGStackXL LLM module.

This file demonstrates different ways to use the LLM components in RAGStackXL.
"""

import os
import asyncio
from dotenv import load_dotenv

from app.llm import (
    LLMConfig,
    LLMFactory,
    LLMProvider,
    ChatMessage,
    ChatRole
)
from app.llm.prompts import PromptTemplate, ChatPromptTemplate, SystemPrompts, create_rag_prompt
from app.llm.chain import LLMChain, SequentialChain


# Load environment variables from .env file
load_dotenv()


async def basic_openai_example():
    """Basic example using OpenAI."""
    print("\n=== Basic OpenAI Example ===")
    
    # Create LLM config
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100
    )
    
    # Create LLM instance
    llm = LLMFactory.create(config)
    
    # Simple completion
    response = await llm.generate("What is retrieval-augmented generation?")
    print(f"Response: {response.content}")
    print(f"Tokens: {response.total_tokens}")
    
    # Chat completion
    messages = [
        ChatMessage.system("You are a helpful, concise assistant."),
        ChatMessage.user("What is retrieval-augmented generation in 2 sentences?")
    ]
    
    response = await llm.chat(messages)
    print(f"\nChat response: {response.content}")
    print(f"Tokens: {response.total_tokens}")


async def prompt_template_example():
    """Example using prompt templates."""
    print("\n=== Prompt Template Example ===")
    
    # Create LLM config
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create LLM instance
    llm = LLMFactory.create(config)
    
    # Simple prompt template
    template = PromptTemplate(
        "Explain {concept} to {audience} in {num_paragraphs} paragraphs."
    )
    
    # Format the template
    prompt = template.format(
        concept="neural networks",
        audience="a high school student",
        num_paragraphs=2
    )
    
    print(f"Formatted prompt: {prompt}")
    
    # Use the template with LLM
    response = await llm.generate(prompt)
    print(f"\nResponse: {response.content}")
    
    # Chat prompt template
    chat_template = ChatPromptTemplate([
        {"role": "system", "content": "You are a helpful teacher."},
        {"role": "user", "content": template}
    ])
    
    # Format as chat messages
    messages = chat_template.format_messages(
        concept="deep learning",
        audience="a beginner programmer",
        num_paragraphs=1
    )
    
    # Convert to ChatMessage objects
    chat_messages = [
        ChatMessage(role=msg["role"], content=msg["content"])
        for msg in messages
    ]
    
    # Use with LLM
    response = await llm.chat(chat_messages)
    print(f"\nChat response: {response.content}")


async def chain_example():
    """Example using LLM chains."""
    print("\n=== LLM Chain Example ===")
    
    # Create LLM config
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create LLM instance
    llm = LLMFactory.create(config)
    
    # Create a simple chain
    template = PromptTemplate("Summarize this text in one paragraph: {text}")
    chain = LLMChain(llm=llm, prompt=template, output_key="summary")
    
    # Run the chain
    result = await chain.execute(
        text="Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of retrieval-based and generation-based approaches in natural language processing. In RAG, a model first retrieves relevant documents or passages from a knowledge base and then uses those retrievals as additional context when generating a response. This approach helps ground the model's outputs in factual information and reduces hallucinations. RAG can be used with various retrieval mechanisms, from simple keyword matching to sophisticated vector search, and can be fine-tuned for specific domains or tasks."
    )
    
    print(f"Summary: {result['summary']}")
    print(f"Execution time: {result['execution_time']:.2f} seconds")
    print(f"Tokens used: {result['tokens']['total']}")
    
    # Create a RAG chain using pre-built templates
    rag_prompt = create_rag_prompt()
    rag_chain = LLMChain(llm=llm, prompt=rag_prompt, output_key="answer")
    
    # Run the RAG chain
    result = await rag_chain.execute(
        question="What is the capital of France?",
        context="France is a country in Western Europe. Its capital is Paris, which is known for landmarks like the Eiffel Tower and the Louvre Museum."
    )
    
    print(f"\nRAG Answer: {result['answer']}")


async def sequential_chain_example():
    """Example using sequential chains."""
    print("\n=== Sequential Chain Example ===")
    
    # Create LLM config
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create LLM instance
    llm = LLMFactory.create(config)
    
    # Create first chain: translation to French
    template1 = PromptTemplate(
        "Translate this text to French: {input_text}"
    )
    chain1 = LLMChain(llm=llm, prompt=template1, output_key="french_text")
    
    # Create second chain: translation to German
    template2 = PromptTemplate(
        "Translate this text from French to German: {french_text}"
    )
    chain2 = LLMChain(llm=llm, prompt=template2, output_key="german_text")
    
    # Create a sequential chain
    seq_chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["input_text"],
        output_variables=["french_text", "german_text"]
    )
    
    # Run the sequential chain
    result = await seq_chain.execute(input_text="Hello, world!")
    
    print(f"Original text: Hello, world!")
    print(f"French translation: {result['french_text']}")
    print(f"German translation: {result['german_text']}")
    print(f"Total execution time: {result['execution_time']:.2f} seconds")


async def streaming_example():
    """Example using streaming responses."""
    print("\n=== Streaming Example ===")
    
    # Create LLM config
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create LLM instance
    llm = LLMFactory.create(config)
    
    # Create a chain
    template = PromptTemplate("Write a short story about {topic} in {style} style.")
    chain = LLMChain(llm=llm, prompt=template)
    
    print("Streaming response:")
    print("-----------------")
    
    # Stream the response
    async for chunk in chain.stream(topic="a robot learning to paint", style="humorous"):
        print(chunk, end="", flush=True)
    
    print("\n-----------------")


async def main():
    """Run all examples."""
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file with your API key or set it in your environment.")
        return
    
    await basic_openai_example()
    await prompt_template_example()
    await chain_example()
    await sequential_chain_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main()) 