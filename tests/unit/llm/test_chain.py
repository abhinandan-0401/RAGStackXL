"""
Unit tests for LLM Chains.
"""

import unittest
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from typing import List, Dict, Any

from app.llm.chain import LLMChain, SequentialChain
from app.llm.interfaces import LLM, LLMConfig, LLMResponse, ChatMessage
from app.llm.prompts import PromptTemplate, ChatPromptTemplate
from app.core.interfaces import EventType


class MockLLM(LLM):
    """Mock LLM for testing."""
    
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
            content="Mock response",
            model=self.config.model_name,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
    
    async def generate_stream(self, prompt, **kwargs):
        """Mock generate_stream method."""
        yield "Mock"
        yield " streamed"
        yield " response"
    
    async def chat(self, messages, **kwargs):
        """Mock chat method."""
        self.chat_calls.append((messages, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(
            content="Mock chat response",
            model=self.config.model_name,
            usage={"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
        )
    
    async def chat_stream(self, messages, **kwargs):
        """Mock chat_stream method."""
        yield "Mock"
        yield " streamed"
        yield " chat"
        yield " response"


@pytest.mark.asyncio
class TestLLMChain:
    """Test cases for LLMChain."""
    
    async def test_basic_text_chain(self):
        """Test basic text prompt chain."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create chain
        template = PromptTemplate("Hello, {name}!")
        chain = LLMChain(llm=llm, prompt=template)
        
        # Check properties
        assert chain.input_variables == ["name"]
        assert not chain.is_chat_prompt
        
        # Execute chain
        result = await chain.execute(name="World")
        
        # Check LLM was called correctly
        assert len(llm.generate_calls) == 1
        assert llm.generate_calls[0][0] == "Hello, World!"
        
        # Check result
        assert result[chain.output_key] == "Mock response"
        assert result["raw_output"] == "Mock response"
        assert "execution_time" in result
        assert result["tokens"]["prompt"] == 10
        assert result["tokens"]["completion"] == 5
        assert result["tokens"]["total"] == 15
        assert "inputs" in result
        assert result["inputs"]["name"] == "World"
    
    async def test_chat_chain(self):
        """Test chat prompt chain."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create chain
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PromptTemplate("Hello, {name}!")}
        ]
        chat_template = ChatPromptTemplate(messages)
        chain = LLMChain(llm=llm, prompt=chat_template)
        
        # Check properties
        assert chain.input_variables == ["name"]
        assert chain.is_chat_prompt
        
        # Execute chain
        result = await chain.execute(name="World")
        
        # Check LLM was called correctly
        assert len(llm.chat_calls) == 1
        messages = llm.chat_calls[0][0]
        assert len(messages) == 2
        assert messages[0].role.value == "system"
        assert messages[0].content == "You are a helpful assistant."
        assert messages[1].role.value == "user"
        assert messages[1].content == "Hello, World!"
        
        # Check result
        assert result[chain.output_key] == "Mock chat response"
    
    async def test_run_shortcut(self):
        """Test the run shortcut method."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create chain
        template = PromptTemplate("Hello, {name}!")
        chain = LLMChain(llm=llm, prompt=template)
        
        # Execute chain with run shortcut
        result = await chain.run(name="World")
        
        # Check result is just the output
        assert result == "Mock response"
    
    async def test_stream(self):
        """Test streaming from a chain."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create text chain
        template = PromptTemplate("Hello, {name}!")
        chain = LLMChain(llm=llm, prompt=template)
        
        # Stream from chain
        chunks = []
        async for chunk in chain.stream(name="World"):
            chunks.append(chunk)
        
        # Check chunks
        assert chunks == ["Mock", " streamed", " response"]
        
        # Create chat chain
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PromptTemplate("Hello, {name}!")}
        ]
        chat_template = ChatPromptTemplate(messages)
        chat_chain = LLMChain(llm=llm, prompt=chat_template)
        
        # Stream from chat chain
        chunks = []
        async for chunk in chat_chain.stream(name="World"):
            chunks.append(chunk)
        
        # Check chunks
        assert chunks == ["Mock", " streamed", " chat", " response"]
    
    async def test_output_parser(self):
        """Test using an output parser."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create a mock response
        llm.responses.append(LLMResponse(
            content="42",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
        ))
        
        # Create parser function
        def parse_int(text):
            return int(text)
        
        # Create chain with parser
        template = PromptTemplate("What is the answer to {question}?")
        chain = LLMChain(
            llm=llm,
            prompt=template,
            output_parser=parse_int,
            output_key="answer"
        )
        
        # Execute chain
        result = await chain.execute(question="life, the universe, and everything")
        
        # Check parsed result
        assert result["answer"] == 42
        assert isinstance(result["answer"], int)
        assert result["raw_output"] == "42"
    
    async def test_output_parser_error(self):
        """Test error handling in output parser."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create a mock response that will cause parser error
        llm.responses.append(LLMResponse(
            content="not an integer",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        ))
        
        # Create parser function that will raise an error
        def parse_int(text):
            return int(text)
        
        # Create chain with parser
        template = PromptTemplate("What is the answer to {question}?")
        chain = LLMChain(
            llm=llm, 
            prompt=template,
            output_parser=parse_int,
            output_key="answer"
        )
        
        # Mock logger
        with patch("app.llm.chain.log") as mock_log:
            # Execute chain (should not raise exception)
            result = await chain.execute(question="life, the universe, and everything")
            
            # Check error was logged
            mock_log.error.assert_called_once()
            assert "Error parsing output" in mock_log.error.call_args[0][0]
            
            # Check result falls back to raw output
            assert result["answer"] == "not an integer"
            assert result["raw_output"] == "not an integer"
    
    async def test_metadata(self):
        """Test adding metadata to chain."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create chain with metadata
        template = PromptTemplate("Hello, {name}!")
        metadata = {"purpose": "greeting", "version": "1.0"}
        chain = LLMChain(llm=llm, prompt=template, metadata=metadata)
        
        # Execute chain
        result = await chain.execute(name="World")
        
        # Check metadata in result
        assert "metadata" in result
        assert result["metadata"] == metadata
    
    async def test_event_publishing(self):
        """Test event publishing."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create chain
        template = PromptTemplate("Hello, {name}!")
        chain = LLMChain(llm=llm, prompt=template)
        
        # Mock event_bus
        with patch("app.llm.chain.event_bus") as mock_event_bus:
            # Execute chain
            result = await chain.execute(name="World")
            
            # Check event was published
            mock_event_bus.publish.assert_called_once()
            event_arg = mock_event_bus.publish.call_args[0][0]
            assert event_arg.event_type == EventType.GENERATION_COMPLETED
            payload = event_arg.payload
            assert payload["chain_type"] == "LLMChain"
            assert payload["output_key"] == "text"
            assert "execution_time" in payload
            assert payload["model"] == "test-model"


@pytest.mark.asyncio
class TestSequentialChain:
    """Test cases for SequentialChain."""
    
    async def test_basic_sequential_chain(self):
        """Test basic sequential chain execution."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create two chains
        template1 = PromptTemplate("Translate '{input_text}' to French")
        chain1 = LLMChain(llm=llm, prompt=template1, output_key="french_text")
        
        # Mock response for first chain
        llm.responses.append(LLMResponse(
            content="Bonjour le monde",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        ))
        
        template2 = PromptTemplate("Translate '{french_text}' to German")
        chain2 = LLMChain(llm=llm, prompt=template2, output_key="german_text")
        
        # Mock response for second chain
        llm.responses.append(LLMResponse(
            content="Hallo Welt",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        ))
        
        # Create sequential chain
        seq_chain = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["input_text"],
            output_variables=["french_text", "german_text"]
        )
        
        # Execute chain
        result = await seq_chain.execute(input_text="Hello world")
        
        # Check LLM calls
        assert len(llm.generate_calls) == 2
        assert llm.generate_calls[0][0] == "Translate 'Hello world' to French"
        assert llm.generate_calls[1][0] == "Translate 'Bonjour le monde' to German"
        
        # Check result
        assert result["french_text"] == "Bonjour le monde"
        assert result["german_text"] == "Hallo Welt"
        assert "execution_time" in result
        assert "chain_results" in result
        assert "chain_0" in result["chain_results"]
        assert "chain_1" in result["chain_results"]
    
    async def test_run_shortcut(self):
        """Test the run shortcut method."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create two chains
        template1 = PromptTemplate("Process '{input_text}'")
        chain1 = LLMChain(llm=llm, prompt=template1, output_key="intermediate")
        
        # Mock response for first chain
        llm.responses.append(LLMResponse(
            content="Intermediate result",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
        ))
        
        template2 = PromptTemplate("Finalize '{intermediate}'")
        chain2 = LLMChain(llm=llm, prompt=template2, output_key="final")
        
        # Mock response for second chain
        llm.responses.append(LLMResponse(
            content="Final result",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        ))
        
        # Create sequential chain
        seq_chain = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["input_text"],
            output_variables=["final"]  # Only want the final output
        )
        
        # Execute chain with run shortcut
        result = await seq_chain.run(input_text="Input")
        
        # Check result contains only the output variables
        assert "final" in result
        assert result["final"] == "Final result"
        assert "intermediate" not in result
        assert "execution_time" not in result
        assert "chain_results" not in result
    
    async def test_missing_input_variables(self):
        """Test error handling for missing input variables."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create a simple chain
        template = PromptTemplate("Process '{input_text}'")
        chain = LLMChain(llm=llm, prompt=template, output_key="output")
        
        # Create sequential chain
        seq_chain = SequentialChain(
            chains=[chain],
            input_variables=["input_text"],
            output_variables=["output"]
        )
        
        # Execute with missing input
        with pytest.raises(ValueError):
            await seq_chain.execute(wrong_input="Input")
    
    async def test_missing_intermediate_variables(self):
        """Test error handling for missing intermediate variables."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create two chains with mismatched variables
        template1 = PromptTemplate("Process '{input_text}'")
        chain1 = LLMChain(llm=llm, prompt=template1, output_key="result1")
        
        template2 = PromptTemplate("Process '{missing_variable}'")
        chain2 = LLMChain(llm=llm, prompt=template2, output_key="result2")
        
        # Create sequential chain
        seq_chain = SequentialChain(
            chains=[chain1, chain2],
            input_variables=["input_text"],
            output_variables=["result1", "result2"]
        )
        
        # Execute chain and check for error
        with pytest.raises(ValueError):
            await seq_chain.execute(input_text="Input")
    
    async def test_metadata(self):
        """Test adding metadata to sequential chain."""
        # Create mock LLM
        config = LLMConfig(provider="test", model_name="test-model")
        llm = MockLLM(config)
        
        # Create a simple chain
        template = PromptTemplate("Process '{input_text}'")
        chain = LLMChain(llm=llm, prompt=template, output_key="output")
        
        # Mock a response
        llm.responses.append(LLMResponse(
            content="Processed result",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        ))
        
        # Create sequential chain with metadata
        metadata = {"pipeline": "test-pipeline", "version": "1.0"}
        seq_chain = SequentialChain(
            chains=[chain],
            input_variables=["input_text"],
            output_variables=["output"],
            metadata=metadata
        )
        
        # Execute chain
        result = await seq_chain.execute(input_text="Input")
        
        # Check metadata in result
        assert "metadata" in result
        assert result["metadata"] == metadata 