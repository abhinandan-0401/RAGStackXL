"""
LLM Chains for RAGStackXL.

This module provides the LLMChain implementation for chaining together
prompts and LLMs to create complex generation pipelines.
"""

from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
import asyncio
import time
from datetime import datetime

from app.llm.interfaces import LLM, LLMConfig, LLMResponse, ChatMessage, ChatRole
from app.llm.prompts import PromptTemplate, ChatPromptTemplate
from app.core.interfaces import EventType, event_bus, Event
from app.utils.logging import log


class LLMChain:
    """
    Chain for connecting an LLM with a prompt template.
    
    This class provides a way to execute an LLM with a prompt template,
    handling variable substitution and result processing.
    """
    
    def __init__(
        self,
        llm: LLM,
        prompt: Union[str, PromptTemplate, ChatPromptTemplate],
        output_key: str = "text",
        output_parser: Optional[Callable[[str], Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an LLM chain.
        
        Args:
            llm: LLM to use for generation
            prompt: Prompt template or string to use
            output_key: Key to use for the output in the result dictionary
            output_parser: Function to parse the output
            metadata: Additional metadata
        """
        self.llm = llm
        self.output_key = output_key
        self.output_parser = output_parser
        self.metadata = metadata or {}
        
        # Process prompt
        if isinstance(prompt, str):
            self.prompt = PromptTemplate(prompt)
            self.is_chat_prompt = False
        elif isinstance(prompt, ChatPromptTemplate):
            self.prompt = prompt
            self.is_chat_prompt = True
        elif isinstance(prompt, PromptTemplate):
            self.prompt = prompt
            self.is_chat_prompt = False
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        # Get input variables from prompt
        self.input_variables = self.prompt.input_variables
    
    async def run(self, **kwargs) -> Any:
        """
        Run the chain with the given inputs.
        
        This is a simplified interface that returns only the output value.
        
        Args:
            **kwargs: Values for the prompt variables
            
        Returns:
            Chain output
        """
        result = await self.execute(**kwargs)
        return result[self.output_key]
    
    async def stream(self, **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream the chain execution.
        
        This allows for streaming the results as they are generated.
        
        Args:
            **kwargs: Values for the prompt variables
            
        Yields:
            Text chunks as they are generated
        """
        # Format prompt
        if self.is_chat_prompt:
            formatted_messages = self.prompt.format_messages(**kwargs)
            # Convert to ChatMessage objects
            messages = [
                ChatMessage(
                    role=msg["role"],
                    content=msg["content"]
                )
                for msg in formatted_messages
            ]
            
            # Stream from chat API
            async for chunk in self.llm.chat_stream(messages):
                yield chunk
        else:
            # Format as string
            formatted_prompt = self.prompt.format(**kwargs)
            
            # Stream from completion API
            async for chunk in self.llm.generate_stream(formatted_prompt):
                yield chunk
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the chain with the given inputs.
        
        Args:
            **kwargs: Values for the prompt variables
            
        Returns:
            Dictionary with the chain results
        """
        start_time = time.time()
        
        try:
            # Format prompt
            if self.is_chat_prompt:
                formatted_messages = self.prompt.format_messages(**kwargs)
                # Convert to ChatMessage objects
                messages = [
                    ChatMessage(
                        role=msg["role"],
                        content=msg["content"]
                    )
                    for msg in formatted_messages
                ]
                
                # Call LLM chat API
                response = await self.llm.chat(messages)
            else:
                # Format as string
                formatted_prompt = self.prompt.format(**kwargs)
                
                # Call LLM completion API
                response = await self.llm.generate(formatted_prompt)
            
            # Process output
            output = response.content
            
            # Parse output if parser is provided
            if self.output_parser:
                try:
                    parsed_output = self.output_parser(output)
                except Exception as e:
                    log.error(f"Error parsing output: {e}")
                    parsed_output = output
            else:
                parsed_output = output
            
            # Calculate metrics
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create result dictionary
            result = {
                self.output_key: parsed_output,
                "raw_output": output,
                "execution_time": execution_time,
                "tokens": {
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                    "total": response.total_tokens
                },
                "model": response.model,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add inputs to result (except for potentially large context)
            sanitized_inputs = {k: v for k, v in kwargs.items() if k != "context" and not isinstance(v, (list, dict))}
            result["inputs"] = sanitized_inputs
            
            # Add metadata
            result["metadata"] = self.metadata
            
            # Publish event
            self._publish_generation_event(result)
            
            return result
            
        except Exception as e:
            log.error(f"Error executing LLM chain: {e}")
            raise
    
    def _publish_generation_event(self, result: Dict[str, Any]) -> None:
        """
        Publish event for chain generation.
        
        Args:
            result: Chain result
        """
        event_bus.publish(
            Event(
                event_type=EventType.GENERATION_COMPLETED,
                payload={
                    "chain_type": self.__class__.__name__,
                    "output_key": self.output_key,
                    "execution_time": result["execution_time"],
                    "tokens": result["tokens"],
                    "model": result["model"]
                }
            )
        )


class SequentialChain:
    """
    Chain that runs multiple chains in sequence.
    
    This class provides a way to execute multiple chains sequentially,
    passing the outputs of one chain as inputs to the next.
    """
    
    def __init__(
        self,
        chains: List[LLMChain],
        input_variables: List[str],
        output_variables: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a sequential chain.
        
        Args:
            chains: List of chains to execute in sequence
            input_variables: List of input variables
            output_variables: List of output variables
            metadata: Additional metadata
        """
        self.chains = chains
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.metadata = metadata or {}
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute all chains in sequence.
        
        Args:
            **kwargs: Values for the chain input variables
            
        Returns:
            Dictionary with all chain results
        """
        start_time = time.time()
        
        # Check for missing input variables
        missing_vars = set(self.input_variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing input variables: {', '.join(missing_vars)}")
        
        # Initialize state with input variables
        state = {**kwargs}
        results = {}
        
        # Execute each chain, passing state
        for i, chain in enumerate(self.chains):
            # Get required inputs for this chain
            chain_inputs = {}
            for var_name in chain.input_variables:
                if var_name not in state:
                    raise ValueError(f"Missing variable for chain {i}: {var_name}")
                chain_inputs[var_name] = state[var_name]
            
            # Execute chain
            chain_result = await chain.execute(**chain_inputs)
            
            # Update state with chain result
            state[chain.output_key] = chain_result[chain.output_key]
            
            # Store full result
            results[f"chain_{i}"] = chain_result
        
        # Create final result
        result = {
            var: state[var] for var in self.output_variables if var in state
        }
        
        # Add execution stats
        end_time = time.time()
        execution_time = end_time - start_time
        
        result["execution_time"] = execution_time
        result["chain_results"] = results
        result["metadata"] = self.metadata
        
        return result
    
    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the sequential chain and return only the output variables.
        
        Args:
            **kwargs: Values for the chain input variables
            
        Returns:
            Dictionary with output variables
        """
        result = await self.execute(**kwargs)
        return {var: result[var] for var in self.output_variables if var in result} 