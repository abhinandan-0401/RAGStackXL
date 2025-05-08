"""
Prompt template system for RAGStackXL.

This module provides tools for structured prompt engineering, allowing for
easy creation and management of prompts for LLMs.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import re
import json
from string import Formatter

from app.utils.logging import log


class PromptTemplate:
    """
    Template for generating prompts.
    
    This class provides a way to create templates for prompts with variables
    that can be filled in at runtime.
    """
    
    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
        template_format: str = "f-string",
        validate_template: bool = True
    ):
        """
        Initialize a prompt template.
        
        Args:
            template: Template string with variables
            input_variables: List of variable names in the template
            template_format: Format of the template ('f-string' or 'jinja2')
            validate_template: Whether to validate the template
        """
        self.template = template
        self.template_format = template_format
        
        # Auto-detect input variables if not provided
        self._input_variables = self._extract_variables() if input_variables is None else input_variables
        
        # Validate the template
        if validate_template:
            self._validate_template()
    
    def _extract_variables(self) -> List[str]:
        """
        Extract variables from the template.
        
        Returns:
            List of variable names
        """
        if self.template_format == "f-string":
            # Use string.Formatter to extract variables from f-string format
            formatter = Formatter()
            variables = [field_name for _, field_name, _, _ in formatter.parse(self.template) if field_name]
            return list(set(variables))
        else:
            # Simple regex-based extraction for other formats
            pattern = r"{{\s*(\w+)\s*}}"
            return list(set(re.findall(pattern, self.template)))
    
    def _validate_template(self) -> None:
        """
        Validate the template.
        
        Raises:
            ValueError: If the template is invalid
        """
        # Extract variables from the template
        template_vars = self._extract_variables()
        
        # Check that all input variables are in the template
        for var in self._input_variables:
            if var not in template_vars:
                log.warning(f"Input variable '{var}' not found in template")
        
        # Check for variables in the template that are not in input_variables
        for var in template_vars:
            if var not in self._input_variables:
                log.warning(f"Template variable '{var}' not declared in input_variables")
    
    @property
    def input_variables(self) -> List[str]:
        """Get the input variables for the template."""
        return self._input_variables
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the given values.
        
        Args:
            **kwargs: Values for the template variables
            
        Returns:
            Formatted string
            
        Raises:
            KeyError: If a required variable is missing
        """
        # Check for missing variables
        missing_vars = set(self._input_variables) - set(kwargs.keys())
        if missing_vars:
            raise KeyError(f"Missing variables: {', '.join(missing_vars)}")
        
        # Format the template
        if self.template_format == "f-string":
            try:
                return self.template.format(**kwargs)
            except KeyError as e:
                raise KeyError(f"Error formatting template: {e}")
        else:
            # Simple replacement for other formats
            result = self.template
            for key, value in kwargs.items():
                pattern = r"{{(\s*)" + re.escape(key) + r"(\s*)}}"
                result = re.sub(pattern, str(value), result)
            return result
    
    def __call__(self, **kwargs) -> str:
        """
        Format the template with the given values.
        
        This allows the template to be called like a function.
        
        Args:
            **kwargs: Values for the template variables
            
        Returns:
            Formatted string
        """
        return self.format(**kwargs)


class ChatPromptTemplate:
    """
    Template for generating chat prompts.
    
    This class provides a way to create templates for multi-message chat prompts
    with variables that can be filled in at runtime.
    """
    
    def __init__(
        self,
        message_templates: List[Dict[str, Union[str, PromptTemplate]]],
        input_variables: Optional[List[str]] = None
    ):
        """
        Initialize a chat prompt template.
        
        Args:
            message_templates: List of message templates
                Each message should have 'role' and 'content' keys
                'content' can be a string or PromptTemplate
            input_variables: List of input variables across all templates
        """
        self.message_templates = message_templates
        
        # Process message templates
        self._process_templates()
        
        # Extract all input variables if not provided
        self._input_variables = input_variables or self._extract_all_variables()
    
    def _process_templates(self) -> None:
        """Process message templates to ensure they are all PromptTemplates."""
        for i, message in enumerate(self.message_templates):
            # Validate message structure
            if "role" not in message:
                raise ValueError(f"Message {i} is missing 'role'")
            if "content" not in message:
                raise ValueError(f"Message {i} is missing 'content'")
            
            # Convert string content to PromptTemplate
            if isinstance(message["content"], str):
                message["content"] = PromptTemplate(message["content"])
    
    def _extract_all_variables(self) -> List[str]:
        """
        Extract all variables from all templates.
        
        Returns:
            List of all variable names
        """
        all_vars = set()
        for message in self.message_templates:
            if isinstance(message["content"], PromptTemplate):
                all_vars.update(message["content"].input_variables)
        return list(all_vars)
    
    @property
    def input_variables(self) -> List[str]:
        """Get all input variables across all templates."""
        return self._input_variables
    
    def format_messages(self, **kwargs) -> List[Dict[str, str]]:
        """
        Format all message templates with the given values.
        
        Args:
            **kwargs: Values for the template variables
            
        Returns:
            List of formatted messages with role and content
        """
        messages = []
        for message_template in self.message_templates:
            # Get the content template
            content_template = message_template["content"]
            
            # Format the content
            try:
                # Extract only the variables needed for this template
                if isinstance(content_template, PromptTemplate):
                    template_vars = content_template.input_variables
                    template_kwargs = {k: v for k, v in kwargs.items() if k in template_vars}
                    content = content_template.format(**template_kwargs)
                else:
                    content = content_template
                
                # Create the formatted message
                messages.append({
                    "role": message_template["role"],
                    "content": content
                })
            except KeyError as e:
                log.error(f"Error formatting chat message: {e}")
                raise
        
        return messages
    
    def format(self, **kwargs) -> str:
        """
        Format all messages and return as a string.
        
        This is mainly for debugging purposes, as chat models typically
        expect structured message objects.
        
        Args:
            **kwargs: Values for the template variables
            
        Returns:
            String representation of all messages
        """
        messages = self.format_messages(**kwargs)
        return "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    def __call__(self, **kwargs) -> List[Dict[str, str]]:
        """
        Format all message templates with the given values.
        
        This allows the template to be called like a function.
        
        Args:
            **kwargs: Values for the template variables
            
        Returns:
            List of formatted messages
        """
        return self.format_messages(**kwargs)


# Common system prompts
class SystemPrompts:
    """Collection of common system prompts."""
    
    DEFAULT = "You are a helpful, accurate, and concise assistant."
    
    RAG = """You are a helpful, accurate, and concise assistant. You will be given:
1. A user question
2. Relevant context to help answer the question

Use the context to provide a comprehensive, factual answer to the user's question.
If the context doesn't contain the information needed to fully answer the question, 
say so clearly and don't make up information.
Always cite your sources by referring to the document ID if available."""

    SUMMARY = """You are a helpful, accurate, and concise summarization assistant. 
Your task is to create clear, factual summaries of the provided content.
Focus on extracting the key points, main ideas, and essential information.
Be objective and maintain the tone of the original text."""

    QA = """You are a helpful question-answering assistant. 
Your goal is to provide accurate, factual answers to the user's questions.
If you don't know the answer or don't have enough information, say so clearly.
Don't make up information or guess when you're uncertain."""


# Helper functions to create common templates
def create_rag_prompt(include_system_prompt: bool = True) -> ChatPromptTemplate:
    """
    Create a RAG prompt template.
    
    Args:
        include_system_prompt: Whether to include the system prompt
        
    Returns:
        Chat prompt template for RAG
    """
    messages = []
    
    if include_system_prompt:
        messages.append({
            "role": "system",
            "content": SystemPrompts.RAG
        })
    
    messages.extend([
        {
            "role": "user",
            "content": PromptTemplate(
                "Question: {question}\n\nContext: {context}"
            )
        }
    ])
    
    return ChatPromptTemplate(messages)


def create_summary_prompt(include_system_prompt: bool = True) -> ChatPromptTemplate:
    """
    Create a summarization prompt template.
    
    Args:
        include_system_prompt: Whether to include the system prompt
        
    Returns:
        Chat prompt template for summarization
    """
    messages = []
    
    if include_system_prompt:
        messages.append({
            "role": "system",
            "content": SystemPrompts.SUMMARY
        })
    
    messages.extend([
        {
            "role": "user",
            "content": PromptTemplate(
                "Summarize the following text: {text}"
            )
        }
    ])
    
    return ChatPromptTemplate(messages)


def create_qa_prompt(include_system_prompt: bool = True) -> ChatPromptTemplate:
    """
    Create a Q&A prompt template.
    
    Args:
        include_system_prompt: Whether to include the system prompt
        
    Returns:
        Chat prompt template for Q&A
    """
    messages = []
    
    if include_system_prompt:
        messages.append({
            "role": "system",
            "content": SystemPrompts.QA
        })
    
    messages.extend([
        {
            "role": "user",
            "content": PromptTemplate("{question}")
        }
    ])
    
    return ChatPromptTemplate(messages) 