"""
Unit tests for prompt templates.
"""

import unittest
import pytest
from unittest.mock import patch

from app.llm.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemPrompts,
    create_rag_prompt,
    create_summary_prompt,
    create_qa_prompt
)


class TestPromptTemplate(unittest.TestCase):
    """Test cases for PromptTemplate."""
    
    def test_basic_template(self):
        """Test basic template creation and formatting."""
        # Create a template with explicit variables
        template = PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name"]
        )
        
        # Check properties
        self.assertEqual(template.template, "Hello, {name}!")
        self.assertEqual(template.input_variables, ["name"])
        
        # Test formatting
        result = template.format(name="World")
        self.assertEqual(result, "Hello, World!")
        
        # Test call syntax
        result = template(name="Universe")
        self.assertEqual(result, "Hello, Universe!")
    
    def test_auto_detect_variables(self):
        """Test automatic detection of variables."""
        template = PromptTemplate("My name is {name} and I am {age} years old.")
        
        # Check detected variables (order may vary as it uses a set internally)
        self.assertIn("name", template.input_variables)
        self.assertIn("age", template.input_variables)
        self.assertEqual(len(template.input_variables), 2)
        
        # Test formatting
        result = template.format(name="Alice", age=30)
        self.assertEqual(result, "My name is Alice and I am 30 years old.")
    
    def test_missing_variables(self):
        """Test error handling for missing variables."""
        template = PromptTemplate("Hello, {name}!")
        
        # Should raise a KeyError for missing variables
        with self.assertRaises(KeyError):
            template.format()
        
        with self.assertRaises(KeyError):
            template.format(wrong_name="Alice")
    
    @patch("app.llm.prompts.log")
    def test_validation_warnings(self, mock_log):
        """Test validation warnings."""
        # Variable in input_variables but not in template
        PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name", "age"],
            validate_template=True
        )
        mock_log.warning.assert_called_with("Input variable 'age' not found in template")
        
        mock_log.reset_mock()
        
        # Variable in template but not in input_variables
        PromptTemplate(
            template="Hello, {name}!",
            input_variables=[],
            validate_template=True
        )
        mock_log.warning.assert_called_once_with("Template variable 'name' not declared in input_variables")


class TestChatPromptTemplate(unittest.TestCase):
    """Test cases for ChatPromptTemplate."""
    
    def test_basic_chat_template(self):
        """Test basic chat template creation and formatting."""
        # Create a template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PromptTemplate("My name is {name}.")}
        ]
        chat_template = ChatPromptTemplate(messages)
        
        # Check properties
        self.assertEqual(len(chat_template.message_templates), 2)
        self.assertEqual(chat_template.input_variables, ["name"])
        
        # Test formatting
        formatted = chat_template.format_messages(name="Alice")
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["role"], "system")
        self.assertEqual(formatted[0]["content"], "You are a helpful assistant.")
        self.assertEqual(formatted[1]["role"], "user")
        self.assertEqual(formatted[1]["content"], "My name is Alice.")
        
        # Test call syntax
        formatted = chat_template(name="Bob")
        self.assertEqual(formatted[1]["content"], "My name is Bob.")
    
    def test_missing_variables(self):
        """Test error handling for missing variables."""
        messages = [
            {"role": "user", "content": PromptTemplate("My name is {name} and I am {age} years old.")}
        ]
        chat_template = ChatPromptTemplate(messages)
        
        # Should raise a KeyError for missing variables
        with self.assertRaises(KeyError):
            chat_template.format_messages(name="Alice")
    
    def test_string_content_conversion(self):
        """Test automatic conversion of string content to PromptTemplate."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        chat_template = ChatPromptTemplate(messages)
        
        # Check that string content was converted to PromptTemplate
        self.assertIsInstance(messages[0]["content"], PromptTemplate)
        self.assertIsInstance(messages[1]["content"], PromptTemplate)
    
    def test_invalid_message_structure(self):
        """Test error handling for invalid message structure."""
        # Missing role
        messages = [
            {"content": "Hello!"}
        ]
        with self.assertRaises(ValueError):
            ChatPromptTemplate(messages)
        
        # Missing content
        messages = [
            {"role": "user"}
        ]
        with self.assertRaises(ValueError):
            ChatPromptTemplate(messages)
    
    def test_format_string_output(self):
        """Test formatting as a string."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PromptTemplate("My name is {name}.")}
        ]
        chat_template = ChatPromptTemplate(messages)
        
        # Format as string
        formatted = chat_template.format(name="Alice")
        expected = "system: You are a helpful assistant.\n\nuser: My name is Alice."
        self.assertEqual(formatted, expected)


class TestHelperFunctions(unittest.TestCase):
    """Test cases for prompt helper functions."""
    
    def test_system_prompts(self):
        """Test system prompts constants."""
        self.assertTrue(SystemPrompts.DEFAULT)
        self.assertTrue(SystemPrompts.RAG)
        self.assertTrue(SystemPrompts.SUMMARY)
        self.assertTrue(SystemPrompts.QA)
    
    def test_create_rag_prompt(self):
        """Test creating a RAG prompt."""
        # With system prompt
        prompt = create_rag_prompt(include_system_prompt=True)
        formatted = prompt.format_messages(question="What is RAG?", context="RAG stands for Retrieval Augmented Generation.")
        
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["role"], "system")
        self.assertEqual(formatted[0]["content"], SystemPrompts.RAG)
        self.assertEqual(formatted[1]["role"], "user")
        self.assertTrue("Question: What is RAG?" in formatted[1]["content"])
        self.assertTrue("Context: RAG stands for Retrieval Augmented Generation." in formatted[1]["content"])
        
        # Without system prompt
        prompt = create_rag_prompt(include_system_prompt=False)
        formatted = prompt.format_messages(question="What is RAG?", context="RAG stands for Retrieval Augmented Generation.")
        
        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]["role"], "user")
    
    def test_create_summary_prompt(self):
        """Test creating a summary prompt."""
        prompt = create_summary_prompt()
        formatted = prompt.format_messages(text="This is a text to summarize.")
        
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["role"], "system")
        self.assertEqual(formatted[0]["content"], SystemPrompts.SUMMARY)
        self.assertEqual(formatted[1]["role"], "user")
        self.assertTrue("Summarize the following text: This is a text to summarize." in formatted[1]["content"])
    
    def test_create_qa_prompt(self):
        """Test creating a QA prompt."""
        prompt = create_qa_prompt()
        formatted = prompt.format_messages(question="What is the meaning of life?")
        
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["role"], "system")
        self.assertEqual(formatted[0]["content"], SystemPrompts.QA)
        self.assertEqual(formatted[1]["role"], "user")
        self.assertEqual(formatted[1]["content"], "What is the meaning of life?") 