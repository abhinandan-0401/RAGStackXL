"""
Unit tests for the settings module.
"""
import os
import pytest
from pathlib import Path

from app.config.settings import Settings


def test_settings_initialization():
    """Test that settings can be initialized with default values."""
    settings = Settings()
    assert settings.PROJECT_NAME == "RAGStackXL"
    assert settings.VERSION == "0.1.0"
    assert settings.ENVIRONMENT == "development"


def test_settings_environment_validation():
    """Test environment validation."""
    # Valid environments should not raise an error
    Settings(ENVIRONMENT="development")
    Settings(ENVIRONMENT="staging")
    Settings(ENVIRONMENT="production")
    
    # Invalid environment should raise a ValueError
    with pytest.raises(ValueError):
        Settings(ENVIRONMENT="invalid")


def test_settings_directories_creation(tmp_path):
    """Test that directories are created by the settings class."""
    # Create a temporary directory for testing
    test_dir = tmp_path / "test_settings"
    test_dir.mkdir()
    
    # Create test subdirectories
    docs_dir = str(test_dir / "documents")
    vectordb_dir = str(test_dir / "vectordb")
    
    # Initialize settings with test directories
    settings = Settings(
        BASE_DIR=str(test_dir),
        DOCUMENT={"DATA_DIR": docs_dir},
        VECTORDB={"PERSIST_DIRECTORY": vectordb_dir},
    )
    
    # Call the method to create directories
    settings.create_directories()
    
    # Check that directories were created
    assert Path(docs_dir).exists()
    assert Path(vectordb_dir).exists()
    assert Path(test_dir / "logs").exists() 