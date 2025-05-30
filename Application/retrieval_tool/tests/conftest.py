"""
Test configuration and fixtures for the retrieval tool tests.
"""

import os
from pathlib import Path
import pytest
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Load environment variables
    load_dotenv()
    
    # Verify required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    yield  # Run the test
    
    # Cleanup after test (if needed)
    pass 