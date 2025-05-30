"""
Test package for the retrieval tool.

This package contains comprehensive tests for the retrieval tool, including:
- Unit tests for individual components
- Integration tests for the full pipeline
- Performance benchmarks
- Error case validation
- Threshold and fallback testing
- Filter exact-match validation

All tests are designed to maintain pipeline integrity by running in isolation.
"""

from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# Ensure test data directory exists
TEST_DATA_DIR.mkdir(exist_ok=True)

# Test constants
TEST_COLLECTION = "test_knowledge_base"
TEST_SECTIONS = [
    "1 Introduction",
    "2 Troubleshooting",
    "3 Case Studies",
    "4 Best Practices"
]

# Test utilities
def create_test_document(content: str, metadata: dict) -> dict:
    """Create a test document with content and metadata."""
    return {
        "page_content": content,
        "metadata": metadata
    }

def create_test_chunks() -> list:
    """Create a set of test chunks with known overlaps."""
    return [
        create_test_document(
            "This is the first chunk of the article. It contains some content.",
            {
                "section": "1 Introduction",
                "chunk_index": 0,
                "article_url": "http://test.com/article1",
                "article_title": "Test Article 1"
            }
        ),
        create_test_document(
            "It contains some content. This is the second chunk with overlap.",
            {
                "section": "1 Introduction",
                "chunk_index": 1,
                "article_url": "http://test.com/article1",
                "article_title": "Test Article 1"
            }
        ),
        # Add more test chunks as needed
    ] 