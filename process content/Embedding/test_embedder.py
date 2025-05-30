import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from embedder import (
    load_documents,
    initialize_embedding_model,
    normalize_embeddings,
    create_vector_store,
    calculate_file_hash
)

# Test data
SAMPLE_DOCUMENT = {
    "page_content": "Test content",
    "metadata": {
        "url": "http://example.com",
        "section": "Test Section",
        "article_index": 0,
        "chunk_index": 0,
        "token_count": 10
    }
}

@pytest.fixture
def sample_chunks_file(tmp_path):
    """Create a temporary chunks.json file for testing."""
    chunks_file = tmp_path / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump([SAMPLE_DOCUMENT], f)
    return str(chunks_file)

def test_load_documents(sample_chunks_file):
    """Test document loading functionality."""
    documents = load_documents(sample_chunks_file)
    assert len(documents) == 1
    assert documents[0].page_content == "Test content"
    assert documents[0].metadata["url"] == "http://example.com"
    assert documents[0].metadata["section"] == "Test Section"

def test_load_documents_invalid_format(tmp_path):
    """Test document loading with invalid format."""
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, 'w', encoding='utf-8') as f:
        json.dump({"not": "a list"}, f)
    
    with pytest.raises(ValueError, match="Input JSON must be a list of documents"):
        load_documents(str(invalid_file))

def test_load_documents_missing_metadata(tmp_path):
    """Test document loading with missing metadata."""
    invalid_doc = {
        "page_content": "Test content",
        "metadata": {
            "url": "http://example.com"
            # Missing required fields
        }
    }
    invalid_file = tmp_path / "invalid_metadata.json"
    with open(invalid_file, 'w', encoding='utf-8') as f:
        json.dump([invalid_doc], f)
    
    with pytest.raises(ValueError, match="Missing required metadata fields"):
        load_documents(str(invalid_file))

@patch('embedder.OpenAIEmbeddings')
def test_initialize_embedding_model(mock_embeddings):
    """Test embedding model initialization."""
    mock_embeddings.return_value = MagicMock()
    model = initialize_embedding_model()
    assert model is not None
    mock_embeddings.assert_called_once_with(
        model="text-embedding-3-small",
        openai_api_key=None,
        chunk_size=100
    )

def test_normalize_embeddings():
    """Test embedding normalization."""
    # Create sample embeddings
    sample_embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    normalized = normalize_embeddings(sample_embeddings)
    
    # Check normalization
    assert len(normalized) == 2
    assert len(normalized[0]) == 3
    
    # Verify L2 norm is 1 for each embedding
    for emb in normalized:
        norm = sum(x * x for x in emb) ** 0.5
        assert abs(norm - 1.0) < 1e-6

@patch('embedder.Chroma')
@patch('embedder.calculate_file_hash')
def test_create_vector_store(mock_hash, mock_chroma, tmp_path):
    """Test vector store creation."""
    # Setup
    mock_hash.return_value = "test_hash"
    mock_chroma.from_embeddings.return_value = MagicMock()
    
    # Create test data
    documents = [MagicMock(metadata={"token_count": 10})]
    embeddings = [[1.0, 2.0, 3.0]]
    
    # Create vector store
    vector_store = create_vector_store(documents, embeddings)
    
    # Verify
    assert vector_store is not None
    mock_chroma.from_embeddings.assert_called_once()
    
    # Check metadata
    call_kwargs = mock_chroma.from_embeddings.call_args[1]
    assert call_kwargs["collection_name"] == "knowledge_base"
    assert call_kwargs["collection_metadata"]["version"] == "1.0.0"
    assert call_kwargs["collection_metadata"]["source_file_hash"] == "test_hash"
    assert call_kwargs["collection_metadata"]["normalized"] is True

def test_calculate_file_hash(tmp_path):
    """Test file hash calculation."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    # Calculate hash
    file_hash = calculate_file_hash(str(test_file))
    
    # Verify hash is a valid SHA-256 hash
    assert len(file_hash) == 64
    assert all(c in "0123456789abcdef" for c in file_hash)

if __name__ == "__main__":
    pytest.main([__file__]) 