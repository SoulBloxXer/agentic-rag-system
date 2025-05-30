"""
Configuration settings for the retrieval tool.
"""

from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
EMBEDDINGS_DIR = BASE_DIR / "Embeddings"
CHUNKS_DIR = BASE_DIR / "Chunks"

# Vector store settings (matching embedder.py)
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
NORMALIZED = True
EMBEDDING_CHUNK_SIZE = 100

# Vector store configuration dictionary
VECTOR_STORE_SETTINGS: Dict[str, Any] = {
    "embedding_model": EMBEDDING_MODEL,
    "chunk_size": EMBEDDING_CHUNK_SIZE,
    "collection_name": COLLECTION_NAME,
    "embedding_dim": EMBEDDING_DIM,
    "normalized": NORMALIZED,
    "collection_metadata": {
        "version": "1.0.0",
        "embedding_model": EMBEDDING_MODEL,
        "normalized": NORMALIZED,
        "embedding_dim": EMBEDDING_DIM
    }
}

# Search parameters
DEFAULT_K = 10  # Number of initial results to retrieve
MIN_OVERLAP_WORDS = 20  # Minimum words for overlap detection
SEMANTIC_WEIGHT = 0.7  # Weight for semantic score in combined scoring
KEYWORD_WEIGHT = 0.3  # Weight for keyword score in combined scoring

# Configurable thresholds
TIGHT_THRESHOLD_PERCENTILE = 10  # Can be overridden
FALLBACK_THRESHOLD_PERCENTILE = 20  # Can be overridden
MIN_RELEVANCE_SCORE = 0.3  # Minimum score for a chunk to be considered relevant

# Threshold adjustment constants for adaptive threshold logic
TIGHT_DISTRIBUTION_ADJUSTMENT = 0.03  # For std dev < 0.05
SPREAD_DISTRIBUTION_ADJUSTMENT = 0.08  # For std dev > 0.2
BALANCED_DISTRIBUTION_ADJUSTMENT = 0.05  # For other cases

# Logging settings
LOG_DIR = BASE_DIR / "logs"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
LOG_ROTATION = "100 MB"
LOG_RETENTION = "1 week"

# Cache settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # Cache time-to-live in seconds

# Error messages
ERROR_MESSAGES: Dict[str, str] = {
    "vector_store_empty": "Vector store is empty! Please run embedder.py first.",
    "invalid_filter": "Invalid filter format or no exact match found. Use 'section:1 Case Studies' format.",
    "section_not_found": "Section '{section}' not found in knowledge base (exact match required)",
    "no_results": "No relevant results found for query",
    "invalid_query": "Query cannot be empty",
    "threshold_error": "Error adjusting search threshold: {reason}",
    "fallback_triggered": "Using fallback threshold: {reason}"
}

def get_log_path(log_name: str) -> Path:
    """Get the full path for a log file."""
    return LOG_DIR / f"{log_name}.log"

def get_vector_store_path() -> Path:
    """
    Get the path to the vector store.
    
    Returns:
        Path: Path to the vector store directory
        
    Raises:
        ValueError: If the vector store directory doesn't exist
    """
    if not EMBEDDINGS_DIR.exists():
        raise ValueError(f"Vector store directory not found: {EMBEDDINGS_DIR}")
    return EMBEDDINGS_DIR

def get_chunks_path() -> Path:
    """Get the path to the chunks directory."""
    return CHUNKS_DIR 