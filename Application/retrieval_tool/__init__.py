"""
Retrieval tool module for semantic search and context assembly.
Provides robust, efficient, and well-documented retrieval capabilities.

Components:
- Retriever: Core semantic search with adaptive thresholds and exact metadata filtering
- ContextAssembler: Chunk deduplication and raw metadata preservation
- SearchMetrics: Performance tracking and analysis
"""

from .retriever import Retriever, SearchResult, DeduplicationStats
from .exceptions import (
    RetrievalError,
    NoResultsFound,
    InvalidFilter,
    SectionNotFound,
    VectorStoreError,
    ThresholdError
)
from .metrics import SearchMetrics
from .config import (
    DEFAULT_K,
    MIN_OVERLAP_WORDS,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    TIGHT_THRESHOLD_PERCENTILE,
    FALLBACK_THRESHOLD_PERCENTILE,
    MIN_RELEVANCE_SCORE
)

__all__ = [
    'Retriever',
    'SearchResult',
    'DeduplicationStats',
    'SearchMetrics',
    'RetrievalError',
    'NoResultsFound',
    'InvalidFilter',
    'SectionNotFound',
    'VectorStoreError',
    'ThresholdError',
    'DEFAULT_K',
    'MIN_OVERLAP_WORDS',
    'SEMANTIC_WEIGHT',
    'KEYWORD_WEIGHT',
    'TIGHT_THRESHOLD_PERCENTILE',
    'FALLBACK_THRESHOLD_PERCENTILE',
    'MIN_RELEVANCE_SCORE'
]

"""
Retrieval tool package initialization.
""" 