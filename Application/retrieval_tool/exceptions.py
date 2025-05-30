"""
Custom exceptions for the retrieval tool.
"""

class RetrievalError(Exception):
    """Base exception for retrieval errors."""
    pass

class NoResultsFound(RetrievalError):
    """Raised when no relevant results are found."""
    pass

class InvalidFilter(RetrievalError):
    """Raised when filter format is invalid or section not found."""
    pass

class SectionNotFound(InvalidFilter):
    """Raised when section filter doesn't match any known section."""
    pass

class VectorStoreError(RetrievalError):
    """Raised for vector store related errors."""
    pass

class ThresholdError(RetrievalError):
    """Raised when threshold adjustment fails."""
    pass 