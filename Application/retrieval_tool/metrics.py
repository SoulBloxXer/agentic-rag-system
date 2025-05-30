"""
Search metrics tracking and analysis.
"""

from typing import Dict, Set, Optional, Any, List
from datetime import datetime, UTC
import hashlib
from loguru import logger
from .config import get_log_path
from contextlib import contextmanager

class SearchMetrics:
    """Track and analyze search performance metrics."""
    
    def __init__(self, log_name: str = "search_metrics"):
        """Initialize metrics tracking.
        
        Args:
            log_name: Name of the log file (without extension)
        """
        self.query_id = None
        self.query_hash = None
        self.start_time = None
        self.metrics = {
            "query": "",
            "total_articles": 0,
            "total_chunks": 0,
            "relevant_chunks": 0,
            "distance_scores": [],
            "keyword_scores": [],
            "combined_scores": [],
            "response_time": 0,
            "feedback": None,
            "used_chunks": set(),
            "threshold_used": None,
            "threshold_type": None,  # "tight" or "fallback"
            "fallback_reason": None,
            "filter_applied": None
        }
        
        # Set up logging
        logger.add(
            get_log_path(log_name),
            rotation="100 MB",
            retention="1 week",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    
    @contextmanager
    def track_query(self, query: str, filter_dict: Optional[Dict[str, str]] = None):
        """Context manager for tracking a query.
        
        Args:
            query: The search query
            filter_dict: Optional filter dictionary (e.g., {"section": "1 Case Studies"})
        """
        try:
            self.start_query(query, filter_dict)
            yield
        finally:
            self.end_query()
    
    def record_error(self, error_message: str) -> None:
        """Record an error in the metrics.
        
        Args:
            error_message: The error message to record
        """
        self.metrics["error"] = error_message
        logger.error(f"Error recorded: {error_message}")
    
    def start_query(self, query: str, filter_dict: Optional[Dict[str, str]] = None) -> None:
        """Start tracking a new query.
        
        Args:
            query: The search query
            filter_dict: Optional filter dictionary (e.g., {"section": "1 Case Studies"})
        """
        self.query_id = datetime.now(UTC).isoformat()
        self.query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
        self.start_time = datetime.now(UTC)
        self.metrics["query"] = query
        self.metrics["filter_applied"] = filter_dict
        logger.info(f"Starting query {self.query_hash}")
        if filter_dict:
            logger.info(f"Filter applied: {filter_dict}")
    
    def end_query(self) -> Dict[str, Any]:
        """End tracking and log metrics.
        
        Returns:
            Dict containing the final metrics
        """
        if self.start_time:
            self.metrics["response_time"] = (datetime.now(UTC) - self.start_time).total_seconds()
        
        # Convert sets to lists for JSON serialization
        metrics_for_logging = self.metrics.copy()
        metrics_for_logging["used_chunks"] = list(self.metrics["used_chunks"])
        
        # Log to metrics log
        logger.bind(query_id=self.query_id, query_hash=self.query_hash).info(
            "Query Metrics",
            extra={
                "metrics": metrics_for_logging,
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
        
        return metrics_for_logging
    
    def add_feedback(self, rating: int, feedback: Optional[str] = None) -> None:
        """Add human feedback for the query.
        
        Args:
            rating: Rating from 1-5
            feedback: Optional feedback comment
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
            
        self.metrics["feedback"] = {
            "rating": rating,
            "comment": feedback,
            "timestamp": datetime.now(UTC).isoformat()
        }
        logger.bind(query_id=self.query_id).info(
            f"Feedback received: {rating}/5 - {feedback}"
        )
    
    def track_chunk_usage(self, chunk_id: str, score: float) -> None:
        """Track which chunks were used in the response.
        
        Args:
            chunk_id: Unique identifier for the chunk
            score: Combined relevance score
        """
        self.metrics["used_chunks"].add(chunk_id)
        self.metrics["combined_scores"].append(score)
    
    def update_threshold(self, threshold: float, threshold_type: str, fallback_reason: Optional[str] = None) -> None:
        """Update the threshold used for this query.
        
        Args:
            threshold: The threshold value used
            threshold_type: Either "tight" or "fallback"
            fallback_reason: Optional explanation if fallback was used
        """
        self.metrics["threshold_used"] = threshold
        self.metrics["threshold_type"] = threshold_type
        self.metrics["fallback_reason"] = fallback_reason
        
        if threshold_type == "fallback":
            logger.info(f"Using fallback threshold: {threshold:.4f} - Reason: {fallback_reason}")
        else:
            logger.debug(f"Using tight threshold: {threshold:.4f}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current metrics.
        
        Returns:
            Dict containing key metrics
        """
        return {
            "query_id": self.query_id,
            "query_hash": self.query_hash,
            "total_articles": self.metrics["total_articles"],
            "total_chunks": self.metrics["total_chunks"],
            "relevant_chunks": self.metrics["relevant_chunks"],
            "response_time": self.metrics["response_time"],
            "threshold_used": self.metrics["threshold_used"],
            "threshold_type": self.metrics["threshold_type"],
            "fallback_reason": self.metrics["fallback_reason"],
            "filter_applied": self.metrics["filter_applied"]
        } 