"""
Test script for the retrieval tool with full metrics tracking.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import sys
import json

# Add parent directory to path so we can import the retrieval tool
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from Application.retrieval_tool.retriever import Retriever
from Application.retrieval_tool.metrics import SearchMetrics

def setup_logging():
    """Set up logging for the test."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Set up log file
    log_file = log_dir / "test_retriever.log"
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="100 MB"
    )
    
    return log_file

def format_metrics(metrics: dict) -> str:
    """Format metrics dictionary for logging."""
    return json.dumps(metrics, indent=2)

def test_retriever():
    """Test the full retrieval tool with metrics tracking."""
    # Set up logging
    log_file = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting retrieval tool test with metrics")
    logger.info("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Verify OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    logger.info("✅ Environment checks passed")
    
    # Initialize retriever with metrics
    logger.info("\nInitializing retriever...")
    metrics = SearchMetrics()
    retriever = Retriever(metrics=metrics)
    logger.info("✅ Retriever initialized")
    
    # Test basic search
    logger.info("\n" + "=" * 80)
    logger.info("Testing basic search...")
    logger.info("=" * 80)
    
    test_queries = [
        "What are common shrink wrap issues?",
        "How to fix sealing problems? | section: Troubleshooting",
        "Best practices for heat sealing"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        try:
            # Perform search using the full pipeline
            result = retriever.search(query)
            
            # Log complete results
            logger.info("\n" + "=" * 80)
            logger.info("COMPLETE SEARCH RESULTS")
            logger.info("=" * 80)
            
            # Log all chunks with their indices
            logger.info("\nRetrieved Articles:")
            logger.info("-" * 80)
            for i, chunk in enumerate(result.chunks, 1):
                logger.info(f"\nArticle {i}:")
                logger.info("-" * 40)
                
                # Extract chunk indices from the content
                chunk_lines = chunk.split('\n')
                chunk_indices = []
                for line in chunk_lines:
                    if "chunk_index:" in line:
                        chunk_indices.append(line.split("chunk_index:")[1].strip())
                
                logger.info(f"Chunk Indices: {', '.join(chunk_indices)}")
                logger.info(chunk)
            
            # Log complete metrics
            logger.info("\n" + "=" * 80)
            logger.info("COMPLETE METRICS")
            logger.info("=" * 80)
            logger.info(format_metrics(result.metrics))
            
            # Log filter information
            if result.filter_applied:
                logger.info("\nFilter Applied:")
                logger.info("-" * 40)
                logger.info(format_metrics(result.filter_applied))
            
            # Log fallback information
            if result.fallback_reason:
                logger.info("\nFallback Information:")
                logger.info("-" * 40)
                logger.info(f"Reason: {result.fallback_reason}")
            
            # Log summary
            logger.info("\n" + "=" * 80)
            logger.info("SEARCH SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total Articles: {result.metrics['total_articles']}")
            logger.info(f"Total Chunks: {result.metrics['total_chunks']}")
            logger.info(f"Response Time: {result.metrics['response_time']:.2f}s")
            logger.info(f"Threshold Type: {result.metrics['threshold_type']}")
            
            # Log deduplication stats
            dedup_stats = result.metrics['deduplication_stats']
            logger.info("\nDeduplication Summary:")
            logger.info(f"Chunks with Overlap: {dedup_stats['chunks_with_overlap']}")
            logger.info(f"Total Overlap Removed: {dedup_stats['total_overlap_removed']} words")
            logger.info(f"Articles Processed: {dedup_stats['articles_processed']}")
            
        except Exception as e:
            logger.error(f"❌ Error with query '{query}': {str(e)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Test completed successfully!")
    logger.info(f"Full log available at: {log_file}")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_retriever() 