"""
Context assembler for the retrieval tool.
Handles chunk deduplication and assembly while preserving raw metadata format.
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, UTC
from collections import defaultdict
from loguru import logger
from langchain.docstore.document import Document

from .config import MIN_OVERLAP_WORDS
from .metrics import SearchMetrics

@dataclass
class ChunkStats:
    """Statistics about chunk processing."""
    total_chunks: int = 0
    chunks_with_overlap: int = 0
    total_overlap_removed: int = 0
    unique_articles: int = 0
    unique_sections: int = 0

class ContextAssembler:
    """Assembles and deduplicates chunks while preserving raw metadata format."""
    
    def __init__(self, metrics: Optional[SearchMetrics] = None):
        """Initialize the context assembler.
        
        Args:
            metrics: Optional SearchMetrics instance for tracking performance
        """
        self.metrics = metrics
        self.stats = ChunkStats()
        self._processed_chunks: List[Document] = []
        self._article_metadata: Dict[str, Dict] = {}
    
    def find_overlap(self, prev_content: str, current_content: str) -> Tuple[str, int]:
        """Find overlapping content between chunks.
        
        Args:
            prev_content: Content of the previous chunk
            current_content: Content of the current chunk
            
        Returns:
            Tuple of (overlap text, overlap length in words)
        """
        words_prev = prev_content.split()
        words_current = current_content.split()
        
        # Look for overlapping sequences
        for i in range(len(words_prev)):
            for j in range(len(words_current)):
                if words_prev[i] == words_current[j]:
                    # Check if we have a sequence match
                    overlap = []
                    k = 0
                    while (i + k < len(words_prev) and 
                           j + k < len(words_current) and 
                           words_prev[i + k] == words_current[j + k]):
                        overlap.append(words_prev[i + k])
                        k += 1
                    
                    if len(overlap) >= MIN_OVERLAP_WORDS:
                        overlap_text = " ".join(overlap)
                        return overlap_text, len(overlap)
        
        return "", 0
    
    def deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove overlapping content between chunks while preserving metadata.
        
        Args:
            chunks: List of Document objects with metadata from vector store
            
        Returns:
            List of deduplicated Document objects
        """
        if not chunks:
            return chunks
        
        # Sort chunks by article and chunk index
        sorted_chunks = sorted(
            chunks,
            key=lambda x: (x.metadata["url"], x.metadata["chunk_index"])
        )
        
        # Track unique articles and sections
        self.stats.unique_articles = len({c.metadata["url"] for c in sorted_chunks})
        self.stats.unique_sections = len({c.metadata["section"] for c in sorted_chunks})
        
        # Deduplicate chunks
        deduped_chunks = []
        prev_content = ""
        current_article = None
        
        for chunk in sorted_chunks:
            self.stats.total_chunks += 1
            
            # If we're starting a new article, reset prev_content
            if current_article != chunk.metadata["url"]:
                current_article = chunk.metadata["url"]
                prev_content = ""
            
            # Find and handle overlap
            current_content = chunk.page_content
            overlap, overlap_length = self.find_overlap(prev_content, current_content)
            
            if overlap:
                self.stats.chunks_with_overlap += 1
                self.stats.total_overlap_removed += overlap_length
                
                # Remove overlapping content from current chunk
                if overlap in current_content:
                    current_content = current_content.replace(overlap, "", 1).strip()
                    if current_content:  # Only add if there's non-overlapping content
                        deduped_chunks.append(Document(
                            page_content=current_content,
                            metadata=chunk.metadata  # Preserve original metadata
                        ))
            else:
                deduped_chunks.append(chunk)  # Keep original Document object
            
            prev_content = chunk.page_content
        
        # Store processed chunks
        self._processed_chunks = deduped_chunks
        
        # Log deduplication stats
        if self.metrics:
            self.metrics.track_chunk_usage(
                total_chunks=self.stats.total_chunks,
                chunks_with_overlap=self.stats.chunks_with_overlap,
                total_overlap_removed=self.stats.total_overlap_removed
            )
        
        logger.info(
            f"Deduplication complete: {self.stats.total_chunks} chunks processed, "
            f"{self.stats.chunks_with_overlap} had overlap, "
            f"{self.stats.total_overlap_removed} words removed"
        )
        
        return deduped_chunks
    
    def format_chunks_for_llm(self, chunks: List[Document]) -> List[str]:
        """Format chunks for the LLM while preserving metadata.
        
        Args:
            chunks: List of Document objects with metadata
            
        Returns:
            List of formatted chunk strings with embedded metadata
        """
        formatted_chunks = []
        
        # Group chunks by article
        articles = defaultdict(list)
        for chunk in chunks:
            url = chunk.metadata["url"]
            articles[url].append(chunk)
        
        # Sort chunks within each article by chunk_index
        for url in articles:
            articles[url].sort(key=lambda x: x.metadata["chunk_index"])
        
        # Format each article's chunks
        for url, article_chunks in articles.items():
            # Get article metadata from first chunk
            first_chunk = article_chunks[0]
            title = url.split("/")[-1].replace("-", " ").title()
            section = first_chunk.metadata["section"]
            
            # Format each chunk with metadata
            for chunk in article_chunks:
                # Create metadata header
                header = (
                    f"--- Article: {title} ---\n"
                    f"Section: {section}\n"
                    f"Source: {url}\n"
                    f"---\n"
                )
                
                # Combine header with content
                formatted_chunk = header + chunk.page_content
                formatted_chunks.append(formatted_chunk)
        
        return formatted_chunks
    
    def get_assembly_stats(self) -> Dict:
        """Get statistics about the context assembly process.
        
        Returns:
            Dictionary containing assembly statistics
        """
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_chunks": self.stats.total_chunks,
            "chunks_with_overlap": self.stats.chunks_with_overlap,
            "total_overlap_removed": self.stats.total_overlap_removed,
            "unique_articles": self.stats.unique_articles,
            "unique_sections": self.stats.unique_sections,
            "final_chunk_count": len(self._processed_chunks)
        }
    
    def log_assembly_details(self):
        """Log detailed information about the context assembly process."""
        stats = self.get_assembly_stats()
        
        logger.info("\n" + "="*80)
        logger.info("CONTEXT ASSEMBLY DETAILS")
        logger.info("="*80)
        
        logger.info("\nAssembly Statistics:")
        logger.info(f"Total chunks processed: {stats['total_chunks']}")
        logger.info(f"Chunks with overlap: {stats['chunks_with_overlap']}")
        logger.info(f"Total overlap removed: {stats['total_overlap_removed']} words")
        logger.info(f"Unique articles: {stats['unique_articles']}")
        logger.info(f"Unique sections: {stats['unique_sections']}")
        logger.info(f"Final chunk count: {stats['final_chunk_count']}")
        
        logger.info("\n" + "="*80)

def prepare_chunks_for_llm(chunks: List[Document], metrics: Optional[SearchMetrics] = None) -> Tuple[List[str], Dict]:
    """
    Prepare chunks for the LLM by deduplicating and assembling them.
    
    Args:
        chunks: List of Document objects with metadata from vector store
        metrics: Optional SearchMetrics instance for tracking performance
    
    Returns:
        Tuple of (formatted chunks with embedded metadata, assembly statistics)
    """
    assembler = ContextAssembler(metrics)
    
    # First deduplicate chunks while preserving metadata
    deduped_chunks = assembler.deduplicate_chunks(chunks)
    
    # Then format chunks for the LLM
    formatted_chunks = assembler.format_chunks_for_llm(deduped_chunks)
    
    # Get stats and log details
    stats = assembler.get_assembly_stats()
    assembler.log_assembly_details()
    
    return formatted_chunks, stats 