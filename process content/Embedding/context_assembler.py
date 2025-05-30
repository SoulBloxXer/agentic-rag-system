from typing import List, Dict, Set, Tuple
from collections import defaultdict
from langchain.docstore.document import Document
from loguru import logger
import re
from datetime import datetime, UTC

class ContextAssembler:
    """Assembles and prepares context for the LLM from search results."""
    
    def __init__(self):
        self.assembled_context = ""
        self.source_metadata = {}  # Track unique sources and their metadata
        self.chunk_metadata = []   # Track which chunks were used
        self.overlap_stats = {     # Track overlap handling statistics
            "total_chunks": 0,
            "chunks_with_overlap": 0,
            "total_overlap_removed": 0
        }
    
    def find_overlap(self, prev_content: str, current_content: str, min_overlap: int = 20) -> Tuple[str, int]:
        """Find overlapping content between chunks and return the overlap and its length."""
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
                    
                    if len(overlap) >= min_overlap:
                        overlap_text = " ".join(overlap)
                        return overlap_text, len(overlap)
        
        return "", 0
    
    def deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove overlapping content between chunks while preserving context."""
        if not chunks:
            return chunks
        
        # Sort chunks by article and chunk index
        sorted_chunks = sorted(chunks, key=lambda x: (x.metadata["url"], x.metadata["chunk_index"]))
        deduped_chunks = []
        current_article = None
        prev_content = ""
        
        for chunk in sorted_chunks:
            self.overlap_stats["total_chunks"] += 1
            
            # If we're starting a new article, reset prev_content
            if current_article != chunk.metadata["url"]:
                current_article = chunk.metadata["url"]
                prev_content = ""
            
            # Find and handle overlap
            current_content = chunk.page_content
            overlap, overlap_length = self.find_overlap(prev_content, current_content)
            
            if overlap:
                self.overlap_stats["chunks_with_overlap"] += 1
                self.overlap_stats["total_overlap_removed"] += overlap_length
                
                # Remove overlapping content from current chunk
                current_content = current_content[len(overlap):].strip()
                if current_content:  # Only add if there's non-overlapping content
                    deduped_chunks.append(Document(
                        page_content=current_content,
                        metadata=chunk.metadata
                    ))
            else:
                deduped_chunks.append(chunk)
            
            prev_content = chunk.page_content
        
        return deduped_chunks
    
    def deduplicate_metadata(self, chunks: List[Document]) -> Dict[str, Dict]:
        """Deduplicate and organize metadata by article."""
        metadata_by_article = defaultdict(lambda: {
            "url": "",
            "section": "",
            "chunk_indices": set(),
            "relevance_scores": [],
            "total_chunks": 0
        })
        
        for chunk in chunks:
            url = chunk.metadata["url"]
            metadata_by_article[url].update({
                "url": url,
                "section": chunk.metadata["section"],
                "chunk_indices": metadata_by_article[url]["chunk_indices"].union({chunk.metadata["chunk_index"]}),
                "total_chunks": len(metadata_by_article[url]["chunk_indices"])
            })
            
            if "relevance_score" in chunk.metadata:
                metadata_by_article[url]["relevance_scores"].append(chunk.metadata["relevance_score"])
        
        # Convert sets to sorted lists for better readability
        for article in metadata_by_article.values():
            article["chunk_indices"] = sorted(list(article["chunk_indices"]))
            if article["relevance_scores"]:
                article["avg_relevance"] = sum(article["relevance_scores"]) / len(article["relevance_scores"])
        
        return dict(metadata_by_article)
    
    def assemble_context(self, chunks: List[Document], include_metadata: bool = True) -> str:
        """Assemble chunks into a coherent context for the LLM."""
        # Debug: Log initial chunk count and unique URLs
        initial_urls = {chunk.metadata["url"] for chunk in chunks}
        logger.debug(f"Initial chunks: {len(chunks)}, Unique URLs: {len(initial_urls)}")
        
        # First deduplicate chunks
        deduped_chunks = self.deduplicate_chunks(chunks)
        
        # Debug: Log after deduplication
        deduped_urls = {chunk.metadata["url"] for chunk in deduped_chunks}
        logger.debug(f"After deduplication - Chunks: {len(deduped_chunks)}, Unique URLs: {len(deduped_urls)}")
        
        # Group chunks by article
        articles = defaultdict(list)
        for chunk in deduped_chunks:
            url = chunk.metadata["url"]
            articles[url].append(chunk)
        
        # Debug: Log article grouping
        logger.debug(f"Articles grouped: {len(articles)}")
        
        # Sort chunks within each article by chunk_index
        for url in articles:
            articles[url].sort(key=lambda x: x.metadata["chunk_index"])
        
        # Assemble context with clear article boundaries
        context_blocks = []
        for url, chunks in articles.items():
            # Add article header with metadata
            if include_metadata:
                # Get article title from URL
                article_title = url.split("/")[-1].replace("-", " ").title()
                
                # Format section name more clearly
                section_parts = chunks[0].metadata["section"].split(" ", 1)
                if len(section_parts) == 2:
                    section_num, section_name = section_parts
                    section = f"Section {section_num}: {section_name}"
                else:
                    section = chunks[0].metadata["section"]
                
                # Create a more informative header with dashes and clear boundaries
                header = f"\n--- Article: {article_title} ---\n"
                header += f"{section}\n"
                header += f"Source: {url}\n"
                if "relevance_score" in chunks[0].metadata:
                    header += f"Relevance score: {chunks[0].metadata['relevance_score']:.2f}\n"
                header += "---\n"  # Clear closing boundary for metadata
                context_blocks.append(header)
            
            # Join chunks with proper spacing
            content = "\n".join(chunk.page_content for chunk in chunks)
            context_blocks.append(content)
        
        # Store the assembled context
        self.assembled_context = "\n\n".join(context_blocks)
        
        # Store metadata for reference - using deduped_chunks instead of original chunks
        self.source_metadata = self.deduplicate_metadata(deduped_chunks)
        self.chunk_metadata = [
            {
                "url": chunk.metadata["url"],
                "section": chunk.metadata["section"],
                "chunk_index": chunk.metadata["chunk_index"],  # Keep for internal tracking
                "relevance_score": chunk.metadata.get("relevance_score", None)
            }
            for chunk in deduped_chunks
        ]
        
        # Debug: Log final metadata counts
        logger.debug(f"Final metadata - Source metadata articles: {len(self.source_metadata)}, Chunk metadata URLs: {len(set(c['url'] for c in self.chunk_metadata))}")
        
        return self.assembled_context
    
    def get_assembly_stats(self) -> Dict:
        """Get statistics about the context assembly process."""
        # Debug: Log all URLs in chunk metadata
        urls = set(chunk["url"] for chunk in self.chunk_metadata)
        logger.debug(f"URLs in chunk metadata: {urls}")
        
        # Count unique articles by URL
        unique_articles = len(urls)
        logger.debug(f"Counted unique articles: {unique_articles}")
        
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_articles": unique_articles,
            "total_chunks": self.overlap_stats["total_chunks"],
            "chunks_with_overlap": self.overlap_stats["chunks_with_overlap"],
            "total_overlap_removed": self.overlap_stats["total_overlap_removed"],
            "articles_by_section": self._get_section_stats(),
            "context_length": len(self.assembled_context),
            "chunk_metadata": self.chunk_metadata,
            "source_metadata": self.source_metadata
        }
    
    def _get_section_stats(self) -> Dict[str, int]:
        """Get count of articles by section."""
        section_stats = defaultdict(int)
        for article in self.source_metadata.values():
            section_stats[article["section"]] += 1
        return dict(section_stats)
    
    def log_assembly_details(self):
        """Log detailed information about the context assembly process."""
        stats = self.get_assembly_stats()
        
        logger.info("\n" + "="*80)
        logger.info("CONTEXT ASSEMBLY DETAILS")
        logger.info("="*80)
        
        logger.info("\nAssembly Statistics:")
        logger.info(f"Total articles: {stats['total_articles']}")
        logger.info(f"Total chunks processed: {stats['total_chunks']}")
        logger.info(f"Chunks with overlap: {stats['chunks_with_overlap']}")
        logger.info(f"Total overlap removed: {stats['total_overlap_removed']} words")
        logger.info(f"Final context length: {stats['context_length']} characters")
        
        logger.info("\nArticles by Section:")
        for section, count in stats["articles_by_section"].items():
            logger.info(f"  {section}: {count} articles")
        
        logger.info("\nSource Articles:")
        for url, metadata in stats["source_metadata"].items():
            logger.info(f"\n  URL: {url}")
            logger.info(f"  Section: {metadata['section']}")
            logger.info(f"  Chunks used: {metadata['chunk_indices']}")
            if "avg_relevance" in metadata:
                logger.info(f"  Average relevance: {metadata['avg_relevance']:.2f}")
        
        logger.info("\n" + "="*80)
        logger.info("ASSEMBLED CONTEXT PREVIEW")
        logger.info("="*80)
        logger.info("\n" + self.assembled_context[:500] + "...\n")
        logger.info("="*80)

def prepare_context_for_llm(chunks: List[Document], include_metadata: bool = True) -> Tuple[str, Dict]:
    """
    Prepare context for the LLM from search results.
    
    Args:
        chunks: List of Document objects from the search
        include_metadata: Whether to include metadata in the context
    
    Returns:
        Tuple of (assembled context string, assembly statistics)
    """
    assembler = ContextAssembler()
    context = assembler.assemble_context(chunks, include_metadata)
    stats = assembler.get_assembly_stats()
    assembler.log_assembly_details()
    
    return context, stats

# Example usage:
if __name__ == "__main__":
    # Configure logging
    logger.add(
        "process content/Embedding/logs/context_assembly.log",
        rotation="100 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # This would typically be used with search results from test_queries.py
    # Example:
    # from test_queries import smart_search, load_vector_store
    # vector_store = load_vector_store()
    # chunks, _ = smart_search(vector_store, "your query here")
    # context, stats = prepare_context_for_llm(chunks) 