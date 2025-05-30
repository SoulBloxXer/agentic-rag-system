"""
Core semantic search implementation for the retrieval tool.
Handles query processing, search execution, and result formatting.
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from datetime import datetime
import re
from collections import defaultdict

from langchain.docstore.document import Document
from langchain_chroma import Chroma
import numpy as np

from .metrics import SearchMetrics
from .config import (
    DEFAULT_K, MIN_OVERLAP_WORDS, SEMANTIC_WEIGHT, KEYWORD_WEIGHT,
    TIGHT_THRESHOLD_PERCENTILE, FALLBACK_THRESHOLD_PERCENTILE,
    MIN_RELEVANCE_SCORE, TIGHT_DISTRIBUTION_ADJUSTMENT,
    SPREAD_DISTRIBUTION_ADJUSTMENT, BALANCED_DISTRIBUTION_ADJUSTMENT,
    COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_DIM, NORMALIZED,
    CACHE_ENABLED, CACHE_TTL,
    get_vector_store_path,
    VECTOR_STORE_SETTINGS
)

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structured search result with metadata and metrics."""
    query: str
    chunks: List[str]
    metrics: Dict
    filter_applied: Optional[Dict] = None
    fallback_reason: Optional[str] = None

@dataclass
class DeduplicationStats:
    """Statistics about chunk deduplication."""
    total_chunks: int = 0
    chunks_with_overlap: int = 0
    total_overlap_removed: int = 0
    articles_processed: int = 0

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

class Retriever:
    """
    Core semantic search implementation with strict adaptive thresholds
    and user-friendly section filtering.
    """
    
    def __init__(self, metrics: Optional[SearchMetrics] = None):
        """
        Initialize the retriever.
        
        Args:
            metrics: Optional SearchMetrics instance for tracking performance
        """
        self.metrics = metrics or SearchMetrics()
        self._vector_store = None
        self._section_map = None  # Cache for section name mapping
        self._dedup_stats = DeduplicationStats()  # Track deduplication statistics
        
        # Load environment variables
        load_dotenv()
        
        # Initialize vector store
        self._initialize_vector_store()
        
    def _initialize_vector_store(self) -> None:
        """
        Initialize the vector store and section mapping.
        
        Raises:
            VectorStoreError: If vector store initialization fails
            ValueError: If required environment variables are missing
        """
        try:
            # Verify OpenAI API key is available
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            # Initialize embedding function
            embedding_function = OpenAIEmbeddings(
                model=VECTOR_STORE_SETTINGS["embedding_model"],
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                chunk_size=VECTOR_STORE_SETTINGS["chunk_size"]
            )
            
            # Get vector store path
            persist_directory = get_vector_store_path()
            if not persist_directory.exists():
                raise VectorStoreError(f"Vector store directory not found: {persist_directory}")
            
            # Initialize vector store
            self._vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_function,
                persist_directory=str(persist_directory)
            )
            
            # Verify connection and collection
            try:
                collection = self._vector_store._collection
                count = collection.count()
                if count == 0:
                    raise VectorStoreError("Vector store is empty")
                logger.info(f"Successfully connected to vector store with {count} documents")
            except Exception as e:
                raise VectorStoreError(f"Failed to verify vector store connection: {str(e)}")
            
            # Build section mapping for user-friendly filtering
            self._build_section_map()
            
        except ValueError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise VectorStoreError(f"Vector store initialization failed: {str(e)}")
    
    def _build_section_map(self) -> None:
        """Build mapping of partial section names to full names."""
        try:
            # Get all unique sections from vector store
            sections = set()
            for doc in self._vector_store.get():
                # Chroma returns a dict with 'metadata' key
                if isinstance(doc, dict) and 'metadata' in doc:
                    if 'section' in doc['metadata']:
                        sections.add(doc['metadata']['section'])
                # Handle Document objects
                elif hasattr(doc, 'metadata') and 'section' in doc.metadata:
                    sections.add(doc.metadata['section'])
            
            # Build mapping (e.g., "Case Studies" -> "1 Case Studies")
            self._section_map = {}
            for full_name in sections:
                # Remove numbers and clean
                base_name = re.sub(r'^\d+\s*', '', full_name).strip().lower()
                self._section_map[base_name] = full_name
                
            logger.info(f"Built section map with {len(self._section_map)} entries")
            
        except Exception as e:
            logger.error(f"Failed to build section map: {str(e)}")
            raise VectorStoreError("Failed to build section mapping")
    
    def _match_section(self, section_name: str) -> str:
        """
        Match user-friendly section name to full section name.
        
        Args:
            section_name: User-provided section name
            
        Returns:
            Full section name with index
            
        Raises:
            SectionNotFound: If no matching section is found
        """
        if not section_name:
            return None
            
        # Try exact match first
        if section_name in self._section_map.values():
            return section_name
            
        # Try case-insensitive match
        base_name = section_name.lower().strip()
        if base_name in self._section_map:
            return self._section_map[base_name]
            
        # If no match, raise with available sections
        available = sorted(self._section_map.values())
        raise SectionNotFound(
            f"Section '{section_name}' not found. Available sections: {available}"
        )
    
    def _calculate_threshold(self, scores: List[float], is_fallback: bool = False) -> float:
        """
        Calculate adaptive threshold based on score distribution.
        
        Args:
            scores: List of similarity scores
            is_fallback: Whether this is for fallback search
            
        Returns:
            Calculated threshold value
        """
        if not scores:
            raise ThresholdError("No scores provided for threshold calculation")
            
        # Calculate base threshold
        percentile = FALLBACK_THRESHOLD_PERCENTILE if is_fallback else TIGHT_THRESHOLD_PERCENTILE
        base_threshold = np.percentile(scores, percentile)
        
        # Calculate distribution metrics
        std_dev = np.std(scores)
        
        # Apply distribution-based adjustment
        if std_dev < 0.05:  # Tight distribution
            adjustment = TIGHT_DISTRIBUTION_ADJUSTMENT
        elif std_dev > 0.2:  # Spread distribution
            adjustment = SPREAD_DISTRIBUTION_ADJUSTMENT
        else:  # Balanced distribution
            adjustment = BALANCED_DISTRIBUTION_ADJUSTMENT
            
        # Calculate final threshold
        threshold = min(0.85, base_threshold + adjustment)
        
        logger.info(
            f"Calculated {'fallback' if is_fallback else 'tight'} threshold: "
            f"{threshold:.3f} (base: {base_threshold:.3f}, std: {std_dev:.3f})"
        )
        
        return threshold
    
    def _filter_by_threshold(
        self,
        docs: List[Document],
        scores: List[float],
        threshold: float
    ) -> Tuple[List[Document], List[float]]:
        """
        Filter documents by threshold and maintain article-level context.
        
        Args:
            docs: List of documents with scores
            scores: List of similarity scores
            threshold: Threshold to filter by
            
        Returns:
            Filtered documents and scores, maintaining article context
        """
        if not docs or not scores:
            return [], []
            
        # Get unique articles from above-threshold chunks and their max scores
        article_scores = defaultdict(float)
        for doc, score in zip(docs, scores):
            if "url" in doc.metadata:
                article_scores[doc.metadata["url"]] = max(article_scores[doc.metadata["url"]], score)
        
        # Sort articles by their max scores
        sorted_articles = sorted(
            article_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get ALL chunks from relevant articles in score order
        all_chunks = []
        all_scores = []
        
        # First, get all chunks from the vector store for relevant articles
        for url, max_score in sorted_articles:
            # Query the vector store for all chunks from this article
            article_chunks = self._vector_store.get(
                where={"url": url},
                include=["documents", "metadatas"]
            )
            
            # Add all chunks from this article
            for doc, metadata in zip(article_chunks["documents"], article_chunks["metadatas"]):
                chunk = Document(
                    page_content=doc,
                    metadata=metadata
                )
                all_chunks.append(chunk)
                # Use the original score if available, otherwise use the article's max score
                original_score = next(
                    (score for d, score in zip(docs, scores) 
                     if d.metadata.get("url") == url 
                     and d.metadata.get("chunk_index") == metadata.get("chunk_index")),
                    max_score  # Use article's max score for chunks without original scores
                )
                all_scores.append(original_score)
        
        # Sort chunks by article URL (which maintains score order) and chunk index
        sorted_pairs = sorted(
            zip(all_chunks, all_scores),
            key=lambda x: (
                # First sort by article URL to maintain score order
                next((i for i, (url, _) in enumerate(sorted_articles) if url == x[0].metadata.get("url", "")), 0),
                # Then sort by chunk index within each article
                x[0].metadata.get("chunk_index", 0)
            )
        )
        
        # Unzip the sorted pairs
        filtered_docs, filtered_scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        return list(filtered_docs), list(filtered_scores)
    
    def _perform_search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        k: int = DEFAULT_K
    ) -> Tuple[List[Document], List[float], Dict]:
        """
        Perform semantic search with optional filtering.
        
        Args:
            query: Search query
            filter_dict: Optional filter dictionary
            k: Number of results to retrieve
            
        Returns:
            Tuple of (documents, scores, metrics)
        """
        try:
            # Handle section filtering
            if filter_dict and "section" in filter_dict:
                section = self._match_section(filter_dict["section"])
                if section:
                    filter_dict = {"section": section}
            
            # First, perform semantic search to find relevant chunks
            docs_and_scores = self._vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
            
            if not docs_and_scores:
                return [], [], {"threshold_type": "none", "fallback_reason": "No results found"}
                
            # Unzip results
            docs, scores = zip(*docs_and_scores)
            docs, scores = list(docs), list(scores)
            
            # Calculate threshold
            threshold = self._calculate_threshold(scores)
            
            # Get all chunks from articles that have any chunks above threshold
            filtered_docs, filtered_scores = self._filter_by_threshold(docs, scores, threshold)
            
            # If no results after filtering, try fallback
            if not filtered_docs:
                logger.info("No results after tight threshold, trying fallback")
                
                # Calculate fallback threshold
                fallback_threshold = self._calculate_threshold(scores, is_fallback=True)
                
                # Get all chunks from articles that have any chunks above fallback threshold
                filtered_docs, filtered_scores = self._filter_by_threshold(
                    docs, scores, fallback_threshold
                )
                
                metrics = {
                    "threshold_type": "fallback",
                    "threshold_used": fallback_threshold,
                    "fallback_reason": "No results met tight threshold"
                }
            else:
                metrics = {
                    "threshold_type": "tight",
                    "threshold_used": threshold,
                    "fallback_reason": None
                }
            
            # Log the number of chunks retrieved
            logger.info(
                f"Retrieved {len(filtered_docs)} chunks from {len(set(d.metadata['url'] for d in filtered_docs))} articles"
            )
            
            return filtered_docs, filtered_scores, metrics
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise RetrievalError(f"Search failed: {str(e)}")
    
    def _find_overlap(self, prev_content: str, current_content: str, min_overlap: int = MIN_OVERLAP_WORDS) -> Tuple[str, int]:
        """
        Find overlapping content between chunks and return the overlap and its length.
        
        Args:
            prev_content: Content of previous chunk
            current_content: Content of current chunk
            min_overlap: Minimum number of words to consider as overlap
            
        Returns:
            Tuple of (overlap text, overlap length)
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
                    
                    if len(overlap) >= min_overlap:
                        overlap_text = " ".join(overlap)
                        return overlap_text, len(overlap)
        
        return "", 0

    def _deduplicate_chunks(self, chunks: List[Document]) -> Tuple[List[Document], DeduplicationStats]:
        """
        Remove overlapping content between chunks while preserving context.
        
        Args:
            chunks: List of Document objects sorted by chunk index
            
        Returns:
            Tuple of (deduplicated Document objects, deduplication statistics)
        """
        if not chunks:
            return chunks, DeduplicationStats()
            
        stats = DeduplicationStats()
        stats.total_chunks = len(chunks)
        stats.articles_processed = 1  # Since we process one article at a time
            
        deduped_chunks = []
        prev_content = ""
        
        for chunk in chunks:
            current_content = chunk.page_content
            
            # Find and handle overlap
            overlap, overlap_length = self._find_overlap(prev_content, current_content)
            
            if overlap:
                stats.chunks_with_overlap += 1
                stats.total_overlap_removed += overlap_length
                
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
        
        return deduped_chunks, stats

    def _format_article(self, chunks: List[Document]) -> Tuple[str, DeduplicationStats]:
        """
        Format a complete article from its chunks.
        
        Args:
            chunks: List of Document objects belonging to the same article
            
        Returns:
            Tuple of (formatted article string, deduplication statistics)
        """
        if not chunks:
            return "", DeduplicationStats()
            
        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: x.metadata["chunk_index"])
        
        # Deduplicate chunks to remove overlapping content
        deduped_chunks, stats = self._deduplicate_chunks(sorted_chunks)
        
        # Get article metadata from first chunk
        metadata = deduped_chunks[0].metadata
        
        # Build article header
        header_parts = []
        
        # Add article title if available
        if "title" in metadata:
            header_parts.append(f"Article: {metadata['title']}")
        else:
            # Extract title from URL if not in metadata
            url = metadata.get("url", "")
            title = url.split("/")[-1].replace("-", " ").title()
            header_parts.append(f"Article: {title}")
            
        # Add section if available
        if "section" in metadata:
            # Format section name more clearly
            section_parts = metadata["section"].split(" ", 1)
            if len(section_parts) == 2:
                section_num, section_name = section_parts
                section = f"Section {section_num}: {section_name}"
            else:
                section = metadata["section"]
            header_parts.append(section)
            
        # Add source URL if available
        if "url" in metadata:
            header_parts.append(f"Source: {metadata['url']}")
            
        # Create article header
        header = "--- " + " | ".join(header_parts) + " ---\n\n"
        
        # Join all deduplicated chunk content
        content = "\n".join(chunk.page_content for chunk in deduped_chunks)
        
        return header + content, stats

    def _assemble_articles(self, docs: List[Document]) -> Tuple[List[str], DeduplicationStats]:
        """
        Assemble chunks into complete articles.
        
        Args:
            docs: List of Document objects from search
            
        Returns:
            Tuple of (list of formatted article strings, deduplication statistics)
        """
        # Group chunks by article URL
        articles = defaultdict(list)
        for doc in docs:
            if "url" in doc.metadata:
                articles[doc.metadata["url"]].append(doc)
        
        # Log chunk distribution
        logger.info(f"Found {len(articles)} articles with chunks:")
        for url, chunks in articles.items():
            chunk_indices = sorted(c.metadata.get("chunk_index", 0) for c in chunks)
            logger.info(f"Article {url}: {len(chunks)} chunks with indices {chunk_indices}")
        
        # Format each article and collect stats
        formatted_articles = []
        total_stats = DeduplicationStats()
        
        for url, chunks in articles.items():
            # Sort chunks by index to ensure proper order
            sorted_chunks = sorted(chunks, key=lambda x: x.metadata.get("chunk_index", 0))
            
            # Log chunk sequence
            chunk_indices = [c.metadata.get("chunk_index", 0) for c in sorted_chunks]
            logger.info(f"Processing article {url} with chunks in order: {chunk_indices}")
            
            article, stats = self._format_article(sorted_chunks)
            if article:
                formatted_articles.append(article)
                # Aggregate stats
                total_stats.total_chunks += stats.total_chunks
                total_stats.chunks_with_overlap += stats.chunks_with_overlap
                total_stats.total_overlap_removed += stats.total_overlap_removed
                total_stats.articles_processed += stats.articles_processed
        
        return formatted_articles, total_stats

    def _extract_filter_from_query(self, query: str) -> Tuple[str, Optional[Dict]]:
        """
        Extract section filter from query if present in format '| section: Name'.
        
        Args:
            query: Search query that may contain embedded filter
            
        Returns:
            Tuple of (clean query, filter dictionary if found)
        """
        # Look for section filter pattern
        filter_pattern = r'\|\s*section:\s*([^|]+)'
        match = re.search(filter_pattern, query)
        
        if match:
            # Extract section name and clean query
            section_name = match.group(1).strip()
            clean_query = query[:match.start()].strip()
            
            # Try to match section name
            try:
                full_section = self._match_section(section_name)
                return clean_query, {"section": full_section}
            except SectionNotFound as e:
                logger.warning(f"Section filter not found: {str(e)}")
                return query, None
        
        return query, None

    def search(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        k: int = DEFAULT_K
    ) -> SearchResult:
        """
        Perform semantic search with optional filtering.
        
        Args:
            query: Search query (may include embedded filter like '| section: Name')
            filter_dict: Optional filter dictionary (alternative to embedded filter)
            k: Number of results to retrieve
            
        Returns:
            SearchResult with complete articles (chunks reassembled) and metrics
            
        Raises:
            RetrievalError: For general retrieval errors
            NoResultsFound: When no results are found
            InvalidFilter: For invalid filter format
            SectionNotFound: When section filter doesn't match
            VectorStoreError: For vector store issues
            ThresholdError: For threshold calculation issues
        """
        start_time = datetime.now()
        
        try:
            # Extract filter from query if present
            clean_query, embedded_filter = self._extract_filter_from_query(query)
            
            # Use embedded filter if present, otherwise use filter_dict
            effective_filter = embedded_filter or filter_dict
            
            # Track query in metrics
            with self.metrics.track_query(clean_query):
                # Perform search with clean query and effective filter
                docs, scores, metrics = self._perform_search(clean_query, effective_filter, k)
                
                if not docs:
                    raise NoResultsFound("No relevant results found")
                
                # Assemble chunks into complete articles
                formatted_articles, dedup_stats = self._assemble_articles(docs)
                
                # Calculate response time
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Update metrics with deduplication stats
                metrics.update({
                    "response_time": response_time,
                    "total_articles": len(formatted_articles),
                    "total_chunks": len(docs),
                    "filter_applied": effective_filter,
                    "original_query": query,  # Keep original query with embedded filter
                    "clean_query": clean_query,  # Query without filter
                    "deduplication_stats": {
                        "total_chunks": dedup_stats.total_chunks,
                        "chunks_with_overlap": dedup_stats.chunks_with_overlap,
                        "total_overlap_removed": dedup_stats.total_overlap_removed,
                        "articles_processed": dedup_stats.articles_processed
                    }
                })
                
                # Create result
                result = SearchResult(
                    query=clean_query,  # Use clean query in result
                    chunks=formatted_articles,
                    metrics=metrics,
                    filter_applied=effective_filter
                )
                
                # Log success with deduplication stats
                logger.info(
                    f"Search successful: {len(formatted_articles)} articles, "
                    f"{len(docs)} chunks, threshold: {metrics['threshold_type']}, "
                    f"time: {response_time:.2f}s\n"
                    f"Query: {query} -> {clean_query} "
                    f"(filter: {effective_filter})\n"
                    f"Deduplication: {dedup_stats.chunks_with_overlap}/{dedup_stats.total_chunks} "
                    f"chunks had overlap, removed {dedup_stats.total_overlap_removed} words"
                )
                
                return result
                
        except Exception as e:
            # Log error
            logger.error(f"Search failed: {str(e)}")
            
            # Update metrics
            self.metrics.record_error(str(e))
            
            # Re-raise with context
            if isinstance(e, RetrievalError):
                raise
            raise RetrievalError(f"Search failed: {str(e)}")
    
    def get_available_sections(self) -> List[str]:
        """
        Get list of available section names.
        
        Returns:
            List of full section names
        """
        return sorted(self._section_map.values()) 