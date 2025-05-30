import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from loguru import logger
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import re
import sys
import numpy as np
from datetime import datetime, UTC
import hashlib
import chromadb

# Configure logging
logger.add(
    "process content/Embedding/logs/query_test.log",
    rotation="100 MB",
    retention="1 week",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Add evaluation logging
logger.add(
    "process content/Embedding/logs/evaluation.log",
    rotation="100 MB",
    retention="1 month",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

class SearchMetrics:
    """Track and analyze search performance metrics."""
    
    def __init__(self):
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
            "successful_citations": set()
        }
    
    def start_query(self, query: str):
        """Start tracking a new query."""
        self.query_id = datetime.now(UTC).isoformat()
        self.query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
        self.start_time = datetime.now(UTC)
        self.metrics["query"] = query
        logger.info(f"Starting query {self.query_hash}")
    
    def end_query(self):
        """End tracking and log metrics."""
        if self.start_time:
            self.metrics["response_time"] = (datetime.now(UTC) - self.start_time).total_seconds()
        
        # Log to evaluation log
        logger.bind(query_id=self.query_id, query_hash=self.query_hash).info(
            "Query Metrics",
            extra={
                "metrics": self.metrics,
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
    
    def add_feedback(self, rating: int, feedback: str = None):
        """Add human feedback for the query."""
        self.metrics["feedback"] = {
            "rating": rating,
            "comment": feedback,
            "timestamp": datetime.now(UTC).isoformat()
        }
        logger.bind(query_id=self.query_id).info(
            f"Feedback received: {rating}/5 - {feedback}"
        )
    
    def track_chunk_usage(self, chunk: Document, score: float):
        """Track which chunks were used in the response."""
        chunk_id = f"{chunk.metadata['url']}_{chunk.metadata['chunk_index']}"
        self.metrics["used_chunks"].add(chunk_id)
        self.metrics["combined_scores"].append(score)
    
    def track_citation(self, url: str):
        """Track successful citations."""
        self.metrics["successful_citations"].add(url)

def load_vector_store() -> Chroma:
    """Load the existing vector store."""
    logger.info("Loading vector store from process content/Embedding/embeddings/")
    try:
        # Initialize embeddings with the same configuration as during creation
        embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=None,  # Will use OPENAI_API_KEY from environment
            chunk_size=100  # Process in batches of 100
        )
        
        vector_store = Chroma(
            collection_name="knowledge_base",
            persist_directory="process content/Embedding/embeddings",
            embedding_function=embedding_function
        )
        
        # Verify the collection exists and has documents
        collection = vector_store._collection
        count = collection.count()
        logger.info(f"Successfully loaded vector store with {count} documents")
        
        if count == 0:
            raise ValueError("Vector store is empty! Please run embedder.py first.")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise

def group_results_by_article(results: List[Document]) -> Dict[str, Dict]:
    """Group search results by article URL and deduplicate metadata."""
    articles = defaultdict(lambda: {
        "chunks": [],
        "metadata": None,
        "full_content": ""
    })
    
    for doc in results:
        url = doc.metadata["url"]
        articles[url]["chunks"].append(doc)
        articles[url]["metadata"] = {
            "url": url,
            "section": doc.metadata["section"],
            "article_index": doc.metadata["article_index"]
        }
        articles[url]["full_content"] += doc.page_content + "\n"
    
    return dict(articles)

def get_all_chunks_from_articles(vector_store: Chroma, article_urls: Set[str]) -> Dict[str, List[Document]]:
    """Get all chunks from the specified articles."""
    articles = defaultdict(list)
    
    # Query for all chunks from these articles
    results = vector_store.get(
        where={"url": {"$in": list(article_urls)}},
        include=["documents", "metadatas"]
    )
    
    # Group chunks by article
    for doc, metadata in zip(results["documents"], results["metadatas"]):
        url = metadata["url"]
        articles[url].append(Document(
            page_content=doc,
            metadata=metadata
        ))
    
    # Sort chunks by chunk_index within each article
    for url in articles:
        articles[url].sort(key=lambda x: x.metadata["chunk_index"])
    
    return dict(articles)

def extract_keywords(query: str) -> Set[str]:
    """Extract meaningful keywords from the query."""
    # Remove common words and punctuation
    words = re.findall(r'\b\w+\b', query.lower())
    # Filter out common words (you can expand this list)
    stop_words = {'what', 'are', 'the', 'and', 'for', 'to', 'in', 'on', 'at', 'is', 'was', 'were', 'be', 'been', 'being'}
    return {word for word in words if word not in stop_words and len(word) > 2}

def calculate_keyword_relevance(doc: Document, keywords: Set[str]) -> float:
    """Calculate how many keywords appear in the document."""
    content = doc.page_content.lower()
    matches = sum(1 for keyword in keywords if keyword in content)
    return matches / len(keywords) if keywords else 0

def analyze_overlap(chunks: List[Document]) -> List[Document]:
    """Analyze and handle overlapping content in chunks."""
    if not chunks:
        return chunks
    
    # Sort chunks by index
    sorted_chunks = sorted(chunks, key=lambda x: x.metadata["chunk_index"])
    deduped_chunks = []
    
    for i, chunk in enumerate(sorted_chunks):
        if i == 0:
            deduped_chunks.append(chunk)
            continue
        
        # Get previous chunk's content
        prev_content = sorted_chunks[i-1].page_content
        
        # Find overlap at the start of current chunk
        current_content = chunk.page_content
        overlap = find_overlap(prev_content, current_content)
        
        if overlap:
            # Remove overlapping content from current chunk
            current_content = current_content[len(overlap):].strip()
            if current_content:  # Only add if there's non-overlapping content
                deduped_chunks.append(Document(
                    page_content=current_content,
                    metadata=chunk.metadata
                ))
        else:
            deduped_chunks.append(chunk)
    
    return deduped_chunks

def find_overlap(prev_content: str, current_content: str, min_overlap: int = 20) -> Optional[str]:
    """Find overlapping content between chunks."""
    # Simple implementation - can be improved with more sophisticated text analysis
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
                    return " ".join(overlap)
    
    return None

def calculate_adaptive_threshold(scores: List[float]) -> float:
    """Calculate an adaptive threshold based on score distribution."""
    if not scores:
        return 1.0  # Default threshold if no scores
    
    # Calculate basic statistics
    mean_score = sum(scores) / len(scores)
    std_dev = np.std(scores)
    
    # Use the 20th percentile as base threshold
    percentile_20 = np.percentile(scores, 20)
    
    # Adjust threshold based on distribution
    if std_dev < 0.05:  # Very tight distribution
        return min(1.0, mean_score + 0.1)  # More lenient
    elif std_dev > 0.2:  # Very spread distribution
        return min(1.0, percentile_20 + 0.2)  # More strict
    else:
        return min(1.0, percentile_20 + 0.15)  # Balanced

def smart_search(vector_store: Chroma, query: str, filter_dict: Dict = None, metrics: Optional[SearchMetrics] = None) -> Tuple[List[Document], List[Document]]:
    """Perform smart search with adaptive thresholds and improved fallback."""
    if metrics:
        metrics.start_query(query)
    
    try:
        # Log filter status explicitly
        if filter_dict:
            logger.info("\n" + "="*80)
            logger.info("FILTER STATUS:")
            if "section" in filter_dict:
                logger.info(f"Section filter ACTIVE: {filter_dict['section']}")
                section_docs = vector_store.get(
                    where={"section": filter_dict["section"]},
                    include=["metadatas"]
                )
                logger.info(f"Total documents in this section: {len(section_docs['metadatas'])}")
            if "url" in filter_dict:
                logger.info(f"URL filter ACTIVE: {filter_dict['url']}")
            logger.info("="*80 + "\n")
        else:
            logger.info("\n" + "="*80)
            logger.info("FILTER STATUS: No filters applied - searching all documents")
            logger.info("="*80 + "\n")

        # Extract keywords for hybrid matching
        keywords = extract_keywords(query)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Stage 1: Initial search with tighter thresholds
        initial_results = vector_store.similarity_search_with_score(
            query,
            k=10,  # Reduced from 30 to get more focused results
            filter=filter_dict
        )
        
        if not initial_results:
            logger.info("No results from exact query, trying with main keywords...")
            main_keywords = [k for k in keywords if len(k) > 3]
            if main_keywords:
                keyword_query = " ".join(main_keywords)
                logger.info(f"Trying search with keywords: {keyword_query}")
                initial_results = vector_store.similarity_search_with_score(
                    keyword_query,
                    k=10,
                    filter=filter_dict
                )
        
        if not initial_results:
            logger.warning("No results returned from any search attempt")
            return [], []
        
        # Log initial results and analyze scores
        logger.info("\nDistance Score Analysis (lower is better):")
        scores = [score for _, score in initial_results]
        logger.info(f"Best (lowest) score: {min(scores):.4f}")
        logger.info(f"Worst (highest) score: {max(scores):.4f}")
        logger.info(f"Mean score: {sum(scores)/len(scores):.4f}")
        logger.info(f"Median score: {sorted(scores)[len(scores)//2]:.4f}")
        logger.info(f"Standard deviation: {np.std(scores):.4f}")
        
        # Calculate tighter adaptive threshold
        # Use 10th percentile instead of 20th for stricter filtering
        percentile_10 = np.percentile(scores, 10)
        std_dev = np.std(scores)
        
        # Adjust threshold based on distribution
        if std_dev < 0.05:  # Very tight distribution
            distance_threshold = min(0.85, percentile_10 + 0.05)  # More lenient but still strict
        elif std_dev > 0.2:  # Very spread distribution
            distance_threshold = min(0.9, percentile_10 + 0.1)  # More strict
        else:
            distance_threshold = min(0.85, percentile_10 + 0.08)  # Balanced but strict
        
        logger.info(f"\nAdaptive threshold analysis:")
        logger.info(f"Calculated threshold: {distance_threshold:.4f}")
        
        # Get unique articles from initial results that meet the tighter threshold
        article_urls = {doc.metadata["url"] for doc, score in initial_results if score < distance_threshold}
        
        # If no articles meet the tighter threshold, try with original thresholds as fallback
        if not article_urls:
            logger.info("No articles met tighter threshold, trying with original thresholds...")
            # Calculate original threshold (20th percentile)
            percentile_20 = np.percentile(scores, 20)
            if std_dev < 0.05:
                distance_threshold = min(1.0, mean_score + 0.1)
            elif std_dev > 0.2:
                distance_threshold = min(1.0, percentile_20 + 0.2)
            else:
                distance_threshold = min(1.0, percentile_20 + 0.15)
            
            logger.info(f"Using fallback threshold: {distance_threshold:.4f}")
            article_urls = {doc.metadata["url"] for doc, score in initial_results if score < distance_threshold}
            
            # If still no results, take top 2 results as final fallback
            if not article_urls and len(initial_results) >= 2:
                logger.info("No articles met fallback threshold, using top 2 results")
                article_urls = {doc.metadata["url"] for doc, _ in initial_results[:2]}
        
        # Log filter impact on results
        if filter_dict:
            logger.info("\n" + "="*80)
            logger.info("FILTER IMPACT:")
            logger.info(f"Articles found after filtering: {len(article_urls)}")
            if "section" in filter_dict:
                logger.info(f"Section '{filter_dict['section']}' filter reduced results to {len(article_urls)} articles")
            if "url" in filter_dict:
                logger.info(f"URL filter restricted results to 1 specific article")
            logger.info("="*80 + "\n")
        
        logger.info(f"Found {len(article_urls)} potential articles")
        
        if not article_urls:
            logger.warning("No articles found even with fallback")
            return [], []
        
        # Get ALL chunks from these articles (full article retrieval)
        all_chunks = get_all_chunks_from_articles(vector_store, article_urls)
        
        # Stage 2: Focused search within these articles to identify relevant chunks
        relevant_chunk_urls = set()  # Track URLs of articles with relevant chunks
        chunk_scores_by_url = defaultdict(list)  # Track scores for each chunk by URL
        
        for url, chunks in all_chunks.items():
            # Calculate both semantic and keyword relevance for each chunk
            for chunk in chunks:
                # Get semantic score (if available in initial results)
                semantic_score = next(
                    (score for doc, score in initial_results 
                     if doc.metadata["url"] == url 
                     and doc.metadata["chunk_index"] == chunk.metadata["chunk_index"]),
                    0.5  # Default score if not found
                )
                
                # Calculate keyword relevance
                keyword_score = calculate_keyword_relevance(chunk, keywords)
                
                # Combined score with adjusted weights
                semantic_score_normalized = 1.0 - min(semantic_score, 1.0)
                combined_score = (0.7 * semantic_score_normalized) + (0.3 * keyword_score)
                chunk_scores_by_url[url].append((chunk, combined_score))
            
            # Sort chunks by combined score
            chunk_scores_by_url[url].sort(key=lambda x: x[1], reverse=True)
            
            # If any chunk meets the threshold, mark the article as relevant
            if any(score > 0.3 for _, score in chunk_scores_by_url[url]):
                relevant_chunk_urls.add(url)
        
        # Get all chunks from articles that have any relevant chunks
        relevant_chunks = []
        for url in relevant_chunk_urls:
            # Add all chunks from this article
            relevant_chunks.extend([chunk for chunk, _ in chunk_scores_by_url[url]])
        
        # Sort all chunks by article and chunk index
        all_chunks_list = []
        for url, chunks in all_chunks.items():
            all_chunks_list.extend(sorted(chunks, key=lambda x: x.metadata["chunk_index"]))
        
        # Track metrics
        if metrics:
            metrics.metrics["total_articles"] = len(article_urls)
            metrics.metrics["total_chunks"] = sum(len(chunks) for chunks in all_chunks.values())
            metrics.metrics["relevant_chunks"] = len(relevant_chunks)
            metrics.metrics["distance_scores"] = scores
            metrics.metrics["threshold_used"] = distance_threshold
            
            # Track chunk usage
            for url in relevant_chunk_urls:
                for chunk, score in chunk_scores_by_url[url]:
                    metrics.track_chunk_usage(chunk, score)
        
        # Handle overlapping content
        relevant_chunks = analyze_overlap(relevant_chunks)
        all_chunks_list = analyze_overlap(all_chunks_list)
        
        return relevant_chunks, all_chunks_list
        
    except Exception as e:
        logger.error(f"Error in smart search: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        return [], []
    finally:
        if metrics:
            metrics.end_query()

def run_test_queries(vector_store: Chroma):
    """Run a series of test queries to verify functionality."""
    test_queries = [
        {
            "query": "What are the benefits of shrink wrapping for food businesses?",
            "filter": None,
            "description": "Basic semantic search without filters"
        },
        {
            "query": "Tell me about Celtic Fish and Game's business model",
            "filter": {"section": "1 Case Studies"},
            "description": "Semantic search with section filter"
        },
        {
            "query": "What packaging solutions were provided to Hogsbottom?",
            "filter": {"url": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/case-study-hogsbottom-garden-delights"},
            "description": "Semantic search with URL filter"
        }
    ]
    
    for test in test_queries:
        logger.info("\n" + "="*80)
        logger.info(f"Running test: {test['description']}")
        logger.info(f"Query: {test['query']}")
        if test['filter']:
            logger.info(f"Filter: {test['filter']}")
        
        try:
            # Perform smart search
            relevant_chunks, all_chunks = smart_search(vector_store, test['query'], test['filter'])
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found. This might indicate a problem with the search.")
                continue
            
            # Group chunks by article for display
            articles = defaultdict(list)
            for chunk in all_chunks:
                url = chunk.metadata["url"]
                articles[url].append(chunk)
            
            # Calculate statistics
            total_articles = len(articles)
            total_chunks = sum(len(chunks) for chunks in articles.values())
            total_relevant_chunks = len(relevant_chunks)
            
            # Get article relevance scores
            article_scores = defaultdict(float)
            for chunk in relevant_chunks:
                url = chunk.metadata["url"]
                # Count how many relevant chunks each article has
                article_scores[url] += 1
            
            # Sort articles by relevance
            sorted_articles = sorted(
                articles.items(),
                key=lambda x: article_scores[x[0]],
                reverse=True
            )
            
            # Log detailed results
            logger.info("\nDetailed Results:")
            for i, (url, chunks) in enumerate(sorted_articles, 1):
                relevance_score = article_scores[url]
                logger.info(f"\nArticle {i} (Relevance Score: {relevance_score:.1f}):")
                logger.info(f"URL: {url}")
                logger.info(f"Section: {chunks[0].metadata['section']}")
                logger.info(f"Total chunks in article: {len(chunks)}")
                
                # Show which chunks were most relevant
                relevant_chunk_indices = [
                    chunk.metadata["chunk_index"] 
                    for chunk in relevant_chunks 
                    if chunk.metadata["url"] == url
                ]
                logger.info(f"Most relevant chunk indices: {sorted(relevant_chunk_indices)}")
                
                # Show content preview of the first chunk
                logger.info(f"Content preview: {chunks[0].page_content[:200]}...")
            
            # Log summary
            logger.info("\n" + "="*80)
            logger.info("SEARCH SUMMARY:")
            logger.info(f"Total articles found: {total_articles}")
            logger.info(f"Total chunks across all articles: {total_chunks}")
            logger.info(f"Total relevant chunks identified: {total_relevant_chunks}")
            
            # Show top 3 most relevant articles
            logger.info("\nTop 3 most relevant articles:")
            for i, (url, _) in enumerate(sorted_articles[:3], 1):
                score = article_scores[url]
                logger.info(f"{i}. {url} (Relevance Score: {score:.1f})")
            
            logger.info("="*80 + "\n")
                
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            continue

def get_full_section_name(vector_store: Chroma, section_name: str) -> Optional[str]:
    """Find the full section name (with index) from a partial section name."""
    # Get all unique sections from the vector store
    all_sections = set()
    for doc in vector_store.get()["metadatas"]:
        if "section" in doc:
            all_sections.add(doc["section"])
    
    # Try to find a matching section
    section_name = section_name.lower()
    for full_section in all_sections:
        # Remove the index and compare
        section_without_index = " ".join(full_section.split()[1:]).lower()
        if section_name in section_without_index or section_without_index in section_name:
            return full_section
    
    return None

def interactive_query(vector_store: Chroma):
    """Run interactive queries with feedback collection."""
    metrics = SearchMetrics()
    
    logger.info("\nStarting interactive query mode. Type 'exit' to quit.")
    logger.info("You can also filter by section or URL using these formats:")
    logger.info("  Query: your question here")
    logger.info("  Query with section filter: your question here | section:Section Name")
    logger.info("  Query with URL filter: your question here | url:http://example.com/article")
    logger.info("\nAfter each response, you can provide feedback:")
    logger.info("  Type 'rate X' where X is 1-5 to rate the response")
    logger.info("  Type 'feedback: your comment' to add a comment")
    logger.info("\nFilter status will be explicitly shown for each query")
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter your query (or 'exit' to quit): ").strip()
            
            if user_input.lower() == 'exit':
                logger.info("Exiting interactive query mode.")
                break
            
            # Handle feedback
            if user_input.lower().startswith('rate '):
                try:
                    rating = int(user_input.split()[1])
                    if 1 <= rating <= 5:
                        feedback = input("Add a comment (optional): ").strip()
                        metrics.add_feedback(rating, feedback)
                        continue
                except (ValueError, IndexError):
                    logger.warning("Invalid rating format. Use 'rate X' where X is 1-5")
                    continue
            
            if user_input.lower().startswith('feedback:'):
                metrics.add_feedback(None, user_input[9:].strip())
                continue
            
            # Parse filter if present
            query = user_input
            filter_dict = None
            
            if "|" in user_input:
                query, filter_str = user_input.split("|", 1)
                query = query.strip()
                filter_str = filter_str.strip()
                
                if filter_str.startswith("section:"):
                    section = filter_str[8:].strip()
                    # Find the full section name
                    full_section = get_full_section_name(vector_store, section)
                    if full_section:
                        filter_dict = {"section": full_section}
                        logger.info(f"\nFilter detected: Section filter")
                        logger.info(f"Input section name: '{section}'")
                        logger.info(f"Matched to full section: '{full_section}'")
                    else:
                        logger.warning(f"\nSection filter failed: '{section}' not found")
                        logger.warning("Available sections (without indices):")
                        # Show available sections without their indices
                        all_sections = set()
                        for doc in vector_store.get()["metadatas"]:
                            if "section" in doc:
                                section_without_index = " ".join(doc["section"].split()[1:])
                                all_sections.add(section_without_index)
                        for s in sorted(all_sections):
                            logger.info(f"  - {s}")
                        continue
                elif filter_str.startswith("url:"):
                    url = filter_str[4:].strip()
                    filter_dict = {"url": url}
                    logger.info(f"\nFilter detected: URL filter")
                    logger.info(f"Filtering to specific article: {url}")
            
            # Run the query with metrics
            relevant_chunks, all_chunks = smart_search(vector_store, query, filter_dict, metrics)
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found. Try rephrasing your query.")
                continue
            
            # Group chunks by article for display
            articles = defaultdict(list)
            for chunk in all_chunks:
                url = chunk.metadata["url"]
                articles[url].append(chunk)
            
            # Calculate statistics
            total_articles = len(articles)
            total_chunks = sum(len(chunks) for chunks in articles.values())
            total_relevant_chunks = len(relevant_chunks)
            
            # Get article relevance scores
            article_scores = defaultdict(float)
            for chunk in relevant_chunks:
                url = chunk.metadata["url"]
                article_scores[url] += 1
            
            # Sort articles by relevance
            sorted_articles = sorted(
                articles.items(),
                key=lambda x: article_scores[x[0]],
                reverse=True
            )
            
            # Log summary
            logger.info("\n" + "="*80)
            logger.info("SEARCH SUMMARY:")
            logger.info(f"Total articles found: {total_articles}")
            logger.info(f"Total chunks across all articles: {total_chunks}")
            logger.info(f"Total relevant chunks identified: {total_relevant_chunks}")
            
            # Show top 3 most relevant articles
            logger.info("\nTop 3 most relevant articles:")
            for i, (url, chunks) in enumerate(sorted_articles[:3], 1):
                score = article_scores[url]
                logger.info(f"\n{i}. {url} (Relevance Score: {score:.1f})")
                logger.info(f"   Section: {chunks[0].metadata['section']}")
                logger.info(f"   Content preview: {chunks[0].page_content[:200]}...")
            
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            continue

def debug_list_collections():
    """List all Chroma collections and their document counts for debugging."""
    try:
        client = chromadb.PersistentClient(path="process content/Embedding/embeddings")
        collections = client.list_collections()
        print("\n[DEBUG] Chroma collections in embeddings directory:")
        for col in collections:
            print(f"\nCollection: {col.name}")
            print(f"- Document count: {col.count()}")
            # Get a sample of documents to check for duplicates
            try:
                sample = col.get(limit=5)
                print("- Sample document IDs:", sample['ids'][:5])
                print("- Sample metadatas:", [m.get('url', 'NO_URL') for m in sample['metadatas'][:5]])
            except Exception as e:
                print(f"- Error getting sample: {str(e)}")
    except Exception as e:
        print(f"[DEBUG] Error listing collections: {str(e)}")

def main():
    """Main execution function."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Load vector store
        vector_store = load_vector_store()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--test":
                # Run test queries
                run_test_queries(vector_store)
            elif sys.argv[1] == "--query":
                # Run single query from command line
                if len(sys.argv) < 3:
                    print("Error: Please provide a query after --query")
                    print("Usage: python test_queries.py --query 'your query here'")
                    sys.exit(1)
                query = " ".join(sys.argv[2:])  # Join all arguments after --query
                metrics = SearchMetrics()
                metrics.start_query(query)
                
                # Run the search
                relevant_chunks, all_chunks = smart_search(vector_store, query, None, metrics)
                
                # Group chunks by article for display
                articles = defaultdict(list)
                for chunk in all_chunks:
                    url = chunk.metadata["url"]
                    articles[url].append(chunk)
                
                # Calculate statistics
                total_articles = len(articles)
                total_chunks = sum(len(chunks) for chunks in articles.values())
                total_relevant_chunks = len(relevant_chunks)
                
                # Get article relevance scores
                article_scores = defaultdict(float)
                for chunk in relevant_chunks:
                    url = chunk.metadata["url"]
                    article_scores[url] += 1
                
                # Sort articles by relevance
                sorted_articles = sorted(
                    articles.items(),
                    key=lambda x: article_scores[x[0]],
                    reverse=True
                )
                
                # Print summary
                print("\n" + "="*80)
                print("SEARCH SUMMARY:")
                print(f"Total articles found: {total_articles}")
                print(f"Total chunks across all articles: {total_chunks}")
                print(f"Total relevant chunks identified: {total_relevant_chunks}")
                
                # Show top 3 most relevant articles
                print("\nTop 3 most relevant articles:")
                for i, (url, chunks) in enumerate(sorted_articles[:3], 1):
                    score = article_scores[url]
                    print(f"\n{i}. {url} (Relevance Score: {score:.1f})")
                    print(f"   Section: {chunks[0].metadata['section']}")
                    print(f"   Content preview: {chunks[0].page_content[:200]}...")
                
                print("="*80)
                
                # End metrics tracking
                metrics.end_query()
            else:
                print("Unknown argument. Use:")
                print("  --test     : Run test queries")
                print("  --query    : Run a single query (e.g., --query 'your query here')")
                print("  (no args)  : Run in interactive mode")
                sys.exit(1)
        else:
            # Run interactive mode
            interactive_query(vector_store)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    debug_list_collections()
    main() 