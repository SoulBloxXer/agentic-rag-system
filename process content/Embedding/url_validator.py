import json
from pathlib import Path
from typing import Dict, Set, List, Tuple
from loguru import logger
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse, urljoin
import re
import sys
import traceback

# Configure logging
logger.add(
    "process content/Embedding/logs/url_validation.log",
    rotation="100 MB",
    retention="1 week",
    level="DEBUG",  # Changed to DEBUG for more detailed logging
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def load_vector_store() -> Chroma:
    """Load the existing vector store."""
    logger.info("Loading vector store from process content/Embedding/embeddings/")
    try:
        logger.debug("Initializing OpenAI embeddings...")
        embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=None,  # Will use OPENAI_API_KEY from environment
            chunk_size=100
        )
        
        logger.debug("Initializing Chroma vector store...")
        vector_store = Chroma(
            collection_name="knowledge_base",
            persist_directory="process content/Embedding/embeddings",
            embedding_function=embedding_function
        )
        
        # Verify the collection exists and has documents
        logger.debug("Verifying collection...")
        collection = vector_store._collection
        count = collection.count()
        logger.info(f"Successfully loaded vector store with {count} documents")
        
        if count == 0:
            raise ValueError("Vector store is empty! Please run embedder.py first.")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise

def extract_urls_from_kb(file_path: str) -> Set[str]:
    """Extract URLs from knowledge base file.
    Looks for URLs in markdown-style headers: > **URL:** <http://...>"""
    urls = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find all URLs in the format > **URL:** <http://...>
        url_pattern = r'>\s*\*\*URL:\*\*\s*<([^>]+)>'
        matches = re.finditer(url_pattern, content)
        
        for match in matches:
            url = match.group(1).strip()
            if url:
                urls.add(url)
                
        logger.info(f"Found {len(urls)} unique URLs in knowledge base file")
        return urls
    except Exception as e:
        logger.error(f"Error extracting URLs from knowledge base: {str(e)}")
        raise

def load_correct_urls(file_path: str) -> Dict[str, str]:
    """Load the correct URLs from a JSON file.
    Expected format: {"incorrect_url": "correct_url", ...} or ["url1", "url2", ...]"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Handle both dictionary and list formats
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            # If it's a list, assume all URLs are correct
            return {url: url for url in data}
        else:
            raise ValueError("URL file must be either a dictionary or list")
    except Exception as e:
        logger.error(f"Error loading correct URLs: {str(e)}")
        raise

def get_current_urls(vector_store: Chroma) -> Set[str]:
    """Get all unique URLs from the vector store."""
    try:
        # Get all documents from the vector store
        results = vector_store.get()
        urls = {doc["url"] for doc in results["metadatas"] if "url" in doc}
        logger.info(f"Found {len(urls)} unique URLs in vector store")
        return urls
    except Exception as e:
        logger.error(f"Error getting URLs from vector store: {str(e)}")
        raise

def check_url_status(url: str) -> Tuple[bool, int]:
    """Check if a URL is accessible."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200, response.status_code
    except requests.RequestException as e:
        return False, 0

def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing trailing slashes and converting to lowercase."""
    url = url.lower().rstrip('/')
    # Remove any URL parameters
    url = url.split('?')[0]
    return url

def is_url_shortened(short_url: str, full_url: str) -> bool:
    """Check if a URL appears to be a shortened version of another URL.
    This handles cases where path segments were accidentally removed."""
    short_parts = urlparse(normalize_url(short_url))
    full_parts = urlparse(normalize_url(full_url))
    
    # If domains don't match, it's not a shortening issue
    if short_parts.netloc != full_parts.netloc:
        return False
        
    # Get path segments
    short_segments = [s for s in short_parts.path.split('/') if s]
    full_segments = [s for s in full_parts.path.split('/') if s]
    
    # If short URL has more segments, it's not a shortening issue
    if len(short_segments) >= len(full_segments):
        return False
        
    # Check if short URL's segments appear in sequence in the full URL
    try:
        # Find the starting index of short segments in full segments
        start_idx = full_segments.index(short_segments[0])
        # Check if all short segments appear in sequence
        return full_segments[start_idx:start_idx + len(short_segments)] == short_segments
    except ValueError:
        return False

def find_matching_correct_url(url: str, correct_urls: Dict[str, str]) -> Tuple[str, str]:
    """Find the correct URL that matches or contains the given URL.
    Returns (matched_url, match_type) where match_type is 'exact', 'shortened', or None."""
    normalized_url = normalize_url(url)
    
    # First try exact match
    for correct_url in correct_urls.values():
        if normalize_url(correct_url) == normalized_url:
            return correct_url, 'exact'
            
    # Then try shortened match
    for correct_url in correct_urls.values():
        if is_url_shortened(url, correct_url):
            return correct_url, 'shortened'
            
    return None, None

def validate_urls(kb_urls: Set[str], correct_urls: Dict[str, str]) -> Dict[str, List[dict]]:
    """Validate URLs and compare against correct list."""
    results = {
        "needs_update": [],  # URLs that need to be changed
        "not_found": [],     # URLs in correct list but not in KB
        "valid": []          # URLs that are correct
    }
    
    # Check each URL from the knowledge base
    for url in kb_urls:
        matched_url, match_type = find_matching_correct_url(url, correct_urls)
        
        if matched_url:
            if match_type == 'exact':
                results["valid"].append({
                    "url": url,
                    "status": "valid"
                })
            else:  # shortened
                results["needs_update"].append({
                    "current": url,
                    "correct": matched_url,
                    "status": "shortened"
                })
        else:
            results["needs_update"].append({
                "current": url,
                "status": "no_match"
            })
    
    # Check for URLs in correct list but not in KB
    kb_url_set = {normalize_url(url) for url in kb_urls}
    correct_url_set = {normalize_url(url) for url in correct_urls.values()}
    missing_urls = correct_url_set - kb_url_set
    
    for url in missing_urls:
        results["not_found"].append({
            "url": url,
            "status": "not_found"
        })
    
    return results

def generate_report(results: Dict[str, List[dict]], output_file: str):
    """Generate a detailed report of URL validation results."""
    with open(output_file, 'w') as f:
        f.write("# URL Validation Report\n\n")
        
        # Summary
        f.write("## Summary\n")
        f.write(f"- Total URLs checked: {sum(len(v) for v in results.values())}\n")
        f.write(f"- URLs needing update: {len(results['needs_update'])}\n")
        f.write(f"- Valid URLs: {len(results['valid'])}\n")
        f.write(f"- URLs not found in knowledge base: {len(results['not_found'])}\n\n")
        
        # Detailed results
        f.write("## URLs Needing Update\n")
        for item in results["needs_update"]:
            f.write(f"- Current: {item['current']}\n")
            if 'correct' in item:
                f.write(f"  Correct: {item['correct']}\n")
                f.write(f"  Issue: URL appears to be shortened\n")
            else:
                f.write(f"  Issue: No matching correct URL found\n")
            f.write("\n")
        
        f.write("## URLs Not Found in Knowledge Base\n")
        for item in results["not_found"]:
            f.write(f"- {item['url']}\n\n")
        
        f.write("## Valid URLs\n")
        for item in results["valid"]:
            f.write(f"- {item['url']}\n")

def main():
    """Main execution function."""
    try:
        logger.info("Starting URL validation process...")
        
        # Load environment variables
        logger.debug("Loading environment variables...")
        load_dotenv()
        
        # Check if correct URLs file is provided
        if len(sys.argv) < 2:
            logger.error("Please provide the path to the correct URLs file")
            logger.info("Usage: python url_validator.py <correct_urls.json>")
            sys.exit(1)
        
        correct_urls_path = sys.argv[1]
        kb_path = "process content/knowledge_base_llms.txt"
        
        logger.info(f"Using correct URLs file: {correct_urls_path}")
        logger.info(f"Using knowledge base file: {kb_path}")
        
        # Verify the files exist
        if not Path(correct_urls_path).exists():
            logger.error(f"Correct URLs file not found: {correct_urls_path}")
            sys.exit(1)
        if not Path(kb_path).exists():
            logger.error(f"Knowledge base file not found: {kb_path}")
            sys.exit(1)
        
        # Load URLs from knowledge base and correct URLs list
        logger.debug("Extracting URLs from knowledge base...")
        kb_urls = extract_urls_from_kb(kb_path)
        
        logger.debug("Loading correct URLs from file...")
        correct_urls = load_correct_urls(correct_urls_path)
        
        # Validate URLs
        logger.info("Starting URL validation...")
        results = validate_urls(kb_urls, correct_urls)
        
        # Generate report
        report_path = "process content/Embedding/logs/url_validation_report.md"
        logger.info(f"Generating report at: {report_path}")
        generate_report(results, report_path)
        
        # Log summary
        logger.info("\nURL Validation Summary:")
        logger.info(f"Total URLs checked: {sum(len(v) for v in results.values())}")
        logger.info(f"URLs needing update: {len(results['needs_update'])}")
        logger.info(f"Valid URLs: {len(results['valid'])}")
        logger.info(f"URLs not found in knowledge base: {len(results['not_found'])}")
        logger.info(f"\nDetailed report generated at: {report_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Script failed with error:", exc_info=True)
        sys.exit(1) 