"""
Chunking module for processing the knowledge base into optimized chunks for RAG.
Handles article splitting, metadata extraction, tokenization, and chunking with overlap.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import tiktoken
from loguru import logger
from pydantic import BaseModel, Field
import markdown
from typing_extensions import TypedDict
import sys
import json

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    "chunking.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",  # Show debug messages
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    sys.stderr,  # Also show logs in console
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

class Metadata(TypedDict):
    """Type definition for article metadata."""
    url: str
    section: str
    article_index: int
    chunk_index: Optional[int]
    token_count: Optional[int]

class Document(BaseModel):
    """Represents a chunked document with content and metadata."""
    page_content: str = Field(..., description="The actual content of the chunk")
    metadata: Metadata = Field(..., description="Metadata associated with the chunk")

class Article(BaseModel):
    """Represents a complete article with metadata and content."""
    content: str = Field(..., description="The full article content")
    metadata: Metadata = Field(..., description="Article metadata")
    token_count: int = Field(..., description="Number of tokens in the article")

class Chunker:
    """Main class for handling the chunking process."""
    
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base"  # OpenAI's encoding
    ):
        """
        Initialize the chunker with specified parameters.
        
        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            encoding_name: Name of the tiktoken encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        logger.info(f"Initialized Chunker with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def split_articles(self, raw_text: str) -> List[str]:
        """
        Split raw text into individual articles using '---' delimiter.
        
        Args:
            raw_text: The complete knowledge base text
            
        Returns:
            List of individual article texts
        """
        articles = [article.strip() for article in raw_text.split("---")]
        articles = [article for article in articles if article]  # Remove empty articles
        
        logger.info(f"Split {len(articles)} articles from raw text")
        return articles

    def extract_metadata(self, article: str) -> Tuple[Metadata, str]:
        """
        Extract metadata from article header and return remaining content.
        
        Args:
            article: Raw article text
            
        Returns:
            Tuple of (metadata dict, content without metadata)
        """
        # Extract URL
        url_match = re.search(r">\s*\*\*URL:\*\*\s*<([^>]+)>", article)
        url = url_match.group(1) if url_match else ""
        
        # Extract section
        section_match = re.search(r">\s*\*\*Section:\*\*\s*([^\n]+)", article)
        section = section_match.group(1).strip() if section_match else ""
        
        # Remove metadata from content
        content = re.sub(r">\s*\*\*URL:\*\*.*?\n", "", article)
        content = re.sub(r">\s*\*\*Section:\*\*.*?\n", "", content)
        content = content.strip()
        
        metadata: Metadata = {
            "url": url,
            "section": section,
            "article_index": 0,  # Will be set later
            "chunk_index": None,
            "token_count": None
        }
        
        if not url or not section:
            logger.warning(f"Missing metadata in article: URL={bool(url)}, Section={bool(section)}")
        
        return metadata, content

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text using tiktoken.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def create_chunks(
        self,
        article: Article,
        preserve_formatting: bool = True
    ) -> List[Document]:
        """
        Create overlapping chunks from an article using token-based slicing.
        
        Args:
            article: Article object containing content and metadata
            preserve_formatting: Whether to preserve markdown formatting (handled by decoding)
            
        Returns:
            List of Document objects representing chunks
        """
        logger.debug(f"Starting token-based chunk creation for article {article.metadata['url']}")

        # Tokenize the entire article content
        tokens = self.encoding.encode(article.content)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            # Article is small enough to be a single chunk
            logger.debug("Article is small enough to be a single chunk (token count <= chunk_size)")
            return [Document(
                page_content=article.content,
                metadata={
                    **article.metadata,
                    "chunk_index": 0,
                    "token_count": total_tokens
                }
            )]

        chunks = []
        # Start position in the token list
        token_start = 0
        chunk_index = 0

        while token_start < total_tokens:
            # Calculate the end position for the current chunk (exclusive)
            token_end = min(token_start + self.chunk_size, total_tokens)

            # Get the tokens for the current chunk
            chunk_tokens_slice = tokens[token_start:token_end]
            chunk_token_count = len(chunk_tokens_slice)

            # Decode the tokens back to text content
            chunk_content = self.encoding.decode(chunk_tokens_slice)

            logger.debug(
                f"Created chunk {chunk_index}: token_start={token_start}, "
                f"token_end={token_end}, token_count={chunk_token_count}"
            )

            # Create chunk document
            chunk = Document(
                page_content=chunk_content,
                metadata={
                    **article.metadata,
                    "chunk_index": chunk_index,
                    "token_count": chunk_token_count
                }
            )
            chunks.append(chunk)

            # Calculate the start position for the next chunk
            # Move forward by chunk_size minus overlap
            # Ensure we move forward by at least one token if chunk_size == overlap (though validated against this)
            step_size = self.chunk_size - self.chunk_overlap
            token_start += max(1, step_size) # Ensure at least 1 token advance

            chunk_index += 1

            # Break if the start of the next chunk is beyond the total tokens
            if token_start >= total_tokens:
                break

        logger.info(f"Created {len(chunks)} chunks from article {article.metadata['url']}")
        return chunks

    def _find_chunk_boundary(self, text: str, start_pos: int, target_tokens: int) -> int:
        """
        (This method is less critical now with token-based chunking,
         but kept for potential future use or refinement of chunk ends)
        """
        # The logic below is largely superseded by the token-based slicing in create_chunks.
        # We could potentially use this to adjust chunk_end slightly for sentence breaks
        # *after* token slicing, but for now, the token boundaries are primary.
        
        text_len = len(text)
        if start_pos >= text_len:
            return text_len # Already at the end or beyond

        # Estimate characters per token (rough average)
        chars_per_token = 4
        # Calculate target character position based on token count
        target_char_pos = min(start_pos + target_tokens * chars_per_token, text_len)

        # Define a search window around the target character position
        # Look within +/- half the overlap in characters, but at least a small fixed window
        window_size_chars = max(int(self.chunk_overlap * chars_per_token / 2), 50)
        search_start = max(start_pos, target_char_pos - window_size_chars)
        search_end = min(text_len, target_char_pos + window_size_chars)

        # Search for sentence/paragraph boundaries within the window
        boundaries = []
        # Look for sentence-ending punctuation followed by space
        for match in re.finditer(r'[.!?]\s', text[search_start:search_end]):
             boundaries.append(search_start + match.end())
        # Look for double newlines (paragraph breaks)
        for match in re.finditer(r'\n\n', text[search_start:search_end]):
             boundaries.append(search_start + match.end())

        # Filter boundaries to be after the start_pos
        boundaries = [b for b in boundaries if b > start_pos]
        boundaries.sort()

        best_boundary = -1
        min_distance_to_target = float('inf')

        # Find the boundary closest to the target character position within the window
        for boundary in boundaries:
            distance = abs(boundary - target_char_pos)
            if distance < min_distance_to_target:
                min_distance_to_target = distance
                best_boundary = boundary

        # If a good boundary was found within the window, use it
        if best_boundary != -1:
            return best_boundary

        # If no suitable boundary found in the window, use a position near the target token count
        # Fallback to target_char_pos, ensuring it's at least one char past start_pos
        fallback_boundary = max(start_pos + 1, target_char_pos)

        # Final check: ensure fallback boundary doesn't exceed text length
        return min(fallback_boundary, text_len)

    def process_knowledge_base(self, input_file: Path) -> List[Document]:
        """
        Process the entire knowledge base file into chunks.
        
        Args:
            input_file: Path to the knowledge base file
            
        Returns:
            List of Document objects representing all chunks
        """
        logger.info(f"Processing knowledge base from {input_file}")
        
        # Read input file
        logger.info("Reading input file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        logger.info(f"Successfully read {len(raw_text)} characters")
        
        # Split into articles
        logger.info("Splitting articles...")
        articles = self.split_articles(raw_text)
        all_chunks = []
        
        # Process each article
        logger.info(f"Starting to process {len(articles)} articles...")
        for idx, article_text in enumerate(articles):
            try:
                logger.info(f"Processing article {idx + 1}/{len(articles)}")
                
                # Extract metadata and content
                logger.debug(f"Extracting metadata for article {idx + 1}")
                metadata, content = self.extract_metadata(article_text)
                metadata["article_index"] = idx
                
                # Create article object
                logger.debug(f"Creating article object for article {idx + 1}")
                article = Article(
                    content=content,
                    metadata=metadata,
                    token_count=self.count_tokens(content)
                )
                
                # Create chunks
                logger.debug(f"Creating chunks for article {idx + 1} ({article.token_count} tokens)")
                chunks = self.create_chunks(article)
                all_chunks.extend(chunks)
                
                logger.info(
                    f"Completed article {idx + 1}: {len(chunks)} chunks, "
                    f"{article.token_count} tokens"
                )
                
            except Exception as e:
                logger.error(f"Error processing article {idx + 1}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Completed processing. Created {len(all_chunks)} total chunks")
        return all_chunks

def main():
    """Main entry point for the chunking process."""
    # Get the directory where chunker.py is located
    current_dir = Path(__file__).parent
    # Go up one level to process content directory
    process_content_dir = current_dir.parent
    # Construct path to knowledge base file
    input_file = process_content_dir / "knowledge_base_llms.txt"
    
    if not input_file.exists():
        logger.error(f"Knowledge base file not found at: {input_file}")
        return
    
    # Initialize chunker
    chunker = Chunker()
    
    # Process knowledge base
    chunks = chunker.process_knowledge_base(input_file)
    
    # Save chunks to a JSON file
    output_file = current_dir / "chunks.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Serialize Pydantic models to dictionaries before saving as JSON
            chunks_data = [chunk.model_dump() for chunk in chunks] # Use model_dump() for Pydantic v2+
            # Ensure non-ASCII characters are not escaped
            json.dump(chunks_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved {len(chunks)} chunks to {output_file}")
        # Add a note about token overlap in the log or potentially in the JSON itself if structure allowed
        logger.info("Note: Overlap is calculated based on token counts (64 tokens), which may not always translate to a visually consistent character overlap in the text due to variable token lengths.")
    except Exception as e:
        logger.error(f"Error saving chunks to file: {str(e)}")

    
    # Log statistics
    total_chunks = len(chunks)
    total_tokens = sum(chunk.metadata["token_count"] or 0 for chunk in chunks)
    avg_tokens = total_tokens / total_chunks if total_chunks > 0 else 0
    
    logger.info(f"Chunking Statistics:")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Average tokens per chunk: {avg_tokens:.2f}")

if __name__ == "__main__":
    main() 