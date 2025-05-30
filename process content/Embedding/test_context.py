from test_queries import smart_search, load_vector_store
from context_assembler import prepare_context_for_llm
from loguru import logger
from dotenv import load_dotenv
import sys
from typing import Tuple

def test_context_assembly(query: str, include_metadata: bool = True):
    """Test the context assembly with a real query and show the raw output."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Load the vector store
        logger.info("Loading vector store...")
        vector_store = load_vector_store()
        
        # Perform the search
        logger.info(f"\nPerforming search for query: {query}")
        chunks, _ = smart_search(vector_store, query)
        
        if not chunks:
            logger.error("No chunks found for the query!")
            return
        
        # Prepare the context
        logger.info("\nPreparing context for LLM...")
        context, stats = prepare_context_for_llm(chunks, include_metadata)
        
        # Print the raw context that would be sent to the LLM
        print("\n" + "="*80)
        print("RAW CONTEXT THAT WOULD BE SENT TO LLM:")
        print("="*80)
        print(context)
        print("="*80)
        
        # Print some statistics
        print("\nContext Statistics:")
        print(f"Total articles: {stats['total_articles']}")
        print(f"Total chunks processed: {stats['total_chunks']}")
        print(f"Chunks with overlap: {stats['chunks_with_overlap']}")
        print(f"Total overlap removed: {stats['total_overlap_removed']} words")
        print(f"Final context length: {stats['context_length']} characters")
        
    except Exception as e:
        logger.error(f"Error during context assembly: {str(e)}")
        raise

def find_overlap(prev_content: str, current_content: str, boundary_window: int = 300) -> Tuple[str, int]:
    """
    Find the longest common substring between the end of prev_content and start of current_content.
    Only matches at word boundaries and only looks within a boundary window optimized for
    typical overlap size of ~64 tokens.
    
    Args:
        prev_content: The previous chunk's content
        current_content: The current chunk's content
        boundary_window: How many characters to look at from each end (default 300 to safely
                        cover typical 64-token overlaps)
    """
    # Normalize whitespace and ensure we have spaces at boundaries
    # Only look at the last boundary_window characters of prev_content
    prev = prev_content.rstrip()[-boundary_window:].rstrip() + " "
    # Only look at the first boundary_window characters of current_content
    curr = " " + current_content.lstrip()[:boundary_window].lstrip()
    
    # Rest of the function remains the same...

if __name__ == "__main__":
    # Configure logging
    logger.add(
        "process content/Embedding/logs/context_test.log",
        rotation="100 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Example queries you can try:
        example_queries = [
            "What are the benefits of shrink wrapping for food businesses?",
            "Tell me about Celtic Fish and Game's business model",
            "What packaging solutions were provided to Hogsbottom?",
            "How do I troubleshoot common shrink wrapping problems?"
        ]
        print("\nAvailable example queries:")
        for i, q in enumerate(example_queries, 1):
            print(f"{i}. {q}")
        
        choice = input("\nEnter query number (1-4) or type your own query: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(example_queries):
                query = example_queries[idx]
            else:
                query = choice
        except ValueError:
            query = choice
    
    # Run the test
    test_context_assembly(query) 