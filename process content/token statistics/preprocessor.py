import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import tiktoken
from pathlib import Path

@dataclass
class Article:
    """Represents a single article from the knowledge base."""
    content: str
    url: Optional[str] = None
    section: Optional[str] = None
    token_count: Optional[int] = None

class KnowledgeBasePreprocessor:
    def __init__(self, tokenizer_name: str = "cl100k_base"):
        """Initialize the preprocessor with a specific tokenizer."""
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))
    
    def extract_metadata(self, text: str) -> Dict[str, Optional[str]]:
        """Extract URL and section metadata from article text."""
        metadata = {"url": None, "section": None}
        
        # Extract URL
        url_match = re.search(r"\*\*URL:\*\*\s*(https?://[^\s]+)", text)
        if url_match:
            metadata["url"] = url_match.group(1).strip()
        
        # Extract section
        section_match = re.search(r"\*\*Section:\*\*\s*([^\n]+)", text)
        if section_match:
            metadata["section"] = section_match.group(1).strip()
        
        return metadata
    
    def process_file(self, file_path: str) -> List[Article]:
        """Process the knowledge base file and return a list of Articles."""
        articles = []
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split on the delimiter
        article_texts = content.split('---')
        
        for article_text in article_texts:
            article_text = article_text.strip()
            if not article_text:  # Skip empty articles
                continue
                
            # Extract metadata
            metadata = self.extract_metadata(article_text)
            
            # Create Article object
            article = Article(
                content=article_text,
                url=metadata["url"],
                section=metadata["section"],
                token_count=self.count_tokens(article_text)
            )
            
            articles.append(article)
        
        return articles
    
    def analyze_token_distribution(self, articles: List[Article]) -> Dict[str, float]:
        """Analyze the token distribution of articles."""
        token_counts = [article.token_count for article in articles]
        
        if not token_counts:
            return {
                "mean": 0,
                "median": 0,
                "min": 0,
                "max": 0,
                "std_dev": 0
            }
        
        mean = sum(token_counts) / len(token_counts)
        median = sorted(token_counts)[len(token_counts) // 2]
        std_dev = (sum((x - mean) ** 2 for x in token_counts) / len(token_counts)) ** 0.5
        
        return {
            "mean": mean,
            "median": median,
            "min": min(token_counts),
            "max": max(token_counts),
            "std_dev": std_dev
        }

def main():
    """Main function to demonstrate usage."""
    # Initialize preprocessor
    preprocessor = KnowledgeBasePreprocessor()
    
    # Process the knowledge base file
    try:
        articles = preprocessor.process_file("chunking/knowledge_base_llms.txt")
        
        # Analyze token distribution
        stats = preprocessor.analyze_token_distribution(articles)
        
        print(f"Processed {len(articles)} articles")
        print("\nToken Distribution Statistics:")
        print(f"Mean tokens per article: {stats['mean']:.2f}")
        print(f"Median tokens per article: {stats['median']:.2f}")
        print(f"Min tokens: {stats['min']}")
        print(f"Max tokens: {stats['max']}")
        print(f"Standard deviation: {stats['std_dev']:.2f}")
        
    except FileNotFoundError:
        print("Error: chunking/knowledge_base_llms.txt not found. Please create this file first.")

if __name__ == "__main__":
    main() 