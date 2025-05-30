import pytest
from preprocessor import KnowledgeBasePreprocessor, Article

def test_metadata_extraction():
    preprocessor = KnowledgeBasePreprocessor()
    
    # Test article with metadata
    test_text = """**URL:** https://example.com/article
**Section:** Test Section
This is the article content."""
    
    metadata = preprocessor.extract_metadata(test_text)
    assert metadata["url"] == "https://example.com/article"
    assert metadata["section"] == "Test Section"
    
    # Test article without metadata
    test_text_no_metadata = "This is an article without metadata."
    metadata = preprocessor.extract_metadata(test_text_no_metadata)
    assert metadata["url"] is None
    assert metadata["section"] is None

def test_token_counting():
    preprocessor = KnowledgeBasePreprocessor()
    
    # Test token counting
    test_text = "This is a test sentence."
    token_count = preprocessor.count_tokens(test_text)
    assert isinstance(token_count, int)
    assert token_count > 0

def test_process_file(tmp_path):
    preprocessor = KnowledgeBasePreprocessor()
    
    # Create a temporary test file
    test_file = tmp_path / "test_kb.txt"
    test_content = """**URL:** https://example.com/1
**Section:** Section 1
Article 1 content
---
**URL:** https://example.com/2
**Section:** Section 2
Article 2 content"""
    
    test_file.write_text(test_content)
    
    # Process the file
    articles = preprocessor.process_file(str(test_file))
    
    # Verify results
    assert len(articles) == 2
    assert articles[0].url == "https://example.com/1"
    assert articles[0].section == "Section 1"
    assert articles[1].url == "https://example.com/2"
    assert articles[1].section == "Section 2"
    assert all(isinstance(article.token_count, int) for article in articles)

def test_token_distribution():
    preprocessor = KnowledgeBasePreprocessor()
    
    # Create test articles
    articles = [
        Article(content="Short article", token_count=10),
        Article(content="Medium article", token_count=20),
        Article(content="Long article", token_count=30)
    ]
    
    # Analyze distribution
    stats = preprocessor.analyze_token_distribution(articles)
    
    assert stats["mean"] == 20.0
    assert stats["median"] == 20.0
    assert stats["min"] == 10
    assert stats["max"] == 30
    assert stats["std_dev"] > 0 