"""
Integration tests for the retrieval tool pipeline.

Tests:
- Agentic layer integration
- Article-level retrieval
- Metadata preservation
- Error propagation
- Performance metrics
"""

import pytest
from unittest.mock import Mock, patch
from Application.retrieval_tool import Retriever
from Application.retrieval_tool.tests import create_test_document, create_test_chunks, TEST_SECTIONS

class TestPipelineIntegration:
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for integration testing."""
        store = Mock()
        store.similarity_search.return_value = create_test_chunks()
        store.get_sections.return_value = TEST_SECTIONS
        return store

    @pytest.fixture
    def retriever(self, mock_vector_store):
        """Create a Retriever instance with mock vector store."""
        with patch('langchain_chroma.Chroma', return_value=mock_vector_store):
            return Retriever()

    def test_agentic_layer_integration(self, retriever):
        """Test retriever works with agentic layer requirements."""
        # Test plain text query (from agentic layer)
        results = retriever.search("What are common shrink wrap issues?")
        
        # Verify chunk format matches agentic layer requirements
        assert results.chunks
        chunk = results.chunks[0]
        assert isinstance(chunk, str)  # Must be string for frontend LLM
        assert "--- Article:" in chunk  # Must have article header
        assert "Section:" in chunk  # Must have section info
        assert "Source:" in chunk  # Must have source URL
        
        # Verify metadata is preserved
        assert "Test Article 1" in chunk
        assert "1 Introduction" in chunk
        assert "http://test.com/article1" in chunk

    def test_planner_query_reformulation(self, retriever):
        """Test retriever handles planner's reformulated queries."""
        # Test with reformulated query
        results = retriever.search(
            "What are common shrink wrap issues? | section: Troubleshooting"
        )
        
        # Verify filter was extracted and applied
        assert results.metrics["filter_applied"]["section"] == "2 Troubleshooting"
        assert results.metrics["filter_matched"] in ["exact", "partial"]

    def test_article_level_retrieval(self, retriever, mock_vector_store):
        """Test article-level retrieval maintains context."""
        # Create test chunks from multiple articles
        chunks = create_test_chunks()
        mock_vector_store.similarity_search.return_value = chunks
        
        results = retriever.search("test query")
        
        # Verify all chunks from relevant articles are returned
        assert len(results.chunks) > 0
        # Verify chunks are grouped by article
        article_urls = set()
        for chunk in results.chunks:
            if "Source:" in chunk:
                url = chunk.split("Source:")[1].split("\n")[0].strip()
                article_urls.add(url)
        assert len(article_urls) == 1  # One article in test chunks

    def test_metadata_preservation(self, retriever):
        """Test raw metadata is preserved for citations."""
        results = retriever.search("test query")
        
        # Verify each chunk has complete metadata
        for chunk in results.chunks:
            # Check metadata header format
            assert "--- Article:" in chunk
            assert "Section:" in chunk
            assert "Source:" in chunk
            
            # Verify metadata fields are complete
            metadata_lines = [line for line in chunk.split("\n") 
                            if line.startswith(("---", "Article:", "Section:", "Source:"))]
            assert len(metadata_lines) >= 4  # Header + 3 metadata fields

    def test_metrics_for_decision_making(self, retriever):
        """Test metrics provide necessary info for planner decisions."""
        results = retriever.search("test query")
        metrics = results.metrics
        
        # Verify metrics needed for planner decisions
        assert "threshold_used" in metrics
        assert "threshold_type" in metrics
        assert "filter_applied" in metrics
        assert "deduplication_stats" in metrics
        
        # Verify deduplication stats
        dedup_stats = metrics["deduplication_stats"]
        assert "total_chunks" in dedup_stats
        assert "chunks_with_overlap" in dedup_stats
        assert "total_overlap_removed" in dedup_stats
        assert "articles_processed" in dedup_stats

    def test_error_propagation(self, retriever, mock_vector_store):
        """Test errors are properly propagated to agentic layer."""
        # Test vector store error
        mock_vector_store.similarity_search.side_effect = Exception("Connection error")
        with pytest.raises(Exception) as exc_info:
            retriever.search("test query")
        assert "Connection error" in str(exc_info.value)
        
        # Test no results error
        mock_vector_store.similarity_search.return_value = []
        with pytest.raises(Exception) as exc_info:
            retriever.search("test query")
        assert "No results found" in str(exc_info.value)

    def test_performance_metrics(self, retriever):
        """Test performance metrics are tracked for monitoring."""
        results = retriever.search("test query")
        metrics = results.metrics
        
        # Verify timing metrics
        assert "response_time" in metrics
        assert isinstance(metrics["response_time"], float)
        assert metrics["response_time"] > 0
        
        # Verify threshold metrics
        assert "threshold_used" in metrics
        assert isinstance(metrics["threshold_used"], float)
        assert 0 <= metrics["threshold_used"] <= 1

    @pytest.mark.parametrize("query,expected_format", [
        ("test query", "plain"),  # Plain query
        ("test | section: Introduction", "with_filter"),  # Query with filter
        ("test query with multiple words", "plain"),  # Multi-word query
    ])
    def test_query_format_handling(self, retriever, query, expected_format):
        """Test retriever handles different query formats."""
        results = retriever.search(query)
        
        if expected_format == "with_filter":
            assert "filter_applied" in results.metrics
            assert results.metrics["filter_applied"]["section"] in TEST_SECTIONS
        else:
            assert "filter_applied" not in results.metrics or not results.metrics["filter_applied"] 