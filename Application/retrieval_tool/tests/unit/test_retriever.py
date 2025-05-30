"""
Unit tests for the Retriever class.

Tests core functionality including:
- Search operations
- Section filtering
- Threshold calculations
- Chunk deduplication
- Error handling
"""

import pytest
from unittest.mock import Mock, patch
from Application.retrieval_tool import Retriever, RetrievalError, NoResultsFound, SectionNotFound
from Application.retrieval_tool.tests import create_test_document, create_test_chunks, TEST_SECTIONS

class TestRetriever:
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        store = Mock()
        store.similarity_search.return_value = create_test_chunks()
        store.get_sections.return_value = TEST_SECTIONS
        return store

    @pytest.fixture
    def retriever(self, mock_vector_store):
        """Create a Retriever instance with mock vector store."""
        with patch('langchain_chroma.Chroma', return_value=mock_vector_store):
            return Retriever()

    def test_search_basic(self, retriever):
        """Test basic search functionality."""
        results = retriever.search("test query")
        assert results.chunks
        assert results.metrics
        assert "response_time" in results.metrics
        assert "threshold_used" in results.metrics

    def test_section_filtering_exact(self, retriever):
        """Test exact section name matching."""
        results = retriever.search(
            "test query",
            filter_dict={"section": "1 Introduction"}
        )
        assert results.metrics["filter_applied"]["section"] == "1 Introduction"
        assert results.metrics["filter_matched"] == "exact"

    def test_section_filtering_partial(self, retriever):
        """Test partial section name matching."""
        results = retriever.search(
            "test query",
            filter_dict={"section": "Introduction"}
        )
        assert results.metrics["filter_applied"]["section"] == "1 Introduction"
        assert results.metrics["filter_matched"] == "partial"

    def test_section_filtering_invalid(self, retriever):
        """Test invalid section name handling."""
        with pytest.raises(SectionNotFound):
            retriever.search(
                "test query",
                filter_dict={"section": "Invalid Section"}
            )

    def test_threshold_calculation(self, retriever):
        """Test adaptive threshold calculation."""
        # Test with tight distribution
        scores = [0.85, 0.84, 0.83, 0.82, 0.81]  # std dev < 0.05
        threshold = retriever._calculate_threshold(scores)
        assert 0.80 <= threshold <= 0.85

        # Test with spread distribution
        scores = [0.95, 0.85, 0.75, 0.65, 0.55]  # std dev > 0.2
        threshold = retriever._calculate_threshold(scores)
        assert 0.75 <= threshold <= 0.85

    def test_deduplication(self, retriever):
        """Test chunk deduplication."""
        chunks = create_test_chunks()
        deduped_chunks, stats = retriever._deduplicate_chunks(chunks)
        
        assert len(deduped_chunks) < len(chunks)
        assert stats.total_chunks == len(chunks)
        assert stats.chunks_with_overlap > 0
        assert stats.total_overlap_removed > 0

    def test_no_results(self, retriever, mock_vector_store):
        """Test handling of no results."""
        mock_vector_store.similarity_search.return_value = []
        with pytest.raises(NoResultsFound):
            retriever.search("test query")

    def test_vector_store_error(self, retriever, mock_vector_store):
        """Test vector store error handling."""
        mock_vector_store.similarity_search.side_effect = Exception("Connection error")
        with pytest.raises(RetrievalError):
            retriever.search("test query")

    def test_article_assembly(self, retriever):
        """Test article assembly from chunks."""
        chunks = create_test_chunks()
        articles = retriever._assemble_articles(chunks)
        
        assert len(articles) == 1  # One article from test chunks
        article = articles[0]
        assert "--- Article: Test Article 1 ---" in article
        assert "Section: 1 Introduction" in article
        assert "Source: http://test.com/article1" in article

    def test_metrics_tracking(self, retriever):
        """Test metrics collection and tracking."""
        results = retriever.search("test query")
        metrics = results.metrics
        
        assert "response_time" in metrics
        assert "threshold_used" in metrics
        assert "threshold_type" in metrics
        assert "filter_applied" in metrics
        assert "deduplication_stats" in metrics
        
        dedup_stats = metrics["deduplication_stats"]
        assert "total_chunks" in dedup_stats
        assert "chunks_with_overlap" in dedup_stats
        assert "total_overlap_removed" in dedup_stats
        assert "articles_processed" in dedup_stats

    def test_fallback_threshold(self, retriever, mock_vector_store):
        """Test fallback threshold mechanism."""
        # First call returns no results
        mock_vector_store.similarity_search.side_effect = [
            [],  # First call: no results
            create_test_chunks()  # Second call: some results
        ]
        
        results = retriever.search("test query")
        assert results.metrics["threshold_type"] == "fallback"
        assert "fallback_reason" in results.metrics

    @pytest.mark.parametrize("query,expected_section", [
        ("test | section: Introduction", "1 Introduction"),
        ("test | section: 2 Troubleshooting", "2 Troubleshooting"),
        ("test | section: Case Studies", "3 Case Studies"),
    ])
    def test_query_filter_extraction(self, retriever, query, expected_section):
        """Test filter extraction from query string."""
        results = retriever.search(query)
        assert results.metrics["filter_applied"]["section"] == expected_section 