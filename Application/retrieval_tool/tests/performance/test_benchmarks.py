"""
Performance tests and benchmarks for the retrieval tool.

Tests:
- Search performance
- Deduplication speed
- Memory usage
- Threshold calculation
- Article assembly
"""

import pytest
import time
import psutil
import numpy as np
from unittest.mock import Mock, patch
from Application.retrieval_tool import Retriever
from Application.retrieval_tool.tests import create_test_chunks, TEST_SECTIONS

def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

class TestPerformance:
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store with performance test data."""
        store = Mock()
        # Create a larger set of test chunks for performance testing
        chunks = []
        for i in range(10):  # 10 articles
            for j in range(5):  # 5 chunks per article
                chunks.append({
                    "page_content": f"Article {i} chunk {j} with some content. " * 10,
                    "metadata": {
                        "section": TEST_SECTIONS[i % len(TEST_SECTIONS)],
                        "chunk_index": j,
                        "article_url": f"http://test.com/article{i}",
                        "article_title": f"Test Article {i}"
                    }
                })
        store.similarity_search.return_value = chunks
        store.get_sections.return_value = TEST_SECTIONS
        return store

    @pytest.fixture
    def retriever(self, mock_vector_store):
        """Create a Retriever instance with mock vector store."""
        with patch('langchain_chroma.Chroma', return_value=mock_vector_store):
            return Retriever()

    def test_search_performance(self, retriever):
        """Benchmark search performance."""
        # Measure search time
        start_time = time.time()
        results = retriever.search("test query")
        search_time = time.time() - start_time
        
        # Verify performance metrics
        assert search_time < 1.0  # Should complete within 1 second
        assert results.metrics["response_time"] < 1.0
        
        # Verify memory usage
        memory_usage = get_memory_usage()
        assert memory_usage < 500  # Should use less than 500MB

    def test_deduplication_performance(self, retriever, mock_vector_store):
        """Benchmark deduplication performance."""
        # Create chunks with known overlaps
        chunks = []
        for i in range(5):  # 5 articles
            base_content = f"Base content for article {i}. " * 5
            for j in range(3):  # 3 chunks per article with overlap
                chunks.append({
                    "page_content": f"{base_content} Additional content {j}. " * 3,
                    "metadata": {
                        "section": TEST_SECTIONS[i % len(TEST_SECTIONS)],
                        "chunk_index": j,
                        "article_url": f"http://test.com/article{i}",
                        "article_title": f"Test Article {i}"
                    }
                })
        mock_vector_store.similarity_search.return_value = chunks
        
        # Measure deduplication time
        start_time = time.time()
        results = retriever.search("test query")
        dedup_time = time.time() - start_time
        
        # Verify performance
        assert dedup_time < 0.5  # Should complete within 0.5 seconds
        assert results.metrics["deduplication_stats"]["total_chunks"] > 0
        assert results.metrics["deduplication_stats"]["chunks_with_overlap"] > 0

    def test_threshold_calculation_performance(self, retriever):
        """Benchmark threshold calculation performance."""
        # Generate test scores
        scores = np.random.normal(0.8, 0.1, 1000)  # 1000 scores
        
        # Measure calculation time
        start_time = time.time()
        threshold = retriever._calculate_threshold(scores)
        calc_time = time.time() - start_time
        
        # Verify performance
        assert calc_time < 0.1  # Should complete within 0.1 seconds
        assert 0 <= threshold <= 1

    def test_article_assembly_performance(self, retriever, mock_vector_store):
        """Benchmark article assembly performance."""
        # Create a larger set of chunks
        chunks = []
        for i in range(20):  # 20 articles
            for j in range(10):  # 10 chunks per article
                chunks.append({
                    "page_content": f"Article {i} chunk {j} content. " * 20,
                    "metadata": {
                        "section": TEST_SECTIONS[i % len(TEST_SECTIONS)],
                        "chunk_index": j,
                        "article_url": f"http://test.com/article{i}",
                        "article_title": f"Test Article {i}"
                    }
                })
        mock_vector_store.similarity_search.return_value = chunks
        
        # Measure assembly time
        start_time = time.time()
        results = retriever.search("test query")
        assembly_time = time.time() - start_time
        
        # Verify performance
        assert assembly_time < 1.0  # Should complete within 1 second
        assert len(results.chunks) > 0
        
        # Verify memory usage
        memory_usage = get_memory_usage()
        assert memory_usage < 1000  # Should use less than 1GB

    def test_concurrent_searches(self, retriever):
        """Test performance under concurrent search requests."""
        import concurrent.futures
        
        # Run multiple searches concurrently
        queries = [f"test query {i}" for i in range(10)]
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(retriever.search, query) for query in queries]
            results = [f.result() for f in futures]
        
        total_time = time.time() - start_time
        
        # Verify performance
        assert total_time < 5.0  # Should complete within 5 seconds
        assert len(results) == len(queries)
        
        # Verify memory usage
        memory_usage = get_memory_usage()
        assert memory_usage < 1000  # Should use less than 1GB

    def test_large_result_set(self, retriever, mock_vector_store):
        """Test performance with large result sets."""
        # Create a very large set of chunks
        chunks = []
        for i in range(50):  # 50 articles
            for j in range(20):  # 20 chunks per article
                chunks.append({
                    "page_content": f"Article {i} chunk {j} content. " * 30,
                    "metadata": {
                        "section": TEST_SECTIONS[i % len(TEST_SECTIONS)],
                        "chunk_index": j,
                        "article_url": f"http://test.com/article{i}",
                        "article_title": f"Test Article {i}"
                    }
                })
        mock_vector_store.similarity_search.return_value = chunks
        
        # Measure performance
        start_time = time.time()
        results = retriever.search("test query")
        total_time = time.time() - start_time
        
        # Verify performance
        assert total_time < 2.0  # Should complete within 2 seconds
        assert len(results.chunks) > 0
        
        # Verify memory usage
        memory_usage = get_memory_usage()
        assert memory_usage < 2000  # Should use less than 2GB

    @pytest.mark.parametrize("query_length", [10, 100, 1000])
    def test_query_length_performance(self, retriever, query_length):
        """Test performance with different query lengths."""
        # Generate query of specified length
        query = "test " * (query_length // 5)
        
        # Measure performance
        start_time = time.time()
        results = retriever.search(query)
        search_time = time.time() - start_time
        
        # Verify performance
        assert search_time < 1.0  # Should complete within 1 second
        assert results.metrics["response_time"] < 1.0 