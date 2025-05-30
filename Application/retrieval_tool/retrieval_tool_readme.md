# Retrieval Tool Module

A semantic search and context assembly system designed for integration with an agentic chatbot system. This module provides basic retrieval capabilities with essential metrics tracking.

## Architecture Overview

The module is structured into several key components:

```
Application/Retrieval tool/
├── retriever.py      # Core semantic search implementation
├── context.py        # Context assembly and formatting
├── metrics.py        # Basic performance tracking
├── utils.py          # Helper functions and utilities
├── config.py         # Configuration and constants
└── tests/            # Basic functionality tests
```

### Component Responsibilities

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `Retriever` | Core search functionality | - Semantic search with basic thresholds<br>- Section filtering<br>- Article-level retrieval<br>- Basic result scoring |
| `ContextAssembler` | Post-processing results | - Basic chunk deduplication<br>- Metadata preservation<br>- Article-level context assembly |
| `SearchMetrics` | Basic performance tracking | - Query metrics<br>- Response timing<br>- Basic usage statistics |
| `Utils` | Helper functions | - Text processing<br>- Basic validation |
| `Config` | Settings management | - Search parameters<br>- Basic thresholds<br>- Paths |

## Key Design Decisions

### 1. Search Process
The retrieval tool uses a basic two-stage search process:

1. **Primary Semantic Search**:
   - Converts query to embedding vector
   - Performs similarity search
   - Uses cosine similarity
   - Applies basic thresholds
   - Returns relevant chunks

2. **Fallback Mechanism** (if primary search fails):
   - Uses more conservative matching
   - Returns best matches

### 2. Basic Thresholds
The system uses simple thresholds:

1. **Primary Threshold**:
   - Basic similarity threshold
   - Simple fallback mechanism

2. **Fallback Threshold**:
   - More conservative matching
   - Returns best matches

### 3. Metadata Handling
- Metadata stored in vector store
- Preserved throughout retrieval
- Formatted for LLM consumption

### 4. Article-Level Retrieval
- Returns all chunks from relevant articles
- Maintains article context
- Sorts chunks by index

## API Interface

The module provides a simple interface:

```python
from retrieval_tool import Retriever

# Initialize the retriever
retriever = Retriever()

# Basic search
results = retriever.search(
    query="your search query",
    filter_dict={"section": "Case Studies"}  # Optional
)

# Results format
{
    "query": "original query",
    "chunks": [
        # Document objects with content and metadata
    ],
    "metrics": {
        "response_time": 0.5,
        "threshold_used": 0.75
    }
}
```

## Testing

The `tests/` directory contains basic test suites:

```
Application/Retrieval tool/tests/
├── __init__.py
└── test_vector_store.py  # Basic functionality tests
```

### Test Requirements

1. **Dependencies**:
   ```python
   pytest>=7.0.0
   ```

2. **Running Tests**:
   ```bash
   # Run basic tests
   python -m pytest tests/
   ```

## Integration with Agentic Layer

The module is designed to work with the agentic planner system:

1. **Interface**:
   - Accepts text queries
   - Supports section filtering
   - Returns chunks with metadata
   - Provides basic metrics

2. **Integration Example**:
```python
# Planner LLM calls retriever
results = retriever.search(
    query="What are common shrink wrap issues?",
    filter_dict={"section": "2 Troubleshooting"}  # Optional
)

# Final payload for agentic layer
payload = {
    "user_query": results["query"],
    "retrieved_chunks": results["chunks"],
    "retrieval_skipped": False
}
```

## Future Enhancements

1. **Query Analysis**
   - Query intent detection
   - Query expansion

2. **Result Enhancement**
   - Cross-reference detection
   - Related content suggestion

3. **Performance**
   - Basic caching
   - Optimized operations

## Contributing

1. Follow the code structure
2. Add basic tests
3. Update documentation
4. Maintain type hints
5. Follow error handling patterns

## License

Internal use only - Not for distribution