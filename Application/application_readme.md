# 🤖 Knowledge Base Chatbot Application

## 📁 Directory Structure

```
Application/
├── Chunks/         # Processed and chunked knowledge base data
├── Embeddings/     # Vector embeddings for semantic search
├── Agent/          # Agentic layer implementation
│   ├── planner/    # Planner LLM implementation
│   └── frontend/   # Frontend LLM implementation
└── Retrieval tool/ # Optimized semantic search implementation
```

## 🎯 Implementation Plan

### Phase 1: Retrieval Tool Optimization
First, we'll optimize our semantic search for production use:

1. **Core Search Implementation**
   - Move and refactor code from `test_queries.py`
   - Implement proper error handling
   - Add comprehensive logging
   - Optimize for performance
   - Add caching if needed
   - Make it API-friendly

2. **Search Features**
   - Adaptive threshold search
   - Metadata filtering
   - Chunk deduplication
   - Context assembly
   - Source citation handling

### Phase 2: Agentic Layer Implementation
We'll implement the two-LLM system one at a time:

1. **Planner LLM (First)**
   - Implement in `Agent/planner/`
   - Features:
     - ConversationBufferMemory
     - Query reformulation
     - Retrieval decision making
     - Filter recognition
   - Integration with optimized retrieval tool
   - Testing and validation

2. **Frontend LLM (Second)**
   - Implement in `Agent/frontend/`
   - Features:
     - Full chat history management
     - Context injection
     - Citation handling
     - Natural response generation
   - Integration with Planner LLM
   - Testing and validation

### Phase 3: API Layer
Finally, we'll create a production-ready API:

1. **API Implementation**
   - FastAPI/Flask wrapper
   - Request/response schemas
   - Authentication
   - Rate limiting
   - Error handling
   - Monitoring and logging

2. **Deployment**
   - Containerization
   - Environment configuration
   - Deployment documentation
   - Frontend integration guide

## 🔄 Current Status

✅ Completed:
- Knowledge base processing
- Chunking strategy
- Vector database setup
- Basic semantic search
- Basic vector store testing

🔄 In Progress:
- Agentic layer implementation (Next immediate focus)

## 🛠️ Development Guidelines

### Code Organization
- Each component should be modular and well-documented
- Use type hints and docstrings
- Follow PEP 8 style guide
- Implement proper error handling
- Add comprehensive logging

### Testing
- Unit tests for each component
- Integration tests for the full pipeline
- Performance benchmarks
- Load testing for API

### Documentation
- Code documentation
- API documentation
- Deployment guide
- Integration guide for frontends

## 📝 Next Steps

1. **Immediate Focus: Agentic Layer**
   - Set up `Agent/planner/` structure
   - Implement core planner functionality
   - Integrate with retrieval tool
   - Add basic tests

2. **Following: Frontend LLM**
   - Set up `Agent/frontend/` structure
   - Implement frontend functionality
   - Integrate with planner
   - Add basic tests

3. **Finally: API Layer**
   - Create API wrapper
   - Add authentication
   - Implement monitoring
   - Create deployment docs

## 🔍 Component Details

### Retrieval Tool
The optimized semantic search tool will:
- Use adaptive thresholds for search
- Support metadata filtering
- Handle chunk deduplication
- Assemble context with proper formatting
- Include source citations
- Be optimized for API use

### Retrieval Tool Interface
The retrieval tool provides a simple interface for the planner:
- Accepts plain text queries with optional section filters
- Returns chunks with raw metadata from vector store (no citation generation)
- Preserves exact metadata format from the vector store
- Provides clear error handling and metrics
- Maintains article-level context (returns all chunks from relevant articles)

The planner only needs:
- A list of valid section names for filtering
- The ability to pass plain text queries
- Access to the raw chunks with their metadata

Citation generation and context formatting are handled by the frontend LLM.

### Key Design Decisions

#### 1. Metadata Storage and Handling
- Metadata (section, URL, etc.) is stored separately in the vector store
- Raw metadata is preserved throughout the retrieval process
- Metadata is only formatted into headers at the final step
- No metadata is embedded in chunk content

#### 2. Article-Level Retrieval
- When any chunk from an article is relevant, ALL chunks from that article are retrieved
- This ensures complete context for the LLM
- Chunks are sorted by article URL and chunk index
- Article-level grouping is maintained throughout processing

#### 3. Context Assembly
1. **Retrieval**:
   - Find relevant articles through semantic search
   - Retrieve ALL chunks from relevant articles
   - Preserve raw metadata from vector store

2. **Deduplication**:
   - Remove overlapping content between chunks
   - Maintain article-level grouping
   - Preserve all metadata fields

3. **Formatting**:
   - Group chunks by article
   - Sort chunks within articles
   - Add consistent metadata headers
   - Prepare for LLM consumption

### Planner LLM
The planner will:
- Maintain conversation memory
- Reformulate queries when needed
- Decide when to perform retrieval
- Handle metadata filters
- Return structured payloads

### Frontend LLM
The frontend will:
- Maintain full chat history
- Inject context when needed
- Generate natural responses
- Include proper citations
- Handle conversation flow

## 🚀 Getting Started

1. Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Start with the retrieval tool:
```bash
cd Application/Retrieval\ tool/
# Implementation and testing
```

3. Then move to the planner:
```bash
cd Application/Agent/planner/
# Implementation and testing
```

4. Finally, implement the frontend:
```bash
cd Application/Agent/frontend/
# Implementation and testing
```

## 📚 References
- Project Outline: `project_outline_readme.md`
- Agentic Layer Design: `Agent/agentic_layer_readme.md`
- Original Implementation: `process content/Embedding/` 