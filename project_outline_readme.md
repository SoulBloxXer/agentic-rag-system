# 🧠 Knowledge Base Chatbot with RAG

## 🎯 What is this Project? (Non-Technical Overview)

This project creates a chatbot that can answer questions about shrink wrapping and packaging by using a carefully prepared knowledge base. The key features are:

1. **Accurate Answers**: The chatbot only uses information from our official knowledge base - no made-up or incorrect information
2. **Source Citations**: Every answer includes links to the original articles it used
3. **Context-Aware**: It understands the full context of questions and provides complete, relevant answers
4. **Reliable**: Uses advanced search to find the most relevant information, even when questions are asked in different ways

## 🤔 How Does it Work? (Simple Explanation)

Think of it like a very smart research assistant:

1. When you ask a question, the system:
   - Searches through our knowledge base to find relevant articles
   - Understands the meaning of your question (not just matching words)
   - Finds the most relevant information, even if you ask it differently
   - Combines information from multiple articles when needed
   - Always cites its sources

2. The system is built to be:
   - Accurate: Only uses information from our knowledge base
   - Transparent: Shows where information comes from
   - Helpful: Provides complete, contextual answers
   - Reliable: Works consistently for different types of questions

## 📚 What's in the Knowledge Base?

Our knowledge base contains about 100 articles covering:
- Troubleshooting guides
- Technical specifications
- Case studies
- Best practices
- And more...

Each article is carefully processed to ensure the chatbot can find and use the information effectively.

---

## 🔧 Technical Implementation Details

### Overview
This document outlines a robust Retrieval-Augmented Generation (RAG) strategy for transforming a large `.txt` knowledge base into a vector database with high retrieval accuracy and citation capability using LangChain.

### Knowledge Base Structure
* File: `knowledge_base_llms.txt`
* Contains ~100 articles
* Articles are separated by a consistent delimiter: `---`
* Each article contains Markdown-style metadata at the top, for example:

```markdown
> **URL:** <http://example.com/article>
> **Section:** 19 Eco Friendly
```

### Project Goals
Create an extremely **reliable and transparent chatbot** that can:

1. **Accurately retrieve relevant information** from our knowledge base
2. **Maintain article-level metadata** (specifically the `url` and `section`)
3. **Cite all source URLs** in its final response
4. **Allow metadata to support filtering** during the retrieval pipeline or be passed to the LLM during answer generation

The ultimate goal is to provide a chatbot that:
- Gives accurate, helpful answers
- Only uses information from our knowledge base
- Always shows where information comes from
- Can handle a wide variety of questions
- Provides complete, contextual responses

## 📁 Project Structure and Setup

### Application Directory Structure
```
Application/
├── Chunks/         # Processed and chunked knowledge base data
├── Embeddings/     # Vector embeddings for semantic search
├── Agent/          # Agentic layer implementation
│   ├── planner/    # Planner LLM implementation
│   └── frontend/   # Frontend LLM implementation
└── Retrieval tool/ # Optimized semantic search implementation
```

### Files Created

1. `requirements.txt`
   * Required Python dependencies
   * Run: `pip install -r requirements.txt`

2. `chunking/chunker.py`
   * Implemented chunking script.
   * Features:
     * Splits on `---`
     * Extracts metadata
     * Tokenises and sub-chunks with 256-token chunks and 64-token overlap using token-based slicing.
     * Outputs structured chunks to `chunking/chunks.json`.
     * Includes detailed logging (`chunking/chunking.log`).

3. `test_preprocessor.py`
   * (Existing) Pytest-based unit tests for initial preprocessing steps.

4. `chunking/chunks.json`
   * Output file containing the processed and chunked knowledge base data.

5. `chunking/chunking.log`
   * Log file detailing the chunking process, statistics, and any warnings/errors.

### Getting Started

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## 📆 Project Roadmap

### Phase 1: Data Preprocessing ✅

- Split raw text on `--`
- Extract `url`, `section`
- Output structured article blocks
- Implement logging and statistics

### Phase 2: Chunking Strategy ✅

- Analyse token distribution
- Implement 256-token chunks with 64-token overlap
- Preserve metadata and relationships
- Handle overlapping content

### Phase 3: Vector Database Setup ✅

- Convert chunks to `Document` format
- Generate normalised OpenAI embeddings
- Store in Chroma vector DB with metadata
- Implement token-normalised semantic search

### Phase 4: Retrieval and Context Assembly ✅

- Implement semantic search with adaptive thresholds
- Enable explicit metadata filtering (e.g. `section:` or `url:`)
- Group chunks by article
- Handle overlapping content cleanly
- Prepare structured LLM input
- Implement fallback mechanism for low-scoring queries

### Phase 5: Testing & Validation ✅

- Validate chunking accuracy
- Test retrieval relevance (manual + score-based)
- Verify citation extraction
- Confirm `section:` filter behaviour
- Measure answer coherence and completeness
- Collect and analyse user feedback

### Phase 6: Retrieval Tool Optimization 🔄

- Move and refactor code from `test_queries.py` to `Application/Retrieval tool/`
- Implement proper error handling
- Add comprehensive logging
- Optimize for performance
- Add caching if needed
- Make it API-friendly
- Add unit tests and benchmarks

### Phase 7: Agentic Planning Layer 🔄

This phase introduces a **two-LLM hybrid system** implemented in the `Application/Agent/` directory:

1. **Planner LLM** (First Implementation)
   - Implemented in `Agent/planner/`
   - Uses GPT-4 (temperature 0) to:
     - Track conversation memory
     - Reformulate vague or shorthand queries
     - Decide whether to trigger retrieval
     - Handle metadata filters
   - Returns structured payload:
     ```json
     {
       "user_query": "string",
       "retrieved_chunks": ["chunk 1", "chunk 2", ...],
       "retrieval_skipped": true | false
     }
     ```

2. **Frontend LLM** (Second Implementation)
   - Implemented in `Agent/frontend/`
   - Uses GPT-4 (temperature 0.6) to:
     - Generate natural, helpful responses using retrieved context
     - Maintain full conversation history with the user
     - Support citation via embedded chunk metadata
   - Handles:
     - Context injection
     - Citation formatting
     - Natural response generation
     - Conversation flow

### Phase 8: API Layer (Planned)

- Create FastAPI/Flask wrapper
- Implement request/response schemas
- Add authentication
- Add rate limiting
- Add proper error handling
- Add monitoring and logging
- Create deployment documentation
- Add frontend integration guide

## 🛠️ Implementation Details

### Search Implementation
- Adaptive thresholds based on score distribution:
  - Primary (tighter) thresholds using 10th percentile
  - Fallback to original thresholds (20th percentile) if needed
  - Final fallback to top 2 results
- Combined scoring (70% semantic, 30% keyword)
- Full article retrieval when any chunk is relevant
- Metadata filtering by section and URL
- Overlap detection and handling
- Clear article boundaries with dash-based formatting

### Answer Generation and Citation

Our system employs a sophisticated multi-stage process for generating accurate, well-cited responses:

1. **Chunk Retrieval and Assembly**
   - Retrieve relevant chunks using semantic search with adaptive thresholds
   - Fallback to top results if no articles meet threshold
   - Group chunks by article for context coherence
   - Handle overlapping content through deduplication
   - Maintain chunk ordering within articles
   - Preserve article-level metadata
   - Use combined scoring (70% semantic, 30% keyword)

2. **Context Preparation**
   ```python
   # Example of assembled context structure
   --- Article: Troubleshooting Guide ---
   Section: 2 Technical Guides
   Source: http://example.com/article1
   Relevance score: 0.85
   ---
   [Chunk 1 content]
   [Chunk 2 content]

   --- Article: Technical Specifications ---
   Section: 3 Product Info
   Source: http://example.com/article2
   Relevance score: 0.82
   ---
   [Chunk 1 content]
   [Chunk 2 content]
   ```

3. **Citation and Metadata Handling**
   - Parse and deduplicate source URLs
   - Include section information for context
   - Track chunk relevance scores and adaptive thresholds
   - Maintain article boundaries with dash-based formatting
   - Support filtering by metadata (section and URL)
   - Log threshold calculations and filter impact
   - Preserve full article context when any chunk is relevant
   - Use tighter thresholds with fallback mechanism

## 🔄 Current Status

✅ Completed:
- Knowledge base processing
- Chunking strategy
- Vector database setup
- Basic semantic search
- Testing and validation

🔄 In Progress:
- Retrieval tool optimization
- Agentic layer implementation (Planner LLM first)

📝 Next Steps:
1. Complete retrieval tool optimization
2. Implement Planner LLM
3. Implement Frontend LLM
4. Create API layer
5. Add deployment documentation

## 📚 References
- Agentic Layer Design: `Application/Agent/agentic_layer_readme.md`
- Original Implementation: `process content/Embedding/`