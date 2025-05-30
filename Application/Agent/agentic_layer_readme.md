# 🤖 Agentic Planner Layer — RAG Control System

## 📋 Overview

This module implements a **dedicated LangChain agent** ("Agentic Planner") that sits **on top of a pre-existing semantic retrieval pipeline**, orchestrating intelligent query reformulation, retrieval decision-making, and chunk-payload generation for a downstream frontend LLM.

It does **not** generate final user-facing responses. Instead, it builds a structured payload that includes the current query, any retrieved chunks (if applicable), and decision metadata — allowing the frontend LLM to maintain conversational flow, reuse context, and respond fluently.

---

## ✅ Goals

1. **Separate control from expression**

   * **Planner LLM**: A backend decision-maker (e.g. GPT-4omini @ temperature 0) that handles memory, reformulation, and when/how to call the retriever
   * **Frontend LLM**: A user-facing responder (e.g. GPT-4o @ temperature 0.6) that generates natural, empathetic answers using chunks + full conversation history

2. **Enforce intelligent retrieval logic**

   * Always retrieve unless the query is trivial or continuation of prior topic
   * Never resend previous chunks — let frontend LLM reuse them via memory

3. **Support metadata filtering**

   * Allow optional filters in query (e.g. `section: X`) to scope semantic search

4. **Maintain clean, predictable interfaces**

   * Single structured payload returned every time
   * Explicit flags control frontend logic

---

## 🧠 High-Level System Flow

```
[User Message]
     ↓
[Agentic Planner LLM] (temperature = 0)
  • Maintains short-term memory (LangChain ConversationBufferMemory)
  • Reformulates vague or shorthand queries
  • Decides whether retrieval is required
      ↪ YES: invokes retriever, gets top-k chunks
      ↪ NO: sets retrieval_skipped=True
  • Returns:
      {
        "user_query": "...",
        "retrieved_chunks": [...],  # may be empty
        "retrieval_skipped": true/false
      }
     ↓
[Frontend LLM] (temperature = 0.6)
  • Maintains full chat history (including injected chunks)
  • If new chunks are returned → inject via system message
  • If no new chunks → reuse prior context
  • Generates natural, context-aware reply
  • Extracts citations from metadata in each chunk and includes them in response
     ↓
[UI Layer]
  • Displays assistant response
```

---

## 📦 Payload Schema

```python
{
    "user_query": str,                   # Original user message
    "retrieved_chunks": List[str],      # Raw text chunks (with metadata embedded)
    "retrieval_skipped": bool           # Whether a retrieval tool call was made
}
```

Example:
```python
{
    "user_query": "What if the wrap is breaking at the edges?",
    "retrieved_chunks": [
        "--- Article: Common Shrink Wrap Issues ---\n"
        "Section: 2 Troubleshooting\n"
        "Source: http://...\n"
        "---\n"
        "Shrink wrap may tear at the edges due to...",
        # ... more chunks ...
    ],
    "retrieval_skipped": False
}
```

> Note: The retrieved chunks are simple strings with embedded metadata headers.
> The frontend LLM only needs these three fields to function.
> All other metrics and timing data are for internal monitoring only.

## 🔧 Internal Monitoring (Not Sent to Frontend)

The planner and retriever track detailed metrics for monitoring and debugging:

1. **Retrieval Metrics**:
   - Retrieval time (vector store + scoring)
   - Total response time (end-to-end)
   - Threshold type and value
   - Filter matching details
   - Fallback events

2. **Performance Tracking**:
   - Query processing time
   - Chunk processing time
   - Memory usage
   - Cache hit rates

3. **Error Tracking**:
   - Filter validation results
   - Threshold calculations
   - Fallback reasons
   - Section matching details

These metrics are logged and monitored internally but are not part of the payload sent to the frontend LLM.

---

# 🧱 Planner Agent Responsibilities

### ✅ Reformulation

* Rewrites vague questions using conversation history
* Example:

  * User input: "What are common problems?"
  * Reformulated: "What are common shrink wrap problems?"

### ✅ Filter Recognition

* Looks for any explicitly hinted `section` filters in user query (this wouldn't be common, and is an edge case. Filters must not be inferred from vague references.)
* If present, passes them into retriever in the format `"\<query>" " | section: \<section name>"

  * Will be provided a list of all section names for querying filters correctly
* If not, performs global semantic search

### ✅ Retrieval Decision Logic

```python
if user_query is trivial (e.g. "thanks", "how are you?"):
    retrieval_skipped = True
    retrieved_chunks = []
elif planner detects topic continuation (and no new info needed):
    retrieval_skipped = True
    retrieved_chunks = []
else (a new semantic search is required):
    retrieval_skipped = False
    retrieved_chunks = call_retriever(reformulated_query)
```

> The agent **must not return previous chunks** — reuse is handled by frontend memory.

## Filter Recognition and Handling

The planner LLM handles section filtering through explicit filter recognition:

1. **Explicit Filter Format**:
   - Filters must be explicitly stated in the query using the format: `| section: SectionName`
   - Example: `"What are common shrink wrap issues | section: Troubleshooting"`
   - The planner does NOT attempt to infer filters from vague phrases
   - No implicit filtering (e.g., "tell me about case studies" does not automatically filter to Case Studies section)

2. **Filter Processing**:
   - Planner extracts explicit filters from the query
   - Validates against known section names
   - Passes clean filter to retriever
   - Handles filter-related errors gracefully

3. **Filter Examples**:
   ```python
   # Valid filter formats:
   "query | section: Troubleshooting"
   "query | section: 2 Troubleshooting"
   "query | section: Case Studies"

   # Invalid (will not be filtered):
   "tell me about case studies"  # No explicit filter
   "query | section: unknown"    # Invalid section
   "query | filter: something"   # Wrong filter format
   ```

## Payload Format

The planner returns a structured payload to the frontend LLM:

```python
{
    "user_query": "original user query",
    "retrieved_chunks": [
        # Each chunk is formatted as:
        "--- Article: Article Title ---\n"
        "Section: Full Section Name\n"
        "Source: URL\n"
        "---\n"
        "Content...",
        # ... more chunks ...
    ],
    "retrieval_skipped": False,  # or True if no retrieval needed
    "metrics": {
        "retrieval_time": 0.45,      # Time spent in vector store + scoring
        "total_response_time": 0.52,  # End-to-end time including formatting
        "threshold_used": 0.82,
        "threshold_type": "tight",    # or "fallback"
        "filter_applied": {"section": "2 Troubleshooting"},
        "fallback_reason": null       # or explanation if fallback used
    }
}
```

## Performance Metrics

The system tracks detailed timing metrics to help identify bottlenecks:

1. **Retrieval Time** (`retrieval_time`):
   - Time spent in vector store search
   - Time for similarity scoring
   - Time for threshold calculation
   - Does NOT include formatting or fallback

2. **Total Response Time** (`total_response_time`):
   - End-to-end processing time
   - Includes retrieval time
   - Includes chunk formatting
   - Includes any fallback processing
   - Includes article assembly

3. **Threshold Metrics**:
   - Type of threshold used (tight/fallback)
   - Actual threshold value
   - Reason for fallback if used
   - Distribution statistics

4. **Filter Metrics**:
   - Applied filter details
   - Filter matching time
   - Filter validation results

---

## 🔌 Integration Contract

### Input:

* `chat_history: List[Dict[str, str]]` (from frontend or session memory)
* `user_input: str`

### Output:

* Structured payload (see schema above)

### Example:

```python
{
  "user_query": "What if the wrap is breaking at the edges?",
  "retrieved_chunks": [
    "--- Article: Common Shrink Wrap Issues ---\nSection 2: Troubleshooting\nSource: http://...\n---\nShrink wrap may tear at the edges due to...",
    ...
  ],
  "retrieval_skipped": False
}
```

---

## 🔍 Retriever Tool Requirements

* Accepts **plain-text queries**, including any filters embedded (e.g. `| section: Troubleshooting`)
* Returns top-k chunks with full metadata embedded directly (as a plain string)

> No changes are required to the retriever tool — planner agent reformulates and passes compatible queries.

---

## 🧠 Memory Strategy

* Planner LLM uses **LangChain** `ConversationBufferMemory` chunks. (the **frontend LLM** is the one that maintains full chat history, including prior system messages with injected chunks, allowing it to reuse them across turns.)
* It only influences:

  * Query reformulation
  * Decision to skip retrieval

---

# 🧩 Frontend LLM Design (Summary)

Handled separately. See full frontend readme, but summary is:

* `retrieved_chunks` injected into a `system` at runtime

  * If `retrieved_chunks` is empty, the frontend LLM **reuses previously injected chunks** (retained in system messages from earlier turns). This avoids redundant retrieval.
* Assistant generates final reply based on:

  * Current user input
  * Previous dialogue
  * Context from prior turns

---

## 🛠️ Key Components in Codebase

* `agentic_planner.py`

  * Main class that implements:

    * `plan(user_query, chat_history)`
    * `_reformulate()`
    * `_should_skip_retrieval()`
    * `_extract_filter()`

* `retriever.py`

  * Existing tool-based semantic retriever

* `frontend_llm.py`

  * Where `retrieved_chunks` and original user input are injected into the system message and prompt

---

## ✅ Summary

This agentic planner module upgrades a naive RAG pipeline into a **fully decision-aware control layer**, enabling:

* Memory-aware query reformulation
* Filter-aware semantic search
* Clean separation of retrieval vs generation
* Reduced token waste via intelligent retrieval skipping

It pairs cleanly with any frontend chat LLM and preserves transparent, citation-based responses via chunk-level metadata.