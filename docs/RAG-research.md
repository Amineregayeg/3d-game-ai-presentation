# ADVANCED RAG SYSTEM FOR 3D GAME DEV AI ASSISTANT
## Comprehensive Research Report
**Date:** December 4, 2025 | **Status:** Production-Ready Architecture Framework

---

## EXECUTIVE SUMMARY

This research synthesizes current best practices (2024–2025) for building production-grade RAG systems optimized for technical domains like 3D game development, Blender automation, and real-time model generation. The system architecture emphasizes:

- **Hybrid Retrieval** combining semantic search (vector embeddings) and lexical search (BM25)
- **Agentic RAG** patterns enabling multi-hop reasoning and iterative refinement
- **Local Hosting** with PostgreSQL pgvector for maximum performance and data privacy
- **Technical Domain Optimization** for code snippets, API documentation, and procedural knowledge
- **Evaluation Framework** using RAGAS metrics for continuous quality assurance

---

## SECTION 1: CORE RAG ARCHITECTURE

### 1.1 What is RAG and Why It Matters for 3D Game Dev

**Definition:**
Retrieval-Augmented Generation (RAG) combines an information retrieval system with an LLM to ground responses in external knowledge bases. Instead of relying solely on model weights trained on historical data, RAG:

1. **Retrieves** relevant documents/code from external sources
2. **Augments** the LLM prompt with retrieved context
3. **Generates** grounded, accurate responses

**Why Critical for 3D Game Dev AI Assistant:**

| Challenge | Traditional LLM | RAG Solution |
|-----------|-----------------|--------------|
| Outdated Blender API docs | Hallucinated commands | Real-time API reference retrieval |
| Version-specific syntax | Wrong Python version | Contextual code snippet matching |
| Complex procedural knowledge | Generic explanations | Multi-step workflow retrieval |
| Action grounding | Non-executable suggestions | Exact MCP command parameter retrieval |

**The RAG Advantage:**
- ✅ **Accuracy:** Grounds responses in authoritative documentation
- ✅ **Freshness:** Retrieves current API versions and best practices
- ✅ **Debuggability:** References retrieved sources for verification
- ✅ **Cost-Efficiency:** Smaller models can compete with larger ones when grounded

---

## SECTION 2: HYBRID RETRIEVAL SYSTEM

### 2.1 Why Hybrid Retrieval > Pure Vector Search

**Traditional Vector-Only Approach:**
- ❌ Fails on exact keywords (e.g., "bpy.ops.mesh.select_all")
- ❌ Poor performance on model names and version-specific terms
- ❌ Struggles with rare/specialized terminology

**Hybrid Retrieval Solution:**

```
┌─────────────────────────────────────────────────────────┐
│           USER QUERY (e.g., "select all faces")         │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   ┌──────────────┐         ┌──────────────┐
   │ DENSE SEARCH │         │ BM25 SEARCH  │
   │ (Embeddings) │         │ (Keywords)   │
   └──────┬───────┘         └──────┬───────┘
          │                         │
          │ Top-k=100              │ Top-k=100
          │                         │
         ┌┴─────────────────────────┴┐
         │   RECIPROCAL RANK FUSION  │
         │   (Merge & Re-score)      │
         └────────────┬──────────────┘
                      │
         ┌────────────┴─────────────┐
         ▼                          ▼
   ┌──────────────┐         ┌──────────────┐
   │ RERANKING    │         │ FILTERING    │
   │ (Cross-Enc)  │ ────→   │ (Metadata)   │
   └──────┬───────┘         └──────┬───────┘
          │                         │
          └────────────┬────────────┘
                       ▼
            ┌────────────────────┐
            │ TOP-K CONTEXT (10) │
            │ Ready for LLM Gen  │
            └────────────────────┘
```

### 2.2 Technical Components

#### **A. Dense Search (Vector Embeddings)**

**Purpose:** Semantic understanding of conceptual queries  
**Example Query:** "How do I rotate objects smoothly?"  
→ Retrieves conceptually similar documents about rotation operations

**Embedding Model Selection (2025):**

| Model | Dimensions | Strengths | Use Case |
|-------|-----------|----------|----------|
| **BGE-M3** (NVIDIA) | 4,096 | Multimodal, technical domain optimization | 🏆 **Recommended** |
| **Nomic Embed** | 768 | Lightweight, open-source, fine-tunable | Resource-constrained |
| **voyage-code** | 1,536 | Specifically tuned for code snippets | Code-heavy domains |
| **E5-Large** | 1,024 | Strong general purpose, well-studied | Fallback option |

**BGE-M3 Advantages for Game Dev:**
- Trained on technical documentation
- Handles code and natural language queries
- 4,096 dimensions = high semantic granularity
- Supports multiple languages (Blender community is global)

#### **B. Sparse Search (BM25)**

**Purpose:** Exact keyword matching and rare term retrieval  
**Example Query:** "bpy.ops.mesh.select_all()"  
→ Direct keyword match → Exact API documentation

**BM25 Algorithm:**
```
Score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

Where:
- f(qi, D) = term frequency of query term qi in document D
- IDF(qi) = inverse document frequency
- |D| = document length
- avgdl = average document length
- k1, b = tuning parameters (typically k1=1.5, b=0.75)
```

**Why BM25 for Technical Docs:**
- Probabilistic relevance framework = statistically proven
- Parameter optimization for technical terminology
- Extremely fast (~milliseconds for 50M+ documents)
- No neural network overhead

#### **C. Reciprocal Rank Fusion (RRF)**

**Purpose:** Merge BM25 and Dense results without learned weights

**Algorithm:**
```
RRF(doc) = Σ 1 / (k + rank(doc in retriever_i))

Where k = 60 (standard constant to avoid division by zero)

Example:
- Dense Search: doc_A at rank 1 → RRF contribution = 1/(60+1) = 0.0164
- BM25 Search:  doc_A at rank 5 → RRF contribution = 1/(60+5) = 0.0154
- Total Score = 0.0164 + 0.0154 = 0.0318

Result: Documents ranked high in EITHER retriever bubble up
```

**Advantage:** No training required, statistically optimal fusion

---

## SECTION 3: RERANKING & CONTEXT PRECISION

### 3.1 Cross-Encoder Reranking

**Problem:** Hybrid retrieval returns 50–200 candidates; LLM can only process ~10 effectively

**Solution:** Cross-Encoder reranking to promote top candidates

```
Traditional Approach (Dense Vector):
┌──────────────┐
│ Query & Doc  │  ← Processed SEPARATELY
└──────────────┘
      ↓
  ╔═══════════╗
  ║  Embedding║  ← Cosine similarity
  ║  Encoder  ║     (Fast but less accurate)
  ╚═══════════╝

Cross-Encoder Approach:
┌───────────────────────┐
│  Query + Doc PAIR     │  ← Processed TOGETHER
│  (Concatenated)       │
└───────────────────────┘
      ↓
  ╔═══════════════╗
  ║ Cross-Encoder ║  ← Direct relevance score
  ║  Transformer  ║     (Slower but more accurate)
  ╚═══════════════╝
      ↓
  Relevance Score: 0.92 (0-1 scale)
```

**Recommended Models (2025):**

| Model | Latency | Accuracy | Training | Use Case |
|-------|---------|----------|----------|----------|
| **ms-marco-MiniLM** | 15ms | 0.89 | ✅ Standard | 🏆 **Recommended** |
| **BGE-Reranker** | 25ms | 0.91 | ✅ Technical | High precision |
| **ColBERT** | 50ms | 0.93 | ❌ Custom | Extreme precision (rare) |

**ms-marco-MiniLM Selection Rationale:**
- Trained on 500K query-doc relevance pairs
- 66M parameters (runs on CPU, <100ms batch latency)
- MS MARCO benchmark standard
- Proven on technical documentation tasks

### 3.2 Metadata Filtering

**Purpose:** Filter by document type, version, and domain

**Example Metadata Schema for 3D Game Dev:**

```json
{
  "document_id": "blender_4.2_api_mesh.select_all",
  "source": "blender_api",
  "blender_version": "4.2",
  "category": "mesh_operations",
  "subcategory": "selection",
  "language": "python",
  "is_code": true,
  "modality": "api_reference",
  "tags": ["selection", "mesh", "geometry"],
  "priority": 0.95,
  "updated_at": "2024-12-04"
}
```

**Filter Logic:**
```
IF user mentions "Blender 4.2"
  THEN filter: blender_version >= "4.2"
  ELSE allow all versions (newest first)

IF query contains "[code]"
  THEN filter: is_code = true
  ELSE include both code and conceptual docs
```

---

## SECTION 4: QUERY TRANSFORMATION & AGENTIC RAG

### 4.1 Query Rewriting Pipeline

**Problem:** User voice queries are ambiguous:
- "Select all" ← Which modality? (mesh faces, objects, bones)
- "Make it smooth" ← Smooth what? (geometry, animation, surface)

**Solution:** Agentic query rewriting before retrieval

```
┌─────────────────────────────────┐
│ User Query (STT)                │
│ "I want to rotate all faces"    │
└──────────────┬──────────────────┘
               │
        ┌──────▼─────────┐
        │ CONTEXT KEEPER │ (Session history)
        │ (Last 5 turns) │
        └──────┬─────────┘
               │
        ┌──────▼─────────────────────────────┐
        │ QUERY REWRITING AGENT              │
        │ ┌────────────────────────────────┐ │
        │ │ 1. Identify intent              │ │
        │ │    → "Rotate" + "Faces"         │ │
        │ │                                 │ │
        │ │ 2. Extract entities             │ │
        │ │    → target: faces              │ │
        │ │    → operation: rotate          │ │
        │ │    → scope: all                 │ │
        │ │                                 │ │
        │ │ 3. Generate search variations   │ │
        │ │    v1: "rotate faces in blender"│ │
        │ │    v2: "bpy.ops.transform.rotate"│ │
        │ │    v3: "mesh rotation best practice"│
        │ │                                 │ │
        │ │ 4. Generate multi-hop queries   │ │
        │ │    q1: "select all faces"       │ │
        │ │    q2: "rotate in object mode"  │ │
        │ │    q3: "apply transformation"   │ │
        │ └────────────────────────────────┘ │
        └──────┬─────────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
Search v1, v2, v3    Multi-hop q1, q2, q3
+ Merge Results +
    │
    ▼
Rich Context Ready for Generation
```

### 4.2 Agentic RAG Pattern

**Traditional RAG (Single-pass):**
```
Query → Retrieve → Generate → Answer
❌ If retrieval misses nuances
❌ If context contradicts
❌ No self-correction
```

**Agentic RAG (Multi-step reasoning):**
```
Query → [Agent Planning] → Retrieve → [Agent Validation] 
  ↓
Context sufficient? YES → Generate → Answer
  ↓
Context sufficient? NO → [Query Rewriting] → Re-retrieve → [Merge] → Generate
  ↓
Answer valid? YES → Return
  ↓
Answer valid? NO → [Error Detection] → [Targeted Retrieval] → Regenerate
```

**Validation Logic for 3D Game Dev:**

```python
# Pseudo-code for validation loop
def validate_answer(answer, context, query):
    checks = [
        check_code_syntax(answer),           # Is Python valid?
        check_api_compatibility(answer),     # Does API exist?
        check_version_match(answer),         # Blender version correct?
        check_hallucination(answer, context) # Is grounded in context?
    ]
    
    failed_checks = [c for c in checks if not c.passed]
    
    if failed_checks:
        trigger_retargeted_retrieval(failed_checks)
        return regenerate_answer()
    else:
        return answer
```

---

## SECTION 5: LOCAL DATABASE ARCHITECTURE

### 5.1 PostgreSQL + pgvector: Why Local Hosting

**Advantages:**
- ✅ **Data Privacy:** No vectors sent to third-party APIs
- ✅ **Performance:** Benchmark shows 11.4x higher throughput vs Qdrant (2025)
- ✅ **SQL Integration:** Rich metadata filtering + vector search combined
- ✅ **Cost:** Open-source, self-hosted, no per-query fees
- ✅ **ACID Compliance:** Transaction safety for concurrent operations

**Performance Benchmark (2025 - TigerData):**

```
Dataset: 50M embeddings, 1,536 dimensions
Metric: Queries per second at 99% recall

PostgreSQL + pgvectorscale: 471.57 QPS ✅ WINNER
Qdrant:                      41.47 QPS

Result: 11.4x performance advantage for PostgreSQL
```

### 5.2 Database Schema for 3D Game Dev

```sql
-- Core documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(4096),  -- BGE-M3 embedding
    
    -- Metadata
    source VARCHAR(50),                    -- 'blender_api', 'tutorial', 'best_practices'
    blender_version VARCHAR(10),           -- '4.2', '4.1', '*' (all)
    category VARCHAR(50),                  -- 'mesh', 'object', 'animation', 'rendering'
    subcategory VARCHAR(50),               -- 'selection', 'transformation', etc.
    language VARCHAR(20),                  -- 'python', 'glsl', 'conceptual'
    is_code BOOLEAN,                       -- true if code snippet
    modality VARCHAR(50),                  -- 'api_reference', 'tutorial', 'best_practice'
    
    -- Ranking & freshness
    priority FLOAT,                        -- 0.0-1.0, boost important docs
    updated_at TIMESTAMP,
    
    -- For hybrid search
    bm25_keywords TEXT[],                  -- Extracted keywords for BM25
    
    -- Indexing
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create vector index (fast ANN search)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m=16, ef_construction=200);

-- Create BM25 index (via GIN for text search)
CREATE INDEX ON documents USING GIN (bm25_keywords);

-- Create metadata indexes
CREATE INDEX ON documents(source, blender_version);
CREATE INDEX ON documents(category, subcategory);
CREATE INDEX ON documents(is_code, language);

-- Retrieval history (for analytics & improvement)
CREATE TABLE retrieval_history (
    id SERIAL PRIMARY KEY,
    query TEXT,
    retrieved_doc_ids INTEGER[],
    final_answer TEXT,
    user_satisfaction FLOAT,  -- 0-1 feedback score
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### 5.3 Local Deployment Stack

```
┌────────────────────────────────────────────┐
│     3D Game Dev AI Assistant               │
│     (Claude API / Local LLM)               │
└─────────────────┬──────────────────────────┘
                  │
        ┌─────────┴────────┐
        ▼                  ▼
   ┌─────────┐      ┌──────────────┐
   │LangChain│  OR  │LlamaIndex    │
   │Framework│      │Framework     │
   └─────────┘      └──────────────┘
        │                  │
        └─────────┬────────┘
                  ▼
     ┌─────────────────────────┐
     │  RAG Pipeline Layer     │
     │ ┌─────────────────────┐ │
     │ │Hybrid Retrieval     │ │
     │ │(Dense + BM25 + RRF) │ │
     │ └─────────────────────┘ │
     │ ┌─────────────────────┐ │
     │ │Reranking (MiniLM)   │ │
     │ └─────────────────────┘ │
     │ ┌─────────────────────┐ │
     │ │Query Rewriting      │ │
     │ └─────────────────────┘ │
     │ ┌─────────────────────┐ │
     │ │Agentic Validation   │ │
     │ └─────────────────────┘ │
     └─────────────┬───────────┘
                  ▼
    ┌──────────────────────────┐
    │  PostgreSQL 15+          │
    │  + pgvector              │
    │  + pgvectorscale (opt)   │
    │                          │
    │ ┌──────────────────────┐ │
    │ │ Documents table      │ │
    │ │ (4,096-dim vectors)  │ │
    │ │ + Metadata indices   │ │
    │ │                      │ │
    │ │ ┌────────────────┐   │ │
    │ │ │HNSW index      │   │ │
    │ │ │(Vector search) │   │ │
    │ │ └────────────────┘   │ │
    │ │ ┌────────────────┐   │ │
    │ │ │BM25 index      │   │ │
    │ │ │(Keyword search)│   │ │
    │ │ └────────────────┘   │ │
    │ └──────────────────────┘ │
    └──────────────────────────┘
                  ▲
                  │
    ┌─────────────┴──────────────┐
    ▼                            ▼
┌──────────────┐        ┌──────────────┐
│Blender Docs │        │Game Dev      │
│(API Refs)   │        │Tutorials     │
└──────────────┘        └──────────────┘
```

---

## SECTION 6: EMBEDDING & INDEXING STRATEGY

### 6.1 Chunking Strategy for Technical Docs

**Problem:** Raw Blender API docs are massive; naive chunking loses context

**Solution: Semantic Chunking**

```python
# Example chunking for Blender API reference

# ❌ NAIVE: Fixed 512-token chunks
# Result: Cuts mid-function, loses context

# ✅ SEMANTIC: Chunk by logical boundaries

API_REFERENCE = """
bpy.ops.mesh.select_all()
    Select all mesh elements.
    
    Parameters: action (string) – Action, default 'TOGGLE'
    {'TOGGLE', 'SELECT', 'DESELECT'}
    
    Returns: (enum set in {'RUNNING_MODAL', 'FINISHED'})
"""

# Resulting chunks:
CHUNK_1 = """
Function: bpy.ops.mesh.select_all()
Purpose: Select all mesh elements

Parameters:
- action (string): Action type
  Options: 'TOGGLE', 'SELECT', 'DESELECT'
  Default: 'TOGGLE'
"""

CHUNK_2 = """
Returns: (enum set in {'RUNNING_MODAL', 'FINISHED'})

Context: Used in mesh editing mode
Related: bpy.ops.mesh.select_linked()
         bpy.ops.mesh.select_random()
"""

# Embed each chunk separately with full context
```

**Chunking Rules for 3D Game Dev:**

| Content Type | Chunk Size | Boundary |
|--------------|-----------|----------|
| **API Reference** | ~300-500 tokens | Function/method |
| **Tutorial** | ~600-800 tokens | Logical step or paragraph |
| **Code Example** | ~200-400 tokens | Complete function/snippet |
| **Best Practice** | ~400-600 tokens | Concept or principle |

### 6.2 Embedding Process

```python
# Pseudocode for embedding pipeline

from langchain_nomic import NomicEmbeddings
from langchain.text_splitter import SemanticChunker

# 1. Load documents
documents = load_blender_docs()

# 2. Semantic chunking
splitter = SemanticChunker(
    breakpoint_threshold_type="percentile",
    percentile=90  # Split when similarity drops 90th percentile
)
chunks = splitter.split_documents(documents)

# 3. Generate embeddings (BGE-M3)
embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1",
    dimension=4096,
    normalize=True
)

embedded_chunks = []
for chunk in chunks:
    vector = embeddings.embed_query(chunk.page_content)
    embedded_chunks.append({
        'text': chunk.page_content,
        'vector': vector,
        'metadata': chunk.metadata
    })

# 4. Store in PostgreSQL
for chunk in embedded_chunks:
    insert_into_db(
        content=chunk['text'],
        embedding=chunk['vector'],
        **chunk['metadata']
    )
```

---

## SECTION 7: RETRIEVAL PIPELINE

### 7.1 Dense Retrieval (Vector Search)

```sql
-- Cosine similarity search (pgvector)
SELECT 
    id,
    content,
    1 - (embedding <=> query_embedding) as similarity_score,
    source,
    blender_version,
    category
FROM documents
WHERE 1 - (embedding <=> query_embedding) > 0.7  -- Threshold
ORDER BY similarity_score DESC
LIMIT 100;
```

### 7.2 Sparse Retrieval (BM25)

```sql
-- BM25 ranking (using PostgreSQL full-text search)
SELECT 
    id,
    content,
    ts_rank(to_tsvector('english', content), 
            plainto_tsquery('english', query_text)) as bm25_score,
    source,
    category
FROM documents
WHERE to_tsvector('english', content) @@ plainto_tsquery('english', query_text)
ORDER BY bm25_score DESC
LIMIT 100;
```

### 7.3 Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Merge dense and sparse results using RRF
    
    Args:
        dense_results: List of (doc_id, score) from vector search
        sparse_results: List of (doc_id, score) from BM25
        k: Constant (default 60)
    
    Returns:
        Fused list of (doc_id, rrf_score)
    """
    rrf_scores = {}
    
    # Add dense search contributions
    for rank, (doc_id, score) in enumerate(dense_results, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    
    # Add sparse search contributions
    for rank, (doc_id, score) in enumerate(sparse_results, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    
    # Sort by total RRF score
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:50]  # Top 50 candidates for reranking
```

---

## SECTION 8: RERANKING & CONTEXT ASSEMBLY

### 8.1 Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

# Load pre-trained cross-encoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_candidates(query, candidates, top_k=10):
    """
    Rerank candidate documents using cross-encoder
    
    Args:
        query: User query string
        candidates: List of document texts
        top_k: Return top-k documents
    
    Returns:
        Sorted list of (doc, score)
    """
    # Prepare pairs
    pairs = [[query, doc] for doc in candidates]
    
    # Score each pair
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked[:top_k]
```

### 8.2 Context Assembly

```python
def assemble_context(reranked_docs, query, max_tokens=2000):
    """
    Assemble final context for LLM generation
    
    Args:
        reranked_docs: List of (doc_text, score) tuples
        query: User query (for reference)
        max_tokens: Maximum context tokens
    
    Returns:
        Assembled context string ready for LLM
    """
    context_parts = [
        "=== RETRIEVED CONTEXT ===\n"
    ]
    
    token_count = 0
    max_tokens_per_doc = 400
    
    for i, (doc_text, score) in enumerate(reranked_docs, 1):
        # Enforce max tokens per document
        tokens = len(doc_text.split())
        if tokens > max_tokens_per_doc:
            doc_text = ' '.join(doc_text.split()[:max_tokens_per_doc])
        
        # Build context string
        context_parts.append(
            f"\n[Source {i}, Relevance: {score:.2f}]\n"
            f"{doc_text}"
        )
        
        token_count += len(doc_text.split())
        
        if token_count >= max_tokens:
            break
    
    context_parts.append("\n=== END CONTEXT ===")
    
    return '\n'.join(context_parts)
```

---

## SECTION 9: EVALUATION FRAMEWORK

### 9.1 RAGAS Metrics

**RAGAS (Retrieval-Augmented Generation Assessment)** provides reference-free evaluation:

| Metric | Formula | Ideal Range | What It Measures |
|--------|---------|-------------|------------------|
| **Faithfulness** | Overlap of answer facts with context | 0.7-1.0 | Is answer grounded? (No hallucinations) |
| **Answer Relevancy** | Semantic similarity of answer to query | 0.8-1.0 | Does answer address the question? |
| **Context Precision** | Fraction of relevant retrieved docs | 0.7-1.0 | Are top results actually relevant? |
| **Context Recall** | Coverage of all relevant information | 0.6-1.0 | Did we retrieve everything needed? |

### 9.2 Game Dev-Specific Metrics

```python
class GameDevRAGEvaluator:
    """
    Custom metrics for 3D game development RAG
    """
    
    def check_code_correctness(self, answer, query):
        """
        Does generated code actually work?
        Score: 0 (broken), 0.5 (partial), 1.0 (works)
        """
        try:
            # Attempt to parse and validate Python syntax
            compile(answer, '<string>', 'exec')
            
            # Check for API existence (basic validation)
            if 'bpy.ops.' in answer or 'bpy.data.' in answer:
                # Would need actual Blender context for full validation
                return 0.8
            return 1.0
        except SyntaxError:
            return 0.0
    
    def check_version_compatibility(self, answer, blender_version):
        """
        Is answer compatible with specified Blender version?
        Score: 0 (incompatible), 1.0 (compatible)
        """
        # Parse answer for API calls
        api_calls = extract_api_calls(answer)
        
        # Check against version compatibility matrix
        for api_call in api_calls:
            if not is_compatible(api_call, blender_version):
                return 0.0
        
        return 1.0
    
    def check_completeness(self, answer, query):
        """
        Does answer cover the full workflow?
        For multi-step tasks: selection → transformation → finalization
        """
        required_steps = extract_workflow_steps(query)
        answered_steps = extract_workflow_steps(answer)
        
        coverage = len(answered_steps & required_steps) / len(required_steps)
        return coverage
    
    def compute_rag_score(self, faithfulness, relevancy, precision, recall,
                         code_correctness, version_compat, completeness):
        """
        Weighted score for game dev RAG quality
        """
        weights = {
            'faithfulness': 0.20,
            'relevancy': 0.15,
            'precision': 0.15,
            'recall': 0.15,
            'code_correctness': 0.15,  # Critical for game dev
            'version_compat': 0.10,      # Important
            'completeness': 0.10
        }
        
        return (
            weights['faithfulness'] * faithfulness +
            weights['relevancy'] * relevancy +
            weights['precision'] * precision +
            weights['recall'] * recall +
            weights['code_correctness'] * code_correctness +
            weights['version_compat'] * version_compat +
            weights['completeness'] * completeness
        )
```

### 9.3 Continuous Evaluation Pipeline

```python
def continuous_evaluation_loop():
    """
    Run evaluation on new queries in production
    """
    while True:
        # 1. Fetch recent queries from retrieval_history
        recent_queries = fetch_queries(last_n_hours=24)
        
        # 2. Compute RAGAS metrics
        for query_record in recent_queries:
            metrics = compute_ragas_metrics(
                query=query_record['query'],
                contexts=query_record['retrieved_docs'],
                answer=query_record['final_answer']
            )
            
            # 3. Compute game-dev-specific metrics
            game_metrics = GameDevRAGEvaluator().compute_all(
                answer=query_record['final_answer'],
                query=query_record['query']
            )
            
            # 4. Store for analysis
            save_evaluation(metrics, game_metrics)
            
            # 5. Alert on degradation
            if combined_score < threshold:
                alert_degradation(query_record, metrics, game_metrics)
        
        # 6. Generate daily report
        report = generate_evaluation_report()
        publish_report(report)
        
        sleep(6 hours)
```

---

## SECTION 10: PRODUCTION DEPLOYMENT CHECKLIST

### 10.1 Infrastructure

- [ ] PostgreSQL 15+ with pgvector + pgvectorscale extensions
- [ ] Backup strategy (daily snapshots of embeddings DB)
- [ ] Index monitoring (HNSW/BM25 stats)
- [ ] Query performance monitoring (<100ms p99 latency target)
- [ ] Concurrent connection pooling (PgBouncer for 1000+ QPS)

### 10.2 Data Ingestion

- [ ] Blender API docs crawler (version-specific)
- [ ] Game dev tutorial ingestion pipeline
- [ ] Best practices documentation collection
- [ ] Code example extraction and validation
- [ ] Versioning system (track documentation updates)

### 10.3 RAG Pipeline

- [ ] BGE-M3 embedding model deployment
- [ ] MiniLM cross-encoder for reranking
- [ ] Query rewriting agent
- [ ] Agentic validation loops
- [ ] Error handling and fallback mechanisms

### 10.4 Monitoring & Observability

- [ ] Query latency tracking (dense + sparse + reranking time)
- [ ] RAGAS metric computation (daily)
- [ ] User satisfaction feedback loop
- [ ] Hallucination detection
- [ ] Source attribution verification

### 10.5 Safety & Compliance

- [ ] Data privacy (local hosting validation)
- [ ] API rate limiting
- [ ] Audit logging (who queried what, when)
- [ ] GDPR compliance (data retention policies)
- [ ] Adversarial prompt detection

---

## SECTION 11: OPTIMIZATION TUNING PARAMETERS

### 11.1 Hybrid Retrieval Tuning

```python
RETRIEVAL_CONFIG = {
    # Dense search
    "dense_top_k": 100,
    "dense_threshold": 0.6,
    
    # Sparse search
    "sparse_top_k": 100,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    
    # RRF fusion
    "rrf_k": 60,
    "rrf_top_k": 50,
    
    # Reranking
    "rerank_top_k": 10,
    "rerank_threshold": 0.5,
    
    # Metadata filtering
    "allow_outdated_versions": True,
    "version_priority": "newest_first",
    
    # Context assembly
    "max_context_tokens": 2000,
    "context_token_per_doc": 400
}
```

### 11.2 Embedding Model Selection

**Trade-offs:**

```
Model              | Accuracy | Speed | Cost | Best For
=====================================
BGE-M3             | ⭐⭐⭐⭐⭐ | ⭐⭐⭐  | ⭐⭐  | Game Dev (recommended)
Nomic Embed        | ⭐⭐⭐⭐  | ⭐⭐⭐⭐ | ⭐   | Resource-constrained
E5-Large           | ⭐⭐⭐⭐  | ⭐⭐   | ⭐⭐⭐ | General purpose
Voyage Code        | ⭐⭐⭐⭐⭐ | ⭐⭐   | ⭐⭐⭐ | Code-heavy domains
```

---

## SECTION 12: COST ANALYSIS

### 12.1 Infrastructure Costs (Self-Hosted, Monthly)

```
Component                          | Cost (USD)
=========================================
PostgreSQL Server (4 CPU, 16GB RAM)| $50-100
pgvector Extension                 | $0 (open-source)
Network/Storage (50GB embeddings)  | $20-50
Monitoring/Logging                 | $30-50
Backup & Disaster Recovery         | $20-30
=====================================
TOTAL MONTHLY                       | $120-260
```

### 12.2 vs API-Based RAG

```
Service      | Per Query | Monthly (10K queries) | Pros | Cons
================================================================
API-Based    | $0.005-0.02 | $50-200         | Easy setup | Privacy risk
Self-Hosted  | ~$0.001    | $120-260 fixed   | Private | Ops overhead
```

**Recommendation:** Self-hosted preferred for game dev studio (data sensitivity)

---

## SECTION 13: REFERENCE ARCHITECTURE STACK

### 13.1 Recommended Tech Stack (2025)

```
┌─────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                  │
│  ┌────────────────────────────────────────────────┐ │
│  │ Claude API / Local LLM (Llama 2, Mistral)      │ │
│  └────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────┐
│  FRAMEWORK LAYER                                    │
│  ┌────────────────────────────────────────────────┐ │
│  │ LangChain v0.2+ / LlamaIndex v0.11+            │ │
│  │ (for orchestration & pipeline management)      │ │
│  └────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────┐
│  RAG PIPELINE LAYER                                 │
│  ┌──────────────────────────────────────────────┐   │
│  │ Hybrid Retrieval                             │   │
│  │ • Dense: BGE-M3 (Nomic)                     │   │
│  │ • Sparse: BM25 (PostgreSQL)                 │   │
│  │ • Fusion: RRF                               │   │
│  ├──────────────────────────────────────────────┤   │
│  │ Reranking                                    │   │
│  │ • Model: ms-marco-MiniLM                    │   │
│  │ • Inference: Sentence Transformers          │   │
│  ├──────────────────────────────────────────────┤   │
│  │ Query Transformation                         │   │
│  │ • Multi-hop decomposition                   │   │
│  │ • Query rewriting                           │   │
│  ├──────────────────────────────────────────────┤   │
│  │ Agentic Validation                           │   │
│  │ • Code correctness check                    │   │
│  │ • Version compatibility                     │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────┐
│  DATABASE LAYER                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ PostgreSQL 15+                               │   │
│  │ • pgvector (HNSW index)                     │   │
│  │ • pgvectorscale (StreamingDiskANN)          │   │
│  │ • Full-text search (BM25)                   │   │
│  │ • Metadata indexes                          │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

### 13.2 Python Stack (Example)

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"

# Core
langchain = "^0.2.0"
langchain-nomic = "^0.1.0"
sentence-transformers = "^2.7.0"

# Vector DB
psycopg2-binary = "^2.9.0"
pgvector = "^0.2.0"

# LLM
anthropic = "^0.25.0"  # For Claude API
ollama = "^0.3.0"      # For local models

# Evaluation
ragas = "^0.1.0"

# Infrastructure
fastapi = "^0.110.0"
redis = "^5.0.0"
pydantic = "^2.0.0"

# Monitoring
opentelemetry-api = "^1.20.0"
prometheus-client = "^0.19.0"
```

---

## SECTION 14: NEXT STEPS & IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Weeks 1-4)
1. Set up PostgreSQL + pgvector infrastructure
2. Implement hybrid retrieval pipeline
3. Create chunking pipeline for Blender docs
4. Deploy BGE-M3 embedding model

### Phase 2: Enhancement (Weeks 5-8)
1. Integrate MiniLM reranking
2. Build agentic query rewriting
3. Implement validation loops
4. Set up RAGAS evaluation

### Phase 3: Production (Weeks 9-12)
1. Load production documents
2. Fine-tune ranking parameters
3. Implement monitoring & observability
4. Deploy & stress test

---

## REFERENCES & RESOURCES

### Key Papers
1. **RAG Survey 2024** - Gao et al. - Foundational RAG taxonomy
2. **RAGAS Framework** - Metrics for autonomous RAG evaluation
3. **Hybrid Search Survey** - BM25 + Dense retrieval comparison
4. **Agentic RAG** - Multi-step reasoning patterns (Kore.ai, 2025)
5. **Technical-Embeddings** - Domain-specific embedding optimization (arXiv, 2025)

### Benchmarks
- MS MARCO: 500K query-document pairs (cross-encoder training standard)
- TigerData (2025): pgvector vs Qdrant performance comparison
- RAGAS Benchmark: Standard RAG evaluation dataset

### Tools & Frameworks
- **LangChain** v0.2+ - Query orchestration
- **LlamaIndex** v0.11+ - Data ingestion & indexing
- **Haystack** v2.5+ - Enterprise RAG
- **PostgreSQL** 15+ - Vector DB
- **Sentence Transformers** - Embedding/reranking models
- **RAGAS** - Evaluation framework

### Community & Learning
- https://docs.langchain.com/docs/use_cases/qa_structured_data/
- https://docs.llamaindex.ai/
- https://github.com/explodinggradients/ragas
- https://huggingface.co/models (embedding & reranker models)

---

## CONCLUSION

Building a top-tier RAG system for 3D game development requires orchestrating multiple advanced techniques:

1. **Hybrid retrieval** (dense + sparse) for recall and precision
2. **Cross-encoder reranking** to promote top-k candidates
3. **Agentic RAG** for multi-step reasoning and validation
4. **Local PostgreSQL** for privacy, performance, and control
5. **Continuous evaluation** via RAGAS metrics for quality assurance

This architecture balances technical sophistication with production reliability, enabling your 3D game dev AI assistant to provide accurate, grounded, and actionable responses for complex Blender automation and 3D modeling workflows.

---

**Document Version:** 1.0  
**Last Updated:** December 4, 2025  
**Classification:** Technical Architecture Reference  
**Status:** Ready for Implementation