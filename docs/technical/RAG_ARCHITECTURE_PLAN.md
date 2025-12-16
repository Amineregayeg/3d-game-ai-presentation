# Advanced RAG System for 3D Game Development
## Technical Architecture & Implementation Plan v3.0

**Document Version:** 3.0
**Last Updated:** December 11, 2025
**Status:** Production-Ready Architecture

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 4, 2025 | Initial architecture specification |
| 2.0 | Dec 11, 2025 | Added production sections: Error Handling (17), Security (18), Observability (19), Testing (20), Session Management (21), Operational Runbooks (22) |
| 3.0 | Dec 11, 2025 | **Major upgrade to Fully Agentic Architecture** with OpenAI GPT-5.1 integration, 7-layer agent design, multi-model orchestration |

---

## Executive Summary

This document provides a comprehensive technical specification for building a **production-grade Fully Agentic RAG system** optimized for 3D game development, Blender automation, and procedural asset generation. The architecture combines:

- **Fully Agentic 7-Layer Architecture** with specialized agents per task
- **OpenAI GPT-5.1 Family** for orchestration, generation, and validation
- **Hybrid Retrieval** (dense semantic search + sparse lexical search with RRF fusion)
- **Cross-Encoder Reranking** for context precision
- **Multi-Step Query Decomposition** for complex multi-hop reasoning
- **Local PostgreSQL + pgvector** for data privacy and performance
- **RAGAS Evaluation Framework** for continuous quality assurance
- **Self-Correcting Validation Loop** with automatic retry mechanisms

**Target Performance:**
- Query latency: <3s for complex queries, <500ms for simple queries
- Context precision: >0.85
- Hallucination rate: <5%
- Retrieval accuracy (MRR@10): >0.80
- Multi-step task completion: >90%

---

## 1. Architecture Overview

### 1.1 High-Level System Design

```
┌────────────────────────────────────────────────────────────────┐
│                    USER QUERY (STT Output)                     │
│              e.g., "How do I select all faces in Blender?"    │
└────────────────────────┬─────────────────────────────────────┘
                         │
           ┌─────────────┴──────────────┐
           ▼                            ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ Session Memory   │      │ Query Analysis   │
    │ (Last 5 turns)   │      │ & Rewriting      │
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             └────────────┬────────────┘
                          ▼
            ┌─────────────────────────┐
            │ Agentic Query Transform │
            │ 1. Intent extraction    │
            │ 2. Entity recognition   │
            │ 3. Query variations     │
            │ 4. Multi-hop generation │
            └────────────┬────────────┘
                         │
        ┌────────────────┴─────────────────┐
        ▼                                  ▼
 ┌──────────────────┐            ┌──────────────────┐
 │ HYBRID RETRIEVAL │            │ HYBRID RETRIEVAL │
 │ (Dense + BM25)   │            │ (Dense + BM25)   │
 │                  │            │                  │
 │ Query Variations │            │ Multi-hop Queries│
 │ q1, q2, q3 ...   │            │ q1, q2, q3 ...   │
 └────────┬─────────┘            └────────┬─────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
           ┌──────────────────────────┐
           │ Reciprocal Rank Fusion   │
           │ (RRF - Merge & Score)    │
           │ Top-50 candidates        │
           └──────────────┬───────────┘
                          ▼
           ┌──────────────────────────┐
           │ Cross-Encoder Reranking  │
           │ (MiniLM - Relevance)     │
           │ Top-10 context           │
           └──────────────┬───────────┘
                          ▼
           ┌──────────────────────────┐
           │ Metadata Filtering       │
           │ (Version, domain, type)  │
           └──────────────┬───────────┘
                          ▼
           ┌──────────────────────────┐
           │ Context Assembly         │
           │ Format for LLM Prompt    │
           │ Max 2000 tokens          │
           └──────────────┬───────────┘
                          ▼
           ┌──────────────────────────┐
           │ Agentic Validation       │
           │ 1. Code syntax check     │
           │ 2. API compatibility     │
           │ 3. Version matching      │
           │ 4. Hallucination detect  │
           └──────────────┬───────────┘
                          ▼
           ┌──────────────────────────┐
           │ LLM Generation           │
           │ (Claude / Local LLM)     │
           │ Grounded answer          │
           └──────────────┬───────────┘
                          ▼
           ┌──────────────────────────┐
           │ Answer Validation        │
           │ Pass? → Return           │
           │ Fail? → Re-retrieve      │
           └──────────────────────────┘
```

### 1.2 System Components Overview

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Query Analyzer** | Extract intent & entities | Claude API + Regex |
| **Dense Retrieval** | Semantic search | BGE-M3 embeddings (4,096 dims) |
| **Sparse Retrieval** | Keyword matching | PostgreSQL BM25 |
| **RRF Fusion** | Merge results | Reciprocal Rank Fusion |
| **Reranker** | Precision ranking | MiniLM cross-encoder |
| **Metadata Filter** | Domain filtering | SQL WHERE clauses |
| **Context Assembler** | Format for LLM | Token-aware concatenation |
| **Agentic Validator** | Correctness check | Rule-based + LLM-based |
| **Vector Database** | Store embeddings | PostgreSQL + pgvector |
| **Evaluation Framework** | Quality metrics | RAGAS framework |

---

## 2. Dense Retrieval: Vector Search

### 2.1 Embedding Model: BGE-M3

**Selection Rationale:**
- **4,096 dimensions** = high semantic granularity for technical docs
- **Multi-modal support** = handles text, code, images
- **Technical domain training** = optimized for API docs and code
- **Global language support** = Blender community is multilingual

**Model Specifications:**

```
Architecture: BERT-based with contrastive learning
Dimensions: 4,096
Vocabulary: 250K tokens
Max sequence length: 8,192 tokens
Training data: 1.3B documents (Wikipedia + Common Crawl + technical docs)
Inference latency: ~150ms for 1K documents per query
Memory: ~8GB for full model
Quantization: Supports int8 for 4x memory reduction
```

### 2.2 Embedding Generation Pipeline

```python
class EmbeddingPipeline:
    """
    Generate and cache embeddings for documents
    """

    def __init__(self, model_name="BAAI/bge-m3"):
        # Load BGE-M3 model
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.batch_size = 32
        self.embedding_cache = {}  # In-memory cache

    def generate_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of documents

        Args:
            documents: List of document texts

        Returns:
            (N, 4096) embedding matrix
        """
        embeddings = []

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]

            # Generate embeddings with caching
            batch_embeddings = []
            for doc in batch:
                doc_hash = hash(doc)

                if doc_hash in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[doc_hash])
                else:
                    embedding = self.model.embed_query(doc)
                    batch_embeddings.append(embedding)
                    self.embedding_cache[doc_hash] = embedding

            embeddings.extend(batch_embeddings)

            # Log progress
            print(f"Embedded {min(i + self.batch_size, len(documents))}/{len(documents)}")

        return np.array(embeddings)

    def similarity_search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """
        Find k most similar documents to query

        Uses cosine similarity: sim = (A · B) / (||A|| ||B||)
        """
        query_embedding = self.model.embed_query(query)

        # Retrieve from PostgreSQL
        results = db.query(
            f"""
            SELECT id, content, 1 - (embedding <=> %s::vector) as similarity
            FROM documents
            WHERE 1 - (embedding <=> %s::vector) > {SIMILARITY_THRESHOLD}
            ORDER BY similarity DESC
            LIMIT {k}
            """,
            (query_embedding, query_embedding)
        )

        return [(row['content'], row['similarity']) for row in results]
```

### 2.3 Vector Database Indexing: HNSW

**HNSW (Hierarchical Navigable Small World) Index:**

```
Structure:
- Multi-layer graph (0 to log(N) layers)
- Layer 0: Fully connected (all nodes)
- Higher layers: Sparsely connected
- M = 16 (max connections per node)
- ef_construction = 200 (search width during construction)
- ef = 100 (search width during query)

Performance:
- Insertion: O(log N) expected
- Search: O(log N) with log(N) hops
- Space: O(N × M) ≈ 10-15% overhead

PostgreSQL pgvector Implementation:
"""

-- Create HNSW index for vector search
CREATE INDEX documents_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m=16, ef_construction=200);

-- Query using indexed search
SELECT
    id, content,
    1 - (embedding <=> query_vector) as similarity
FROM documents
WHERE (embedding <=> query_vector) < 0.3  -- Top candidates
ORDER BY similarity DESC
LIMIT 100;
```

### 2.4 Similarity Metrics

```python
def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors

    cos(θ) = (A · B) / (||A|| ||B||)
    Range: [-1, 1] for normalized vectors [0, 1]
    """
    # BGE-M3 produces normalized vectors, so ||A|| = ||B|| = 1
    # Therefore: cos(θ) = A · B (dot product only)

    similarity = np.dot(vec_a, vec_b)

    # For pgvector: use <=> (cosine distance operator)
    # distance = 1 - similarity
    # similarity = 1 - distance

    return similarity

# Similarity score thresholds for game dev domain
THRESHOLDS = {
    'high_confidence': 0.75,    # Use directly
    'medium_confidence': 0.65,  # Candidate for reranking
    'low_confidence': 0.50,     # Fallback/validate
    'discard': 0.40             # Too dissimilar
}
```

---

## 3. Sparse Retrieval: BM25

### 3.1 BM25 Algorithm

**Mathematical Foundation:**

```
BM25 Score(D, Q) = Σ_{i=1}^{n} IDF(q_i) ×
                   (f(q_i, D) × (k1 + 1)) /
                   (f(q_i, D) + k1 × (1 - b + b × |D|/avgdl))

Where:
- q_i = query term i
- D = document
- Q = query terms {q_1, q_2, ..., q_n}
- f(q_i, D) = frequency of q_i in D
- |D| = document length in tokens
- avgdl = average document length in corpus
- k1 = saturation parameter (typically 1.5)
- b = length normalization parameter (typically 0.75)

IDF Calculation:
IDF(q_i) = log((N - n(q_i) + 0.5) / (n(q_i) + 0.5))

Where:
- N = total number of documents
- n(q_i) = number of documents containing q_i
```

**Why BM25 for Game Dev:**
- ✅ Exact keyword matching (critical for API names like `bpy.ops.mesh.select_all`)
- ✅ Rare term detection (specialized terminology)
- ✅ Version-specific filtering (e.g., "Blender 4.2 API")
- ✅ Extremely fast (milliseconds for 50M+ documents)
- ✅ Probabilistically optimal (proven theory)

### 3.2 PostgreSQL Full-Text Search Implementation

```sql
-- Enable full-text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Add text search column to documents
ALTER TABLE documents
ADD COLUMN tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- Create GIN index for fast full-text search
CREATE INDEX documents_tsvector_idx
ON documents
USING GIN (tsv);

-- BM25 relevance ranking query
SELECT
    id,
    content,
    ts_rank(
        tsv,
        plainto_tsquery('english', %s)
    ) as bm25_score,
    source,
    blender_version,
    category
FROM documents
WHERE tsv @@ plainto_tsquery('english', %s)
ORDER BY bm25_score DESC
LIMIT 100;
```

### 3.3 Custom BM25 Implementation

```python
class BM25Retriever:
    """
    Custom BM25 implementation for fine-tuned control
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # Term saturation parameter
        self.b = b    # Length normalization parameter
        self.idf = {}
        self.doc_lengths = []
        self.average_doc_length = 0

    def fit(self, documents: List[str]):
        """
        Build BM25 index from documents

        Args:
            documents: List of document texts
        """
        # Tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]

        # Calculate document frequencies
        doc_frequencies = {}
        for doc in tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

        # Calculate IDF for each term
        N = len(documents)
        for term, freq in doc_frequencies.items():
            idf = math.log((N - freq + 0.5) / (freq + 0.5))
            self.idf[term] = idf

        # Store document lengths
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.average_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        # Build inverted index
        self.inverted_index = {}
        for doc_id, doc in enumerate(tokenized_docs):
            for term in set(doc):
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(doc_id)

    def score(self, query: str, doc_id: int, tokenized_docs: List[List[str]]) -> float:
        """
        Calculate BM25 score for query against document

        Args:
            query: Query string
            doc_id: Document ID
            tokenized_docs: List of tokenized documents

        Returns:
            BM25 score (higher = more relevant)
        """
        score = 0.0
        query_terms = self._tokenize(query)
        doc = tokenized_docs[doc_id]
        doc_length = self.doc_lengths[doc_id]

        for term in query_terms:
            if term not in self.idf:
                continue

            # Calculate term frequency in document
            term_freq = doc.count(term)

            # BM25 formula
            idf = self.idf[term]
            numerator = idf * term_freq * (self.k1 + 1)
            denominator = (term_freq +
                          self.k1 * (1 - self.b + self.b * doc_length / self.average_doc_length))

            score += numerator / denominator

        return score

    def search(self, query: str, k: int = 100) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for query
        """
        # Quick candidate filtering using inverted index
        candidate_docs = set()
        for term in self._tokenize(query):
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])

        # Score candidates
        scores = []
        for doc_id in candidate_docs:
            score = self.score(query, doc_id, self.tokenized_docs)
            if score > 0:
                scores.append((doc_id, score))

        # Return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text: lowercase, remove punctuation, split
        """
        # Remove punctuation, lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())

        # Split on whitespace and special markers (camelCase, snake_case)
        tokens = re.findall(r'\w+', text)

        # Remove stopwords (optional, depends on domain)
        # For game dev, keep all technical terms

        return tokens
```

---

## 4. Retrieval Fusion: Reciprocal Rank Fusion

### 4.1 RRF Algorithm

```python
class Reciprocal RankFusion:
    """
    Merge dense and sparse retrieval results

    Mathematical Foundation:
    RRF(d) = Σ_{r ∈ R} 1 / (k + rank(d, r))

    Where:
    - d = document
    - R = set of retrievers (dense, sparse)
    - rank(d, r) = rank of d in retriever r
    - k = constant (typically 60)
    """

    K = 60  # Constant to avoid division by zero

    @staticmethod
    def fuse(dense_results: List[Tuple[str, float]],
             sparse_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Fuse dense and sparse results using RRF

        Args:
            dense_results: [(doc_id, similarity), ...] from dense search
            sparse_results: [(doc_id, bm25_score), ...] from BM25

        Returns:
            [(doc_id, rrf_score), ...] sorted by fused score
        """
        rrf_scores = {}

        # Add contributions from dense search
        for rank, (doc_id, score) in enumerate(dense_results, 1):
            rrf_contribution = 1 / (ReciprocalRankFusion.K + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_contribution

        # Add contributions from sparse search
        for rank, (doc_id, score) in enumerate(sparse_results, 1):
            rrf_contribution = 1 / (ReciprocalRankFusion.K + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_contribution

        # Sort by total RRF score
        fused = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return fused

# Example: RRF fusion in action
"""
Dense Search Results:
  Rank 1: doc_A (similarity: 0.92)
  Rank 2: doc_B (similarity: 0.87)
  Rank 5: doc_C (similarity: 0.75)

Sparse Search (BM25) Results:
  Rank 1: doc_B (BM25: 12.5)
  Rank 3: doc_A (BM25: 8.3)
  Rank 8: doc_D (BM25: 5.1)

RRF Fusion (k=60):
  doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0154 = 0.0318 ← Ranked high in BOTH
  doc_B: 1/(60+2) + 1/(60+1) = 0.0156 + 0.0164 = 0.0320 ← Ranked high in BOTH
  doc_C: 1/(60+5) + 0 = 0.0143
  doc_D: 0 + 1/(60+8) = 0.0139

Final Ranking: doc_B > doc_A > doc_C > doc_D
"""
```

### 4.2 Why RRF Works

```
Problem with weighted average:
  combined_score = α × dense_score + β × sparse_score
  ❌ Requires learning α, β (domain-dependent tuning)
  ❌ Scores on different scales (0-1 vs raw BM25)

Solution with RRF:
  ✅ No hyperparameters to tune
  ✅ Statistically optimal for ranked combination
  ✅ Captures complementary strengths:
     - Dense: semantic understanding
     - Sparse: exact keyword matching
  ✅ Proven in information retrieval theory
```

---

## 5. Cross-Encoder Reranking

### 5.1 Architecture: MiniLM Cross-Encoder

```python
class CrossEncoderReranker:
    """
    Rerank candidates using MiniLM cross-encoder

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Parameters: 66M
    - Latency: ~15ms per candidate on CPU
    - Accuracy: 0.89 on MS MARCO
    - Trained on 500K query-document relevance pairs
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        self.batch_size = 64

    def rerank(self, query: str,
               candidates: List[str],
               top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rerank candidates for query

        Args:
            query: User query
            candidates: List of candidate document texts
            top_k: Return top-k documents

        Returns:
            [(doc_text, relevance_score), ...] sorted by score
        """
        # Prepare query-document pairs
        pairs = [[query, doc] for doc in candidates]

        # Score pairs in batches
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Sort by score (descending)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]

    def batch_rerank(self, query: str,
                     candidates_batch: List[List[str]],
                     top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Rerank multiple candidate sets (e.g., for multi-hop queries)
        """
        results = []
        for candidates in candidates_batch:
            ranked = self.rerank(query, candidates, top_k)
            results.append(ranked)
        return results
```

### 5.2 Cross-Encoder vs Dense Embedding

```
Comparison:

DENSE EMBEDDING (BGE-M3):
┌──────────────┐
│ Query        │ → Embed → Q_vector (4,096 dims)
└──────────────┘
┌──────────────┐
│ Document     │ → Embed → D_vector (4,096 dims)
└──────────────┘
                    ↓
              Cosine Similarity
              sim = Q · D / (|Q| |D|)

Pros: Fast, pre-compute embeddings
Cons: Query and doc processed independently

CROSS-ENCODER (MiniLM):
┌──────────────────────────┐
│ [CLS] Query [SEP] Doc... │ → Transformer → Relevance Score (0-1)
└──────────────────────────┘
                    ↓
            Direct interaction layer
            (Transformer attention)

Pros: Accurate, captures query-doc interactions
Cons: Slower, must compute per query

RECOMMENDATION: Use dense for fast candidate retrieval (100→50),
                 then cross-encoder for final ranking (50→10)
```

---

## 6. Agentic Query Transformation

### 6.1 Query Analyzer & Rewriter

```python
class QueryTransformer:
    """
    Transform user query into multiple search variations
    for multi-hop retrieval
    """

    def __init__(self, llm=None):
        # Use Claude API for query analysis
        self.llm = llm or AnthropicClient()

    def analyze(self, query: str, session_history: List[str] = None) -> Dict:
        """
        Analyze query to extract intent, entities, and variations

        Args:
            query: User query (from STT)
            session_history: Recent conversation turns

        Returns:
            {
                'intent': str,           # 'rotate_objects', 'select_faces', etc.
                'entities': List[str],   # ['faces', 'rotation', 'smooth']
                'scope': str,            # 'all', 'selected', 'scene'
                'variations': List[str], # Query rewrites for retrieval
                'multi_hop': List[str]   # Sub-queries for multi-step tasks
            }
        """

        prompt = f"""
Analyze this game development query and provide structured output in JSON.

Query: "{query}"
{f"Session context (last 3 turns): {session_history[-3:]}" if session_history else ""}

Extract:
1. intent: What operation is being requested? (e.g., 'select_all_faces', 'apply_modifier')
2. entities: What objects/concepts are involved? (list)
3. scope: What scope? ('all', 'selected', 'scene', 'specific_object')
4. variations: 3-5 alternative phrasings of the same query for retrieval
5. multi_hop: If multi-step, break into sub-queries

Return as valid JSON only, no markdown or explanation.
"""

        response = self.llm.complete(prompt)

        try:
            analysis = json.loads(response)
            return analysis
        except json.JSONDecodeError:
            # Fallback parsing
            return self._fallback_analysis(query)

    def generate_search_queries(self, analysis: Dict) -> List[str]:
        """
        Generate search queries from analysis

        Includes:
        - Direct query
        - Variations with different terminology
        - Technical API names (if code-focused)
        - Best practices versions
        """
        queries = [
            analysis['query']  # Original
        ]

        # Add variations
        queries.extend(analysis.get('variations', []))

        # Add code-focused version if applicable
        if analysis.get('intent') in ['api_call', 'code_snippet']:
            # E.g., "select all faces" → "bpy.ops.mesh.select_all()"
            api_name = self._intent_to_api(analysis['intent'])
            if api_name:
                queries.append(api_name)

        # Add best practices version
        queries.append(f"best practice for {analysis['intent']}")

        return queries

    @staticmethod
    def _intent_to_api(intent: str) -> str:
        """
        Map intent to Blender API call
        """
        mapping = {
            'select_all_faces': 'bpy.ops.mesh.select_all()',
            'rotate_objects': 'bpy.ops.transform.rotate()',
            'apply_modifier': 'bpy.ops.object.modifier_apply()',
            'smooth_surface': 'bpy.ops.object.shade_smooth()',
            'uv_unwrap': 'bpy.ops.uv.unwrap()',
        }
        return mapping.get(intent, '')
```

### 6.2 Multi-Hop Query Decomposition

```python
class MultiHopRAG:
    """
    For complex queries, decompose into sequential retrieval steps

    Example:
    User: "Rotate all faces and apply a smooth shading"

    Decomposition:
      hop_1: "How to select all faces in Blender"
      hop_2: "How to rotate selected faces"
      hop_3: "How to apply smooth shading"
      hop_4: "How to finalize and apply transformations"
    """

    def decompose(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries
        """
        prompt = f"""
The user query requires multiple steps. Decompose it into sequential sub-queries.

Query: "{query}"

Each sub-query should be retrievable independently but form a complete workflow.
Return as JSON list of strings:
["sub_query_1", "sub_query_2", ...]
"""

        response = self.llm.complete(prompt)
        sub_queries = json.loads(response)
        return sub_queries

    def retrieve_multi_hop(self, sub_queries: List[str],
                          retriever) -> Dict[str, List[str]]:
        """
        Retrieve context for each sub-query

        Returns:
            {
                'sub_query_1': [retrieved_docs...],
                'sub_query_2': [retrieved_docs...],
                ...
            }
        """
        results = {}
        for sub_query in sub_queries:
            docs = retriever.retrieve(sub_query, top_k=5)
            results[sub_query] = docs
        return results

    def merge_contexts(self, contexts: Dict[str, List[str]]) -> str:
        """
        Merge multi-hop context into single prompt

        Format for LLM:
        === STEP 1: [sub_query_1] ===
        [Context for step 1]

        === STEP 2: [sub_query_2] ===
        [Context for step 2]
        ...
        """
        merged = ""
        for i, (step_query, docs) in enumerate(contexts.items(), 1):
            merged += f"\n=== STEP {i}: {step_query} ===\n"
            for j, doc in enumerate(docs, 1):
                merged += f"[Source {j}]\n{doc}\n\n"
        return merged
```

---

## 7. Metadata Filtering & Document Schema

### 7.1 Document Schema

```sql
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,

    -- Content
    content TEXT NOT NULL,          -- Full document text
    title VARCHAR(500),             -- Document title
    embedding vector(4096),         -- BGE-M3 embedding

    -- Metadata - Domain
    source VARCHAR(50),             -- 'blender_api', 'tutorial', 'best_practice'
    blender_version VARCHAR(10),    -- '4.2', '4.1', 'all'
    category VARCHAR(50),           -- 'mesh', 'object', 'animation', 'shader', 'rendering'
    subcategory VARCHAR(50),        -- 'selection', 'transformation', 'modifiers'

    -- Metadata - Content Type
    language VARCHAR(20),           -- 'python', 'glsl', 'conceptual'
    is_code BOOLEAN,               -- true if contains code
    modality VARCHAR(50),          -- 'api_reference', 'tutorial', 'example', 'best_practice'
    difficulty VARCHAR(20),        -- 'beginner', 'intermediate', 'advanced'

    -- Ranking
    priority FLOAT DEFAULT 0.5,    -- 0.0-1.0 (boost important docs)
    relevance_score FLOAT,         -- Historical relevance

    -- Temporal
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_date DATE,

    -- Full-text search
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content || ' ' || COALESCE(title, ''))
    ) STORED,

    -- Source tracking
    source_url VARCHAR(1000),
    source_hash VARCHAR(64),       -- SHA-256 for deduplication

    -- Tags for semantic filtering
    tags TEXT[],                    -- ['selection', 'mesh', 'modeling', 'workflow']
    keywords TEXT[]                 -- Extracted keywords for BM25
);

-- Indexes for fast retrieval
CREATE INDEX idx_embedding ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_tsvector ON documents USING GIN (tsv);
CREATE INDEX idx_source_version ON documents(source, blender_version);
CREATE INDEX idx_category ON documents(category, subcategory);
CREATE INDEX idx_is_code_language ON documents(is_code, language);
CREATE INDEX idx_tags ON documents USING GIN (tags);
CREATE INDEX idx_priority_relevance ON documents(priority DESC, relevance_score DESC);

-- Retrieval history for analytics
CREATE TABLE retrieval_history (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    query_embedding vector(4096),
    retrieved_doc_ids BIGINT[],     -- Which docs were retrieved
    final_answer TEXT,
    user_satisfaction FLOAT,        -- 0-1 feedback
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id VARCHAR(100)
);
```

### 7.2 Filtering Logic

```python
class MetadataFilter:
    """
    Filter documents by metadata constraints
    """

    @staticmethod
    def build_where_clause(filters: Dict[str, Any]) -> str:
        """
        Build SQL WHERE clause from filter dict

        Example filters:
        {
            'blender_version': ['4.2', '4.1'],  # OR logic
            'source': 'blender_api',            # AND logic
            'language': 'python',
            'is_code': True,
            'category': 'mesh',
            'difficulty_max': 'intermediate'
        }
        """
        conditions = []

        # Version filtering (with backwards compatibility)
        if 'blender_version' in filters:
            versions = filters['blender_version']
            if versions:
                version_conditions = [
                    f"blender_version = '{v}'" for v in versions
                ]
                # Also include 'all' versions
                version_conditions.append("blender_version = 'all'")
                conditions.append(f"({' OR '.join(version_conditions)})")

        # Source filtering
        if 'source' in filters:
            conditions.append(f"source = '{filters['source']}'")

        # Code vs conceptual
        if 'is_code' in filters:
            conditions.append(f"is_code = {filters['is_code']}")

        # Category filtering
        if 'category' in filters:
            conditions.append(f"category = '{filters['category']}'")

        # Difficulty (max allowed)
        if 'difficulty_max' in filters:
            diff_order = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
            max_level = diff_order.get(filters['difficulty_max'], 3)
            difficulty_mapping = {
                1: "difficulty IN ('beginner')",
                2: "difficulty IN ('beginner', 'intermediate')",
                3: "difficulty IN ('beginner', 'intermediate', 'advanced')"
            }
            conditions.append(difficulty_mapping[max_level])

        # Priority-based filtering (prefer higher priority)
        if conditions:
            where_clause = " AND ".join(conditions)
            return f"WHERE {where_clause}"
        return ""

    @staticmethod
    def extract_filters_from_query(query: str) -> Dict[str, Any]:
        """
        Detect filters mentioned in user query

        Examples:
        - "In Blender 4.2, how to..." → {'blender_version': ['4.2']}
        - "Show me code for..." → {'is_code': True}
        - "Best practice for mesh..." → {'category': 'mesh', 'source': 'best_practice'}
        """
        filters = {}

        # Version detection
        version_pattern = r'Blender\s+(\d+\.\d+)'
        if match := re.search(version_pattern, query):
            filters['blender_version'] = [match.group(1)]

        # Code detection
        if any(word in query.lower() for word in ['code', 'script', 'python', 'api']):
            filters['is_code'] = True

        # Category detection
        categories = {
            'mesh': ['face', 'vertex', 'edge', 'selection'],
            'object': ['object', 'transform', 'position'],
            'shader': ['material', 'node', 'texture'],
            'animation': ['animate', 'keyframe', 'rig']
        }

        query_lower = query.lower()
        for category, keywords in categories.items():
            if any(kw in query_lower for kw in keywords):
                filters['category'] = category
                break

        return filters
```

---

## 8. Context Assembly

### 8.1 Context Formatting

```python
class ContextAssembler:
    """
    Format retrieved documents for LLM consumption
    """

    @staticmethod
    def assemble(reranked_docs: List[Tuple[str, float]],
                 query: str,
                 max_tokens: int = 2000) -> str:
        """
        Assemble final context for prompt

        Args:
            reranked_docs: [(doc_text, relevance_score), ...]
            query: Original user query
            max_tokens: Maximum context length

        Returns:
            Formatted context string
        """
        context_parts = [
            "=== RETRIEVED CONTEXT ===\n",
            f"Query: {query}\n",
            f"Sources: {len(reranked_docs)} documents\n",
            "=" * 40 + "\n"
        ]

        token_count = 0
        max_tokens_per_doc = 400

        for i, (doc_text, score) in enumerate(reranked_docs, 1):
            # Truncate document if too long
            doc_tokens = len(doc_text.split())
            if doc_tokens > max_tokens_per_doc:
                doc_text = ' '.join(doc_text.split()[:max_tokens_per_doc])

            # Format source attribution
            source_line = f"\n[Source {i} | Relevance: {score:.1%}]\n"
            context_parts.append(source_line)
            context_parts.append(doc_text)
            context_parts.append("\n")

            token_count += len(doc_text.split())

            # Stop if context is full
            if token_count >= max_tokens:
                context_parts.append(f"\n[... {len(reranked_docs) - i} more sources available]\n")
                break

        context_parts.append("\n" + "=" * 40)
        context_parts.append("\nRespond based only on the above context.\n")
        context_parts.append("If context is insufficient, state that clearly.\n")
        context_parts.append("Always cite sources when referencing information.\n")

        return ''.join(context_parts)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count (1 token ≈ 4 characters)"""
        return len(text) // 4
```

---

## 9. Agentic Validation Loop

### 9.1 Answer Validation

```python
class AnswerValidator:
    """
    Validate generated answers and trigger re-retrieval if needed
    """

    def __init__(self, llm=None):
        self.llm = llm or AnthropicClient()

    def validate(self, query: str, context: str, answer: str) -> Dict:
        """
        Comprehensive answer validation

        Returns:
            {
                'is_valid': bool,
                'faithfulness_score': float (0-1),
                'issues': List[str],
                'requires_retrieval': bool,
                'suggested_focus': str
            }
        """
        results = {
            'is_valid': True,
            'issues': [],
            'faithfulness_score': 1.0
        }

        # Check 1: Syntax validation (if code answer)
        if self._is_code_answer(answer):
            syntax_check = self._validate_syntax(answer)
            if not syntax_check['valid']:
                results['is_valid'] = False
                results['issues'].append(f"Syntax error: {syntax_check['error']}")

        # Check 2: API compatibility (if Blender API)
        if 'bpy.' in answer:
            api_check = self._validate_blender_api(answer)
            if not api_check['valid']:
                results['is_valid'] = False
                results['issues'].extend(api_check['errors'])

        # Check 3: Grounding (is answer supported by context?)
        grounding_check = self._check_grounding(context, answer, query)
        results['faithfulness_score'] = grounding_check['score']
        if grounding_check['score'] < 0.7:
            results['is_valid'] = False
            results['issues'].append(f"Answer not well grounded in context")

        # Check 4: Hallucination detection
        hallucination_check = self._detect_hallucination(context, answer)
        if hallucination_check['likely_hallucination']:
            results['is_valid'] = False
            results['issues'].append(f"Potential hallucination: {hallucination_check['reason']}")

        # Determine if re-retrieval is needed
        results['requires_retrieval'] = not results['is_valid']
        if results['requires_retrieval']:
            results['suggested_focus'] = self._suggest_retrieval_focus(results['issues'])

        return results

    @staticmethod
    def _is_code_answer(answer: str) -> bool:
        """Check if answer contains code"""
        return any(pattern in answer.lower() for pattern in [
            'bpy.', 'def ', 'class ', 'import ', '```python'
        ])

    @staticmethod
    def _validate_syntax(answer: str) -> Dict:
        """Validate Python syntax"""
        try:
            # Extract code blocks
            code_blocks = re.findall(r'```python\n(.*?)\n```', answer, re.DOTALL)
            for code in code_blocks:
                compile(code, '<answer>', 'exec')
            return {'valid': True}
        except SyntaxError as e:
            return {
                'valid': False,
                'error': str(e)
            }

    @staticmethod
    def _validate_blender_api(answer: str) -> Dict:
        """Check Blender API calls"""
        # Simplified check - would need full API spec for production
        known_apis = [
            'bpy.ops.mesh', 'bpy.ops.object', 'bpy.ops.uv',
            'bpy.ops.transform', 'bpy.context', 'bpy.data'
        ]

        errors = []
        for api_call in re.findall(r'bpy\.\w+(\.\w+)*', answer):
            # Would validate against Blender 4.2 API spec
            pass

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def _check_grounding(self, context: str, answer: str, query: str) -> Dict:
        """
        Check if answer is grounded in context

        Uses Claude to assess factual grounding
        """
        prompt = f"""
Given the query, context, and answer, assess how well the answer is grounded in context.

Query: {query}

Context:
{context}

Answer:
{answer}

Rate on 0-1 scale: How much of the answer is supported by the context?
Return JSON: {{"score": 0.85, "explanation": "..."}}
"""
        response = self.llm.complete(prompt)
        result = json.loads(response)
        return result

    @staticmethod
    def _detect_hallucination(context: str, answer: str) -> Dict:
        """
        Simple hallucination detection

        Heuristics:
        - Specific API calls not in context
        - Version claims not in context
        - Contradictions with context
        """
        # Check for ungrounded API calls
        api_calls_in_context = set(re.findall(r'bpy\.\w+(\.\w+)*', context))
        api_calls_in_answer = set(re.findall(r'bpy\.\w+(\.\w+)*', answer))

        hallucinated_apis = api_calls_in_answer - api_calls_in_context

        if hallucinated_apis:
            return {
                'likely_hallucination': True,
                'reason': f"API calls not in context: {hallucinated_apis}"
            }

        return {'likely_hallucination': False}

    @staticmethod
    def _suggest_retrieval_focus(issues: List[str]) -> str:
        """Suggest what to retrieve to fix issues"""
        if 'Syntax error' in issues[0]:
            return "correct_blender_python_syntax"
        elif 'API' in issues[0]:
            return "blender_api_reference"
        elif 'hallucination' in issues[0]:
            return "authoritative_documentation"
        return "general_documentation"
```

### 9.2 Retrieval Loop

```python
class AgenticRAGPipeline:
    """
    Full agentic RAG with validation loop
    """

    def __init__(self, retriever, reranker, validator, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.validator = validator
        self.llm = llm
        self.max_retries = 3

    def generate(self, query: str) -> Tuple[str, Dict]:
        """
        Generate answer with agentic validation loop

        Returns:
            (answer, metadata)
        """
        metadata = {
            'attempts': 0,
            'retrieval_iterations': 0,
            'validation_results': []
        }

        for attempt in range(self.max_retries):
            metadata['attempts'] = attempt + 1

            # Retrieve context
            context, retrieval_metadata = self._retrieve_context(query)
            metadata['retrieval_iterations'] += 1

            # Generate answer
            answer = self._generate_answer(query, context)

            # Validate answer
            validation = self.validator.validate(query, context, answer)
            metadata['validation_results'].append(validation)

            # If valid, return
            if validation['is_valid']:
                metadata['success'] = True
                return answer, metadata

            # If invalid, try targeted re-retrieval
            if attempt < self.max_retries - 1:
                focus = validation['suggested_focus']
                print(f"Answer validation failed. Re-retrieving with focus: {focus}")
                # Query rewriting based on validation failures
                query = self._rewrite_query(query, validation['issues'])

        # Return best-effort answer after max retries
        metadata['success'] = False
        metadata['note'] = "Answer validation failed after max retries"
        return answer, metadata

    def _retrieve_context(self, query: str) -> Tuple[str, Dict]:
        """Retrieve and rerank context"""
        # Dense retrieval
        dense_results = self.retriever.dense_search(query, k=100)

        # Sparse retrieval
        sparse_results = self.retriever.sparse_search(query, k=100)

        # RRF fusion
        fused_results = self.retriever.fuse(dense_results, sparse_results)

        # Reranking
        top_candidates = [doc for doc, _ in fused_results[:50]]
        reranked = self.reranker.rerank(query, top_candidates, top_k=10)

        # Assemble context
        from_context_assembler import ContextAssembler
        context = ContextAssembler.assemble(reranked, query)

        return context, {
            'dense_count': len(dense_results),
            'sparse_count': len(sparse_results),
            'final_context_docs': len(reranked)
        }

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer from context"""
        prompt = f"""You are a helpful 3D game development assistant.
Answer the user's question based ONLY on the provided context.

{context}

User Question: {query}

Provide a clear, actionable answer. If code, format with ```python blocks.
Always cite which context source you're using.
"""
        answer = self.llm.complete(prompt)
        return answer

    @staticmethod
    def _rewrite_query(original_query: str, issues: List[str]) -> str:
        """Rewrite query to address validation failures"""
        if "Syntax error" in issues[0]:
            return f"{original_query} correct Python syntax"
        elif "API" in issues[0]:
            return f"{original_query} official Blender API"
        return original_query
```

---

## 10. Evaluation Framework

### 10.1 RAGAS Metrics

```python
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

class RAGEvaluator:
    """
    Evaluate RAG quality using RAGAS framework
    """

    def __init__(self):
        self.metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall
        }

    def evaluate(self, query: str,
                context_docs: List[str],
                answer: str,
                reference_answer: str = None) -> Dict[str, float]:
        """
        Evaluate answer quality

        RAGAS Metrics:
        - Faithfulness (0-1): Is answer grounded in context? No hallucinations?
        - Answer Relevancy (0-1): Does answer address the question?
        - Context Precision (0-1): Fraction of retrieved context actually relevant
        - Context Recall (0-1): Coverage of relevant information
        """

        scores = {}

        # Faithfulness: Are claims in answer supported by context?
        scores['faithfulness'] = self.metrics['faithfulness'].score(
            {'answer': answer, 'contexts': context_docs}
        )

        # Answer Relevancy: Does answer match query intent?
        scores['answer_relevancy'] = self.metrics['answer_relevancy'].score(
            {'answer': answer, 'question': query}
        )

        # Context Precision: How much context is actually relevant?
        scores['context_precision'] = self.metrics['context_precision'].score(
            {'question': query, 'contexts': context_docs, 'answer': answer}
        )

        # Context Recall: Did we retrieve all relevant information?
        if reference_answer:
            scores['context_recall'] = self.metrics['context_recall'].score(
                {'question': query, 'contexts': context_docs, 'ground_truth': reference_answer}
            )

        # Composite score
        scores['ragas_score'] = (
            0.25 * scores['faithfulness'] +
            0.25 * scores['answer_relevancy'] +
            0.25 * scores['context_precision'] +
            0.25 * scores.get('context_recall', 0.5)
        )

        return scores
```

### 10.2 Game Dev-Specific Metrics

```python
class GameDevMetrics:
    """
    Custom metrics for 3D game development domain
    """

    @staticmethod
    def check_code_correctness(answer: str) -> float:
        """
        0: Broken (SyntaxError)
        0.5: Partial (compiles but may not run)
        1.0: Correct (syntactically valid Python)
        """
        code_blocks = re.findall(r'```python\n(.*?)\n```', answer, re.DOTALL)

        if not code_blocks:
            return 1.0  # Non-code answer

        for code in code_blocks:
            try:
                compile(code, '<answer>', 'exec')
            except SyntaxError:
                return 0.0

        return 1.0

    @staticmethod
    def check_api_existence(answer: str, blender_version: str = '4.2') -> float:
        """
        0: Uses non-existent APIs
        1.0: All APIs exist and are correct
        """
        api_calls = re.findall(r'bpy\.\w+(\.\w+)*\(\)', answer)

        # Load Blender API spec for version
        known_apis = load_blender_api_spec(blender_version)

        valid_count = sum(1 for api in api_calls if api in known_apis)
        return valid_count / len(api_calls) if api_calls else 1.0

    @staticmethod
    def check_completeness(answer: str, query: str) -> float:
        """
        For multi-step tasks, does answer cover all steps?

        Example:
        Query: "Select all faces and rotate them"
        Answer should cover:
          - Entering face selection mode
          - Selecting all faces
          - Rotating
          - Finalizing transformation
        """
        required_keywords = extract_workflow_keywords(query)
        answer_keywords = set(answer.lower().split())

        coverage = len(required_keywords & answer_keywords) / len(required_keywords)
        return coverage

    @staticmethod
    def check_version_compatibility(answer: str,
                                   blender_version: str) -> float:
        """
        0: Uses APIs/syntax incompatible with version
        1.0: Compatible with specified version
        """
        api_calls = re.findall(r'bpy\.\w+(\.\w+)*', answer)
        api_specs = load_blender_api_spec(blender_version)

        for api in api_calls:
            version_info = api_specs.get(api, {})
            if version_info.get('min_version') > blender_version:
                return 0.0
            if version_info.get('deprecated'):
                return 0.7

        return 1.0

    @staticmethod
    def composite_game_dev_score(
        faithfulness: float,
        relevancy: float,
        precision: float,
        code_correctness: float,
        api_existence: float,
        completeness: float,
        version_compat: float) -> float:
        """
        Weighted score prioritizing correctness for game dev domain
        """
        return (
            0.15 * faithfulness +      # Content grounding
            0.10 * relevancy +         # Query matching
            0.10 * precision +         # Context quality
            0.20 * code_correctness +  # Critical: syntax validity
            0.20 * api_existence +     # Critical: API correctness
            0.15 * completeness +      # Full workflow coverage
            0.10 * version_compat      # Version compatibility
        )
```

---

## 11. Performance Targets

### 11.1 Latency Budget

```
Query Processing Pipeline:

Stage                               | Budget  | Target
=====================================|=========|========
1. Query Embedding (BGE-M3)        | 150ms   | 100ms
2. Dense Vector Search (pgvector)  | 50ms    | 30ms
3. Sparse BM25 Search              | 50ms    | 40ms
4. RRF Fusion                      | 20ms    | 10ms
5. Reranking (MiniLM × 50 docs)    | 750ms   | 600ms
6. Context Assembly                | 50ms    | 30ms
7. LLM Generation (streaming)      | 2000ms  | 1500ms
                                    |---------|--------
TOTAL LATENCY (p95)                | 3600ms  | 2310ms
(NOT including network/rendering)

Target: <200ms for retrieval only
Target: <3s end-to-end with LLM
```

### 11.2 Quality Targets

```
Metric                              | Target
===================================|========
RAGAS Faithfulness                  | >0.85
RAGAS Answer Relevancy              | >0.80
RAGAS Context Precision             | >0.75
RAGAS Context Recall                | >0.70
Code Correctness (game dev)         | >0.95
API Existence (Blender)             | >0.90
Completeness (multi-step tasks)     | >0.80
Hallucination Rate                  | <5%
MRR@10 (retrieval)                  | >0.80
```

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

- [ ] PostgreSQL 15+ setup with pgvector extension
- [ ] Document schema design and table creation
- [ ] BGE-M3 embedding model deployment (CPU or GPU)
- [ ] Chunking pipeline for Blender documentation
- [ ] Initial embedding generation (Blender API + tutorials)
- [ ] HNSW index creation and optimization

### Phase 2: Retrieval (Weeks 5-8)

- [ ] Dense retrieval implementation
- [ ] BM25 sparse retrieval implementation
- [ ] RRF fusion algorithm
- [ ] Metadata filtering system
- [ ] Query analysis and rewriting
- [ ] Multi-hop retrieval decomposition

### Phase 3: Reranking & Quality (Weeks 9-12)

- [ ] MiniLM cross-encoder deployment
- [ ] Context assembly module
- [ ] Agentic validation loop
- [ ] RAGAS evaluation setup
- [ ] Game dev-specific metrics
- [ ] Error handling and fallbacks

### Phase 4: Integration (Weeks 13-16)

- [ ] LangChain/LlamaIndex integration
- [ ] Session memory and conversation context
- [ ] Monitoring and observability
- [ ] Load testing and performance tuning
- [ ] Safety and rate limiting
- [ ] Production deployment

---

## 13. Database Performance Optimization

### 13.1 Index Strategy

```sql
-- HNSW Vector Index (dense search)
CREATE INDEX idx_embedding_hnsw ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m=16, ef_construction=200);

-- GIN Full-Text Index (BM25)
CREATE INDEX idx_tsv_gin ON documents
USING GIN (tsv);

-- Metadata Composite Indexes
CREATE INDEX idx_source_version_priority ON documents
(source, blender_version, priority DESC)
WHERE is_active = true;

CREATE INDEX idx_category_updated ON documents
(category, updated_at DESC);

-- Query analytics
CREATE INDEX idx_session_timestamp ON retrieval_history
(session_id, timestamp DESC);

-- VACUUM ANALYZE for statistics
VACUUM ANALYZE documents;
```

### 13.2 Connection Pooling

```python
import psycopg_pool

# Use PgBouncer for 1000+ QPS
pool = psycopg_pool.AsyncConnectionPool(
    "postgresql://user:password@localhost/ragdb",
    min_size=10,
    max_size=100,
    timeout=5
)

async def query_with_pool(query_text):
    async with pool.connection() as conn:
        result = await conn.execute(query_text)
        return result
```

---

## 14. File Structure

```
rag-system/
├── config/
│   ├── embedding_config.yaml       # BGE-M3 settings
│   ├── retrieval_config.yaml       # Dense/sparse tuning
│   ├── reranker_config.yaml        # MiniLM settings
│   ├── database_config.yaml        # PostgreSQL connection
│   └── evaluation_config.yaml      # RAGAS thresholds
│
├── src/
│   ├── __init__.py
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── models.py              # BGE-M3 wrapper
│   │   ├── chunker.py             # Semantic chunking
│   │   └── ingestion.py           # Document embedding pipeline
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py               # Vector search (pgvector)
│   │   ├── sparse.py              # BM25 search
│   │   ├── fusion.py              # RRF fusion
│   │   ├── filters.py             # Metadata filtering
│   │   └── query_transform.py     # Query analysis & rewriting
│   │
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── cross_encoder.py       # MiniLM reranker
│   │   └── context_assembly.py    # Context formatting
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── answer_validator.py    # Answer validation
│   │   ├── game_dev_metrics.py    # Domain-specific checks
│   │   └── hallucination_detector.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_evaluator.py     # RAGAS framework
│   │   ├── metrics.py             # Metric implementations
│   │   └── reporting.py           # Evaluation reports
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql             # Database schema
│   │   ├── connection.py          # Connection pool
│   │   └── migrations.py          # Schema migrations
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py        # Full RAG orchestration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── metrics.py
│       └── constants.py
│
├── data/
│   ├── blender_api/               # Blender documentation
│   ├── tutorials/                 # Game dev tutorials
│   ├── best_practices/            # Best practice guides
│   └── embeddings/                # Pre-computed embeddings (if cached)
│
├── scripts/
│   ├── ingest_documents.py        # Load docs into database
│   ├── evaluate_rag.py            # Run RAGAS evaluation
│   ├── optimize_indexes.py        # Database optimization
│   └── load_test.py               # Performance testing
│
├── tests/
│   ├── test_dense_retrieval.py
│   ├── test_sparse_retrieval.py
│   ├── test_fusion.py
│   ├── test_reranking.py
│   ├── test_validation.py
│   └── test_end_to_end.py
│
├── requirements.txt
├── pyproject.toml
├── README.md
└── DEPLOYMENT.md
```

---

## 15. Technology Stack

### Python Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.10"

# Core RAG
langchain = "^0.2.0"
langchain-nomic = "^0.1.0"
sentence-transformers = "^2.7.0"

# Vector Database
psycopg2-binary = "^2.9.0"
pgvector = "^0.2.0"

# Embedding Models
huggingface-hub = "^0.21.0"
transformers = "^4.36.0"

# Evaluation
ragas = "^0.1.0"
datasets = "^2.16.0"

# LLM
anthropic = "^0.25.0"  # Claude API
ollama = "^0.3.0"      # Local models

# Infrastructure
fastapi = "^0.110.0"
uvicorn = "^0.28.0"
redis = "^5.0.0"
pydantic = "^2.0.0"

# Monitoring
opentelemetry-api = "^1.20.0"
prometheus-client = "^0.19.0"
loguru = "^0.7.0"

# Utilities
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
pandas = "^2.1.0"
```

---

## 16. Deployment Considerations

### 16.1 Hardware Requirements

```
Minimum:
- CPU: 4-core (8 preferred)
- RAM: 16GB (32GB for production)
- Storage: 500GB SSD (document + embeddings)
- GPU: Optional (for faster embedding inference)

Production:
- CPU: 16-core
- RAM: 64GB
- Storage: 2TB SSD with replication
- GPU: NVIDIA A10 or better (optional)

Network:
- PostgreSQL connection pool: 100 concurrent
- Query throughput: 100 QPS target
```

### 16.2 Scalability

```
Horizontal Scaling:

1. Read Replicas
   - Multiple PostgreSQL read replicas for query distribution
   - Connection pooling across replicas

2. Caching Layer
   - Redis cache for embedding queries
   - TTL: 1 hour (docs change infrequently)

3. Embedding Service
   - Separate embedding inference server
   - Batch processing for bulk operations

4. LLM Service
   - Use Claude API (Anthropic handles scaling)
   - Or local LLM with Ollama + load balancing
```

---

## References

### Key Papers & Resources

1. **RAG Survey (2024)** - Gao et al., "A Survey on Retrieval-Augmented Text Generation"
2. **RAGAS Framework** - Metrics for autonomous RAG evaluation
3. **Hybrid Search** - Luan et al., "Dense Passage Retrieval for Open-Domain Question Answering"
4. **Cross-Encoder** - Thakur et al., "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
5. **HNSW Index** - Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search"
6. **BM25** - Robertson et al., "Okapi BM25 Model for Information Retrieval"

### Tools & Frameworks

- **LangChain** - https://docs.langchain.com/
- **LlamaIndex** - https://docs.llamaindex.ai/
- **PostgreSQL pgvector** - https://github.com/pgvector/pgvector
- **Sentence Transformers** - https://www.sbert.net/
- **RAGAS** - https://github.com/explodinggradients/ragas
- **Hugging Face** - https://huggingface.co/models

---

## 17. Error Handling & Resilience

### 17.1 Circuit Breaker Pattern

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 3       # Successes to close from half-open
    timeout_seconds: int = 30        # Time in open state before half-open
    half_open_max_calls: int = 3     # Max calls in half-open state

class CircuitBreaker:
    """
    Circuit breaker for external service calls (embedding model, LLM, database)

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, reject immediately with fallback
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def call(self, func, fallback=None, *args, **kwargs):
        """
        Execute function with circuit breaker protection

        Args:
            func: Function to execute
            fallback: Fallback function if circuit is open
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    return self._execute_fallback(fallback, *args, **kwargs)

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    return self._execute_fallback(fallback, *args, **kwargs)
                self.half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            if fallback:
                return self._execute_fallback(fallback, *args, **kwargs)
            raise

    def _record_success(self):
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset on success

    def _record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = datetime.now() - self.last_failure_time
        return elapsed > timedelta(seconds=self.config.timeout_seconds)

    def _execute_fallback(self, fallback, *args, **kwargs):
        if fallback:
            return fallback(*args, **kwargs)
        raise CircuitOpenError(f"Circuit {self.name} is open")

class CircuitOpenError(Exception):
    pass

# Circuit breakers for each external service
CIRCUITS = {
    'embedding_model': CircuitBreaker('embedding_model', CircuitBreakerConfig(
        failure_threshold=3, timeout_seconds=60
    )),
    'llm_api': CircuitBreaker('llm_api', CircuitBreakerConfig(
        failure_threshold=5, timeout_seconds=30
    )),
    'database': CircuitBreaker('database', CircuitBreakerConfig(
        failure_threshold=3, timeout_seconds=15
    )),
    'reranker': CircuitBreaker('reranker', CircuitBreakerConfig(
        failure_threshold=5, timeout_seconds=45
    ))
}
```

### 17.2 Retry Logic with Exponential Backoff

```python
import asyncio
import random
from functools import wraps
from typing import Type, Tuple, Callable

class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

def with_retry(config: RetryConfig = None):
    """
    Decorator for retry with exponential backoff

    Delay formula: min(max_delay, base_delay * (exponential_base ^ attempt))
    With jitter: delay * (0.5 + random())
    """
    config = config or RetryConfig()

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        break

                    delay = min(
                        config.max_delay,
                        config.base_delay * (config.exponential_base ** attempt)
                    )

                    if config.jitter:
                        delay *= (0.5 + random.random())

                    await asyncio.sleep(delay)

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        break

                    delay = min(
                        config.max_delay,
                        config.base_delay * (config.exponential_base ** attempt)
                    )

                    if config.jitter:
                        delay *= (0.5 + random.random())

                    time.sleep(delay)

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator

# Retry configurations for different operations
RETRY_CONFIGS = {
    'embedding': RetryConfig(
        max_retries=3,
        base_delay=0.5,
        retryable_exceptions=(TimeoutError, ConnectionError)
    ),
    'database': RetryConfig(
        max_retries=5,
        base_delay=0.2,
        max_delay=10.0,
        retryable_exceptions=(ConnectionError, TimeoutError)
    ),
    'llm': RetryConfig(
        max_retries=2,
        base_delay=2.0,
        max_delay=30.0,
        retryable_exceptions=(TimeoutError, RateLimitError)
    )
}
```

### 17.3 Fallback Strategies

```python
from dataclasses import dataclass
from typing import List, Optional, Any
from abc import ABC, abstractmethod

class FallbackStrategy(ABC):
    """Base class for fallback strategies"""

    @abstractmethod
    def execute(self, context: dict) -> Any:
        pass

class CachedResponseFallback(FallbackStrategy):
    """Return cached response for similar queries"""

    def __init__(self, cache_client, similarity_threshold: float = 0.85):
        self.cache = cache_client
        self.threshold = similarity_threshold

    def execute(self, context: dict) -> Optional[dict]:
        query = context.get('query')
        if not query:
            return None

        # Find similar cached query
        cached = self.cache.find_similar(query, threshold=self.threshold)
        if cached:
            return {
                'answer': cached['answer'],
                'source': 'cache',
                'confidence': cached['similarity'],
                'warning': 'Response from cache - may not be current'
            }
        return None

class DegradedModeFallback(FallbackStrategy):
    """Provide degraded but functional response"""

    def execute(self, context: dict) -> dict:
        query = context.get('query', '')

        return {
            'answer': self._generate_degraded_response(query),
            'source': 'degraded_mode',
            'confidence': 0.3,
            'warning': 'Service temporarily degraded. Response may be limited.'
        }

    def _generate_degraded_response(self, query: str) -> str:
        # Use simple keyword matching for basic responses
        keywords = {
            'select': 'For selection operations, try using bpy.ops.object.select_all() or mesh selection operators.',
            'rotate': 'For rotation, use bpy.ops.transform.rotate() or modify obj.rotation_euler.',
            'material': 'Materials can be created with bpy.data.materials.new() and assigned via obj.data.materials.append().'
        }

        for keyword, response in keywords.items():
            if keyword in query.lower():
                return response

        return "I'm experiencing technical difficulties. Please try again in a few moments."

class BM25OnlyFallback(FallbackStrategy):
    """Fall back to BM25-only retrieval when dense search fails"""

    def __init__(self, bm25_retriever):
        self.retriever = bm25_retriever

    def execute(self, context: dict) -> Optional[List[dict]]:
        query = context.get('query')
        if not query:
            return None

        # Use BM25 sparse retrieval only
        results = self.retriever.search(query, k=10)
        return results if results else None

class FallbackChain:
    """Chain multiple fallback strategies"""

    def __init__(self, strategies: List[FallbackStrategy]):
        self.strategies = strategies

    def execute(self, context: dict) -> Any:
        for strategy in self.strategies:
            try:
                result = strategy.execute(context)
                if result is not None:
                    return result
            except Exception:
                continue

        # Ultimate fallback
        return {
            'answer': "Service temporarily unavailable. Please try again later.",
            'source': 'error_fallback',
            'confidence': 0.0
        }

# Configure fallback chain for RAG pipeline
RAG_FALLBACK_CHAIN = FallbackChain([
    CachedResponseFallback(cache_client=None),  # Inject cache
    BM25OnlyFallback(bm25_retriever=None),      # Inject retriever
    DegradedModeFallback()
])
```

### 17.4 Timeout Management

```python
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

@dataclass
class TimeoutConfig:
    """Timeout configuration for each operation"""
    query_embedding: float = 5.0      # 5 seconds
    dense_search: float = 3.0         # 3 seconds
    sparse_search: float = 2.0        # 2 seconds
    reranking: float = 10.0           # 10 seconds (50 docs)
    llm_generation: float = 30.0      # 30 seconds
    total_request: float = 60.0       # 60 seconds total

TIMEOUTS = TimeoutConfig()

@asynccontextmanager
async def timeout_context(seconds: float, operation: str):
    """
    Async context manager with timeout

    Usage:
        async with timeout_context(5.0, "embedding"):
            result = await embed_query(query)
    """
    try:
        yield await asyncio.wait_for(
            asyncio.create_task(asyncio.sleep(0)),  # Placeholder
            timeout=seconds
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation '{operation}' timed out after {seconds}s")

class TimeoutManager:
    """Manage timeouts across the RAG pipeline"""

    def __init__(self, config: TimeoutConfig = None):
        self.config = config or TIMEOUTS
        self.start_time = None

    def start_request(self):
        self.start_time = asyncio.get_event_loop().time()

    def remaining_time(self) -> float:
        if self.start_time is None:
            return self.config.total_request

        elapsed = asyncio.get_event_loop().time() - self.start_time
        return max(0, self.config.total_request - elapsed)

    def get_timeout(self, operation: str) -> float:
        """Get timeout for operation, capped by remaining request time"""
        operation_timeout = getattr(self.config, operation, 10.0)
        return min(operation_timeout, self.remaining_time())
```

### 17.5 Error Classification & Handling

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ErrorSeverity(Enum):
    LOW = "low"           # Recoverable, use fallback
    MEDIUM = "medium"     # Degraded response possible
    HIGH = "high"         # Request failed, retry later
    CRITICAL = "critical" # System issue, alert required

class ErrorCategory(Enum):
    NETWORK = "network"
    DATABASE = "database"
    MODEL = "model"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"

@dataclass
class RAGError:
    """Structured error for RAG pipeline"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    operation: str
    original_exception: Optional[Exception] = None
    retry_after: Optional[int] = None  # Seconds

    def to_dict(self) -> dict:
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'operation': self.operation,
            'retry_after': self.retry_after
        }

class ErrorHandler:
    """Centralized error handling for RAG pipeline"""

    ERROR_MAPPING = {
        ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
        TimeoutError: (ErrorCategory.NETWORK, ErrorSeverity.LOW),
        MemoryError: (ErrorCategory.INTERNAL, ErrorSeverity.CRITICAL),
        ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.LOW),
    }

    @classmethod
    def handle(cls, exception: Exception, operation: str) -> RAGError:
        """Convert exception to structured RAGError"""

        category, severity = cls.ERROR_MAPPING.get(
            type(exception),
            (ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM)
        )

        # Check for rate limiting
        if hasattr(exception, 'retry_after'):
            return RAGError(
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                message=str(exception),
                operation=operation,
                original_exception=exception,
                retry_after=exception.retry_after
            )

        return RAGError(
            category=category,
            severity=severity,
            message=str(exception),
            operation=operation,
            original_exception=exception
        )

    @classmethod
    def should_retry(cls, error: RAGError) -> bool:
        """Determine if error is retryable"""
        return error.severity in (ErrorSeverity.LOW, ErrorSeverity.MEDIUM)

    @classmethod
    def should_alert(cls, error: RAGError) -> bool:
        """Determine if error requires alerting"""
        return error.severity == ErrorSeverity.CRITICAL
```

---

## 18. Security & Authentication

### 18.1 Authentication Layer

```python
from datetime import datetime, timedelta
from typing import Optional
import jwt
import hashlib
import secrets

@dataclass
class AuthConfig:
    """Authentication configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_key_length: int = 32

class AuthenticationManager:
    """Handle API authentication"""

    def __init__(self, config: AuthConfig):
        self.config = config

    def create_access_token(self, user_id: str, scopes: List[str] = None) -> str:
        """Create JWT access token"""
        expire = datetime.utcnow() + timedelta(
            minutes=self.config.access_token_expire_minutes
        )

        payload = {
            'sub': user_id,
            'exp': expire,
            'iat': datetime.utcnow(),
            'scopes': scopes or ['read', 'query']
        }

        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and its hash"""
        api_key = secrets.token_urlsafe(self.config.api_key_length)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return api_key, key_hash

    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        computed_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return secrets.compare_digest(computed_hash, stored_hash)

class AuthenticationError(Exception):
    pass
```

### 18.2 Rate Limiting

```python
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
import redis

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max concurrent requests

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API requests

    Algorithm:
    - Bucket fills at rate of `requests_per_minute / 60` tokens per second
    - Each request consumes 1 token
    - Bucket has max capacity of `burst_limit`
    """

    def __init__(self, config: RateLimitConfig, redis_client: redis.Redis = None):
        self.config = config
        self.redis = redis_client
        self.local_buckets: Dict[str, dict] = defaultdict(lambda: {
            'tokens': config.burst_limit,
            'last_update': time.time()
        })

    def is_allowed(self, user_id: str) -> tuple[bool, dict]:
        """
        Check if request is allowed

        Returns:
            (allowed: bool, metadata: dict)
        """
        if self.redis:
            return self._check_redis(user_id)
        return self._check_local(user_id)

    def _check_local(self, user_id: str) -> tuple[bool, dict]:
        bucket = self.local_buckets[user_id]
        now = time.time()

        # Refill tokens based on time elapsed
        time_passed = now - bucket['last_update']
        refill_rate = self.config.requests_per_minute / 60.0
        new_tokens = min(
            self.config.burst_limit,
            bucket['tokens'] + (time_passed * refill_rate)
        )

        bucket['tokens'] = new_tokens
        bucket['last_update'] = now

        # Check if we have tokens
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True, {
                'remaining': int(bucket['tokens']),
                'limit': self.config.requests_per_minute,
                'reset_at': now + (60 / refill_rate)
            }

        return False, {
            'remaining': 0,
            'limit': self.config.requests_per_minute,
            'retry_after': int((1 - bucket['tokens']) / refill_rate)
        }

    def _check_redis(self, user_id: str) -> tuple[bool, dict]:
        """Redis-based distributed rate limiting"""
        key = f"rate_limit:{user_id}"
        pipe = self.redis.pipeline()

        now = time.time()
        window_start = now - 60  # 1 minute window

        # Remove old entries and count recent ones
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, 60)

        results = pipe.execute()
        request_count = results[1]

        if request_count < self.config.requests_per_minute:
            return True, {
                'remaining': self.config.requests_per_minute - request_count - 1,
                'limit': self.config.requests_per_minute
            }

        return False, {
            'remaining': 0,
            'limit': self.config.requests_per_minute,
            'retry_after': 60
        }

class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after
```

### 18.3 Input Validation & Sanitization

```python
import re
from typing import Optional
from pydantic import BaseModel, validator, Field

class QueryRequest(BaseModel):
    """Validated query request schema"""

    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, max_length=100)
    filters: Optional[dict] = None
    max_results: int = Field(10, ge=1, le=50)

    @validator('query')
    def sanitize_query(cls, v):
        # Remove potentially dangerous characters
        v = re.sub(r'[<>{}]', '', v)

        # Prevent SQL injection patterns
        sql_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|create|alter)\s",
            r"(?i)(--|;|\/\*|\*\/)",
            r"(?i)(or|and)\s+\d+\s*=\s*\d+"
        ]
        for pattern in sql_patterns:
            if re.search(pattern, v):
                raise ValueError("Invalid query pattern detected")

        # Limit consecutive whitespace
        v = re.sub(r'\s+', ' ', v).strip()

        return v

    @validator('session_id')
    def validate_session_id(cls, v):
        if v is None:
            return v
        # Only allow alphanumeric and hyphens
        if not re.match(r'^[a-zA-Z0-9-]+$', v):
            raise ValueError("Invalid session ID format")
        return v

    @validator('filters')
    def validate_filters(cls, v):
        if v is None:
            return v

        allowed_keys = {
            'blender_version', 'source', 'category',
            'is_code', 'difficulty', 'language'
        }

        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Unknown filter key: {key}")

        return v

class InputSanitizer:
    """Additional input sanitization utilities"""

    @staticmethod
    def sanitize_for_embedding(text: str) -> str:
        """Sanitize text before embedding generation"""
        # Remove null bytes
        text = text.replace('\x00', '')

        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize('NFKC', text)

        # Limit length
        max_chars = 8000  # BGE-M3 max context
        if len(text) > max_chars:
            text = text[:max_chars]

        return text

    @staticmethod
    def sanitize_code_output(code: str) -> str:
        """Sanitize code before returning to user"""
        # Remove any embedded secrets patterns
        secret_patterns = [
            r'(?i)(api_key|apikey|secret|password|token)\s*=\s*["\'][^"\']+["\']',
            r'(?i)bearer\s+[a-zA-Z0-9._-]+',
        ]

        for pattern in secret_patterns:
            code = re.sub(pattern, '[REDACTED]', code)

        return code
```

### 18.4 Data Encryption

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionManager:
    """Handle data encryption for sensitive fields"""

    def __init__(self, master_key: str):
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'rag_system_salt',  # Use env variable in production
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher = Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    def encrypt_embedding(self, embedding: list) -> bytes:
        """Encrypt embedding vector (for sensitive documents)"""
        import pickle
        serialized = pickle.dumps(embedding)
        return self.cipher.encrypt(serialized)

    def decrypt_embedding(self, encrypted: bytes) -> list:
        """Decrypt embedding vector"""
        import pickle
        decrypted = self.cipher.decrypt(encrypted)
        return pickle.loads(decrypted)

# Fields that should be encrypted at rest
ENCRYPTED_FIELDS = {
    'documents': ['content'],  # Encrypt document content if sensitive
    'retrieval_history': ['query', 'final_answer'],
    'user_sessions': ['context']
}
```

### 18.5 Access Control

```python
from enum import Enum
from typing import Set, List
from functools import wraps

class Permission(Enum):
    READ = "read"
    QUERY = "query"
    ADMIN = "admin"
    WRITE = "write"
    DELETE = "delete"
    EVALUATE = "evaluate"

class Role(Enum):
    VIEWER = "viewer"           # Read-only access
    USER = "user"               # Query access
    DEVELOPER = "developer"     # Full query + evaluation
    ADMIN = "admin"             # All permissions

ROLE_PERMISSIONS = {
    Role.VIEWER: {Permission.READ},
    Role.USER: {Permission.READ, Permission.QUERY},
    Role.DEVELOPER: {Permission.READ, Permission.QUERY, Permission.EVALUATE, Permission.WRITE},
    Role.ADMIN: set(Permission)
}

class AccessControlManager:
    """Role-based access control for RAG system"""

    def __init__(self):
        self.user_roles: dict[str, Role] = {}

    def assign_role(self, user_id: str, role: Role):
        self.user_roles[user_id] = role

    def get_permissions(self, user_id: str) -> Set[Permission]:
        role = self.user_roles.get(user_id, Role.VIEWER)
        return ROLE_PERMISSIONS[role]

    def check_permission(self, user_id: str, required: Permission) -> bool:
        permissions = self.get_permissions(user_id)
        return required in permissions

def require_permission(permission: Permission):
    """Decorator to enforce permission requirements"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, user_id: str = None, **kwargs):
            if user_id is None:
                raise PermissionError("User ID required")

            acl = AccessControlManager()
            if not acl.check_permission(user_id, permission):
                raise PermissionError(
                    f"Permission denied: {permission.value} required"
                )

            return func(*args, user_id=user_id, **kwargs)
        return wrapper
    return decorator

class PermissionError(Exception):
    pass
```

---

## 19. Observability & Monitoring

### 19.1 Structured Logging

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict
from contextvars import ContextVar

# Request context for correlation
request_id_var: ContextVar[str] = ContextVar('request_id', default='unknown')
user_id_var: ContextVar[str] = ContextVar('user_id', default='anonymous')

class StructuredLogger:
    """JSON structured logging for observability"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)

    def _build_log_entry(self, level: str, message: str, **kwargs) -> Dict:
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'request_id': request_id_var.get(),
            'user_id': user_id_var.get(),
            'service': 'rag_system',
            **kwargs
        }

    def info(self, message: str, **kwargs):
        entry = self._build_log_entry('INFO', message, **kwargs)
        self.logger.info(json.dumps(entry))

    def error(self, message: str, exception: Exception = None, **kwargs):
        entry = self._build_log_entry('ERROR', message, **kwargs)
        if exception:
            entry['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception)
            }
        self.logger.error(json.dumps(entry))

    def query(self, query: str, latency_ms: float, results_count: int, **kwargs):
        """Log query execution"""
        entry = self._build_log_entry('INFO', 'query_executed',
            query=query[:200],  # Truncate for logs
            latency_ms=latency_ms,
            results_count=results_count,
            **kwargs
        )
        self.logger.info(json.dumps(entry))

    def retrieval(self, stage: str, docs_count: int, latency_ms: float, **kwargs):
        """Log retrieval stage"""
        entry = self._build_log_entry('INFO', f'retrieval_{stage}',
            stage=stage,
            docs_count=docs_count,
            latency_ms=latency_ms,
            **kwargs
        )
        self.logger.info(json.dumps(entry))

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()

# Global logger instance
logger = StructuredLogger('rag')
```

### 19.2 Metrics Collection (Prometheus)

```python
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Request metrics
REQUESTS_TOTAL = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'rag_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Retrieval metrics
RETRIEVAL_LATENCY = Histogram(
    'rag_retrieval_latency_seconds',
    'Retrieval latency by stage',
    ['stage'],  # dense, sparse, fusion, rerank
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

DOCUMENTS_RETRIEVED = Histogram(
    'rag_documents_retrieved',
    'Number of documents retrieved',
    ['stage'],
    buckets=[1, 5, 10, 25, 50, 100]
)

# Quality metrics
VALIDATION_SCORE = Histogram(
    'rag_validation_score',
    'Answer validation scores',
    ['metric'],  # faithfulness, relevancy, precision
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

HALLUCINATION_DETECTED = Counter(
    'rag_hallucination_detected_total',
    'Hallucinations detected',
    ['severity']
)

# System metrics
CIRCUIT_BREAKER_STATE = Gauge(
    'rag_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half_open)',
    ['service']
)

CACHE_HIT_RATE = Gauge(
    'rag_cache_hit_rate',
    'Cache hit rate',
    ['cache_type']  # embedding, query, response
)

# Model metrics
EMBEDDING_GENERATION_TIME = Histogram(
    'rag_embedding_generation_seconds',
    'Embedding generation time',
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0]
)

LLM_TOKENS_USED = Counter(
    'rag_llm_tokens_total',
    'LLM tokens used',
    ['type']  # input, output
)

class MetricsCollector:
    """Centralized metrics collection"""

    @staticmethod
    def record_request(endpoint: str, status: str, latency: float):
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    @staticmethod
    def record_retrieval(stage: str, docs_count: int, latency: float):
        RETRIEVAL_LATENCY.labels(stage=stage).observe(latency)
        DOCUMENTS_RETRIEVED.labels(stage=stage).observe(docs_count)

    @staticmethod
    def record_validation(scores: dict):
        for metric, score in scores.items():
            VALIDATION_SCORE.labels(metric=metric).observe(score)

    @staticmethod
    def record_circuit_state(service: str, state: str):
        state_map = {'closed': 0, 'open': 1, 'half_open': 2}
        CIRCUIT_BREAKER_STATE.labels(service=service).set(state_map.get(state, 0))
```

### 19.3 Distributed Tracing (OpenTelemetry)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from contextlib import contextmanager
from typing import Optional

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to OTLP collector
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317")
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

class RAGTracer:
    """Distributed tracing for RAG pipeline"""

    def __init__(self):
        self.tracer = trace.get_tracer("rag_system")

    @contextmanager
    def trace_operation(self, name: str, attributes: dict = None):
        """Trace an operation"""
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            try:
                yield span
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def trace_query(self, query: str):
        """Start query trace"""
        return self.trace_operation("rag.query", {
            "query.length": len(query),
            "query.preview": query[:100]
        })

    def trace_embedding(self, text_length: int):
        """Trace embedding generation"""
        return self.trace_operation("rag.embedding", {
            "text.length": text_length
        })

    def trace_retrieval(self, stage: str, k: int):
        """Trace retrieval stage"""
        return self.trace_operation(f"rag.retrieval.{stage}", {
            "retrieval.k": k,
            "retrieval.stage": stage
        })

    def trace_reranking(self, candidates: int, top_k: int):
        """Trace reranking"""
        return self.trace_operation("rag.reranking", {
            "reranking.candidates": candidates,
            "reranking.top_k": top_k
        })

    def trace_llm(self, prompt_length: int):
        """Trace LLM generation"""
        return self.trace_operation("rag.llm", {
            "llm.prompt_length": prompt_length
        })

rag_tracer = RAGTracer()
```

### 19.4 Health Checks

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List
import asyncio

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    latency_ms: float
    message: str = ""

class HealthChecker:
    """Health check system for RAG components"""

    def __init__(self, db_pool, embedding_model, llm_client):
        self.db_pool = db_pool
        self.embedding_model = embedding_model
        self.llm_client = llm_client

    async def check_all(self) -> Dict:
        """Run all health checks"""
        checks = await asyncio.gather(
            self.check_database(),
            self.check_embedding_model(),
            self.check_llm(),
            self.check_cache(),
            return_exceptions=True
        )

        results = {}
        overall_status = HealthStatus.HEALTHY

        for check in checks:
            if isinstance(check, Exception):
                results['error'] = str(check)
                overall_status = HealthStatus.UNHEALTHY
            else:
                results[check.name] = check.__dict__
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

        return {
            'status': overall_status.value,
            'components': results,
            'timestamp': datetime.utcnow().isoformat()
        }

    async def check_database(self) -> ComponentHealth:
        """Check PostgreSQL connectivity"""
        start = time.time()
        try:
            async with self.db_pool.connection() as conn:
                await conn.execute("SELECT 1")
            latency = (time.time() - start) * 1000

            status = HealthStatus.HEALTHY if latency < 100 else HealthStatus.DEGRADED
            return ComponentHealth(
                name="database",
                status=status,
                latency_ms=latency
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=-1,
                message=str(e)
            )

    async def check_embedding_model(self) -> ComponentHealth:
        """Check embedding model availability"""
        start = time.time()
        try:
            # Quick test embedding
            _ = self.embedding_model.embed_query("health check")
            latency = (time.time() - start) * 1000

            status = HealthStatus.HEALTHY if latency < 500 else HealthStatus.DEGRADED
            return ComponentHealth(
                name="embedding_model",
                status=status,
                latency_ms=latency
            )
        except Exception as e:
            return ComponentHealth(
                name="embedding_model",
                status=HealthStatus.UNHEALTHY,
                latency_ms=-1,
                message=str(e)
            )

    async def check_llm(self) -> ComponentHealth:
        """Check LLM API availability"""
        start = time.time()
        try:
            # Minimal API call
            _ = await self.llm_client.complete("Hi", max_tokens=1)
            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="llm",
                status=HealthStatus.HEALTHY,
                latency_ms=latency
            )
        except Exception as e:
            return ComponentHealth(
                name="llm",
                status=HealthStatus.UNHEALTHY,
                latency_ms=-1,
                message=str(e)
            )

    async def check_cache(self) -> ComponentHealth:
        """Check Redis cache"""
        # Implementation depends on cache client
        return ComponentHealth(
            name="cache",
            status=HealthStatus.HEALTHY,
            latency_ms=1.0
        )

# Health endpoints
"""
GET /health          → Overall health status
GET /health/live     → Liveness probe (is process running?)
GET /health/ready    → Readiness probe (can handle requests?)
"""
```

### 19.5 Alerting Rules

```yaml
# prometheus/alerts.yaml
groups:
  - name: rag_alerts
    rules:
      # High error rate
      - alert: RAGHighErrorRate
        expr: |
          sum(rate(rag_requests_total{status="error"}[5m]))
          / sum(rate(rag_requests_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "RAG error rate above 5%"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # High latency
      - alert: RAGHighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(rag_request_latency_seconds_bucket[5m])) by (le)
          ) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RAG p95 latency above 5s"

      # Circuit breaker open
      - alert: RAGCircuitBreakerOpen
        expr: rag_circuit_breaker_state > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker open for {{ $labels.service }}"

      # Low cache hit rate
      - alert: RAGLowCacheHitRate
        expr: rag_cache_hit_rate < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate below 50%"

      # High hallucination rate
      - alert: RAGHighHallucinationRate
        expr: |
          sum(rate(rag_hallucination_detected_total[1h]))
          / sum(rate(rag_requests_total[1h])) > 0.1
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Hallucination rate above 10%"

      # Database connection pool exhausted
      - alert: RAGDatabasePoolExhausted
        expr: pg_stat_activity_count > 90
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
```

---

## 20. Testing Strategy

### 20.1 Unit Tests

```python
import pytest
from unittest.mock import Mock, patch
import numpy as np

class TestEmbeddingPipeline:
    """Unit tests for embedding generation"""

    @pytest.fixture
    def embedding_pipeline(self):
        with patch('embedding.models.HuggingFaceEmbeddings'):
            pipeline = EmbeddingPipeline()
            pipeline.model.embed_query = Mock(
                return_value=np.random.rand(4096).tolist()
            )
            return pipeline

    def test_generate_single_embedding(self, embedding_pipeline):
        result = embedding_pipeline.generate_embeddings(["test document"])
        assert result.shape == (1, 4096)

    def test_generate_batch_embeddings(self, embedding_pipeline):
        docs = ["doc1", "doc2", "doc3"]
        result = embedding_pipeline.generate_embeddings(docs)
        assert result.shape == (3, 4096)

    def test_embedding_caching(self, embedding_pipeline):
        doc = "test document"
        embedding_pipeline.generate_embeddings([doc])
        embedding_pipeline.generate_embeddings([doc])

        # Should only call model once due to caching
        assert embedding_pipeline.model.embed_query.call_count == 1

    def test_embedding_normalization(self, embedding_pipeline):
        result = embedding_pipeline.generate_embeddings(["test"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01  # Should be normalized

class TestBM25Retriever:
    """Unit tests for BM25 retrieval"""

    @pytest.fixture
    def bm25_retriever(self):
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.fit([
            "How to select all faces in Blender",
            "Rotating objects in 3D space",
            "Applying smooth shading to mesh",
            "Blender Python API reference"
        ])
        return retriever

    def test_search_returns_results(self, bm25_retriever):
        results = bm25_retriever.search("select faces", k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_search_relevance_ordering(self, bm25_retriever):
        results = bm25_retriever.search("select faces", k=4)
        # First result should be most relevant
        assert results[0][1] >= results[1][1]

    def test_idf_calculation(self, bm25_retriever):
        # Common words should have lower IDF
        assert bm25_retriever.idf.get('blender', 0) < bm25_retriever.idf.get('smooth', float('inf'))

    def test_empty_query(self, bm25_retriever):
        results = bm25_retriever.search("", k=5)
        assert len(results) == 0

class TestRRFFusion:
    """Unit tests for RRF fusion"""

    def test_fusion_combines_results(self):
        dense = [("doc_a", 0.9), ("doc_b", 0.8), ("doc_c", 0.7)]
        sparse = [("doc_b", 12.5), ("doc_a", 8.3), ("doc_d", 5.1)]

        fused = ReciprocalRankFusion.fuse(dense, sparse)

        # Both doc_a and doc_b should be top-ranked
        top_docs = [doc for doc, _ in fused[:2]]
        assert "doc_a" in top_docs
        assert "doc_b" in top_docs

    def test_rrf_score_calculation(self):
        dense = [("doc_a", 0.9)]  # Rank 1
        sparse = [("doc_a", 10.0)]  # Rank 1

        fused = ReciprocalRankFusion.fuse(dense, sparse)

        # RRF score for doc_a: 1/(60+1) + 1/(60+1) = 2 * 0.0164 ≈ 0.0328
        expected_score = 2 * (1 / 61)
        assert abs(fused[0][1] - expected_score) < 0.001

class TestCrossEncoderReranker:
    """Unit tests for cross-encoder reranking"""

    @pytest.fixture
    def reranker(self):
        with patch('reranking.cross_encoder.CrossEncoder'):
            reranker = CrossEncoderReranker()
            reranker.model.predict = Mock(
                return_value=[0.9, 0.7, 0.5]
            )
            return reranker

    def test_rerank_returns_top_k(self, reranker):
        candidates = ["doc1", "doc2", "doc3"]
        results = reranker.rerank("query", candidates, top_k=2)
        assert len(results) == 2

    def test_rerank_ordering(self, reranker):
        candidates = ["doc1", "doc2", "doc3"]
        results = reranker.rerank("query", candidates, top_k=3)

        # Should be sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
```

### 20.2 Integration Tests

```python
import pytest
import asyncio
from testcontainers.postgres import PostgresContainer

class TestRAGPipelineIntegration:
    """Integration tests for full RAG pipeline"""

    @pytest.fixture(scope="class")
    def postgres_container(self):
        with PostgresContainer("pgvector/pgvector:pg15") as postgres:
            yield postgres

    @pytest.fixture
    def rag_pipeline(self, postgres_container):
        # Initialize with test database
        db_url = postgres_container.get_connection_url()
        pipeline = AgenticRAGPipeline(
            database_url=db_url,
            embedding_model="test",
            llm_client=MockLLMClient()
        )

        # Seed test data
        pipeline.seed_test_data()

        return pipeline

    @pytest.mark.asyncio
    async def test_end_to_end_query(self, rag_pipeline):
        """Test full query flow"""
        query = "How do I select all faces in Blender?"

        answer, metadata = await rag_pipeline.generate(query)

        assert answer is not None
        assert len(answer) > 0
        assert metadata['success'] is True

    @pytest.mark.asyncio
    async def test_retrieval_pipeline(self, rag_pipeline):
        """Test retrieval returns relevant documents"""
        query = "rotate mesh objects"

        context, metadata = await rag_pipeline._retrieve_context(query)

        assert 'rotate' in context.lower() or 'transform' in context.lower()
        assert metadata['final_context_docs'] > 0

    @pytest.mark.asyncio
    async def test_validation_loop_triggers(self, rag_pipeline):
        """Test validation loop on bad answer"""
        # Configure mock to return invalid answer first
        rag_pipeline.llm.set_responses([
            "Invalid answer with bpy.invalid.api()",  # Will fail validation
            "Valid answer with bpy.ops.mesh.select_all()"  # Will pass
        ])

        answer, metadata = await rag_pipeline.generate("select faces")

        assert metadata['attempts'] > 1  # Should have retried
        assert metadata['success'] is True

    @pytest.mark.asyncio
    async def test_metadata_filtering(self, rag_pipeline):
        """Test metadata filters are applied"""
        query = "Blender 4.2 API for mesh selection"

        context, metadata = await rag_pipeline._retrieve_context(
            query,
            filters={'blender_version': ['4.2']}
        )

        # Should only retrieve Blender 4.2 docs
        # Verification depends on test data setup

class TestDatabaseIntegration:
    """Integration tests for database operations"""

    @pytest.fixture
    def db_client(self, postgres_container):
        return DatabaseClient(postgres_container.get_connection_url())

    def test_vector_search(self, db_client):
        """Test pgvector similarity search"""
        # Insert test vector
        test_embedding = np.random.rand(4096).tolist()
        db_client.insert_document(
            content="test document",
            embedding=test_embedding
        )

        # Search with same vector should return it
        results = db_client.vector_search(test_embedding, k=1)
        assert len(results) == 1
        assert results[0]['content'] == "test document"

    def test_bm25_search(self, db_client):
        """Test PostgreSQL full-text search"""
        db_client.insert_document(
            content="Blender mesh selection tutorial"
        )

        results = db_client.bm25_search("mesh selection", k=5)
        assert any("mesh" in r['content'].lower() for r in results)
```

### 20.3 Performance Tests

```python
import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    """Performance and load tests"""

    @pytest.fixture
    def performance_pipeline(self):
        # Use production-like configuration
        return AgenticRAGPipeline(
            database_url=PERFORMANCE_DB_URL,
            embedding_model=PRODUCTION_MODEL
        )

    def test_embedding_latency(self, performance_pipeline):
        """Test embedding generation stays under budget"""
        queries = ["test query " * i for i in range(1, 11)]
        latencies = []

        for query in queries:
            start = time.time()
            performance_pipeline.embed_query(query)
            latencies.append((time.time() - start) * 1000)

        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        assert p95 < 150, f"Embedding p95 latency {p95}ms exceeds 150ms budget"

    def test_retrieval_latency(self, performance_pipeline):
        """Test retrieval pipeline latency"""
        queries = [
            "select all faces",
            "rotate mesh objects",
            "apply smooth shading",
            "UV unwrap tutorial",
            "Blender Python API"
        ]

        latencies = []
        for query in queries:
            start = time.time()
            performance_pipeline._retrieve_context(query)
            latencies.append((time.time() - start) * 1000)

        p95 = statistics.quantiles(latencies, n=20)[18]
        assert p95 < 200, f"Retrieval p95 latency {p95}ms exceeds 200ms budget"

    def test_concurrent_queries(self, performance_pipeline):
        """Test handling concurrent requests"""
        queries = ["test query"] * 50

        start = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(
                performance_pipeline.generate,
                queries
            ))
        total_time = time.time() - start

        # Should handle 50 queries in under 30 seconds (parallelized)
        assert total_time < 30

        # All queries should succeed
        success_rate = sum(1 for _, meta in results if meta['success']) / len(results)
        assert success_rate > 0.95

    def test_memory_usage(self, performance_pipeline):
        """Test memory doesn't leak under load"""
        import tracemalloc

        tracemalloc.start()

        # Run 100 queries
        for i in range(100):
            performance_pipeline.generate(f"test query {i}")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be under 2GB for this workload
        assert peak < 2 * 1024 * 1024 * 1024

class TestLoadTest:
    """Load testing for production readiness"""

    def test_sustained_load(self, performance_pipeline):
        """Test sustained QPS for 5 minutes"""
        target_qps = 10
        duration_seconds = 300

        results = {
            'success': 0,
            'failure': 0,
            'latencies': []
        }

        start_time = time.time()
        query_count = 0

        while time.time() - start_time < duration_seconds:
            query_start = time.time()

            try:
                answer, meta = performance_pipeline.generate("test query")
                if meta['success']:
                    results['success'] += 1
                else:
                    results['failure'] += 1
            except Exception:
                results['failure'] += 1

            results['latencies'].append((time.time() - query_start) * 1000)
            query_count += 1

            # Rate limit to target QPS
            elapsed = time.time() - start_time
            expected_queries = elapsed * target_qps
            if query_count > expected_queries:
                time.sleep(0.1)

        # Verify success rate
        success_rate = results['success'] / (results['success'] + results['failure'])
        assert success_rate > 0.99, f"Success rate {success_rate} below 99%"

        # Verify latency
        p99 = statistics.quantiles(results['latencies'], n=100)[98]
        assert p99 < 5000, f"p99 latency {p99}ms exceeds 5s"
```

### 20.4 Test Fixtures & Data

```python
# tests/fixtures/test_documents.py

TEST_DOCUMENTS = [
    {
        "content": "To select all faces in Blender, enter Edit Mode, press 'A' to select all, then press '3' for face selection mode. Alternatively, use bpy.ops.mesh.select_all(action='SELECT') in Python.",
        "source": "blender_api",
        "category": "mesh",
        "blender_version": "4.2",
        "is_code": True
    },
    {
        "content": "Rotating objects in Blender: Select object, press 'R' for rotation. Constrain to axis with 'X', 'Y', or 'Z'. Use bpy.ops.transform.rotate() for scripting.",
        "source": "tutorial",
        "category": "object",
        "blender_version": "all",
        "is_code": True
    },
    {
        "content": "Smooth shading can be applied with Right-click > Shade Smooth. In Python: bpy.ops.object.shade_smooth(). For auto-smooth, enable it in Object Data Properties.",
        "source": "best_practice",
        "category": "mesh",
        "blender_version": "4.2",
        "is_code": True
    }
]

TEST_QUERIES = [
    {
        "query": "How do I select all faces in Blender?",
        "expected_category": "mesh",
        "expected_keywords": ["select", "face", "edit mode"]
    },
    {
        "query": "Rotate an object using Python",
        "expected_category": "object",
        "expected_keywords": ["rotate", "bpy", "transform"]
    }
]

# Test embeddings (pre-computed for consistency)
TEST_EMBEDDINGS = {
    "select_faces": [0.1, 0.2, ...],  # 4096-dim vector
    "rotate_object": [0.3, 0.1, ...],
}
```

---

## 21. Session Management

### 21.1 Session State Architecture

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import redis

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)

@dataclass
class SessionState:
    """Full session state"""
    session_id: str
    user_id: Optional[str]
    turns: List[ConversationTurn] = field(default_factory=list)
    context_window: List[str] = field(default_factory=list)  # Retrieved docs
    active_filters: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # Session metadata
    total_queries: int = 0
    total_tokens_used: int = 0
    avg_latency_ms: float = 0.0

class SessionManager:
    """
    Manage conversation sessions with memory

    Features:
    - Sliding window context (last N turns)
    - Context compression for long sessions
    - Automatic session expiry
    - Cross-request state persistence
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        max_turns: int = 10,
        session_ttl_hours: int = 24,
        context_window_size: int = 5
    ):
        self.redis = redis_client
        self.max_turns = max_turns
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.context_window_size = context_window_size

    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create new session"""
        import uuid
        session_id = str(uuid.uuid4())

        state = SessionState(
            session_id=session_id,
            user_id=user_id
        )

        self._save_state(state)
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve session state"""
        key = f"session:{session_id}"
        data = self.redis.get(key)

        if data is None:
            return None

        return self._deserialize_state(data)

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict = None
    ):
        """Add conversation turn to session"""
        state = self.get_session(session_id)
        if state is None:
            raise SessionNotFoundError(session_id)

        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

        state.turns.append(turn)
        state.last_activity = datetime.utcnow()
        state.total_queries += 1 if role == 'user' else 0

        # Trim old turns if exceeds max
        if len(state.turns) > self.max_turns:
            state.turns = state.turns[-self.max_turns:]

        self._save_state(state)

    def get_context_for_query(self, session_id: str) -> str:
        """
        Get conversation context for RAG query

        Returns formatted context string for LLM
        """
        state = self.get_session(session_id)
        if state is None:
            return ""

        # Get last N turns
        recent_turns = state.turns[-self.context_window_size:]

        context_parts = ["=== CONVERSATION HISTORY ==="]
        for turn in recent_turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            context_parts.append(f"\n{role_label}: {turn.content}")

        context_parts.append("\n=== END HISTORY ===")

        return '\n'.join(context_parts)

    def update_filters(self, session_id: str, filters: Dict):
        """Update active filters for session"""
        state = self.get_session(session_id)
        if state:
            state.active_filters.update(filters)
            self._save_state(state)

    def compress_context(self, session_id: str, llm_client):
        """
        Compress old context for long sessions

        Uses LLM to summarize older turns
        """
        state = self.get_session(session_id)
        if state is None or len(state.turns) < self.max_turns:
            return

        # Summarize older turns
        old_turns = state.turns[:-self.context_window_size]
        old_context = '\n'.join([
            f"{t.role}: {t.content}" for t in old_turns
        ])

        summary_prompt = f"""
Summarize this conversation history concisely, preserving key information:

{old_context}

Summary (2-3 sentences):
"""

        summary = llm_client.complete(summary_prompt)

        # Replace old turns with summary turn
        summary_turn = ConversationTurn(
            role='system',
            content=f"[Previous conversation summary: {summary}]",
            timestamp=datetime.utcnow()
        )

        state.turns = [summary_turn] + state.turns[-self.context_window_size:]
        self._save_state(state)

    def _save_state(self, state: SessionState):
        """Save session state to Redis"""
        key = f"session:{state.session_id}"
        data = self._serialize_state(state)
        self.redis.setex(
            key,
            int(self.session_ttl.total_seconds()),
            data
        )

    def _serialize_state(self, state: SessionState) -> str:
        """Serialize session state to JSON"""
        return json.dumps({
            'session_id': state.session_id,
            'user_id': state.user_id,
            'turns': [
                {
                    'role': t.role,
                    'content': t.content,
                    'timestamp': t.timestamp.isoformat(),
                    'metadata': t.metadata
                }
                for t in state.turns
            ],
            'active_filters': state.active_filters,
            'created_at': state.created_at.isoformat(),
            'last_activity': state.last_activity.isoformat(),
            'total_queries': state.total_queries,
            'total_tokens_used': state.total_tokens_used
        })

    def _deserialize_state(self, data: str) -> SessionState:
        """Deserialize JSON to session state"""
        obj = json.loads(data)

        return SessionState(
            session_id=obj['session_id'],
            user_id=obj['user_id'],
            turns=[
                ConversationTurn(
                    role=t['role'],
                    content=t['content'],
                    timestamp=datetime.fromisoformat(t['timestamp']),
                    metadata=t.get('metadata', {})
                )
                for t in obj['turns']
            ],
            active_filters=obj.get('active_filters', {}),
            created_at=datetime.fromisoformat(obj['created_at']),
            last_activity=datetime.fromisoformat(obj['last_activity']),
            total_queries=obj.get('total_queries', 0),
            total_tokens_used=obj.get('total_tokens_used', 0)
        )

class SessionNotFoundError(Exception):
    def __init__(self, session_id: str):
        super().__init__(f"Session not found: {session_id}")
```

### 21.2 Context Window Management

```python
class ContextWindowManager:
    """
    Manage context window for LLM prompts

    Strategies:
    - Sliding window (most recent N turns)
    - Summarization (compress old context)
    - Selective retention (keep important turns)
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        reserve_for_response: int = 1000
    ):
        self.max_tokens = max_tokens
        self.reserve = reserve_for_response
        self.available_tokens = max_tokens - reserve

    def build_context(
        self,
        session: SessionState,
        retrieved_docs: List[str],
        query: str
    ) -> str:
        """
        Build context that fits within token budget

        Priority:
        1. Current query
        2. Most relevant retrieved docs
        3. Recent conversation turns
        4. Session summary (if compressed)
        """
        context_parts = []
        tokens_used = 0

        # 1. Add query (always included)
        query_section = f"Current Query: {query}\n"
        tokens_used += self._estimate_tokens(query_section)
        context_parts.append(query_section)

        # 2. Add retrieved docs (high priority)
        doc_budget = int(self.available_tokens * 0.6)  # 60% for docs
        doc_tokens = 0

        context_parts.append("\n=== RETRIEVED CONTEXT ===\n")
        for i, doc in enumerate(retrieved_docs):
            doc_text = f"[Source {i+1}]\n{doc}\n"
            doc_tok = self._estimate_tokens(doc_text)

            if doc_tokens + doc_tok > doc_budget:
                break

            context_parts.append(doc_text)
            doc_tokens += doc_tok

        tokens_used += doc_tokens

        # 3. Add conversation history
        history_budget = self.available_tokens - tokens_used
        history_tokens = 0

        context_parts.append("\n=== CONVERSATION HISTORY ===\n")
        for turn in reversed(session.turns[-5:]):  # Last 5 turns
            turn_text = f"{turn.role}: {turn.content}\n"
            turn_tok = self._estimate_tokens(turn_text)

            if history_tokens + turn_tok > history_budget:
                break

            context_parts.insert(-1, turn_text)  # Insert before last
            history_tokens += turn_tok

        return ''.join(context_parts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation (4 chars per token)"""
        return len(text) // 4
```

---

## 22. Operational Runbooks

### 22.1 Deployment Checklist

```markdown
# RAG System Deployment Checklist

## Pre-Deployment

- [ ] All tests passing (unit, integration, performance)
- [ ] Database migrations applied to staging
- [ ] Environment variables configured
- [ ] Secrets rotated (API keys, JWT secret)
- [ ] Backup of current production state
- [ ] Rollback plan documented

## Database Setup

- [ ] PostgreSQL 15+ installed
- [ ] pgvector extension enabled
- [ ] Schema migrations applied
- [ ] HNSW indexes created
- [ ] Connection pool configured (min: 10, max: 100)
- [ ] Read replicas configured (if scaling)

## Model Deployment

- [ ] BGE-M3 model downloaded and cached
- [ ] MiniLM cross-encoder downloaded
- [ ] GPU allocation verified (if applicable)
- [ ] Model warm-up queries executed
- [ ] Embedding generation latency verified (<150ms)

## Infrastructure

- [ ] Redis cache running
- [ ] Prometheus/Grafana configured
- [ ] Log aggregation setup (ELK/Loki)
- [ ] Health check endpoints responding
- [ ] Rate limiting configured
- [ ] SSL/TLS certificates valid

## Post-Deployment Verification

- [ ] End-to-end test query successful
- [ ] Metrics appearing in dashboard
- [ ] No errors in logs
- [ ] Latency within SLA (<5s p95)
- [ ] Circuit breakers in closed state
```

### 22.2 Incident Response Procedures

```markdown
# Incident Response Runbook

## Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 | System down | 15 min | Database unreachable, all queries failing |
| P2 | Degraded | 1 hour | High latency, partial failures |
| P3 | Minor | 4 hours | Single component issues, non-critical |
| P4 | Low | Next business day | Performance optimization, minor bugs |

## P1: System Down

### Symptoms
- All queries returning errors
- Health check failing
- Circuit breakers all open

### Immediate Actions (First 15 minutes)
1. **Assess scope**: Check all health endpoints
   ```bash
   curl -s http://localhost:8000/health | jq
   ```

2. **Check database connectivity**
   ```bash
   psql -h localhost -U rag_user -d ragdb -c "SELECT 1"
   ```

3. **Check model services**
   ```bash
   curl -s http://localhost:8001/health  # Embedding service
   ```

4. **Review recent changes**
   - Deployment within last hour?
   - Configuration change?
   - Infrastructure change?

5. **Enable incident channel**
   - Page on-call engineer
   - Create incident ticket
   - Start incident log

### Escalation
- If not resolved in 30 min → Escalate to senior engineer
- If not resolved in 1 hour → Escalate to engineering lead

## P2: High Latency

### Symptoms
- p95 latency > 5 seconds
- Timeout errors increasing
- Users reporting slow responses

### Diagnostic Steps
1. **Identify bottleneck**
   ```sql
   -- Check slow queries
   SELECT query, calls, mean_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

2. **Check resource utilization**
   ```bash
   # CPU/Memory
   top -p $(pgrep -d',' -f "rag_service")

   # GPU (if applicable)
   nvidia-smi
   ```

3. **Review metrics**
   - Embedding latency
   - Reranking latency
   - LLM generation time

### Mitigation
- Scale up resources if CPU/memory bound
- Reduce batch sizes if GPU memory constrained
- Enable degraded mode if LLM API slow
- Clear cache if stale data suspected

## P3: Hallucination Rate Spike

### Symptoms
- Validation failures increasing
- User complaints about incorrect answers
- Hallucination metric > 10%

### Investigation
1. **Sample failed responses**
   ```sql
   SELECT query, answer, validation_score
   FROM retrieval_history
   WHERE validation_score < 0.7
   ORDER BY timestamp DESC
   LIMIT 20;
   ```

2. **Check retrieval quality**
   - Are relevant documents being retrieved?
   - Is context precision dropping?

3. **Review recent document changes**
   - New documents ingested?
   - Document quality issues?

### Resolution
- Increase reranking candidates
- Adjust similarity threshold
- Re-index problematic documents
- Tighten validation rules
```

### 22.3 Scaling Procedures

```markdown
# Scaling Runbook

## Horizontal Scaling

### Add Read Replica (Database)

1. **Create replica**
   ```bash
   # On primary
   pg_basebackup -h primary -D /var/lib/postgresql/replica -U replication -P
   ```

2. **Configure streaming replication**
   ```ini
   # recovery.conf on replica
   standby_mode = 'on'
   primary_conninfo = 'host=primary port=5432 user=replication'
   ```

3. **Update connection pool**
   ```python
   # Add replica to read pool
   READ_REPLICAS = ['replica1:5432', 'replica2:5432']
   ```

4. **Verify replication lag**
   ```sql
   SELECT pg_last_wal_replay_lsn() - pg_last_wal_receive_lsn() AS lag;
   ```

### Scale Embedding Service

1. **Deploy additional instance**
   ```bash
   docker run -d \
     --name embedding-2 \
     -p 8002:8000 \
     -v /models:/models \
     rag-embedding-service
   ```

2. **Add to load balancer**
   ```nginx
   upstream embedding_servers {
       server embedding-1:8000;
       server embedding-2:8000;
   }
   ```

3. **Verify load distribution**
   ```bash
   # Check request distribution
   curl http://load-balancer/stats
   ```

## Vertical Scaling

### Increase Database Resources

1. **Estimate new requirements**
   - Current: 16 cores, 64GB RAM
   - Target: 32 cores, 128GB RAM

2. **Schedule maintenance window**

3. **Resize instance**
   ```bash
   # Cloud provider CLI
   aws rds modify-db-instance \
     --db-instance-identifier ragdb \
     --db-instance-class db.r6g.4xlarge \
     --apply-immediately
   ```

4. **Update PostgreSQL config**
   ```ini
   shared_buffers = 32GB
   effective_cache_size = 96GB
   work_mem = 512MB
   ```

5. **Verify performance improvement**
   - Run benchmark queries
   - Check latency metrics
```

### 22.4 Backup & Recovery

```markdown
# Backup & Recovery Runbook

## Automated Backups

### Daily Database Backup
```bash
#!/bin/bash
# /opt/scripts/backup_db.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/postgresql"

# Full backup with pg_dump
pg_dump -h localhost -U rag_user -d ragdb \
  -F c -f "${BACKUP_DIR}/ragdb_${DATE}.dump"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/ragdb_${DATE}.dump" \
  s3://rag-backups/postgresql/

# Retain last 30 days locally
find ${BACKUP_DIR} -name "*.dump" -mtime +30 -delete
```

### Embedding Backup (Weekly)
```bash
#!/bin/bash
# Backup pre-computed embeddings

pg_dump -h localhost -U rag_user -d ragdb \
  -t documents -c "SELECT id, embedding FROM documents" \
  -F c -f /backups/embeddings_$(date +%Y%m%d).dump
```

## Recovery Procedures

### Full Database Recovery
```bash
# 1. Stop application
systemctl stop rag-service

# 2. Drop and recreate database
psql -U postgres -c "DROP DATABASE ragdb;"
psql -U postgres -c "CREATE DATABASE ragdb;"

# 3. Restore from backup
pg_restore -h localhost -U rag_user -d ragdb \
  /backups/postgresql/ragdb_YYYYMMDD.dump

# 4. Verify indexes
psql -U rag_user -d ragdb -c "\di"

# 5. Restart application
systemctl start rag-service
```

### Point-in-Time Recovery
```bash
# Recover to specific timestamp
pg_restore -h localhost -U rag_user -d ragdb \
  --target-time="2024-01-15 14:30:00" \
  /backups/postgresql/ragdb_20240115.dump
```

### Embedding Re-generation
```bash
# If embeddings corrupted, re-generate from documents
python scripts/regenerate_embeddings.py \
  --batch-size 100 \
  --checkpoint /backups/embeddings_checkpoint.json
```
```

---

## 23. Fully Agentic RAG Architecture

This section describes the upgraded **Fully Agentic Architecture** that transforms the RAG system from a simple retrieve-and-generate pipeline into a sophisticated multi-agent system capable of handling complex, multi-step tasks.

### 23.1 Why Fully Agentic?

| Query Type | Simple RAG | Agentic RAG |
|------------|------------|-------------|
| "What is bpy.ops.mesh?" | ✅ Works | Overkill |
| "Select all faces in edit mode" | ✅ Works | Overkill |
| "Create character, rig, animate walk" | ❌ Fails | ✅ **Excels** |
| "Debug my script that crashes" | ❌ Fails | ✅ **Excels** |
| "Compare UV unwrapping methods" | ⚠️ Partial | ✅ **Excels** |
| Follow-up questions with context | ⚠️ Limited | ✅ **Excels** |

**Blender 3D Game AI Assistant requires agentic capabilities** because:
1. Users ask multi-step questions (model → rig → animate)
2. Tasks require tool orchestration (code generation + validation + execution)
3. Complex queries need decomposition and planning
4. Iterative refinement based on validation feedback

### 23.2 Seven-Layer Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                         │
│                   "Create a rigged character with walk cycle"                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: ORCHESTRATOR AGENT                                                 │
│  ═══════════════════════════════════════════════════════════════════════════│
│  Model: GPT-5.1 (reasoning_effort="medium")                                  │
│                                                                              │
│  Responsibilities:                                                           │
│  • Parse complex multi-step queries                                          │
│  • Create execution plan with dependencies                                   │
│  • Dispatch to specialized agents                                            │
│  • Handle failures and re-planning                                           │
│  • Maintain conversation state                                               │
│                                                                              │
│  LLM Requirements: ⭐⭐⭐⭐⭐ Reasoning, ⭐⭐⭐⭐ Tool Use, ⭐⭐⭐ Speed           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: QUERY ANALYSIS AGENT                                               │
│  ═══════════════════════════════════════════════════════════════════════════│
│  Model: GPT-5-mini (reasoning_effort="low")                                  │
│                                                                              │
│  Sub-Tasks:                                                                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                │
│  │ Intent Extract  │ │ Entity Extract  │ │ Query Decompose │                │
│  │ "create_rig"    │ │ [character,     │ │ 1. Model char   │                │
│  │ "animate_walk"  │ │  armature,      │ │ 2. Add armature │                │
│  └─────────────────┘ │  walk_cycle]    │ │ 3. Bind weights │                │
│                      └─────────────────┘ │ 4. Animate walk │                │
│  ┌─────────────────┐ ┌─────────────────┐ └─────────────────┘                │
│  │ Query Expansion │ │ Filter Detect   │                                    │
│  │ • Synonyms      │ │ • version: 4.2  │                                    │
│  │ • Related terms │ │ • category:     │                                    │
│  └─────────────────┘ │   rigging       │                                    │
│                      └─────────────────┘                                    │
│  LLM Requirements: ⭐⭐⭐⭐ Speed, ⭐⭐⭐ Reasoning, ⭐⭐⭐ Structured Output     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: RETRIEVAL AGENT (Hybrid Search)                                    │
│  ═══════════════════════════════════════════════════════════════════════════│
│  NO LLM - Pure Algorithmic (Fast, Deterministic)                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         PARALLEL RETRIEVAL                             │ │
│  │  ┌─────────────────────┐         ┌─────────────────────┐              │ │
│  │  │   DENSE SEARCH      │         │   SPARSE SEARCH     │              │ │
│  │  │   BGE-M3 Embeddings │         │   BM25 Algorithm    │              │ │
│  │  │   4096 dimensions   │         │   TF-IDF weighting  │              │ │
│  │  │   HNSW Index        │         │   GIN Index         │              │ │
│  │  │   ~50ms latency     │         │   ~20ms latency     │              │ │
│  │  └──────────┬──────────┘         └──────────┬──────────┘              │ │
│  │             └───────────┬───────────────────┘                          │ │
│  │                         ▼                                              │ │
│  │              ┌─────────────────────┐                                   │ │
│  │              │   RRF FUSION (k=60) │                                   │ │
│  │              └─────────────────────┘                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  Requirements: Low latency (<100ms), High recall, Deterministic              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: RERANKING AGENT                                                    │
│  ═══════════════════════════════════════════════════════════════════════════│
│  Model: BAAI/bge-reranker-v2-m3 (Cross-Encoder, self-hosted)                │
│  Fallback: GPT-5-nano for LLM-based reranking                               │
│                                                                              │
│  Process:                                                                    │
│  • Score each (query, document) pair                                         │
│  • Re-order by relevance score                                               │
│  • Return top-K most relevant                                                │
│                                                                              │
│  Latency: ~150ms for 30 documents                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: CONTEXT ASSEMBLY AGENT                                             │
│  ═══════════════════════════════════════════════════════════════════════════│
│  Model: GPT-5-mini (reasoning_effort="none")                                 │
│                                                                              │
│  Tasks:                                                                      │
│  • Relevance filtering (remove score < 0.5)                                  │
│  • Context compression (extract key passages)                                │
│  • Context ordering (by sub-query relevance)                                 │
│  • Metadata enrichment (add citations, versions)                             │
│                                                                              │
│  LLM Requirements: ⭐⭐⭐⭐ Speed, ⭐⭐⭐ Summarization                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: GENERATION AGENT (The Brain)                                       │
│  ═══════════════════════════════════════════════════════════════════════════│
│  Model: GPT-5.1 (reasoning_effort="medium" or "high" for complex)           │
│                                                                              │
│  This is the CORE layer - generates the final answer                        │
│                                                                              │
│  Input:                                                                      │
│  • Original query + decomposed sub-queries                                   │
│  • Assembled context (retrieved + compressed docs)                           │
│  • Conversation history (session memory)                                     │
│  • User preferences & constraints                                            │
│                                                                              │
│  Processing:                                                                 │
│  • Chain-of-thought reasoning                                                │
│  • Multi-step code generation (if needed)                                    │
│  • Citation tracking (which doc supports each claim)                         │
│  • Confidence scoring per statement                                          │
│                                                                              │
│  Output:                                                                     │
│  • Answer with inline citations [1], [2]                                     │
│  • Code blocks (if applicable)                                               │
│  • Confidence score (0-1)                                                    │
│  • Follow-up suggestions                                                     │
│                                                                              │
│  LLM Requirements: ⭐⭐⭐⭐⭐ Reasoning, ⭐⭐⭐⭐⭐ Coding, ⭐⭐⭐⭐ Instruction     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7: VALIDATION AGENT (Quality Gate)                                    │
│  ═══════════════════════════════════════════════════════════════════════════│
│  Model: GPT-5-mini (reasoning_effort="low")                                  │
│                                                                              │
│  Validation Checks:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. FAITHFULNESS CHECK (Hallucination Detection)                     │    │
│  │    • Is every claim supported by retrieved context?                 │    │
│  │    • Score: 0.0 - 1.0                                               │    │
│  │                                                                     │    │
│  │ 2. RELEVANCY CHECK                                                  │    │
│  │    • Does the answer address the query?                             │    │
│  │    • Score: 0.0 - 1.0                                               │    │
│  │                                                                     │    │
│  │ 3. COMPLETENESS CHECK                                               │    │
│  │    • Are there gaps in the answer?                                  │    │
│  │    • Score: 0.0 - 1.0                                               │    │
│  │                                                                     │    │
│  │ 4. CODE VALIDATION (if applicable)                                  │    │
│  │    • Syntax check, API correctness, version compatibility           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Decision Logic:                                                             │
│  • Score >= 0.8 → PASS → Return to user                                      │
│  • Score < 0.8  → RETRY → Back to Layer 5/6 (max 3 attempts)                │
│  • 3 failures   → RETURN with warnings                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 23.3 OpenAI GPT-5 Model Family Configuration

#### December 2025 Model Lineup

| Model | Input $/1M | Output $/1M | Context | Best For |
|-------|------------|-------------|---------|----------|
| **GPT-5.1** | $1.25 | $10.00 | 400K | Orchestration, Generation |
| **GPT-5-mini** | $0.25 | $2.00 | 400K | Analysis, Validation |
| **GPT-5-nano** | $0.05 | $0.40 | 128K | High-volume, simple tasks |
| **GPT-5.1-Codex** | $1.50 | $12.00 | 400K | Code-heavy workflows |

**Sources**: [OpenAI GPT-5.1 Announcement](https://openai.com/index/gpt-5-1/), [GPT-5 API Docs](https://platform.openai.com/docs/models/gpt-5)

#### GPT-5.1 API Configuration

```python
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# GPT-5.1 with reasoning effort control
class GPT51Config:
    """
    GPT-5.1 Configuration for Agentic RAG

    reasoning_effort options:
    - "none": No reasoning tokens, fastest (default in 5.1)
    - "minimal": Very few reasoning tokens, fast
    - "low": Some reasoning, balanced
    - "medium": Moderate reasoning, recommended for complex tasks
    - "high": Deep reasoning, slowest but most accurate
    """

    # Layer-specific configurations
    ORCHESTRATOR = {
        "model": "gpt-5.1",
        "reasoning_effort": "medium",
        "max_tokens": 4096,
        "temperature": 0.3,
    }

    QUERY_ANALYZER = {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "max_tokens": 1024,
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    CONTEXT_ASSEMBLER = {
        "model": "gpt-5-mini",
        "reasoning_effort": "none",
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    GENERATOR = {
        "model": "gpt-5.1",
        "reasoning_effort": "medium",  # or "high" for complex queries
        "max_tokens": 8192,
        "temperature": 0.4,
    }

    VALIDATOR = {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "max_tokens": 1024,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
```

#### GPT-5.1 Tools Configuration

```python
# GPT-5.1 native tools for agentic workflows
TOOLS_CONFIG = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": "Search the knowledge base for relevant Blender documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 10},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "blender_version": {"type": "string"},
                            "category": {"type": "string"},
                            "doc_type": {"type": "string"}
                        }
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_blender_code",
            "description": "Validate Python code for Blender API compatibility",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "blender_version": {"type": "string", "default": "4.2"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "decompose_task",
            "description": "Break down a complex task into sub-tasks",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "max_steps": {"type": "integer", "default": 5}
                },
                "required": ["task"]
            }
        }
    }
]

# Using GPT-5.1's native apply_patch tool for code editing
CODING_TOOLS = [
    {"type": "apply_patch"},  # Native GPT-5.1 tool for code edits
    {"type": "shell"}         # Native GPT-5.1 tool for command execution
]
```

### 23.4 Agent Implementation

#### Orchestrator Agent

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import json

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SubTask:
    id: str
    description: str
    dependencies: List[str]
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class ExecutionPlan:
    query: str
    intent: str
    sub_tasks: List[SubTask]
    current_task_index: int = 0

class OrchestratorAgent:
    """
    Layer 1: Orchestrates the entire agentic RAG pipeline

    Responsibilities:
    - Parse user query
    - Create execution plan
    - Coordinate specialized agents
    - Handle failures and re-planning
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.config = GPT51Config.ORCHESTRATOR

    async def create_execution_plan(
        self,
        query: str,
        session_context: Optional[str] = None
    ) -> ExecutionPlan:
        """
        Analyze query and create execution plan with sub-tasks
        """
        system_prompt = """You are an orchestrator agent for a Blender 3D AI assistant.

        Analyze the user's query and create an execution plan.

        For simple queries (single concept, direct question):
        - Create a single retrieval + generation task

        For complex queries (multi-step, requires decomposition):
        - Break into ordered sub-tasks with dependencies
        - Each sub-task should be independently retrievable

        Return JSON:
        {
            "intent": "simple" | "complex",
            "analysis": "brief analysis of the query",
            "sub_tasks": [
                {
                    "id": "task_1",
                    "description": "what to retrieve/do",
                    "dependencies": [],
                    "type": "retrieval" | "generation" | "validation"
                }
            ]
        }
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nSession context: {session_context or 'None'}"}
        ]

        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            reasoning_effort=self.config["reasoning_effort"],
            response_format={"type": "json_object"},
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"]
        )

        plan_data = json.loads(response.choices[0].message.content)

        sub_tasks = [
            SubTask(
                id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
                status=TaskStatus.PENDING
            )
            for t in plan_data["sub_tasks"]
        ]

        return ExecutionPlan(
            query=query,
            intent=plan_data["intent"],
            sub_tasks=sub_tasks
        )

    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Execute the plan by coordinating specialized agents
        """
        results = {}

        for task in plan.sub_tasks:
            # Check dependencies
            deps_met = all(
                results.get(dep, {}).get("status") == "completed"
                for dep in task.dependencies
            )

            if not deps_met:
                task.status = TaskStatus.FAILED
                task.error = "Dependencies not met"
                continue

            task.status = TaskStatus.IN_PROGRESS

            try:
                # Route to appropriate agent
                if task.type == "retrieval":
                    result = await self.retrieval_agent.execute(task)
                elif task.type == "generation":
                    result = await self.generation_agent.execute(task, results)
                elif task.type == "validation":
                    result = await self.validation_agent.execute(task, results)

                task.status = TaskStatus.COMPLETED
                task.result = result
                results[task.id] = {"status": "completed", "result": result}

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                results[task.id] = {"status": "failed", "error": str(e)}

                # Re-plan if critical task failed
                if self._is_critical_task(task):
                    return await self._handle_failure(plan, task, e)

        return self._compile_final_response(plan, results)
```

#### Query Analysis Agent

```python
class QueryAnalysisAgent:
    """
    Layer 2: Analyzes and decomposes user queries
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.config = GPT51Config.QUERY_ANALYZER

    async def analyze(self, query: str) -> Dict[str, Any]:
        """
        Extract intent, entities, and generate query variations
        """
        system_prompt = """Analyze this Blender-related query and extract:

        1. intent: The primary action/goal (e.g., "select_faces", "create_material", "animate_character")
        2. entities: Key concepts mentioned (API names, tools, objects, operations)
        3. blender_version: If specified, extract version (default: null)
        4. category: mesh, animation, materials, scripting, modeling, rigging, rendering
        5. query_variations: 3 alternative phrasings for better retrieval
        6. is_multi_step: Does this require multiple operations?
        7. sub_queries: If multi-step, break into individual queries

        Return valid JSON only."""

        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            reasoning_effort=self.config["reasoning_effort"],
            response_format={"type": "json_object"},
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"]
        )

        return json.loads(response.choices[0].message.content)
```

#### Generation Agent

```python
class GenerationAgent:
    """
    Layer 6: Generates final answers with citations
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.config = GPT51Config.GENERATOR

    async def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        sub_results: Dict[str, Any],
        session_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate grounded answer with citations
        """
        # Format context with citation markers
        formatted_context = self._format_context_with_citations(context)

        system_prompt = f"""You are an expert Blender 3D assistant. Generate accurate, helpful answers based on the provided context.

        RULES:
        1. Only use information from the provided context
        2. Add citations [1], [2], etc. for each claim
        3. If generating code, ensure it's valid for Blender Python API
        4. If the context doesn't contain enough information, say so
        5. For multi-step tasks, provide clear numbered instructions

        CONTEXT:
        {formatted_context}

        Previous conversation: {json.dumps(session_history[-5:] if session_history else [])}
        """

        # Determine reasoning effort based on query complexity
        reasoning = "high" if sub_results.get("is_complex") else "medium"

        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            reasoning_effort=reasoning,
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"]
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "citations": self._extract_citations(answer, context),
            "confidence": self._calculate_confidence(response),
            "reasoning_tokens": response.usage.reasoning_tokens if hasattr(response.usage, 'reasoning_tokens') else 0
        }

    def _format_context_with_citations(self, context: List[Dict]) -> str:
        """Format documents with citation markers"""
        formatted = []
        for i, doc in enumerate(context, 1):
            formatted.append(f"[{i}] Source: {doc.get('source', 'Unknown')}")
            formatted.append(f"    Version: {doc.get('version', 'N/A')}")
            formatted.append(f"    Content: {doc['content'][:1000]}")
            formatted.append("")
        return "\n".join(formatted)
```

#### Validation Agent

```python
class ValidationAgent:
    """
    Layer 7: Validates generated answers for quality
    """

    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.config = GPT51Config.VALIDATOR
        self.max_retries = 3

    async def validate(
        self,
        query: str,
        answer: str,
        context: List[Dict[str, Any]],
        attempt: int = 1
    ) -> Dict[str, Any]:
        """
        Validate answer for faithfulness, relevancy, and completeness
        """
        system_prompt = """You are a validation agent. Evaluate the answer quality.

        Score each dimension 0.0 to 1.0:

        1. faithfulness: Is every claim supported by the context? (check for hallucinations)
        2. relevancy: Does the answer address the user's query?
        3. completeness: Are all parts of the query answered?
        4. code_validity: If code is present, is it syntactically correct and uses valid Blender API?

        Return JSON:
        {
            "faithfulness": 0.0-1.0,
            "relevancy": 0.0-1.0,
            "completeness": 0.0-1.0,
            "code_validity": 0.0-1.0 or null if no code,
            "composite_score": weighted average,
            "issues": ["list of specific issues found"],
            "pass": true if composite >= 0.8
        }
        """

        validation_input = f"""
        QUERY: {query}

        ANSWER: {answer}

        CONTEXT DOCUMENTS:
        {json.dumps([{"content": d["content"][:500], "source": d.get("source")} for d in context])}
        """

        response = await self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": validation_input}
            ],
            reasoning_effort=self.config["reasoning_effort"],
            response_format={"type": "json_object"},
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"]
        )

        validation_result = json.loads(response.choices[0].message.content)
        validation_result["attempt"] = attempt

        return validation_result

    async def validate_with_retry(
        self,
        query: str,
        generation_agent: GenerationAgent,
        context: List[Dict],
        session_history: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Validate and retry generation if needed
        """
        for attempt in range(1, self.max_retries + 1):
            # Generate answer
            generation_result = await generation_agent.generate(
                query, context, {}, session_history
            )

            # Validate
            validation = await self.validate(
                query,
                generation_result["answer"],
                context,
                attempt
            )

            if validation["pass"]:
                return {
                    "answer": generation_result["answer"],
                    "citations": generation_result["citations"],
                    "validation": validation,
                    "attempts": attempt,
                    "status": "success"
                }

            # Add feedback for next attempt
            if attempt < self.max_retries:
                context.append({
                    "content": f"VALIDATION FEEDBACK (attempt {attempt}): {', '.join(validation['issues'])}",
                    "source": "validation_feedback"
                })

        # Return best effort after max retries
        return {
            "answer": generation_result["answer"],
            "citations": generation_result["citations"],
            "validation": validation,
            "attempts": self.max_retries,
            "status": "warning",
            "warning": "Answer may have quality issues. Please verify."
        }
```

### 23.5 Cost Estimation per Query

| Layer | Model | Avg Tokens | Cost/Query |
|-------|-------|------------|------------|
| Orchestrator | GPT-5.1 | ~2,000 | ~$0.025 |
| Query Analysis | GPT-5-mini | ~500 | ~$0.001 |
| Retrieval | None | - | $0 |
| Reranking | BGE-reranker | - | $0 (self-hosted) |
| Context Assembly | GPT-5-mini | ~1,000 | ~$0.002 |
| Generation | GPT-5.1 | ~4,000 | ~$0.045 |
| Validation | GPT-5-mini | ~800 | ~$0.002 |

**Total: ~$0.075 per simple query, ~$0.15-0.20 for complex queries with retries**

### 23.6 Query Routing Logic

```python
class QueryRouter:
    """
    Route queries to appropriate pipeline based on complexity
    """

    SIMPLE_INTENTS = [
        "definition", "lookup", "single_api", "syntax"
    ]

    COMPLEX_INTENTS = [
        "multi_step", "comparison", "tutorial", "workflow",
        "debug", "optimization", "create_full"
    ]

    async def route(self, query: str, analysis: Dict) -> str:
        """
        Determine pipeline: 'simple' or 'agentic'

        Simple: Direct retrieval + generation (faster, cheaper)
        Agentic: Full 7-layer pipeline (complex queries)
        """
        intent = analysis.get("intent", "")
        is_multi_step = analysis.get("is_multi_step", False)
        entity_count = len(analysis.get("entities", []))

        # Heuristics for routing
        if is_multi_step:
            return "agentic"

        if any(ci in intent.lower() for ci in self.COMPLEX_INTENTS):
            return "agentic"

        if entity_count > 3:
            return "agentic"

        if "?" in query and query.count(",") > 2:
            return "agentic"

        return "simple"
```

---

## Conclusion

This Advanced RAG System combines state-of-the-art retrieval techniques with a **Fully Agentic Architecture** powered by OpenAI's GPT-5.1 model family. The 7-layer design enables:

**Key Takeaways:**
- ✅ **Fully Agentic Architecture** with specialized agents per task
- ✅ **OpenAI GPT-5.1** for orchestration and generation with reasoning control
- ✅ **GPT-5-mini** for fast analysis and validation tasks
- ✅ **Hybrid retrieval** (dense + sparse + RRF) for optimal recall and precision
- ✅ **Cross-encoder reranking** for context quality
- ✅ **Self-correcting validation loops** with up to 3 retry attempts
- ✅ **Query routing** to optimize cost (simple vs agentic pipeline)
- ✅ **Local PostgreSQL deployment** for privacy and control
- ✅ **Continuous RAGAS evaluation** for quality assurance
- ✅ **Domain-specific validation** for Blender API correctness

---

**Document Version:** 3.0
**Last Updated:** December 11, 2025
**Classification:** Technical Architecture Reference
**Status:** Production-Ready Specification

### Version History
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 4, 2025 | Initial architecture specification |
| 2.0 | Dec 11, 2025 | Added production sections: Error Handling (17), Security (18), Observability (19), Testing (20), Session Management (21), Operational Runbooks (22) |
| 3.0 | Dec 11, 2025 | Major upgrade: Fully Agentic Architecture with OpenAI GPT-5.1, 7-layer agent design |
