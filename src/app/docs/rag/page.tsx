"use client";

import { DocPageLayout } from "@/components/docs/DocPageLayout";

const ragIcon = (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125" />
  </svg>
);

const phases = [
  {
    name: "Phase 1: Foundation",
    duration: "Weeks 1-4",
    tasks: [
      "Set up PostgreSQL with pgvector extension",
      "Design document schema with metadata",
      "Implement chunking strategies (semantic, recursive)",
      "Create document ingestion pipeline",
      "Set up BGE-M3 embedding service",
    ],
    milestone: "Documents stored with vector embeddings",
  },
  {
    name: "Phase 2: Retrieval",
    duration: "Weeks 5-8",
    tasks: [
      "Implement HNSW index for vector search",
      "Build BM25 sparse retrieval with Elasticsearch",
      "Create hybrid retrieval combining dense + sparse",
      "Implement RRF (Reciprocal Rank Fusion)",
      "Add query expansion and rewriting",
    ],
    milestone: "Hybrid retrieval returns relevant documents",
  },
  {
    name: "Phase 3: Reranking & Quality",
    duration: "Weeks 9-12",
    tasks: [
      "Integrate MiniLM cross-encoder reranker",
      "Implement document deduplication",
      "Add relevance score calibration",
      "Create answer validation loop",
      "Implement citation extraction",
    ],
    milestone: "Reranking improves precision@10 by 20%",
  },
  {
    name: "Phase 4: Integration",
    duration: "Weeks 13-16",
    tasks: [
      "Build agentic query transformation",
      "Implement multi-hop reasoning",
      "Set up RAGAS evaluation framework",
      "Create monitoring dashboard",
      "Performance optimization and caching",
    ],
    milestone: "Production-ready RAG pipeline",
  },
];

const sections = [
  {
    title: "Overview",
    content: `The Advanced RAG (Retrieval-Augmented Generation) system provides intelligent knowledge retrieval for the AI game assistant. It combines dense vector search with sparse BM25 retrieval, cross-encoder reranking, and agentic query processing.

Key capabilities:
- Hybrid retrieval (dense + sparse) for comprehensive coverage
- Cross-encoder reranking for precision optimization
- Multi-hop reasoning for complex queries
- Real-time document updates
- RAGAS-based quality evaluation`,
  },
  {
    title: "Embedding Architecture",
    content: `The system uses BGE-M3 (BAAI General Embedding) for dense vector representations:

1. Model Specifications:
   - Dimension: 4,096 (configurable)
   - Max sequence: 8,192 tokens
   - Multi-lingual support

2. Vector Storage:
   - PostgreSQL with pgvector extension
   - HNSW index for approximate nearest neighbor
   - Parameters: M=16, ef_construction=200

3. Chunking Strategy:
   - Semantic chunking by topic/section
   - Recursive splitting for long documents
   - Overlap: 10-15% between chunks
   - Target chunk size: 512 tokens`,
    code: `# BGE-M3 Embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    batch_size=32
)`,
  },
  {
    title: "Hybrid Retrieval",
    content: `The retrieval system combines multiple strategies for optimal recall:

1. Dense Retrieval (Vector Search):
   - Cosine similarity with HNSW index
   - Top-k: 50 candidates
   - Threshold: 0.7 minimum similarity

2. Sparse Retrieval (BM25):
   - Elasticsearch backend
   - TF-IDF weighted scoring
   - Field boosting for titles/headers

3. Reciprocal Rank Fusion (RRF):
   - Combines rankings from both methods
   - k=60 constant for score normalization
   - Deduplication by document ID`,
    code: `# RRF Score Calculation
def rrf_score(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])`,
  },
  {
    title: "Cross-Encoder Reranking",
    content: `MiniLM cross-encoder provides fine-grained relevance scoring:

1. Model: ms-marco-MiniLM-L-6-v2
   - Input: (query, document) pairs
   - Output: relevance score [0, 1]

2. Reranking Process:
   - Score all RRF candidates
   - Sort by cross-encoder score
   - Apply score threshold (0.5)
   - Return top-k (default: 5)

3. Performance:
   - ~10ms per candidate
   - Batch processing for efficiency
   - GPU acceleration available`,
  },
  {
    title: "Agentic Query Processing",
    content: `The system uses agentic loops for complex query handling:

1. Query Analysis:
   - Intent classification
   - Entity extraction
   - Complexity assessment

2. Query Transformation:
   - Decomposition for multi-hop queries
   - Expansion with synonyms/related terms
   - Reformulation for better retrieval

3. Validation Loop:
   - Check answer completeness
   - Verify source citations
   - Retry with refined query if needed`,
  },
  {
    title: "RAGAS Evaluation",
    content: `Quality metrics using the RAGAS framework:

1. Faithfulness: Does the answer use only retrieved context?
2. Answer Relevancy: Does the answer address the query?
3. Context Precision: Are retrieved documents relevant?
4. Context Recall: Are all necessary documents retrieved?

Target scores:
- Faithfulness: > 0.9
- Answer Relevancy: > 0.85
- Context Precision: > 0.8
- Context Recall: > 0.75`,
  },
];

const apiReference = [
  {
    endpoint: "/api/rag/query",
    method: "POST",
    description: "Execute a RAG query and return answer with citations",
    parameters: [
      { name: "query", type: "string", description: "The user's question" },
      { name: "top_k", type: "int", description: "Number of documents to retrieve (default: 5)" },
      { name: "rerank", type: "bool", description: "Enable cross-encoder reranking (default: true)" },
    ],
  },
  {
    endpoint: "/api/rag/ingest",
    method: "POST",
    description: "Ingest documents into the knowledge base",
    parameters: [
      { name: "documents", type: "array", description: "Array of document objects" },
      { name: "chunk_size", type: "int", description: "Target chunk size in tokens (default: 512)" },
    ],
  },
  {
    endpoint: "/api/rag/search",
    method: "POST",
    description: "Search for documents without generating an answer",
    parameters: [
      { name: "query", type: "string", description: "Search query" },
      { name: "method", type: "string", description: "dense, sparse, or hybrid (default: hybrid)" },
    ],
  },
  {
    endpoint: "/api/rag/evaluate",
    method: "POST",
    description: "Evaluate RAG quality using RAGAS metrics",
    parameters: [
      { name: "test_set", type: "array", description: "Array of query-answer-context tuples" },
    ],
  },
];

export default function RAGDocPage() {
  return (
    <DocPageLayout
      title="Advanced RAG"
      subtitle="Retrieval-Augmented Generation System"
      description="A production-grade RAG pipeline combining BGE-M3 dense embeddings, BM25 sparse retrieval, HNSW indexing, cross-encoder reranking, and RAGAS evaluation. Designed for accurate, citation-backed knowledge retrieval."
      gradient="from-emerald-500 to-teal-600"
      accentColor="bg-emerald-500"
      icon={ragIcon}
      presentationLink="/rag"
      technologies={["PostgreSQL", "pgvector", "BGE-M3", "HNSW", "BM25", "Elasticsearch", "MiniLM", "RAGAS"]}
      phases={phases}
      sections={sections}
      apiReference={apiReference}
    />
  );
}
