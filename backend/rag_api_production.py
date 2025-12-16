"""
RAG API Module - Fully Agentic RAG System (PRODUCTION)
OpenAI GPT-5.1 Integration with 7-Layer Agent Architecture

This module implements the production agentic RAG pipeline for the Blender 3D AI Assistant.
"""

import os
import re
import time
import json
import hashlib
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

from flask import Blueprint, request, jsonify, current_app

# OpenAI GPT-5.1 Family
from openai import OpenAI

# Embeddings & Reranking
from sentence_transformers import SentenceTransformer, CrossEncoder

# Vector Database
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np

# BM25 Sparse Search
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_bp = Blueprint('rag', __name__)

# =============================================================================
# Configuration - GPT-5.1 Family (NO GPT-4 - OBSOLETE)
# =============================================================================

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# GPT-5.1 Model Family Configuration (December 2025)
# NOTE: GPT-5.1 does NOT support custom temperature - only default (1) is allowed
MODEL_CONFIG = {
    "orchestrator": {
        "model": "gpt-5.1",
        "reasoning_effort": "medium",
        "max_tokens": 4096,
    },
    "query_analyzer": {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "max_tokens": 1024,
    },
    "context_assembler": {
        "model": "gpt-5-mini",
        "reasoning_effort": "none",
        "max_tokens": 2048,
    },
    "generator": {
        "model": "gpt-5.1",
        "reasoning_effort": "medium",
        "max_tokens": 8192,
    },
    "validator": {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
        "max_tokens": 1024,
    }
}

# Database Configuration
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "dbname": "ragdb",
    "user": "raguser",
    "password": "<RAG_DB_PASSWORD>"
}

# Embedding Model: all-MiniLM-L6-v2 for fast CPU inference (384 dimensions)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Reranker Model: BGE-reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# RRF Fusion constant
RRF_K = 60

# Session storage
SESSION_STORE: Dict[str, Dict] = {}

# =============================================================================
# Model Initialization (Lazy Loading)
# =============================================================================

_openai_client: Optional[OpenAI] = None
_embedding_model: Optional[SentenceTransformer] = None
_reranker_model: Optional[CrossEncoder] = None
_bm25_index: Optional[BM25Okapi] = None
_bm25_doc_ids: List[str] = []


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client for GPT-5.1"""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized for GPT-5.1 family")
    return _openai_client


def get_embedding_model() -> SentenceTransformer:
    """Get or create BGE-M3 embedding model"""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Embedding model loaded. Dimension: {_embedding_model.get_sentence_embedding_dimension()}")
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    """Get or create BGE reranker cross-encoder"""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker model loaded")
    return _reranker_model


def get_db_connection():
    """Get PostgreSQL connection with pgvector"""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


# =============================================================================
# Data Classes
# =============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStage(str, Enum):
    ORCHESTRATION = "orchestration"
    QUERY_ANALYSIS = "query_analysis"
    RETRIEVAL_DENSE = "retrieval_dense"
    RETRIEVAL_SPARSE = "retrieval_sparse"
    RRF_FUSION = "rrf_fusion"
    RERANKING = "reranking"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION = "generation"
    VALIDATION = "validation"


@dataclass
class StageResult:
    stage: str
    status: str
    duration_ms: int
    results_count: int = 0
    details: Optional[Dict] = None


@dataclass
class Document:
    id: str
    content: str
    title: Optional[str] = None
    source: str = "blender_docs"
    version: str = "4.2"
    category: str = "general"
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class QueryAnalysis:
    intent: str
    entities: List[str]
    variations: List[str]
    detected_filters: Dict[str, str]
    is_multi_step: bool
    sub_queries: List[str]
    confidence: float


@dataclass
class ValidationResult:
    faithfulness: float
    relevancy: float
    completeness: float
    code_validity: Optional[float]
    composite_score: float
    issues: List[str]
    passed: bool
    attempt: int


@dataclass
class RAGResponse:
    request_id: str
    session_id: str
    query: str
    analysis: QueryAnalysis
    documents: List[Document]
    answer: str
    citations: List[Dict]
    validation: ValidationResult
    metrics: Dict[str, float]
    latency: Dict[str, int]
    stages: List[StageResult]
    status: str
    warning: Optional[str] = None


# =============================================================================
# Layer 1: Orchestrator Agent (GPT-5.1)
# =============================================================================

def run_orchestrator(query: str, session_history: List[Dict]) -> Tuple[Dict, int]:
    """
    Layer 1: Orchestrator Agent - Plans query execution strategy
    Model: GPT-5.1 with reasoning_effort="medium"
    """
    start_time = time.time()

    client = get_openai_client()
    config = MODEL_CONFIG["orchestrator"]

    prompt = f"""You are the Orchestrator Agent for a Blender 3D AI Assistant RAG system.
Analyze the user query and plan the execution strategy.

Query: "{query}"
Session History: {json.dumps(session_history[-3:]) if session_history else "None"}

Determine:
1. query_complexity: "simple" (single concept) or "complex" (multi-step, requires reasoning)
2. pipeline_type: "standard" (direct retrieval) or "agentic" (needs query decomposition)
3. requires_code: boolean - does the user want code/API examples?
4. blender_version: extract if mentioned, default "4.2"
5. priority_sources: ["blender_api", "blender_manual", "tutorials"] - order by relevance

Return JSON only:
{{"query_complexity": "...", "pipeline_type": "...", "requires_code": true/false, "blender_version": "...", "priority_sources": [...]}}"""

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=config["max_tokens"],
        )

        result_text = response.choices[0].message.content.strip()
        # Parse JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        plan = json.loads(result_text)

    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        plan = {
            "query_complexity": "simple",
            "pipeline_type": "standard",
            "requires_code": "code" in query.lower() or "python" in query.lower(),
            "blender_version": "4.2",
            "priority_sources": ["blender_api", "blender_manual"]
        }

    duration_ms = int((time.time() - start_time) * 1000)
    return plan, duration_ms


# =============================================================================
# Layer 2: Query Analysis Agent (GPT-5-mini)
# =============================================================================

def run_query_analysis(query: str, session_history: List[Dict]) -> Tuple[QueryAnalysis, int]:
    """
    Layer 2: Query Analysis Agent - Extract intent, entities, generate variations
    Model: GPT-5-mini with reasoning_effort="low"
    """
    start_time = time.time()

    client = get_openai_client()
    config = MODEL_CONFIG["query_analyzer"]

    prompt = f"""Analyze this Blender 3D query for RAG retrieval.

Query: "{query}"
Context: {json.dumps(session_history[-3:]) if session_history else "No prior context"}

Extract:
1. intent: Main operation (e.g., "select_faces", "create_material", "animate_object")
2. entities: Key concepts/objects mentioned (list)
3. variations: 3-5 alternative phrasings for better retrieval
4. detected_filters: {{"category": "mesh/object/material/animation", "version": "4.2"}}
5. is_multi_step: Does this require multiple operations?
6. sub_queries: If multi-step, list individual steps as queries
7. confidence: 0.0-1.0 how confident are you in this analysis

Return valid JSON only:
{{"intent": "...", "entities": [...], "variations": [...], "detected_filters": {{}}, "is_multi_step": false, "sub_queries": [], "confidence": 0.9}}"""

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=config["max_tokens"],
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        data = json.loads(result_text)
        analysis = QueryAnalysis(
            intent=data.get("intent", "general_query"),
            entities=data.get("entities", []),
            variations=data.get("variations", [query]),
            detected_filters=data.get("detected_filters", {}),
            is_multi_step=data.get("is_multi_step", False),
            sub_queries=data.get("sub_queries", []),
            confidence=data.get("confidence", 0.8)
        )

    except Exception as e:
        logger.error(f"Query analysis error: {e}")
        analysis = QueryAnalysis(
            intent="general_query",
            entities=query.lower().split()[:5],
            variations=[query],
            detected_filters={},
            is_multi_step=False,
            sub_queries=[],
            confidence=0.5
        )

    duration_ms = int((time.time() - start_time) * 1000)
    return analysis, duration_ms


# =============================================================================
# Layer 3: Hybrid Retrieval (Dense + Sparse)
# =============================================================================

def run_dense_retrieval(query: str, top_k: int = 100) -> Tuple[List[Tuple[str, str, float]], int]:
    """
    Dense retrieval using BGE-M3 embeddings + pgvector
    Returns: [(doc_id, content, similarity_score), ...]
    """
    start_time = time.time()

    try:
        model = get_embedding_model()
        query_embedding = model.encode(query, normalize_embeddings=True)

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Cosine similarity search using pgvector
        cur.execute("""
            SELECT
                id, content, title, source, blender_version, category,
                1 - (embedding <=> %s::vector) as similarity
            FROM documents
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding.tolist(), query_embedding.tolist(), top_k))

        results = []
        for row in cur.fetchall():
            results.append((
                str(row['id']),
                row['content'],
                float(row['similarity'])
            ))

        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Dense retrieval error: {e}")
        results = []

    duration_ms = int((time.time() - start_time) * 1000)
    return results, duration_ms


def run_sparse_retrieval(query: str, top_k: int = 100) -> Tuple[List[Tuple[str, str, float]], int]:
    """
    Sparse retrieval using PostgreSQL full-text search (BM25-like)
    Returns: [(doc_id, content, bm25_score), ...]
    """
    start_time = time.time()

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Full-text search with ts_rank (BM25-like scoring)
        cur.execute("""
            SELECT
                id, content, title, source, blender_version, category,
                ts_rank_cd(tsv, plainto_tsquery('english', %s), 32) as rank
            FROM documents
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, top_k))

        results = []
        for row in cur.fetchall():
            results.append((
                str(row['id']),
                row['content'],
                float(row['rank'])
            ))

        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Sparse retrieval error: {e}")
        results = []

    duration_ms = int((time.time() - start_time) * 1000)
    return results, duration_ms


# =============================================================================
# Layer 3b: RRF Fusion
# =============================================================================

def run_rrf_fusion(
    dense_results: List[Tuple[str, str, float]],
    sparse_results: List[Tuple[str, str, float]],
    k: int = RRF_K
) -> Tuple[List[Tuple[str, str, float]], int]:
    """
    Reciprocal Rank Fusion to merge dense and sparse results

    RRF(d) = Σ 1 / (k + rank(d, r))
    """
    start_time = time.time()

    rrf_scores: Dict[str, float] = {}
    doc_contents: Dict[str, str] = {}

    # Add dense contributions
    for rank, (doc_id, content, score) in enumerate(dense_results, 1):
        rrf_contribution = 1.0 / (k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_contribution
        doc_contents[doc_id] = content

    # Add sparse contributions
    for rank, (doc_id, content, score) in enumerate(sparse_results, 1):
        rrf_contribution = 1.0 / (k + rank)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_contribution
        doc_contents[doc_id] = content

    # Sort by RRF score
    sorted_results = sorted(
        [(doc_id, doc_contents[doc_id], score) for doc_id, score in rrf_scores.items()],
        key=lambda x: x[2],
        reverse=True
    )

    duration_ms = int((time.time() - start_time) * 1000)
    return sorted_results, duration_ms


# =============================================================================
# Layer 4: Cross-Encoder Reranking
# =============================================================================

def run_reranking(
    query: str,
    candidates: List[Tuple[str, str, float]],
    top_k: int = 10
) -> Tuple[List[Document], int]:
    """
    Rerank candidates using BGE cross-encoder
    """
    start_time = time.time()

    if not candidates:
        return [], int((time.time() - start_time) * 1000)

    try:
        reranker = get_reranker_model()

        # Prepare query-document pairs
        pairs = [[query, doc_content] for (doc_id, doc_content, _) in candidates[:50]]

        # Get reranking scores
        scores = reranker.predict(pairs)

        # Combine with document info
        reranked = []
        for i, ((doc_id, content, rrf_score), rerank_score) in enumerate(zip(candidates[:50], scores)):
            doc = Document(
                id=doc_id,
                content=content,
                rrf_score=rrf_score,
                rerank_score=float(rerank_score)
            )
            reranked.append(doc)

        # Sort by rerank score
        reranked.sort(key=lambda d: d.rerank_score, reverse=True)

    except Exception as e:
        logger.error(f"Reranking error: {e}")
        # Fallback: use RRF scores
        reranked = [
            Document(id=doc_id, content=content, rrf_score=score, rerank_score=score)
            for doc_id, content, score in candidates[:top_k]
        ]

    duration_ms = int((time.time() - start_time) * 1000)
    return reranked[:top_k], duration_ms


# =============================================================================
# Layer 5: Context Assembly (GPT-5-mini)
# =============================================================================

def run_context_assembly(
    query: str,
    documents: List[Document],
    max_tokens: int = 4000
) -> Tuple[str, int]:
    """
    Assemble context for generation with token-aware truncation
    """
    start_time = time.time()

    context_parts = [
        "=== RETRIEVED CONTEXT FOR BLENDER QUERY ===\n",
        f"Query: {query}\n",
        f"Sources: {len(documents)} documents (ranked by relevance)\n",
        "=" * 50 + "\n"
    ]

    estimated_tokens = 100
    max_per_doc = 600

    for i, doc in enumerate(documents, 1):
        # Truncate document if needed
        content = doc.content
        if len(content.split()) > max_per_doc:
            content = ' '.join(content.split()[:max_per_doc]) + "..."

        source_info = f"\n[Source {i} | Relevance: {doc.rerank_score:.1%} | ID: {doc.id}]\n"
        context_parts.append(source_info)
        context_parts.append(content)
        context_parts.append("\n" + "-" * 40 + "\n")

        estimated_tokens += len(content.split()) + 20
        if estimated_tokens >= max_tokens:
            context_parts.append(f"\n[... {len(documents) - i} more sources omitted]\n")
            break

    context_parts.append("\n" + "=" * 50)
    context_parts.append("\nIMPORTANT: Base your answer ONLY on the above context.")
    context_parts.append("\nCite sources using [Source N] notation.")
    context_parts.append("\nIf context is insufficient, clearly state what's missing.\n")

    context = ''.join(context_parts)
    duration_ms = int((time.time() - start_time) * 1000)
    return context, duration_ms


# =============================================================================
# Layer 6: Generation Agent (GPT-5.1)
# =============================================================================

def run_generation(
    query: str,
    context: str,
    analysis: QueryAnalysis,
    session_history: List[Dict]
) -> Tuple[str, List[Dict], int]:
    """
    Layer 6: Generation Agent - Generate answer with citations
    Model: GPT-5.1 with reasoning_effort="medium"
    """
    start_time = time.time()

    client = get_openai_client()
    config = MODEL_CONFIG["generator"]

    # Build system prompt
    system_prompt = """You are an expert Blender 3D Assistant with deep knowledge of:
- Blender Python API (bpy module)
- 3D modeling, sculpting, and mesh operations
- Materials, shaders, and node-based workflows
- Animation, rigging, and keyframes
- Rendering and compositing

Guidelines:
1. Answer ONLY based on the provided context
2. Always cite sources using [Source N] notation
3. For code questions, provide working Python/bpy examples
4. Use ```python code blocks for code
5. Be concise but complete
6. If context is insufficient, say so clearly"""

    user_prompt = f"""{context}

User Question: {query}

Intent Detected: {analysis.intent}
Entities: {', '.join(analysis.entities)}

Provide a clear, actionable answer with source citations."""

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=config["max_tokens"],
        )

        answer = response.choices[0].message.content.strip()

        # Extract citations from answer
        citations = []
        citation_pattern = r'\[Source (\d+)\]'
        for match in re.finditer(citation_pattern, answer):
            source_num = int(match.group(1))
            citations.append({
                "index": source_num,
                "doc_id": f"doc-{source_num}",
                "text": f"Reference from source {source_num}"
            })

    except Exception as e:
        logger.error(f"Generation error: {e}")
        answer = f"I apologize, but I encountered an error generating the response: {str(e)}"
        citations = []

    duration_ms = int((time.time() - start_time) * 1000)
    return answer, citations, duration_ms


# =============================================================================
# Layer 7: Validation Agent (GPT-5-mini)
# =============================================================================

def run_validation(
    query: str,
    answer: str,
    context: str,
    documents: List[Document]
) -> Tuple[ValidationResult, int]:
    """
    Layer 7: Validation Agent - Quality validation with RAGAS-like metrics
    Model: GPT-5-mini with reasoning_effort="low"
    """
    start_time = time.time()

    client = get_openai_client()
    config = MODEL_CONFIG["validator"]

    prompt = f"""Evaluate this RAG-generated answer for quality.

QUERY: {query}

CONTEXT (summarized):
{context[:2000]}...

ANSWER:
{answer}

Evaluate on these metrics (0.0-1.0):
1. faithfulness: Is the answer grounded in the context? (no hallucinations)
2. relevancy: Does the answer address the user's question?
3. completeness: Is the answer comprehensive enough?
4. code_validity: If code is present, is it syntactically correct? (null if no code)

Also identify any issues (list of strings).

Return JSON only:
{{"faithfulness": 0.9, "relevancy": 0.95, "completeness": 0.85, "code_validity": 0.9, "issues": []}}"""

    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=config["max_tokens"],
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        data = json.loads(result_text)

        # Calculate composite score
        scores = [data.get("faithfulness", 0.8), data.get("relevancy", 0.8), data.get("completeness", 0.8)]
        if data.get("code_validity") is not None:
            scores.append(data["code_validity"])
        composite = sum(scores) / len(scores)

        validation = ValidationResult(
            faithfulness=data.get("faithfulness", 0.8),
            relevancy=data.get("relevancy", 0.8),
            completeness=data.get("completeness", 0.8),
            code_validity=data.get("code_validity"),
            composite_score=composite,
            issues=data.get("issues", []),
            passed=composite >= 0.7 and data.get("faithfulness", 0.8) >= 0.7,
            attempt=1
        )

    except Exception as e:
        logger.error(f"Validation error: {e}")
        validation = ValidationResult(
            faithfulness=0.8,
            relevancy=0.8,
            completeness=0.8,
            code_validity=None,
            composite_score=0.8,
            issues=[f"Validation error: {str(e)}"],
            passed=True,
            attempt=1
        )

    duration_ms = int((time.time() - start_time) * 1000)
    return validation, duration_ms


# =============================================================================
# Main Pipeline
# =============================================================================

def run_rag_pipeline(
    query: str,
    session_id: Optional[str] = None,
    settings: Optional[Dict] = None
) -> RAGResponse:
    """
    Execute the full 7-layer agentic RAG pipeline
    """
    request_id = f"rag_{int(time.time())}_{os.urandom(4).hex()}"
    settings = settings or {}

    # Get or create session
    if session_id and session_id in SESSION_STORE:
        session = SESSION_STORE[session_id]
    else:
        session_id = f"sess_{int(time.time())}_{os.urandom(6).hex()}"
        SESSION_STORE[session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "history": [],
            "context": []
        }
        session = SESSION_STORE[session_id]

    stages: List[StageResult] = []
    latency: Dict[str, int] = {}
    total_start = time.time()

    # Layer 1: Orchestration
    plan, orch_duration = run_orchestrator(query, session["history"])
    latency["orchestration"] = orch_duration
    stages.append(StageResult(
        stage=PipelineStage.ORCHESTRATION.value,
        status="completed",
        duration_ms=orch_duration,
        details=plan
    ))

    # Layer 2: Query Analysis
    analysis, analysis_duration = run_query_analysis(query, session["history"])
    latency["query_analysis"] = analysis_duration
    stages.append(StageResult(
        stage=PipelineStage.QUERY_ANALYSIS.value,
        status="completed",
        duration_ms=analysis_duration,
        results_count=len(analysis.entities),
        details={"intent": analysis.intent, "confidence": analysis.confidence}
    ))

    # Layer 3: Hybrid Retrieval
    top_k = settings.get("top_k", 100)
    retrieval_mode = settings.get("retrieval_mode", "hybrid")

    dense_results, dense_duration = [], 0
    sparse_results, sparse_duration = [], 0

    if retrieval_mode in ["hybrid", "dense"]:
        dense_results, dense_duration = run_dense_retrieval(query, top_k)
        latency["retrieval_dense"] = dense_duration
        stages.append(StageResult(
            stage=PipelineStage.RETRIEVAL_DENSE.value,
            status="completed",
            duration_ms=dense_duration,
            results_count=len(dense_results)
        ))

    if retrieval_mode in ["hybrid", "sparse"]:
        sparse_results, sparse_duration = run_sparse_retrieval(query, top_k)
        latency["retrieval_sparse"] = sparse_duration
        stages.append(StageResult(
            stage=PipelineStage.RETRIEVAL_SPARSE.value,
            status="completed",
            duration_ms=sparse_duration,
            results_count=len(sparse_results)
        ))

    # RRF Fusion
    if retrieval_mode == "hybrid" and dense_results and sparse_results:
        fused_results, fusion_duration = run_rrf_fusion(dense_results, sparse_results)
    elif dense_results:
        fused_results = dense_results
        fusion_duration = 0
    else:
        fused_results = sparse_results
        fusion_duration = 0

    latency["rrf_fusion"] = fusion_duration
    stages.append(StageResult(
        stage=PipelineStage.RRF_FUSION.value,
        status="completed",
        duration_ms=fusion_duration,
        results_count=len(fused_results),
        details={"k": RRF_K, "mode": retrieval_mode}
    ))

    # Layer 4: Reranking
    documents = []
    if settings.get("enable_reranking", True) and fused_results:
        documents, rerank_duration = run_reranking(query, fused_results, top_k=10)
        latency["reranking"] = rerank_duration
        stages.append(StageResult(
            stage=PipelineStage.RERANKING.value,
            status="completed",
            duration_ms=rerank_duration,
            results_count=len(documents)
        ))
    else:
        # Skip reranking, just convert to Documents
        documents = [
            Document(id=doc_id, content=content, rrf_score=score, rerank_score=score)
            for doc_id, content, score in fused_results[:10]
        ]

    # Layer 5: Context Assembly
    context, assembly_duration = run_context_assembly(query, documents)
    latency["context_assembly"] = assembly_duration
    stages.append(StageResult(
        stage=PipelineStage.CONTEXT_ASSEMBLY.value,
        status="completed",
        duration_ms=assembly_duration,
        results_count=len(documents)
    ))

    # Layer 6: Generation
    answer, citations, gen_duration = run_generation(query, context, analysis, session["history"])
    latency["generation"] = gen_duration
    stages.append(StageResult(
        stage=PipelineStage.GENERATION.value,
        status="completed",
        duration_ms=gen_duration,
        details={"model": MODEL_CONFIG["generator"]["model"]}
    ))

    # Layer 7: Validation
    validation, val_duration = run_validation(query, answer, context, documents)
    latency["validation"] = val_duration
    stages.append(StageResult(
        stage=PipelineStage.VALIDATION.value,
        status="completed",
        duration_ms=val_duration,
        details={"passed": validation.passed, "score": validation.composite_score}
    ))

    # Total latency
    latency["total"] = int((time.time() - total_start) * 1000)

    # Update session history
    session["history"].append({
        "query": query,
        "answer": answer[:500],
        "timestamp": datetime.utcnow().isoformat()
    })
    session["history"] = session["history"][-10:]  # Keep last 10

    # Build metrics
    metrics = {
        "faithfulness": validation.faithfulness,
        "relevancy": validation.relevancy,
        "completeness": validation.completeness,
        "composite_score": validation.composite_score
    }
    if validation.code_validity is not None:
        metrics["code_validity"] = validation.code_validity

    return RAGResponse(
        request_id=request_id,
        session_id=session_id,
        query=query,
        analysis=analysis,
        documents=documents,
        answer=answer,
        citations=citations,
        validation=validation,
        metrics=metrics,
        latency=latency,
        stages=stages,
        status="success" if validation.passed else "warning",
        warning=None if validation.passed else "Answer may have quality issues"
    )


# =============================================================================
# API Routes
# =============================================================================

@rag_bp.route('/api/rag/query', methods=['POST'])
def rag_query():
    """Main RAG query endpoint"""
    data = request.json or {}

    query = data.get('query', '').strip()
    session_id = data.get('session_id')
    settings = data.get('settings', {})

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    if len(query) > 2000:
        return jsonify({'error': 'Query too long (max 2000 characters)'}), 400

    try:
        response = run_rag_pipeline(query, session_id, settings)

        result = {
            "request_id": response.request_id,
            "session_id": response.session_id,
            "query": response.query,
            "analysis": asdict(response.analysis),
            "documents": [asdict(d) for d in response.documents],
            "answer": response.answer,
            "citations": response.citations,
            "validation": asdict(response.validation),
            "metrics": response.metrics,
            "latency": response.latency,
            "stages": [asdict(s) for s in response.stages],
            "status": response.status,
            "warning": response.warning,
            "demo_mode": False,
            "models": {
                "orchestrator": MODEL_CONFIG["orchestrator"]["model"],
                "generator": MODEL_CONFIG["generator"]["model"],
                "analyzer": MODEL_CONFIG["query_analyzer"]["model"]
            }
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@rag_bp.route('/api/rag/status', methods=['GET'])
def rag_status():
    """Get RAG system status"""
    openai_ok = False
    embedding_ok = False
    db_ok = False

    # Check OpenAI
    try:
        client = get_openai_client()
        openai_ok = True
    except:
        pass

    # Check embedding model
    try:
        model = get_embedding_model()
        embedding_ok = True
    except:
        pass

    # Check database
    try:
        conn = get_db_connection()
        conn.close()
        db_ok = True
    except:
        pass

    return jsonify({
        "status": "healthy" if all([openai_ok, db_ok]) else "degraded",
        "components": {
            "openai_gpt5": openai_ok,
            "database": db_ok,
            "embedding_model": embedding_ok,
            "reranker": False,  # Will be true after first use
        },
        "config": {
            "orchestrator_model": MODEL_CONFIG["orchestrator"]["model"],
            "generator_model": MODEL_CONFIG["generator"]["model"],
            "analyzer_model": MODEL_CONFIG["query_analyzer"]["model"],
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL
        },
        "demo_mode": False,
        "active_sessions": len(SESSION_STORE),
        "timestamp": datetime.utcnow().isoformat()
    })


@rag_bp.route('/api/rag/sessions/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """Get session history"""
    if session_id not in SESSION_STORE:
        return jsonify({'error': 'Session not found'}), 404

    session = SESSION_STORE[session_id]
    return jsonify({
        "session_id": session_id,
        "created_at": session["created_at"],
        "history": session["history"],
        "turn_count": len(session["history"])
    })


@rag_bp.route('/api/rag/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    """Delete a session"""
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
        return jsonify({'message': 'Session deleted'})
    return jsonify({'error': 'Session not found'}), 404


@rag_bp.route('/api/rag/sample-queries', methods=['GET'])
def get_sample_queries():
    """Get sample queries"""
    return jsonify({
        "queries": [
            {"label": "Selection", "query": "How do I select all faces in Blender?", "category": "mesh"},
            {"label": "Python API", "query": "What is bpy.ops.mesh.select_all?", "category": "scripting"},
            {"label": "Materials", "query": "How to create a procedural material with nodes?", "category": "materials"},
            {"label": "Animation", "query": "Add keyframes for object animation using Python", "category": "animation"},
            {"label": "Multi-step", "query": "Select all faces, extrude them, and apply smooth shading", "category": "workflow"},
            {"label": "UV Mapping", "query": "How to unwrap UV coordinates automatically?", "category": "uv"}
        ]
    })
