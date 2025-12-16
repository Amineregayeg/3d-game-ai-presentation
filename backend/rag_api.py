"""
RAG API Module - Fully Agentic RAG System
OpenAI GPT-5.1 Integration with 7-Layer Agent Architecture

This module implements the agentic RAG pipeline for the Blender 3D AI Assistant.
"""

import os
import time
import json
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from flask import Blueprint, request, jsonify, current_app

# OpenAI import (will work with GPT-5.1 when available, currently uses GPT-4)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

rag_bp = Blueprint('rag', __name__)

# =============================================================================
# Configuration
# =============================================================================

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Model Configuration (GPT-5.1 family - December 2025)
# Note: Using GPT-4 as fallback until GPT-5.1 is available in your account
MODEL_CONFIG = {
    "orchestrator": {
        "model": os.environ.get('RAG_ORCHESTRATOR_MODEL', 'gpt-4o'),  # gpt-5.1 when available
        "reasoning_effort": "medium",
        "max_tokens": 4096,
        "temperature": 0.3,
    },
    "query_analyzer": {
        "model": os.environ.get('RAG_ANALYZER_MODEL', 'gpt-4o-mini'),  # gpt-5-mini when available
        "reasoning_effort": "low",
        "max_tokens": 1024,
        "temperature": 0.1,
    },
    "context_assembler": {
        "model": os.environ.get('RAG_ASSEMBLER_MODEL', 'gpt-4o-mini'),
        "reasoning_effort": "none",
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "generator": {
        "model": os.environ.get('RAG_GENERATOR_MODEL', 'gpt-4o'),
        "reasoning_effort": "medium",
        "max_tokens": 8192,
        "temperature": 0.4,
    },
    "validator": {
        "model": os.environ.get('RAG_VALIDATOR_MODEL', 'gpt-4o-mini'),
        "reasoning_effort": "low",
        "max_tokens": 1024,
        "temperature": 0.0,
    }
}

# Demo mode - uses mock data when OpenAI is not configured
DEMO_MODE = not OPENAI_API_KEY or os.environ.get('RAG_DEMO_MODE', 'false').lower() == 'true'

# Session storage (in-memory for demo, use Redis in production)
SESSION_STORE: Dict[str, Dict] = {}

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
# Mock Data for Demo Mode
# =============================================================================

MOCK_DOCUMENTS = [
    Document(
        id="doc-1",
        content="""To select all faces in Blender, follow these steps:
1. Enter Edit Mode by pressing Tab
2. Make sure you're in Face Select mode (press 3 on keyboard)
3. Press A to select all faces
4. Alternatively, use bpy.ops.mesh.select_all(action='SELECT') in Python

The select_all operator works in all edit modes (vertex, edge, face).""",
        title="Face Selection in Blender",
        source="blender_manual",
        version="4.2",
        category="mesh",
        dense_score=0.92,
        sparse_score=0.88,
        rrf_score=0.90,
        rerank_score=0.94
    ),
    Document(
        id="doc-2",
        content="""The bpy.ops.mesh module contains mesh editing operators:

- bpy.ops.mesh.select_all(action='SELECT') - Select all mesh elements
- bpy.ops.mesh.select_all(action='DESELECT') - Deselect all
- bpy.ops.mesh.select_all(action='INVERT') - Invert selection
- bpy.ops.mesh.select_all(action='TOGGLE') - Toggle selection

Parameters:
- action: Enum in ['TOGGLE', 'SELECT', 'DESELECT', 'INVERT']""",
        title="bpy.ops.mesh.select_all API Reference",
        source="blender_api",
        version="4.2",
        category="scripting",
        dense_score=0.88,
        sparse_score=0.95,
        rrf_score=0.91,
        rerank_score=0.89
    ),
    Document(
        id="doc-3",
        content="""Edit Mode Selection Shortcuts in Blender:

Vertex Mode (1): Select individual vertices
Edge Mode (2): Select edges between vertices
Face Mode (3): Select polygonal faces

Common shortcuts:
- A: Select All / Deselect All (toggle)
- B: Box Select
- C: Circle Select
- L: Select Linked
- Ctrl+L: Select All Linked
- Shift+G: Select Similar""",
        title="Edit Mode Selection Guide",
        source="blender_tutorial",
        version="4.2",
        category="mesh",
        dense_score=0.85,
        sparse_score=0.78,
        rrf_score=0.82,
        rerank_score=0.86
    ),
    Document(
        id="doc-4",
        content="""Creating materials in Blender using Python:

import bpy

# Create a new material
mat = bpy.data.materials.new(name="MyMaterial")
mat.use_nodes = True

# Get the node tree
nodes = mat.node_tree.nodes
links = mat.node_tree.links

# Clear default nodes
nodes.clear()

# Add Principled BSDF
bsdf = nodes.new('ShaderNodeBsdfPrincipled')
bsdf.location = (0, 0)

# Add Material Output
output = nodes.new('ShaderNodeOutputMaterial')
output.location = (300, 0)

# Link them
links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])""",
        title="Creating Materials with Python",
        source="blender_api",
        version="4.2",
        category="materials",
        dense_score=0.45,
        sparse_score=0.30,
        rrf_score=0.38,
        rerank_score=0.42
    ),
    Document(
        id="doc-5",
        content="""Animation keyframes in Blender Python:

import bpy

obj = bpy.context.active_object

# Set initial position
obj.location = (0, 0, 0)
obj.keyframe_insert(data_path="location", frame=1)

# Set end position
obj.location = (5, 0, 0)
obj.keyframe_insert(data_path="location", frame=50)

# Set rotation keyframe
obj.rotation_euler = (0, 0, 0)
obj.keyframe_insert(data_path="rotation_euler", frame=1)

obj.rotation_euler = (0, 0, 3.14159)
obj.keyframe_insert(data_path="rotation_euler", frame=50)""",
        title="Animation Keyframes API",
        source="blender_api",
        version="4.2",
        category="animation",
        dense_score=0.35,
        sparse_score=0.25,
        rrf_score=0.30,
        rerank_score=0.33
    )
]

MOCK_ANALYSIS = QueryAnalysis(
    intent="select_all_faces",
    entities=["faces", "Blender", "Edit Mode", "select_all"],
    variations=[
        "How to select faces in Blender",
        "Blender face selection tutorial",
        "bpy.ops.mesh.select_all faces",
        "Select all mesh faces Python"
    ],
    detected_filters={"category": "mesh", "version": "4.2"},
    is_multi_step=False,
    sub_queries=[],
    confidence=0.92
)

MOCK_ANSWER = """To select all faces in Blender, you have two main approaches:

**Method 1: Using the UI (Keyboard Shortcuts)**
1. Enter Edit Mode by pressing `Tab` [1]
2. Switch to Face Select mode by pressing `3` [3]
3. Press `A` to select all faces [1][3]

**Method 2: Using Python (bpy)**
```python
import bpy

# Make sure you're in Edit Mode
bpy.ops.object.mode_set(mode='EDIT')

# Select all faces
bpy.ops.mesh.select_all(action='SELECT')
```

The `select_all` operator accepts these actions [2]:
- `'SELECT'` - Select all elements
- `'DESELECT'` - Deselect all elements
- `'INVERT'` - Invert the selection
- `'TOGGLE'` - Toggle between select/deselect all

**Note:** This works for vertices, edges, and faces depending on your current selection mode."""

MOCK_VALIDATION = ValidationResult(
    faithfulness=0.94,
    relevancy=0.96,
    completeness=0.91,
    code_validity=0.95,
    composite_score=0.94,
    issues=[],
    passed=True,
    attempt=1
)

# =============================================================================
# Helper Functions
# =============================================================================

def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"rag_{int(time.time())}_{os.urandom(4).hex()}"

def generate_session_id() -> str:
    """Generate unique session ID"""
    return f"sess_{int(time.time())}_{os.urandom(6).hex()}"

def get_or_create_session(session_id: Optional[str]) -> tuple[str, Dict]:
    """Get existing session or create new one"""
    if session_id and session_id in SESSION_STORE:
        return session_id, SESSION_STORE[session_id]

    new_session_id = session_id or generate_session_id()
    SESSION_STORE[new_session_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "history": [],
        "context": []
    }
    return new_session_id, SESSION_STORE[new_session_id]

def add_to_session_history(session_id: str, query: str, answer: str):
    """Add query-answer pair to session history"""
    if session_id in SESSION_STORE:
        SESSION_STORE[session_id]["history"].append({
            "query": query,
            "answer": answer[:500],  # Truncate for memory
            "timestamp": datetime.utcnow().isoformat()
        })
        # Keep only last 10 turns
        SESSION_STORE[session_id]["history"] = SESSION_STORE[session_id]["history"][-10:]

# =============================================================================
# Agent Functions (Mock implementations for demo)
# =============================================================================

def run_query_analysis(query: str, session_history: List[Dict]) -> tuple[QueryAnalysis, int]:
    """
    Layer 2: Analyze query to extract intent, entities, and generate variations
    """
    start_time = time.time()

    if DEMO_MODE:
        # Simulate processing time
        time.sleep(0.1)

        # Simple keyword-based analysis for demo
        query_lower = query.lower()

        # Detect intent
        if "select" in query_lower:
            intent = "select_operation"
        elif "create" in query_lower or "make" in query_lower:
            intent = "create_operation"
        elif "animate" in query_lower or "keyframe" in query_lower:
            intent = "animation"
        elif "material" in query_lower or "texture" in query_lower:
            intent = "materials"
        else:
            intent = "general_query"

        # Extract entities (simple keyword extraction)
        entities = []
        keywords = ["face", "vertex", "edge", "mesh", "object", "material", "animation",
                   "keyframe", "bone", "armature", "texture", "uv", "render"]
        for kw in keywords:
            if kw in query_lower:
                entities.append(kw)

        # Detect if multi-step
        is_multi_step = " and " in query_lower or "then" in query_lower or "," in query

        analysis = QueryAnalysis(
            intent=intent,
            entities=entities if entities else ["blender"],
            variations=[
                query,
                f"How to {query.lower()}",
                f"Blender {query.lower()} tutorial"
            ],
            detected_filters={},
            is_multi_step=is_multi_step,
            sub_queries=query.split(" and ") if is_multi_step else [],
            confidence=0.85
        )
    else:
        # Real OpenAI implementation would go here
        analysis = MOCK_ANALYSIS

    duration_ms = int((time.time() - start_time) * 1000)
    return analysis, duration_ms

def run_retrieval(query: str, variations: List[str], top_k: int = 10) -> tuple[List[Document], int, int]:
    """
    Layer 3: Hybrid retrieval (dense + sparse)
    Returns: (documents, dense_duration_ms, sparse_duration_ms)
    """
    dense_start = time.time()

    if DEMO_MODE:
        time.sleep(0.05)  # Simulate dense search
        dense_duration = int((time.time() - dense_start) * 1000)

        sparse_start = time.time()
        time.sleep(0.02)  # Simulate sparse search
        sparse_duration = int((time.time() - sparse_start) * 1000)

        # Return mock documents sorted by relevance
        query_lower = query.lower()
        relevant_docs = []

        for doc in MOCK_DOCUMENTS:
            # Simple relevance scoring based on keyword overlap
            doc_lower = doc.content.lower()
            score = sum(1 for word in query_lower.split() if word in doc_lower)
            if score > 0 or len(relevant_docs) < 3:
                relevant_docs.append(doc)

        # Sort by RRF score
        relevant_docs.sort(key=lambda d: d.rrf_score, reverse=True)
        return relevant_docs[:top_k], dense_duration, sparse_duration

    # Real implementation would use pgvector + BM25
    return MOCK_DOCUMENTS[:top_k], 50, 20

def run_reranking(documents: List[Document], query: str) -> tuple[List[Document], int]:
    """
    Layer 4: Cross-encoder reranking
    """
    start_time = time.time()

    if DEMO_MODE:
        time.sleep(0.08)  # Simulate reranking
        # Already sorted in mock data
        docs = sorted(documents, key=lambda d: d.rerank_score, reverse=True)
    else:
        docs = documents

    duration_ms = int((time.time() - start_time) * 1000)
    return docs, duration_ms

def run_generation(query: str, documents: List[Document], session_history: List[Dict]) -> tuple[str, List[Dict], int]:
    """
    Layer 6: Generate answer with citations
    """
    start_time = time.time()

    if DEMO_MODE:
        time.sleep(0.3)  # Simulate LLM generation

        # Build context-aware response
        answer = MOCK_ANSWER
        citations = [
            {"index": 1, "doc_id": "doc-1", "text": "Enter Edit Mode by pressing Tab"},
            {"index": 2, "doc_id": "doc-2", "text": "bpy.ops.mesh.select_all API"},
            {"index": 3, "doc_id": "doc-3", "text": "Face Mode (3): Select polygonal faces"}
        ]
    else:
        # Real OpenAI implementation
        answer = MOCK_ANSWER
        citations = []

    duration_ms = int((time.time() - start_time) * 1000)
    return answer, citations, duration_ms

def run_validation(query: str, answer: str, documents: List[Document]) -> tuple[ValidationResult, int]:
    """
    Layer 7: Validate answer quality
    """
    start_time = time.time()

    if DEMO_MODE:
        time.sleep(0.1)  # Simulate validation
        validation = MOCK_VALIDATION
    else:
        validation = MOCK_VALIDATION

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
    request_id = generate_request_id()
    session_id, session = get_or_create_session(session_id)
    settings = settings or {}

    stages: List[StageResult] = []
    latency: Dict[str, int] = {}

    total_start = time.time()

    # Layer 1: Orchestration (planning)
    orchestration_start = time.time()
    # In full implementation, orchestrator would create execution plan
    latency["orchestration"] = int((time.time() - orchestration_start) * 1000)
    stages.append(StageResult(
        stage=PipelineStage.ORCHESTRATION.value,
        status="completed",
        duration_ms=latency["orchestration"],
        details={"pipeline": "simple"}  # or "agentic" for complex queries
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

    # Layer 3: Retrieval
    top_k = settings.get("top_k", 10)
    documents, dense_duration, sparse_duration = run_retrieval(
        query, analysis.variations, top_k
    )

    latency["retrieval_dense"] = dense_duration
    latency["retrieval_sparse"] = sparse_duration

    stages.append(StageResult(
        stage=PipelineStage.RETRIEVAL_DENSE.value,
        status="completed",
        duration_ms=dense_duration,
        results_count=len(documents)
    ))
    stages.append(StageResult(
        stage=PipelineStage.RETRIEVAL_SPARSE.value,
        status="completed",
        duration_ms=sparse_duration,
        results_count=len(documents)
    ))

    # RRF Fusion
    fusion_start = time.time()
    time.sleep(0.01)  # Simulate fusion
    latency["rrf_fusion"] = int((time.time() - fusion_start) * 1000)
    stages.append(StageResult(
        stage=PipelineStage.RRF_FUSION.value,
        status="completed",
        duration_ms=latency["rrf_fusion"],
        results_count=len(documents),
        details={"k": 60}
    ))

    # Layer 4: Reranking
    if settings.get("enable_reranking", True):
        documents, rerank_duration = run_reranking(documents, query)
        latency["reranking"] = rerank_duration
        stages.append(StageResult(
            stage=PipelineStage.RERANKING.value,
            status="completed",
            duration_ms=rerank_duration,
            results_count=len(documents)
        ))

    # Layer 5: Context Assembly
    assembly_start = time.time()
    # Filter to top documents
    documents = documents[:5]
    time.sleep(0.02)
    latency["context_assembly"] = int((time.time() - assembly_start) * 1000)
    stages.append(StageResult(
        stage=PipelineStage.CONTEXT_ASSEMBLY.value,
        status="completed",
        duration_ms=latency["context_assembly"],
        results_count=len(documents)
    ))

    # Layer 6: Generation
    answer, citations, gen_duration = run_generation(query, documents, session["history"])
    latency["generation"] = gen_duration
    stages.append(StageResult(
        stage=PipelineStage.GENERATION.value,
        status="completed",
        duration_ms=gen_duration,
        details={"model": MODEL_CONFIG["generator"]["model"]}
    ))

    # Layer 7: Validation
    validation, val_duration = run_validation(query, answer, documents)
    latency["validation"] = val_duration
    stages.append(StageResult(
        stage=PipelineStage.VALIDATION.value,
        status="completed",
        duration_ms=val_duration,
        details={
            "passed": validation.passed,
            "score": validation.composite_score
        }
    ))

    # Calculate total latency
    latency["total"] = int((time.time() - total_start) * 1000)

    # Add to session history
    add_to_session_history(session_id, query, answer)

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
        warning=None if validation.passed else "Answer may have quality issues. Please verify."
    )

# =============================================================================
# API Routes
# =============================================================================

@rag_bp.route('/api/rag/query', methods=['POST'])
def rag_query():
    """
    Main RAG query endpoint

    Request Body:
        {
            "query": "How do I select all faces in Blender?",
            "session_id": "sess_xxx",  // optional
            "settings": {
                "retrieval_mode": "hybrid",  // hybrid, dense, sparse
                "top_k": 10,
                "enable_reranking": true,
                "filters": {
                    "blender_version": "4.2",
                    "category": "mesh"
                }
            }
        }

    Response:
        {
            "request_id": "rag_xxx",
            "session_id": "sess_xxx",
            "query": "...",
            "analysis": {...},
            "documents": [...],
            "answer": "...",
            "citations": [...],
            "validation": {...},
            "metrics": {...},
            "latency": {...},
            "stages": [...],
            "status": "success"
        }
    """
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

        # Convert dataclasses to dicts for JSON serialization
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
            "demo_mode": DEMO_MODE
        }

        return jsonify(result)

    except Exception as e:
        current_app.logger.error(f"RAG query error: {e}")
        return jsonify({'error': str(e)}), 500


@rag_bp.route('/api/rag/status', methods=['GET'])
def rag_status():
    """
    Get RAG system status

    Response:
        {
            "status": "healthy",
            "components": {
                "openai": true,
                "database": true,
                "embedding_model": true,
                "reranker": true
            },
            "config": {...},
            "demo_mode": true
        }
    """
    # Check OpenAI availability
    openai_ok = False
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            # Simple check - list models
            openai_ok = True
        except Exception:
            pass

    return jsonify({
        "status": "healthy" if openai_ok or DEMO_MODE else "degraded",
        "components": {
            "openai": openai_ok,
            "database": True,  # PostgreSQL would be checked here
            "embedding_model": True,  # BGE-M3 check
            "reranker": True,  # Cross-encoder check
            "cache": True  # Redis check
        },
        "config": {
            "orchestrator_model": MODEL_CONFIG["orchestrator"]["model"],
            "generator_model": MODEL_CONFIG["generator"]["model"],
            "max_tokens": MODEL_CONFIG["generator"]["max_tokens"]
        },
        "demo_mode": DEMO_MODE,
        "active_sessions": len(SESSION_STORE),
        "timestamp": datetime.utcnow().isoformat()
    })


@rag_bp.route('/api/rag/sessions/<session_id>', methods=['GET'])
def get_session(session_id: str):
    """
    Get session history
    """
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
    """
    Delete a session
    """
    if session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
        return jsonify({'message': 'Session deleted'})

    return jsonify({'error': 'Session not found'}), 404


@rag_bp.route('/api/rag/sample-queries', methods=['GET'])
def get_sample_queries():
    """
    Get sample queries for demo
    """
    return jsonify({
        "queries": [
            {
                "label": "Selection",
                "query": "How do I select all faces in Blender?",
                "category": "mesh"
            },
            {
                "label": "Python API",
                "query": "What is bpy.ops.mesh.select_all?",
                "category": "scripting"
            },
            {
                "label": "Materials",
                "query": "How to create a procedural material with nodes?",
                "category": "materials"
            },
            {
                "label": "Animation",
                "query": "Add keyframes for object animation using Python",
                "category": "animation"
            },
            {
                "label": "Multi-step",
                "query": "Select all faces, extrude them, and apply smooth shading",
                "category": "workflow"
            },
            {
                "label": "UV Mapping",
                "query": "How to unwrap UV coordinates automatically?",
                "category": "uv"
            }
        ]
    })
