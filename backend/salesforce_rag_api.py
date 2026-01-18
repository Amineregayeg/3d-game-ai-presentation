"""
Salesforce RAG API Module - Agentic RAG System for Salesforce Consultant
Adapted from rag_api_production.py for Salesforce documentation

This module implements a 7-layer agentic RAG pipeline for the Salesforce Virtual Assistant.
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
import logging

from flask import Blueprint, request, jsonify

# OpenAI
from openai import OpenAI

# Embeddings & Reranking
from sentence_transformers import SentenceTransformer, CrossEncoder

# Vector Database (FAISS-based)
import numpy as np
import pickle
from pathlib import Path
try:
    import faiss
except ImportError:
    faiss = None
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

salesforce_rag_bp = Blueprint('salesforce_rag', __name__)

# =============================================================================
# Configuration
# =============================================================================

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Model Configuration
MODEL_CONFIG = {
    "orchestrator": "gpt-4o",
    "query_analyzer": "gpt-4o-mini",
    "generator": "gpt-4o",
    "validator": "gpt-4o-mini"
}

# Data Directory (FAISS-based storage)
DATA_DIR = Path(__file__).parent / "salesforce_data"

# Lazy-loaded data stores
_faiss_index = None
_documents = None
_bm25_index = None
_tokenized_docs = None

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Reranker Model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# RRF Fusion constant
RRF_K = 60

# Session storage
SESSION_STORE: Dict[str, Dict] = {}

# =============================================================================
# Salesforce Consultant Persona
# =============================================================================

SALESFORCE_CONSULTANT_PROMPT = """You are Alex, a senior Salesforce consultant with 15+ years of hands-on experience. You've helped 200+ orgs optimize their Salesforce implementations.

RESPONSE RULES (CRITICAL):
1. Get straight to the point - NO filler phrases like "Based on my analysis" or "That's a great question"
2. Start your answer with the solution, not background
3. Be specific - give exact field names, API names, step numbers
4. Sound like you're talking to a colleague, not lecturing
5. Use "I recommend..." or "Here's what I'd do..." - take ownership

FORMAT GUIDELINES:
- Short paragraphs (2-3 sentences max)
- Use bullet points for steps or lists
- Include Apex/SOQL snippets when helpful (keep them concise)
- End with a concrete next step or question, never generic advice

TONE:
- Direct and confident ("Do this" not "You might want to consider")
- Friendly but professional (not overly formal or robotic)
- Acknowledge complexity honestly ("This can be tricky because...")
- Don't over-explain basics unless asked

EXAMPLE GOOD RESPONSE:
"To track overdue opportunities, add a formula checkbox field:

`Is_Overdue__c = CloseDate < TODAY() && NOT(IsClosed)`

Then create a list view filtering on this field. Want me to walk you through the automation to notify owners?"

EXAMPLE BAD RESPONSE:
"Based on my analysis of Salesforce best practices, tracking overdue opportunities is indeed a common requirement that many organizations face. There are several approaches you could consider..."

Ground your answers in the retrieved documentation. If you're unsure, say so directly."""

# =============================================================================
# Model Initialization (Lazy Loading)
# =============================================================================

_openai_client: Optional[OpenAI] = None
_embedding_model: Optional[SentenceTransformer] = None
_reranker_model: Optional[CrossEncoder] = None


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    return _openai_client


def get_embedding_model() -> SentenceTransformer:
    """Get or create embedding model"""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        # Force CPU to avoid CUDA compatibility issues
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        logger.info(f"Embedding model loaded. Dimension: {_embedding_model.get_sentence_embedding_dimension()}")
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    """Get or create reranker cross-encoder"""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        # Force CPU to avoid CUDA compatibility issues
        _reranker_model = CrossEncoder(RERANKER_MODEL, device="cpu")
        logger.info("Reranker model loaded")
    return _reranker_model


def load_faiss_index():
    """Load FAISS index from disk"""
    global _faiss_index
    if _faiss_index is None:
        index_path = DATA_DIR / "salesforce.index"
        if index_path.exists() and faiss:
            _faiss_index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index with {_faiss_index.ntotal} vectors")
    return _faiss_index


def load_documents():
    """Load documents metadata from disk"""
    global _documents
    if _documents is None:
        docs_path = DATA_DIR / "salesforce_documents.json"
        if docs_path.exists():
            with open(docs_path, "r") as f:
                _documents = json.load(f)
            logger.info(f"Loaded {len(_documents)} documents")
    return _documents


def load_bm25_index():
    """Load BM25 index from disk"""
    global _bm25_index, _tokenized_docs
    if _bm25_index is None:
        bm25_path = DATA_DIR / "salesforce_bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                _bm25_index, _tokenized_docs = pickle.load(f)
            logger.info("Loaded BM25 index")
    return _bm25_index, _tokenized_docs


def is_data_available():
    """Check if FAISS data is available"""
    index_path = DATA_DIR / "salesforce.index"
    docs_path = DATA_DIR / "salesforce_documents.json"
    return index_path.exists() and docs_path.exists()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QueryAnalysis:
    original_query: str
    intent: str
    entities: List[str]
    salesforce_objects: List[str]
    features: List[str]
    query_variations: List[str]
    is_code_query: bool
    is_apex_query: bool
    is_soql_query: bool
    is_flow_query: bool
    confidence: float
    # Action detection fields
    is_action_request: bool = False
    action_type: Optional[str] = None  # query, create, update, delete
    action_object: Optional[str] = None  # Account, Contact, etc.
    action_params: Optional[Dict] = None  # Field values or query conditions


@dataclass
class RetrievedDocument:
    id: int
    title: str
    content: str
    source: str
    category: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    is_apex: bool = False
    is_soql: bool = False


@dataclass
class RAGResponse:
    request_id: str
    session_id: str
    query: str
    analysis: Optional[Dict] = None
    documents: List[Dict] = field(default_factory=list)
    answer: str = ""
    citations: List[Dict] = field(default_factory=list)
    mcp_suggestions: List[Dict] = field(default_factory=list)
    mcp_result: Optional[Dict] = None  # Result of executed MCP action
    validation: Optional[Dict] = None
    metrics: Optional[Dict] = None
    latency: Dict = field(default_factory=dict)
    stages: List[Dict] = field(default_factory=list)
    status: str = "success"


# =============================================================================
# RAG Pipeline Functions
# =============================================================================

def analyze_query(query: str) -> QueryAnalysis:
    """Analyze the query to understand intent and extract entities"""
    start_time = time.time()

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=MODEL_CONFIG["query_analyzer"],
            messages=[
                {
                    "role": "system",
                    "content": """You are a Salesforce query analyzer. Analyze the user's question and extract:
1. intent: The main goal (configuration, development, troubleshooting, best_practices, integration, security, reporting, automation)
2. entities: Key concepts mentioned
3. salesforce_objects: Standard/custom objects mentioned (Account, Contact, Opportunity, Case, Lead, Custom__c, etc.)
4. features: Salesforce features mentioned (Flow, Apex, SOQL, LWC, Reports, Sharing, API, etc.)
5. query_variations: 2-3 alternative phrasings of the question
6. is_code_query: Whether the user wants code examples
7. is_apex_query: Whether Apex code is needed
8. is_soql_query: Whether SOQL is needed
9. is_flow_query: Whether Flow configuration is needed
10. confidence: How confident you are in the analysis (0-1)
11. is_action_request: TRUE if user wants to EXECUTE an action on their org (show records, create, update, delete), FALSE if just asking a question
12. action_type: If is_action_request=true, one of: "query", "create", "update", "delete". null otherwise.
13. action_object: If is_action_request=true, the Salesforce object (Account, Contact, Opportunity, etc.)
14. action_params: If is_action_request=true, extract parameters. For query: {"fields": [...], "conditions": "..."}. For create: field values.

Examples:
- "Show me all accounts" → is_action_request=true, action_type="query", action_object="Account", action_params={"fields":["Id","Name"]}
- "Create a contact named John Smith for Acme" → is_action_request=true, action_type="create", action_object="Contact", action_params={"FirstName":"John","LastName":"Smith"}
- "List contacts with email containing @gmail" → is_action_request=true, action_type="query", action_object="Contact", action_params={"fields":["Id","Name","Email"],"conditions":"Email LIKE '%@gmail%'"}
- "What is a custom object?" → is_action_request=false

Respond in JSON format."""
                },
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )

        result = json.loads(response.choices[0].message.content)
        duration = int((time.time() - start_time) * 1000)
        logger.info(f"Query analysis completed in {duration}ms")

        return QueryAnalysis(
            original_query=query,
            intent=result.get("intent", "general"),
            entities=result.get("entities", []),
            salesforce_objects=result.get("salesforce_objects", []),
            features=result.get("features", []),
            query_variations=result.get("query_variations", []),
            is_code_query=result.get("is_code_query", False),
            is_apex_query=result.get("is_apex_query", False),
            is_soql_query=result.get("is_soql_query", False),
            is_flow_query=result.get("is_flow_query", False),
            confidence=result.get("confidence", 0.8),
            is_action_request=result.get("is_action_request", False),
            action_type=result.get("action_type"),
            action_object=result.get("action_object"),
            action_params=result.get("action_params")
        )

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        # Return default analysis
        return QueryAnalysis(
            original_query=query,
            intent="general",
            entities=[],
            salesforce_objects=[],
            features=[],
            query_variations=[query],
            is_code_query="code" in query.lower() or "apex" in query.lower(),
            is_apex_query="apex" in query.lower(),
            is_soql_query="soql" in query.lower() or "query" in query.lower(),
            is_flow_query="flow" in query.lower(),
            confidence=0.5,
            is_action_request=False,
            action_type=None,
            action_object=None,
            action_params=None
        )


def execute_mcp_action(analysis: QueryAnalysis) -> Optional[Dict]:
    """Execute MCP action if the query is an action request"""
    if not analysis.is_action_request or not analysis.action_type:
        return None

    try:
        from salesforce_service import get_salesforce_service
        service = get_salesforce_service()

        action_type = analysis.action_type.lower()
        obj_name = analysis.action_object or "Account"
        params = analysis.action_params or {}

        logger.info(f"Executing MCP action: {action_type} on {obj_name} with params: {params}")

        if action_type == "query":
            # Build SOQL query
            fields = params.get("fields", ["Id", "Name"])
            if isinstance(fields, list):
                fields_str = ", ".join(fields)
            else:
                fields_str = "Id, Name"

            conditions = params.get("conditions", "")
            soql = f"SELECT {fields_str} FROM {obj_name}"
            if conditions:
                soql += f" WHERE {conditions}"
            soql += " LIMIT 10"

            result = service.query(soql)
            return {
                "action": "query",
                "success": True,
                "object": obj_name,
                "soql": soql,
                "totalSize": result.get("totalSize", 0),
                "records": result.get("records", [])
            }

        elif action_type == "create":
            result = service.create_record(obj_name, params)
            return {
                "action": "create",
                "success": True,
                "object": obj_name,
                "id": result.get("id"),
                "message": f"Successfully created {obj_name} record"
            }

        elif action_type == "update":
            record_id = params.pop("Id", params.pop("id", None))
            if not record_id:
                return {"action": "update", "success": False, "error": "Record ID required for update"}
            result = service.update_record(obj_name, record_id, params)
            return {
                "action": "update",
                "success": True,
                "object": obj_name,
                "id": record_id,
                "message": f"Successfully updated {obj_name} record"
            }

        elif action_type == "delete":
            record_id = params.get("Id") or params.get("id")
            if not record_id:
                return {"action": "delete", "success": False, "error": "Record ID required for delete"}
            result = service.delete_record(obj_name, record_id)
            return {
                "action": "delete",
                "success": True,
                "object": obj_name,
                "id": record_id,
                "message": f"Successfully deleted {obj_name} record"
            }

        return None

    except Exception as e:
        logger.error(f"MCP action failed: {e}")
        return {
            "action": analysis.action_type,
            "success": False,
            "error": str(e)
        }


def dense_retrieval(query: str, top_k: int = 50) -> List[RetrievedDocument]:
    """Perform dense vector retrieval using FAISS"""
    start_time = time.time()

    index = load_faiss_index()
    documents = load_documents()

    if not index or not documents:
        logger.warning("No FAISS index available, returning empty results")
        return []

    try:
        model = get_embedding_model()
        query_embedding = model.encode(query, normalize_embeddings=True)
        query_vector = np.array([query_embedding]).astype('float32')

        # Search FAISS index
        scores, indices = index.search(query_vector, min(top_k, len(documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(documents):
                doc = documents[idx]
                results.append(RetrievedDocument(
                    id=doc.get('id', idx),
                    title=doc.get('title', 'Untitled'),
                    content=doc.get('content', ''),
                    source=doc.get('source', 'unknown'),
                    category=doc.get('category', 'general'),
                    dense_score=float(score),
                    is_apex=doc.get('is_apex', False),
                    is_soql=doc.get('is_soql', False)
                ))

        duration = int((time.time() - start_time) * 1000)
        logger.info(f"Dense retrieval: {len(results)} docs in {duration}ms")
        return results

    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
        return []


def sparse_retrieval(query: str, top_k: int = 50) -> List[RetrievedDocument]:
    """Perform sparse (BM25) retrieval"""
    start_time = time.time()

    bm25, tokenized_docs = load_bm25_index()
    documents = load_documents()

    if not bm25 or not documents:
        logger.warning("No BM25 index available, returning empty results")
        return []

    try:
        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = documents[idx]
                results.append(RetrievedDocument(
                    id=doc.get('id', idx),
                    title=doc.get('title', 'Untitled'),
                    content=doc.get('content', ''),
                    source=doc.get('source', 'unknown'),
                    category=doc.get('category', 'general'),
                    sparse_score=float(scores[idx]),
                    is_apex=doc.get('is_apex', False),
                    is_soql=doc.get('is_soql', False)
                ))

        duration = int((time.time() - start_time) * 1000)
        logger.info(f"Sparse retrieval: {len(results)} docs in {duration}ms")
        return results

    except Exception as e:
        logger.error(f"Sparse retrieval failed: {e}")
        return []


def rrf_fusion(dense_results: List[RetrievedDocument],
               sparse_results: List[RetrievedDocument],
               k: int = RRF_K) -> List[RetrievedDocument]:
    """Combine results using Reciprocal Rank Fusion"""
    start_time = time.time()

    # Build ID to document mapping
    doc_map: Dict[int, RetrievedDocument] = {}
    rrf_scores: Dict[int, float] = {}

    # Process dense results
    for rank, doc in enumerate(dense_results, 1):
        doc_map[doc.id] = doc
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1.0 / (k + rank)

    # Process sparse results
    for rank, doc in enumerate(sparse_results, 1):
        if doc.id not in doc_map:
            doc_map[doc.id] = doc
        else:
            # Merge scores
            doc_map[doc.id].sparse_score = doc.sparse_score
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1.0 / (k + rank)

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Build result list with RRF scores
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        doc.rrf_score = rrf_scores[doc_id]
        results.append(doc)

    duration = int((time.time() - start_time) * 1000)
    logger.info(f"RRF fusion: {len(results)} docs in {duration}ms")
    return results


def rerank_documents(query: str, documents: List[RetrievedDocument], top_k: int = 10) -> List[RetrievedDocument]:
    """Rerank documents using cross-encoder"""
    start_time = time.time()

    if not documents:
        return []

    try:
        reranker = get_reranker_model()

        # Prepare pairs for reranking
        pairs = [[query, doc.content[:1000]] for doc in documents[:50]]  # Limit to top 50

        # Get rerank scores
        scores = reranker.predict(pairs)

        # Update documents with rerank scores
        for i, score in enumerate(scores):
            if i < len(documents):
                documents[i].rerank_score = float(score)

        # Sort by rerank score
        documents.sort(key=lambda x: x.rerank_score, reverse=True)

        duration = int((time.time() - start_time) * 1000)
        logger.info(f"Reranking: {len(documents[:top_k])} docs in {duration}ms")
        return documents[:top_k]

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return documents[:top_k]


def generate_answer(query: str, analysis: QueryAnalysis,
                    documents: List[RetrievedDocument], session_history: List[Dict] = None,
                    system_prompt: str = None, expertise_level: str = "intermediate") -> Tuple[str, List[Dict]]:
    """Generate answer using GPT with avatar-specific persona"""
    start_time = time.time()

    # Use provided system prompt or fallback to default
    persona_prompt = system_prompt or SALESFORCE_CONSULTANT_PROMPT

    # Add expertise-specific guidance
    expertise_guidance = ""
    if expertise_level == "beginner":
        expertise_guidance = "\n\nIMPORTANT: This user is a beginner. Use simple language, avoid jargon, explain concepts step by step, and be encouraging. Do NOT include complex code examples unless specifically asked."
    elif expertise_level == "advanced":
        expertise_guidance = "\n\nIMPORTANT: This is an advanced user. Feel free to include technical details, Apex code examples, architecture considerations, and discuss governor limits or performance implications."

    full_prompt = persona_prompt + expertise_guidance

    # Build context from documents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"[Source {i}] {doc.title}\n{doc.content[:2000]}")

    context = "\n\n---\n\n".join(context_parts)

    # Build conversation history
    messages = [
        {"role": "system", "content": full_prompt},
        {"role": "system", "content": f"Retrieved Documentation Context:\n\n{context}"}
    ]

    if session_history:
        for turn in session_history[-5:]:  # Last 5 turns
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=MODEL_CONFIG["generator"],
            messages=messages,
            max_tokens=1500,
            temperature=0.4  # Lower for consistent, direct consultant tone
        )

        answer = response.choices[0].message.content

        # Extract citations (simple pattern matching)
        citations = []
        citation_pattern = r'\[Source (\d+)\]'
        matches = re.findall(citation_pattern, answer)
        for match in set(matches):
            idx = int(match) - 1
            if 0 <= idx < len(documents):
                citations.append({
                    "index": int(match),
                    "doc_id": documents[idx].id,
                    "title": documents[idx].title
                })

        duration = int((time.time() - start_time) * 1000)
        logger.info(f"Answer generated in {duration}ms")
        return answer, citations

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"I apologize, but I encountered an error generating the answer: {str(e)}", []


def format_mcp_result_for_answer(mcp_result: Dict) -> str:
    """Format MCP result for inclusion in answer"""
    if not mcp_result:
        return ""

    action = mcp_result.get("action", "unknown")

    if action == "query":
        records = mcp_result.get("records", [])
        if not records:
            return "No records found matching your criteria."

        result_text = f"Found {len(records)} record(s):\n"
        for i, record in enumerate(records[:5], 1):
            # Get display fields (exclude metadata)
            display_fields = {k: v for k, v in record.items() if k != "attributes"}
            result_text += f"{i}. " + ", ".join([f"{k}: {v}" for k, v in display_fields.items()]) + "\n"

        if len(records) > 5:
            result_text += f"... and {len(records) - 5} more records."
        return result_text

    elif action == "create":
        return f"Successfully created {mcp_result.get('object')} record with ID: {mcp_result.get('id')}"

    elif action == "update":
        return f"Successfully updated {mcp_result.get('object')} record: {mcp_result.get('id')}"

    elif action == "delete":
        return f"Successfully deleted {mcp_result.get('object')} record: {mcp_result.get('id')}"

    return str(mcp_result)


def generate_answer_with_mcp(query: str, analysis: QueryAnalysis,
                             documents: List[RetrievedDocument], mcp_result: Dict,
                             session_history: List[Dict] = None, system_prompt: str = None) -> Tuple[str, List[Dict]]:
    """Generate answer that incorporates MCP execution results"""
    start_time = time.time()

    # Use provided system prompt or fallback to default
    persona_prompt = system_prompt or SALESFORCE_CONSULTANT_PROMPT

    # Format MCP result
    mcp_formatted = format_mcp_result_for_answer(mcp_result)

    # Build context from documents
    context_parts = []
    for i, doc in enumerate(documents[:5], 1):  # Fewer docs since we have MCP results
        context_parts.append(f"[Source {i}] {doc.title}\n{doc.content[:1000]}")
    context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "system", "content": f"""I just executed a Salesforce operation for the user. Here are the results:

ACTION EXECUTED: {mcp_result.get('action', 'unknown').upper()}
OBJECT: {mcp_result.get('object', 'N/A')}
SUCCESS: {mcp_result.get('success', False)}

RESULTS:
{mcp_formatted}

Additional context from documentation:
{context}

Present these results conversationally. Start by confirming what was done, show the results clearly, and offer next steps."""}
    ]

    if session_history:
        for turn in session_history[-3:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=MODEL_CONFIG["generator"],
            messages=messages,
            max_tokens=1500,
            temperature=0.4
        )

        answer = response.choices[0].message.content
        duration = int((time.time() - start_time) * 1000)
        logger.info(f"MCP answer generated in {duration}ms")
        return answer, []

    except Exception as e:
        logger.error(f"MCP answer generation failed: {e}")
        # Return formatted result directly if generation fails
        return f"I executed the {mcp_result.get('action')} operation.\n\n{mcp_formatted}", []


def validate_answer(query: str, answer: str, documents: List[RetrievedDocument]) -> Dict:
    """Validate the answer for faithfulness and relevancy"""
    start_time = time.time()

    context = "\n".join([doc.content[:500] for doc in documents[:5]])

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=MODEL_CONFIG["validator"],
            messages=[
                {
                    "role": "system",
                    "content": """Evaluate the answer quality. Return JSON with:
- faithfulness: 0-1, is the answer grounded in the context?
- relevancy: 0-1, does it address the question?
- completeness: 0-1, is it comprehensive?
- has_code: boolean, does it include code when needed?
- issues: list of any issues found"""
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext: {context}\n\nAnswer: {answer}"
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )

        result = json.loads(response.choices[0].message.content)

        composite_score = (
            result.get("faithfulness", 0.8) * 0.4 +
            result.get("relevancy", 0.8) * 0.4 +
            result.get("completeness", 0.8) * 0.2
        )

        duration = int((time.time() - start_time) * 1000)
        logger.info(f"Validation completed in {duration}ms, score: {composite_score:.2f}")

        return {
            "faithfulness": result.get("faithfulness", 0.8),
            "relevancy": result.get("relevancy", 0.8),
            "completeness": result.get("completeness", 0.8),
            "has_code": result.get("has_code", False),
            "composite_score": composite_score,
            "passed": composite_score >= 0.6,
            "issues": result.get("issues", [])
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "faithfulness": 0.7,
            "relevancy": 0.7,
            "completeness": 0.7,
            "composite_score": 0.7,
            "passed": True,
            "issues": []
        }


def suggest_mcp_operations(analysis: QueryAnalysis, answer: str) -> List[Dict]:
    """Suggest Salesforce MCP operations based on the query and answer"""
    suggestions = []

    # Based on intent
    if analysis.intent == "configuration":
        if "custom field" in analysis.original_query.lower():
            suggestions.append({
                "operation": "describe",
                "params": {"objectType": analysis.salesforce_objects[0] if analysis.salesforce_objects else "Account"},
                "description": "Get object metadata to understand existing fields"
            })
        if "create" in analysis.original_query.lower():
            suggestions.append({
                "operation": "createField",
                "params": {"objectType": analysis.salesforce_objects[0] if analysis.salesforce_objects else "Account"},
                "description": "Create a new custom field"
            })

    if analysis.is_soql_query:
        suggestions.append({
            "operation": "query",
            "params": {"soql": "SELECT Id, Name FROM Account LIMIT 10"},
            "description": "Execute sample query"
        })

    if "report" in analysis.original_query.lower():
        suggestions.append({
            "operation": "describeGlobal",
            "params": {},
            "description": "Get list of available objects for reporting"
        })

    return suggestions


# =============================================================================
# Main RAG Pipeline
# =============================================================================

def process_rag_query(query: str, session_id: Optional[str] = None,
                      settings: Optional[Dict] = None,
                      avatar_context: Optional[Dict] = None) -> RAGResponse:
    """Main RAG pipeline for Salesforce queries with avatar-aware responses"""

    request_id = f"sfrag_{int(time.time() * 1000)}"
    session_id = session_id or f"session_{int(time.time() * 1000)}"

    # Extract avatar context with defaults
    avatar_context = avatar_context or {}
    system_prompt = avatar_context.get('system_prompt') or SALESFORCE_CONSULTANT_PROMPT
    expertise_level = avatar_context.get('expertise_level', 'intermediate')
    has_mcp = avatar_context.get('has_mcp', True)
    language = avatar_context.get('language', 'en')
    avatar_id = avatar_context.get('avatar_id')

    logger.info(f"Processing query for avatar: {avatar_id}, expertise: {expertise_level}, has_mcp: {has_mcp}")

    response = RAGResponse(
        request_id=request_id,
        session_id=session_id,
        query=query
    )

    stages = []
    latency = {}
    total_start = time.time()

    try:
        # Stage 1: Query Analysis
        stage_start = time.time()
        analysis = analyze_query(query)
        latency["query_analysis"] = int((time.time() - stage_start) * 1000)
        stages.append({
            "stage": "query_analysis",
            "status": "complete",
            "duration_ms": latency["query_analysis"],
            "details": {
                "intent": analysis.intent,
                "confidence": analysis.confidence,
                "is_action_request": analysis.is_action_request,
                "action_type": analysis.action_type
            }
        })
        response.analysis = asdict(analysis)

        # Stage 1.5: Execute MCP Action if requested AND avatar has MCP capability
        mcp_result = None
        if analysis.is_action_request and has_mcp:
            stage_start = time.time()
            mcp_result = execute_mcp_action(analysis)
            latency["mcp_execution"] = int((time.time() - stage_start) * 1000)
            stages.append({
                "stage": "mcp_execution",
                "status": "complete" if mcp_result and mcp_result.get("success") else "error",
                "duration_ms": latency["mcp_execution"],
                "details": {
                    "action": analysis.action_type,
                    "object": analysis.action_object,
                    "success": mcp_result.get("success") if mcp_result else False
                }
            })
            response.mcp_result = mcp_result
            logger.info(f"MCP action executed: {analysis.action_type} on {analysis.action_object}")

        # Stage 2: Dense Retrieval
        stage_start = time.time()
        dense_docs = dense_retrieval(query, top_k=50)
        latency["dense_retrieval"] = int((time.time() - stage_start) * 1000)
        stages.append({
            "stage": "dense_retrieval",
            "status": "complete",
            "duration_ms": latency["dense_retrieval"],
            "details": {"documents_found": len(dense_docs)}
        })

        # Stage 3: Sparse Retrieval
        stage_start = time.time()
        sparse_docs = sparse_retrieval(query, top_k=50)
        latency["sparse_retrieval"] = int((time.time() - stage_start) * 1000)
        stages.append({
            "stage": "sparse_retrieval",
            "status": "complete",
            "duration_ms": latency["sparse_retrieval"],
            "details": {"documents_found": len(sparse_docs)}
        })

        # Stage 4: RRF Fusion
        stage_start = time.time()
        fused_docs = rrf_fusion(dense_docs, sparse_docs)
        latency["rrf_fusion"] = int((time.time() - stage_start) * 1000)
        stages.append({
            "stage": "rrf_fusion",
            "status": "complete",
            "duration_ms": latency["rrf_fusion"],
            "details": {"fused_documents": len(fused_docs)}
        })

        # Stage 5: Reranking
        stage_start = time.time()
        reranked_docs = rerank_documents(query, fused_docs, top_k=10)
        latency["reranking"] = int((time.time() - stage_start) * 1000)
        stages.append({
            "stage": "reranking",
            "status": "complete",
            "duration_ms": latency["reranking"],
            "details": {"final_documents": len(reranked_docs)}
        })

        # Convert to dict for response
        response.documents = [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "source": doc.source,
                "category": doc.category,
                "scores": {
                    "dense": doc.dense_score,
                    "sparse": doc.sparse_score,
                    "rrf": doc.rrf_score,
                    "rerank": doc.rerank_score
                }
            }
            for doc in reranked_docs
        ]

        # Stage 6: Answer Generation (with avatar persona)
        stage_start = time.time()
        session_history = SESSION_STORE.get(session_id, {}).get("history", [])

        # If MCP action was executed, include results in context
        if mcp_result and mcp_result.get("success"):
            mcp_context = format_mcp_result_for_answer(mcp_result)
            answer, citations = generate_answer_with_mcp(query, analysis, reranked_docs, mcp_result, session_history, system_prompt)
        else:
            # Pass avatar's system prompt and expertise level
            answer, citations = generate_answer(query, analysis, reranked_docs, session_history, system_prompt, expertise_level)

        latency["generation"] = int((time.time() - stage_start) * 1000)
        stages.append({
            "stage": "generation",
            "status": "complete",
            "duration_ms": latency["generation"],
            "details": {"answer_length": len(answer), "citations": len(citations), "has_mcp_result": mcp_result is not None}
        })
        response.answer = answer
        response.citations = citations

        # Stage 7: Validation
        stage_start = time.time()
        validation = validate_answer(query, answer, reranked_docs)
        latency["validation"] = int((time.time() - stage_start) * 1000)
        stages.append({
            "stage": "validation",
            "status": "complete",
            "duration_ms": latency["validation"],
            "details": {"score": validation["composite_score"], "passed": validation["passed"]}
        })
        response.validation = validation
        response.metrics = {
            "faithfulness": validation["faithfulness"],
            "relevancy": validation["relevancy"],
            "completeness": validation["completeness"],
            "composite_score": validation["composite_score"]
        }

        # MCP Suggestions
        response.mcp_suggestions = suggest_mcp_operations(analysis, answer)

        # Update session
        if session_id not in SESSION_STORE:
            SESSION_STORE[session_id] = {"history": [], "created_at": datetime.now().isoformat()}

        SESSION_STORE[session_id]["history"].append({"role": "user", "content": query})
        SESSION_STORE[session_id]["history"].append({"role": "assistant", "content": answer})
        SESSION_STORE[session_id]["updated_at"] = datetime.now().isoformat()

        # Final timing
        latency["total"] = int((time.time() - total_start) * 1000)
        response.latency = latency
        response.stages = stages
        response.status = "success"

        logger.info(f"RAG query completed in {latency['total']}ms")
        return response

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        response.status = "error"
        response.answer = f"I apologize, but I encountered an error: {str(e)}"
        response.latency = {"total": int((time.time() - total_start) * 1000)}
        return response


# =============================================================================
# Demo Mode (No Database)
# =============================================================================

DEMO_DOCUMENTS = [
    {
        "id": 1,
        "title": "Creating Custom Fields on Account",
        "content": "To create a custom field: Setup > Object Manager > Account > Fields & Relationships > New. Select field type, configure properties, set security, add to layouts.",
        "source": "help_docs",
        "category": "configuration",
        "scores": {"dense": 0.92, "sparse": 0.85, "rrf": 0.045, "rerank": 0.95}
    },
    {
        "id": 2,
        "title": "Apex Triggers Best Practices",
        "content": "Use one trigger per object with a handler class. Always bulkify - avoid SOQL/DML in loops. Use Trigger.new, Trigger.oldMap for context.",
        "source": "apex_guide",
        "category": "development",
        "scores": {"dense": 0.88, "sparse": 0.82, "rrf": 0.042, "rerank": 0.91}
    },
    {
        "id": 3,
        "title": "Record-Triggered Flows",
        "content": "Use Before Save for field updates (no DML). Use After Save for related records. Always add entry conditions to filter records.",
        "source": "trailhead",
        "category": "automation",
        "scores": {"dense": 0.85, "sparse": 0.80, "rrf": 0.040, "rerank": 0.88}
    }
]


def process_demo_query(query: str, avatar_context: Optional[Dict] = None) -> Dict:
    """Process query in demo mode without database, with avatar-aware responses"""
    import random

    # Extract avatar context
    avatar_context = avatar_context or {}
    expertise_level = avatar_context.get('expertise_level', 'intermediate')
    has_mcp = avatar_context.get('has_mcp', True)
    language = avatar_context.get('language', 'en')
    avatar_name = avatar_context.get('name', 'your Salesforce consultant')

    # Simulate processing
    time.sleep(0.3)

    # Greeting detection - respond conversationally to greetings
    query_lower = query.lower().strip()
    greeting_keywords = ['hello', 'hi', 'hey', 'bonjour', 'salut', 'help', 'who are you', 'what can you do', 'how can you help']
    is_greeting = any(kw in query_lower for kw in greeting_keywords) and len(query.split()) < 12

    if is_greeting:
        if language == 'fr':
            greeting_response = f"""Bonjour ! Je suis {avatar_name}, votre consultant Salesforce personnel.

Je peux vous aider avec:
- **Configuration Salesforce** - Champs, objets, automatisations
- **Meilleures pratiques** - Flows, Apex, intégrations
- **Questions techniques** - Requêtes SOQL, governor limits
- **Architecture** - Conception de solutions

Comment puis-je vous aider aujourd'hui ?"""
        else:
            greeting_response = f"""Hi there! I'm {avatar_name}, your personal Salesforce consultant.

I can help you with:
- **Salesforce Setup** - Fields, objects, automations
- **Best Practices** - Flows, Apex, integrations
- **Technical Questions** - SOQL queries, governor limits
- **Architecture** - Solution design and recommendations

What would you like to work on today?"""

        return {
            "request_id": f"demo_{int(time.time() * 1000)}",
            "session_id": "demo_session",
            "query": query,
            "analysis": {
                "intent": "greeting",
                "entities": [],
                "salesforce_objects": [],
                "confidence": 0.95
            },
            "documents": [],
            "answer": greeting_response,
            "citations": [],
            "mcp_suggestions": [],
            "processing_time": 0.3,
            "demo_mode": True
        }

    # Simple keyword matching for demo
    docs = DEMO_DOCUMENTS.copy()
    for doc in docs:
        # Boost score if query matches content
        if any(word in doc["content"].lower() for word in query.lower().split()):
            doc["scores"]["rerank"] = min(1.0, doc["scores"]["rerank"] + 0.1)

    docs.sort(key=lambda x: x["scores"]["rerank"], reverse=True)

    # Generate expertise-appropriate demo answer
    if language == 'fr':
        if expertise_level == 'beginner':
            answer = f"""Excellente question ! Voici mon conseil pour "{query}":

Je vais vous expliquer pas à pas:

1. **Première étape** - Allez dans Configuration (icône engrenage)
2. **Cherchez** - Utilisez la barre de recherche
3. **Suivez l'assistant** - Salesforce vous guide

Voulez-vous que j'explique une étape en particulier ?

[Source 1] {docs[0]['title']}"""
        elif expertise_level == 'advanced':
            answer = f"""Analyse technique pour "{query}":

**Architecture recommandée:**
- Utilisez les Custom Metadata Types pour la configuration
- Implémentez via Apex avec bulkification
- Considérez les Governor Limits (100 SOQL, 150 DML)

```apex
// Pattern recommandé
public class AccountHandler {{
    public static void process(List<Account> records) {{
        // Implementation avec collections
    }}
}}
```

Je peux exécuter des opérations MCP directement. Voulez-vous que je requête votre schéma ?

[Source 1] {docs[0]['title']}"""
        else:
            answer = f"""Basé sur les meilleures pratiques Salesforce pour "{query}":

1. **Comprenez le besoin** - Clarifiez les exigences métier
2. **Vérifiez l'existant** - Salesforce a peut-être déjà cette fonctionnalité
3. **Meilleures pratiques** - Privilégiez le déclaratif (Flows) avant le code
4. **Testez** - Validez toujours en sandbox d'abord

Voulez-vous que j'élabore sur un de ces points ?

[Source 1] {docs[0]['title']}"""
    else:
        if expertise_level == 'beginner':
            answer = f"""Great question! Let me explain this simply for "{query}":

Here are the basic steps:

1. **First step** - Go to Setup (the gear icon)
2. **Search** - Use the search bar to find what you need
3. **Follow the wizard** - Salesforce will guide you through

Would you like me to walk you through any of these steps in more detail? I'm here to help!

[Source 1] {docs[0]['title']}"""
        elif expertise_level == 'advanced':
            answer = f"""Technical analysis for "{query}":

**Recommended Architecture:**
- Use Custom Metadata Types for configuration
- Implement via Apex with proper bulkification
- Consider Governor Limits (100 SOQL, 150 DML)
- Apply appropriate sharing model

```apex
// Recommended pattern
public class AccountHandler {{
    public static void process(List<Account> records) {{
        Map<Id, Account> existingRecords = new Map<Id, Account>(
            [SELECT Id, Name FROM Account WHERE Id IN :records]
        );
        // Bulk processing implementation
    }}
}}
```

I can execute MCP operations directly. Want me to query your schema or create test data?

[Source 1] {docs[0]['title']}"""
        else:
            answer = f"""Based on my analysis of your question about "{query}", here's my recommendation:

1. **Understand the requirement** - Make sure you're clear on the business need
2. **Check existing functionality** - Salesforce may already have a feature for this
3. **Follow best practices** - Use declarative solutions (Flows) before code (Apex)
4. **Test thoroughly** - Always validate in a sandbox first

Would you like me to elaborate on any of these points or help with implementation?

[Source 1] {docs[0]['title']}"""

    return {
        "request_id": f"demo_{int(time.time() * 1000)}",
        "session_id": "demo_session",
        "query": query,
        "analysis": {
            "intent": "general",
            "entities": query.split()[:3],
            "salesforce_objects": ["Account"],
            "confidence": 0.85
        },
        "documents": docs,
        "answer": answer,
        "citations": [{"index": 1, "doc_id": docs[0]["id"], "title": docs[0]["title"]}],
        # Only include MCP suggestions if avatar has MCP capability
        "mcp_suggestions": [
            {"operation": "describe", "params": {"objectType": "Account"}, "description": "Get Account metadata"}
        ] if has_mcp else [],
        "validation": {
            "faithfulness": 0.88,
            "relevancy": 0.90,
            "completeness": 0.85,
            "composite_score": 0.88,
            "passed": True
        },
        "metrics": {
            "faithfulness": 0.88,
            "relevancy": 0.90,
            "completeness": 0.85,
            "composite_score": 0.88
        },
        "latency": {
            "query_analysis": 120,
            "dense_retrieval": 85,
            "sparse_retrieval": 45,
            "rrf_fusion": 5,
            "reranking": 180,
            "generation": 450,
            "validation": 95,
            "total": 980
        },
        "stages": [
            {"stage": "query_analysis", "status": "complete", "duration_ms": 120},
            {"stage": "dense_retrieval", "status": "complete", "duration_ms": 85},
            {"stage": "sparse_retrieval", "status": "complete", "duration_ms": 45},
            {"stage": "rrf_fusion", "status": "complete", "duration_ms": 5},
            {"stage": "reranking", "status": "complete", "duration_ms": 180},
            {"stage": "generation", "status": "complete", "duration_ms": 450},
            {"stage": "validation", "status": "complete", "duration_ms": 95}
        ],
        "status": "success",
        "_demo": True
    }


# =============================================================================
# Flask Routes
# =============================================================================

@salesforce_rag_bp.route('/api/salesforce-rag/query', methods=['POST'])
def handle_query():
    """Handle Salesforce RAG query"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id')
        settings = data.get('settings', {})

        # Extract avatar context for persona-aware responses
        avatar_context = {
            'avatar_id': data.get('avatar_id'),
            'name': data.get('avatar_name', data.get('name', 'your Salesforce consultant')),
            'system_prompt': data.get('system_prompt'),
            'expertise_level': data.get('expertise_level', 'intermediate'),
            'has_mcp': data.get('has_mcp', True),
            'language': data.get('language', 'en')
        }

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Check if we have FAISS data available
        if is_data_available():
            # Full RAG pipeline with avatar context
            response = process_rag_query(query, session_id, settings, avatar_context)
            return jsonify(asdict(response))
        else:
            # Demo mode with avatar context
            logger.info("Running in demo mode (no data indexed)")
            response = process_demo_query(query, avatar_context)
            return jsonify(response)

    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@salesforce_rag_bp.route('/api/salesforce-rag/status', methods=['GET'])
def get_status():
    """Get RAG system status"""
    data_available = is_data_available()
    documents = load_documents() if data_available else None
    doc_count = len(documents) if documents else 0

    return jsonify({
        "status": "healthy",
        "data_available": data_available,
        "document_count": doc_count,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "openai_configured": bool(OPENAI_API_KEY),
        "active_sessions": len(SESSION_STORE),
        "demo_mode": not data_available,
        "storage_type": "faiss"
    })


@salesforce_rag_bp.route('/api/salesforce-rag/sessions', methods=['GET'])
def list_sessions():
    """List active sessions"""
    sessions = [
        {
            "id": sid,
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": len(data.get("history", []))
        }
        for sid, data in SESSION_STORE.items()
    ]
    return jsonify({"sessions": sessions})


@salesforce_rag_bp.route('/api/salesforce-rag/sessions/<session_id>', methods=['GET', 'DELETE'])
def handle_session(session_id: str):
    """Get or delete a session"""
    if request.method == 'DELETE':
        if session_id in SESSION_STORE:
            del SESSION_STORE[session_id]
            return jsonify({"success": True, "message": f"Session {session_id} deleted"})
        return jsonify({"error": "Session not found"}), 404

    # GET
    if session_id in SESSION_STORE:
        return jsonify({
            "session_id": session_id,
            **SESSION_STORE[session_id]
        })
    return jsonify({"error": "Session not found"}), 404


@salesforce_rag_bp.route('/api/salesforce-rag/sample-queries', methods=['GET'])
def get_sample_queries():
    """Get sample Salesforce queries for testing"""
    return jsonify({
        "queries": [
            "How do I create a custom field on the Account object?",
            "What are best practices for Apex triggers?",
            "How do I set up a Record-Triggered Flow?",
            "Can you show me how to write a SOQL query with relationships?",
            "How does the Salesforce sharing model work?",
            "What's the best way to integrate with the Salesforce REST API?",
            "How do I create a report that shows opportunities by stage?",
            "What are the differences between Before Save and After Save flows?",
            "How do I build a Lightning Web Component?",
            "What are governor limits in Apex?"
        ]
    })


# For testing
if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(salesforce_rag_bp)
    app.run(debug=True, port=5001)
