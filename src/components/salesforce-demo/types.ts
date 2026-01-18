// Salesforce Demo Types

// Connection Types
export interface SalesforceOrg {
  id: string;
  name: string;
  instanceUrl: string;
  username: string;
  orgType: 'production' | 'sandbox' | 'developer';
  connectedAt: string;
}

export interface SalesforceConnectionStatus {
  connected: boolean;
  connecting: boolean;
  org?: SalesforceOrg;
  error?: string;
}

// MCP Operation Types
export interface SalesforceOperation {
  id: string;
  type: 'query' | 'insert' | 'update' | 'delete' | 'describe' | 'apex' | 'search';
  description: string;
  soql?: string;
  recordCount?: number;
  status: 'pending' | 'running' | 'success' | 'error';
  result?: unknown;
  error?: string;
  timestamp: number;
  duration_ms?: number;
}

// Conversation Types
export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  audioUrl?: string;
  toolCalls?: ToolCall[];
}

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
  output?: unknown;
  status: 'pending' | 'running' | 'success' | 'error';
}

// RAG Types for Salesforce
export interface SalesforceDocument {
  id: string;
  title: string;
  content: string;
  source: 'help_docs' | 'trailhead' | 'apex_guide' | 'admin_guide' | 'community';
  category?: 'configuration' | 'development' | 'automation' | 'security' | 'reporting' | 'integration';
  salesforce_version?: string;
  object_types?: string[];
  features?: string[];
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  dense_score?: number;
  sparse_score?: number;
  rrf_score?: number;
  rerank_score: number;
  chunk_index?: number;
}

export interface RAGStageResult {
  stage: string;
  status: 'pending' | 'running' | 'complete' | 'error';
  duration_ms?: number;
  results_count?: number;
  items_processed?: number;
}

export interface SalesforceRAGMetrics {
  faithfulness: number;
  relevancy: number;
  completeness: number;
  composite_score: number;
}

export interface SalesforceQueryAnalysis {
  original_query: string;
  intent: string;
  entities?: string[];
  confidence: number;
  is_multi_step: boolean;
  suggested_objects?: string[];
}

// ElevenLabs Conversational AI Types
export interface ConversationConfig {
  agentId: string;
  voiceId: string;
  systemPrompt: string;
  knowledgeBase: string[];
}

export interface ConversationState {
  status: 'idle' | 'connecting' | 'connected' | 'speaking' | 'listening' | 'processing' | 'error';
  isUserSpeaking: boolean;
  isAgentSpeaking: boolean;
  transcript: string;
  agentText: string;
}

// Settings Types
export interface SalesforceSettings {
  voiceId: string;
  enableAutoExecute: boolean;
  showDebugInfo: boolean;
  conversationMode: 'voice' | 'text' | 'both';
  theme: 'salesforce' | 'dark';
}

// Consultant Persona
export const CONSULTANT_PERSONA = {
  name: "Alex",
  role: "Senior Salesforce Consultant",
  experience: "15+ years, 10x certified",
  specializations: ["Sales Cloud", "Service Cloud", "CPQ", "Integration"],
  traits: {
    proactive: "Takes initiative, proposes solutions",
    confident: "Uses 'Here's the best approach...'",
    supportive: "Explains reasoning, celebrates wins",
    efficient: "Executes actions directly via MCP",
    educational: "Provides context without over-explaining"
  }
} as const;

// Voice Options (ElevenLabs voices)
export const ELEVENLABS_VOICES = [
  { id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel", gender: "female", style: "Professional" },
  { id: "EXAVITQu4vr4xnSDxMaL", name: "Sarah", gender: "female", style: "Friendly" },
  { id: "TxGEqnHWrfWFTfGW9XjX", name: "Josh", gender: "male", style: "Confident" },
  { id: "VR6AewLTigWG4xSOukaG", name: "Arnold", gender: "male", style: "Authoritative" },
  { id: "pNInz6obpgDQGcFmaJgB", name: "Adam", gender: "male", style: "Warm" },
] as const;

// Alias for backward compatibility
export const CONSULTANT_VOICES = ELEVENLABS_VOICES;

// Pipeline Stages for Salesforce RAG
export const SALESFORCE_PIPELINE_STAGES = [
  { id: "orchestration", label: "Orchestrator", color: "emerald", description: "Plans query execution strategy" },
  { id: "query_analysis", label: "Query Analysis", color: "cyan", description: "Extract Salesforce intent and objects" },
  { id: "retrieval_dense", label: "Dense Search", color: "purple", description: "Semantic vector similarity" },
  { id: "retrieval_sparse", label: "Sparse Search", color: "purple", description: "BM25 keyword retrieval" },
  { id: "rrf_fusion", label: "RRF Fusion", color: "amber", description: "Reciprocal Rank Fusion" },
  { id: "reranking", label: "Reranking", color: "orange", description: "Cross-encoder precision" },
  { id: "context_assembly", label: "Context", color: "blue", description: "Assemble Salesforce context" },
  { id: "generation", label: "Generation", color: "emerald", description: "Generate consultant response" },
  { id: "validation", label: "Validation", color: "green", description: "Verify accuracy and safety" },
] as const;
