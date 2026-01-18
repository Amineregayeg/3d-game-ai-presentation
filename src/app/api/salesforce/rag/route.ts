import { NextRequest, NextResponse } from "next/server";

// Backend URL - configurable via environment variable
const BACKEND_URL = process.env.FLASK_BACKEND_URL || "http://localhost:5000";

interface RAGQueryRequest {
  query: string;
  session_id?: string;
  include_mcp_suggestions?: boolean;
  stream?: boolean;
  // Avatar context for persona-aware responses
  avatar_id?: string;
  system_prompt?: string;
  expertise_level?: "beginner" | "intermediate" | "advanced";
  has_mcp?: boolean;
  language?: "en" | "fr";
}

interface RAGStageResult {
  stage: string;
  status: "pending" | "running" | "complete" | "error";
  duration_ms?: number;
  items_processed?: number;
  error?: string;
}

interface DocumentResult {
  id: string;
  title: string;
  content: string;
  source: string;
  rerank_score: number;
  chunk_index: number;
}

interface QueryAnalysis {
  original_query: string;
  intent: string;
  confidence: number;
  suggested_objects: string[];
  is_multi_step: boolean;
}

interface RAGMetrics {
  faithfulness: number;
  relevancy: number;
  completeness: number;
  composite_score: number;
}

interface RAGResponse {
  answer: string;
  query_analysis: QueryAnalysis;
  documents: DocumentResult[];
  stages: RAGStageResult[];
  metrics: RAGMetrics;
  mcp_suggestions?: Array<{
    operation: string;
    description: string;
    params?: Record<string, unknown>;
  }>;
  session_id: string;
  processing_time_ms: number;
  _demo?: boolean;
}

// Demo mode response for when backend is unavailable
function getDemoRAGResponse(
  query: string,
  expertiseLevel: string = "intermediate",
  hasMcp: boolean = true,
  language: string = "en"
): RAGResponse {
  const intent = query.toLowerCase().includes("field")
    ? "schema_modification"
    : query.toLowerCase().includes("flow")
    ? "automation"
    : query.toLowerCase().includes("apex")
    ? "development"
    : query.toLowerCase().includes("report")
    ? "analytics"
    : "general_inquiry";

  const suggestedObjects = ["Account", "Contact", "Opportunity"].slice(
    0,
    Math.floor(1 + Math.random() * 3)
  );

  // Generate expertise-appropriate response
  let answer: string;
  if (language === "fr") {
    // French responses
    if (expertiseLevel === "beginner") {
      answer = `Excellente question ! Laissez-moi vous expliquer simplement.\n\nPour "${query}", voici les étapes de base :\n\n1. Connectez-vous à votre organisation Salesforce\n2. Utilisez le menu Configuration (l'icône engrenage)\n3. Recherchez la fonctionnalité souhaitée\n\nN'hésitez pas à me demander plus de détails sur n'importe quelle étape !`;
    } else if (expertiseLevel === "advanced") {
      answer = `Analyse technique pour "${query}":\n\n**Architecture recommandée:**\n- Utiliser des Custom Metadata Types pour la configuration\n- Implémenter via Apex avec bulkification\n- Considérer les Governor Limits\n\n\`\`\`apex\n// Exemple de pattern\npublic class ${suggestedObjects[0]}Handler {\n    public static void process(List<${suggestedObjects[0]}> records) {\n        // Implementation\n    }\n}\n\`\`\`\n\nJe peux exécuter des opérations MCP si nécessaire.`;
    } else {
      answer = `Basé sur les meilleures pratiques Salesforce pour "${query}":\n\n1. **Modèle de données**: Considérez les implications\n2. **Sécurité**: Vérifiez les paramètres de partage\n3. **Test**: Utilisez une sandbox avant la production\n\nVoulez-vous que j'exécute une requête SOQL pour analyser vos données ?`;
    }
  } else {
    // English responses
    if (expertiseLevel === "beginner") {
      answer = `Great question! Let me explain this in simple terms.\n\nFor "${query}", here are the basic steps:\n\n1. Log into your Salesforce org\n2. Navigate to Setup (the gear icon)\n3. Search for the feature you need\n4. Follow the guided setup wizard\n\nWould you like me to walk you through any of these steps in more detail? I'm here to help you learn!`;
    } else if (expertiseLevel === "advanced") {
      answer = `Technical analysis for "${query}":\n\n**Recommended Architecture:**\n- Use Custom Metadata Types for configuration\n- Implement via Apex with proper bulkification\n- Consider Governor Limits (100 SOQL, 150 DML)\n- Apply appropriate sharing model\n\n\`\`\`apex\n// Recommended pattern\npublic class ${suggestedObjects[0]}Handler {\n    public static void process(List<${suggestedObjects[0]}> records) {\n        Map<Id, ${suggestedObjects[0]}> existingRecords = new Map<Id, ${suggestedObjects[0]}>(\n            [SELECT Id, Name FROM ${suggestedObjects[0]} WHERE Id IN :records]\n        );\n        // Bulk processing implementation\n    }\n}\n\`\`\`\n\nI can execute MCP operations directly. Want me to query your schema or create test data?`;
    } else {
      answer = `Based on my analysis of Salesforce best practices for "${query}":\n\n1. **Data Model**: Consider the implications for your object relationships\n2. **Security**: Review field-level security and sharing settings\n3. **Testing**: Always test in sandbox before production\n4. **Automation**: Consider using Flow Builder for declarative solutions\n\nWould you like me to run a SOQL query to analyze your current setup, or shall I help create a solution?`;
    }
  }

  return {
    answer,
    query_analysis: {
      original_query: query,
      intent,
      confidence: 0.85 + Math.random() * 0.1,
      suggested_objects: suggestedObjects,
      is_multi_step: query.length > 50,
    },
    documents: [
      {
        id: "doc-1",
        title: "Custom Fields Best Practices",
        content:
          "When creating custom fields, consider field dependencies, page layouts, and record types. Always plan for data migration and ensure proper field-level security settings.",
        source: "help_docs",
        rerank_score: 0.95,
        chunk_index: 0,
      },
      {
        id: "doc-2",
        title: "Trailhead: Data Modeling",
        content:
          "Learn how to design your data model for scalability and maintainability. Understand relationships between objects and when to use lookup vs master-detail.",
        source: "trailhead",
        rerank_score: 0.88,
        chunk_index: 0,
      },
      {
        id: "doc-3",
        title: "Apex Developer Guide",
        content:
          "Use triggers and classes to extend Salesforce functionality. Follow best practices for bulkification and governor limits.",
        source: "apex_guide",
        rerank_score: 0.82,
        chunk_index: 0,
      },
    ],
    stages: [
      { stage: "orchestration", status: "complete", duration_ms: 45, items_processed: 1 },
      { stage: "query_transform", status: "complete", duration_ms: 120, items_processed: 3 },
      { stage: "dense_retrieval", status: "complete", duration_ms: 85, items_processed: 50 },
      { stage: "sparse_retrieval", status: "complete", duration_ms: 65, items_processed: 30 },
      { stage: "fusion", status: "complete", duration_ms: 25, items_processed: 80 },
      { stage: "rerank", status: "complete", duration_ms: 180, items_processed: 20 },
      { stage: "validation", status: "complete", duration_ms: 50, items_processed: 5 },
      { stage: "generation", status: "complete", duration_ms: 800, items_processed: 1 },
      { stage: "evaluation", status: "complete", duration_ms: 100, items_processed: 1 },
    ],
    metrics: {
      faithfulness: 0.92,
      relevancy: 0.88,
      completeness: 0.85,
      composite_score: 0.88,
    },
    // Only include MCP suggestions if avatar has MCP capability
    mcp_suggestions: hasMcp
      ? [
          {
            operation: "describeGlobal",
            description: "List all available Salesforce objects",
          },
          {
            operation: "describe",
            description: `Get metadata for ${suggestedObjects[0]} object`,
            params: { objectType: suggestedObjects[0] },
          },
        ]
      : [], // Beginner avatars don't get MCP suggestions
    session_id: `demo-${Date.now()}`,
    processing_time_ms: 1500,
    _demo: true,
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as RAGQueryRequest;

    if (!body.query || typeof body.query !== "string") {
      return NextResponse.json(
        { error: "Query is required" },
        { status: 400 }
      );
    }

    // Extract avatar context for persona-aware responses
    const expertiseLevel = body.expertise_level || "intermediate";
    const hasMcp = body.has_mcp !== false; // Default to true
    const language = body.language || "en";

    // Try to call the Flask backend
    try {
      // Forward all avatar context to backend
      const backendResponse = await fetch(`${BACKEND_URL}/api/salesforce-rag/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: body.query,
          session_id: body.session_id,
          include_mcp_suggestions: body.include_mcp_suggestions,
          // Avatar context for persona-aware responses
          avatar_id: body.avatar_id,
          system_prompt: body.system_prompt,
          expertise_level: expertiseLevel,
          has_mcp: hasMcp,
          language: language,
        }),
        // Add timeout to prevent hanging (10 min for slow reranking on CPU)
        signal: AbortSignal.timeout(600000), // 10 minute timeout
      });

      if (backendResponse.ok) {
        const data = await backendResponse.json();
        // Map Flask API response format to frontend expected format
        const mappedResponse = {
          ...data,
          query_analysis: data.analysis ? {
            original_query: data.analysis.original_query || data.query,
            intent: data.analysis.intent || "general_inquiry",
            confidence: data.analysis.confidence || 0.85,
            suggested_objects: data.analysis.salesforce_objects || [],
            is_multi_step: data.analysis.query_variations?.length > 1 || false,
          } : undefined,
          // Map documents to include rerank_score
          documents: data.documents?.map((doc: Record<string, unknown>, index: number) => {
            const scores = doc.scores as Record<string, number> | undefined;
            return {
              ...doc,
              rerank_score: scores?.rerank || (doc.rerank_score as number) || 0.8,
              chunk_index: (doc.chunk_index as number) || index,
            };
          }),
          // Map stages format
          stages: data.stages?.map((stage: Record<string, unknown>) => {
            const details = stage.details as Record<string, unknown> | undefined;
            return {
              stage: stage.stage,
              status: stage.status,
              duration_ms: stage.duration_ms,
              items_processed: (details?.documents_found as number) || (details?.fused_documents as number) || 1,
            };
          }),
          // Map metrics
          metrics: data.metrics || data.validation || {
            faithfulness: 0.85,
            relevancy: 0.85,
            completeness: 0.85,
            composite_score: 0.85,
          },
          // Explicitly pass MCP result for operations panel
          mcp_result: data.mcp_result || null,
          latency: data.latency || {},
          processing_time_ms: data.latency?.total || 0,
        };
        return NextResponse.json(mappedResponse);
      } else {
        // Backend returned an error, fall back to demo mode
        console.warn(`Backend returned ${backendResponse.status}, using demo mode`);
        const demoResponse = getDemoRAGResponse(body.query, expertiseLevel, hasMcp, language);
        return NextResponse.json(demoResponse);
      }
    } catch (fetchError) {
      // Backend unavailable, use demo mode
      console.warn("Backend unavailable, using demo mode:", fetchError);
      const demoResponse = getDemoRAGResponse(body.query, expertiseLevel, hasMcp, language);
      return NextResponse.json(demoResponse);
    }
  } catch (error) {
    console.error("RAG API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "RAG query failed" },
      { status: 500 }
    );
  }
}

// GET endpoint for RAG status
export async function GET() {
  try {
    const backendResponse = await fetch(`${BACKEND_URL}/api/salesforce-rag/status`, {
      signal: AbortSignal.timeout(5000),
    });

    if (backendResponse.ok) {
      const data = await backendResponse.json();
      return NextResponse.json(data);
    }
  } catch {
    // Backend unavailable
  }

  // Return demo status
  return NextResponse.json({
    status: "demo",
    message: "Salesforce RAG API running in demo mode",
    backend_connected: false,
    pipeline_stages: [
      "orchestration",
      "query_transform",
      "dense_retrieval",
      "sparse_retrieval",
      "fusion",
      "rerank",
      "validation",
      "generation",
      "evaluation",
    ],
    sample_queries: [
      "How do I create a custom field on the Account object?",
      "What's the best way to implement a record-triggered flow?",
      "How do I write an Apex trigger for Opportunity?",
      "What are the governor limits for SOQL queries?",
    ],
  });
}
