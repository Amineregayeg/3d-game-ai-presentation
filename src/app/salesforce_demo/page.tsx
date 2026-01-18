"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, Settings, Sparkles } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  SalesforceSettings,
  ConversationalInput,
  ConsultantAvatar,
  SalesforceMCPPanel,
  SalesforceEmbed,
  RAGContext,
  ConversationPanel,
  ConsultantGreeting,
  SalesforceConnectionStatus,
  SalesforceOperation,
  ConversationMessage,
  RAGStageResult,
  SalesforceDocument,
  SalesforceRAGMetrics,
  SalesforceQueryAnalysis,
  CONSULTANT_PERSONA,
  ELEVENLABS_VOICES,
} from "@/components/salesforce-demo";
import { useElevenLabsConversation } from "@/hooks";

// ElevenLabs Conversational AI Agent ID
const ELEVENLABS_AGENT_ID = "agent_7001kdqegdr4eyct05t0cawfwxtf";

export default function SalesforceDemo() {
  // ==================== STATE ====================

  // UI State
  const [showGreeting, setShowGreeting] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  // Settings State
  const [selectedVoice, setSelectedVoice] = useState<string>(ELEVENLABS_VOICES[0].id);
  const [autoExecute, setAutoExecute] = useState(true);
  const [debugMode, setDebugMode] = useState(false);

  // Salesforce Connection State
  const [connectionStatus, setConnectionStatus] = useState<SalesforceConnectionStatus>({
    connected: false,
    connecting: false,
  });

  // Operations State
  const [operations, setOperations] = useState<SalesforceOperation[]>([]);
  const [screenshotUrl, setScreenshotUrl] = useState<string | undefined>();

  // Conversation State
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentTranscript, setCurrentTranscript] = useState("");

  // Avatar State
  const [avatarVideoUrl, setAvatarVideoUrl] = useState<string | undefined>();
  const [avatarStatus, setAvatarStatus] = useState<"idle" | "listening" | "thinking" | "speaking">("idle");
  const [currentResponse, setCurrentResponse] = useState("");

  // RAG State
  const [ragActive, setRagActive] = useState(false);
  const [ragCurrentStage, setRagCurrentStage] = useState("");
  const [ragStages, setRagStages] = useState<RAGStageResult[]>([]);
  const [ragAnalysis, setRagAnalysis] = useState<SalesforceQueryAnalysis | undefined>();
  const [ragDocuments, setRagDocuments] = useState<SalesforceDocument[]>([]);
  const [ragMetrics, setRagMetrics] = useState<SalesforceRAGMetrics | undefined>();
  const [ragComplete, setRagComplete] = useState(false);

  // Audio Level State (for visualization)
  const [audioLevel, setAudioLevel] = useState(0);

  // ElevenLabs Conversation Hook
  const {
    state: conversationState,
    startConversation,
    endConversation,
    isConnected,
    isConnecting,
  } = useElevenLabsConversation({
    onUserTranscript: (transcript, isFinal) => {
      setCurrentTranscript(transcript);
      if (isFinal && transcript.trim()) {
        // Add user message when transcript is final
        const userMessage: ConversationMessage = {
          id: `msg-${Date.now()}`,
          role: "user",
          content: transcript,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, userMessage]);
      }
    },
    onAgentTranscript: (transcript) => {
      setCurrentResponse(transcript);
      // Add or update assistant message
      setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant" && Date.now() - lastMsg.timestamp < 5000) {
          // Update last assistant message
          return prev.map((m, i) =>
            i === prev.length - 1 ? { ...m, content: transcript } : m
          );
        } else {
          // Add new assistant message
          return [
            ...prev,
            {
              id: `msg-${Date.now()}`,
              role: "assistant" as const,
              content: transcript,
              timestamp: Date.now(),
            },
          ];
        }
      });
    },
    onAgentAudioStart: () => {
      setAvatarStatus("speaking");
    },
    onAgentAudioEnd: () => {
      setAvatarStatus("idle");
    },
    onStatusChange: (status) => {
      if (status === "connected") {
        setAvatarStatus("listening");
      } else if (status === "idle" || status === "error") {
        setAvatarStatus("idle");
      }
    },
    onError: (error) => {
      console.error("Conversation error:", error);
      const errorMessage: ConversationMessage = {
        id: `msg-${Date.now()}`,
        role: "system",
        content: `Error: ${error}`,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    },
  });

  // Derive conversation active state from hook
  const isConversationActive = isConnected || isConnecting;

  // Update avatar status based on conversation state
  useEffect(() => {
    if (conversationState.isUserSpeaking) {
      setAvatarStatus("listening");
      setAudioLevel(0.5); // Visual feedback
    } else if (conversationState.isAgentSpeaking) {
      setAvatarStatus("speaking");
      setAudioLevel(0);
    } else if (isConnected) {
      setAvatarStatus("listening");
      setAudioLevel(0);
    }
  }, [conversationState.isUserSpeaking, conversationState.isAgentSpeaking, isConnected]);

  // ==================== HANDLERS ====================

  // Handle Salesforce Connection
  const handleSalesforceConnect = useCallback(async () => {
    setConnectionStatus({ connected: false, connecting: true });

    try {
      // In production, this would redirect to OAuth flow
      const response = await fetch("/api/salesforce/connect", {
        method: "POST",
      });

      if (response.ok) {
        const data = await response.json();
        if (data.authUrl) {
          // Redirect to Salesforce OAuth
          window.location.href = data.authUrl;
        }
      } else {
        // Demo mode - simulate connection
        await new Promise((resolve) => setTimeout(resolve, 1500));
        setConnectionStatus({
          connected: true,
          connecting: false,
          org: {
            id: "00D5f000000xxxx",
            name: "Demo Org",
            instanceUrl: "https://demo-dev-ed.my.salesforce.com",
            username: "demo@salesforce.com",
            orgType: "developer",
            connectedAt: new Date().toISOString(),
          },
        });
      }
    } catch (error) {
      console.error("Connection error:", error);
      // Demo mode fallback
      await new Promise((resolve) => setTimeout(resolve, 1500));
      setConnectionStatus({
        connected: true,
        connecting: false,
        org: {
          id: "00D5f000000xxxx",
          name: "Demo Org",
          instanceUrl: "https://demo-dev-ed.my.salesforce.com",
          username: "demo@salesforce.com",
          orgType: "developer",
          connectedAt: new Date().toISOString(),
        },
      });
    }
  }, []);

  // Handle Salesforce Disconnect
  const handleSalesforceDisconnect = useCallback(() => {
    setConnectionStatus({ connected: false, connecting: false });
    setOperations([]);
    setScreenshotUrl(undefined);
  }, []);

  // Start Conversation (ElevenLabs WebSocket)
  const handleStartConversation = useCallback(async () => {
    // Add system message
    const systemMessage: ConversationMessage = {
      id: `msg-${Date.now()}`,
      role: "system",
      content: "Starting conversation...",
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, systemMessage]);

    try {
      // Get signed WebSocket URL from API
      const response = await fetch("/api/elevenlabs/conversation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "start",
          agentId: ELEVENLABS_AGENT_ID,
          voiceId: selectedVoice,
        }),
      });

      if (response.ok) {
        const { wsUrl } = await response.json();
        // Start conversation with the hook (handles audio capture and WebSocket)
        await startConversation(wsUrl);

        // Update system message
        setMessages((prev) => {
          const updated = [...prev];
          const lastSystemIdx = updated.findLastIndex(m => m.role === "system");
          if (lastSystemIdx >= 0) {
            updated[lastSystemIdx] = {
              ...updated[lastSystemIdx],
              content: "Conversation started - speak now!",
            };
          }
          return updated;
        });
      } else {
        throw new Error("Failed to get WebSocket URL");
      }
    } catch (error) {
      console.error("Failed to start conversation:", error);
      const errorMessage: ConversationMessage = {
        id: `msg-${Date.now()}`,
        role: "system",
        content: `Failed to start conversation: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  }, [selectedVoice, startConversation]);

  // End Conversation
  const handleEndConversation = useCallback(() => {
    endConversation();
    setCurrentTranscript("");
    setCurrentResponse("");

    const systemMessage: ConversationMessage = {
      id: `msg-${Date.now()}`,
      role: "system",
      content: "Conversation ended",
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, systemMessage]);
  }, [endConversation]);

  // Handle Text Input
  const handleTextSubmit = useCallback(
    async (text: string) => {
      // Add user message
      const userMessage: ConversationMessage = {
        id: `msg-${Date.now()}`,
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsProcessing(true);
      setAvatarStatus("thinking");

      // Start RAG pipeline
      setRagActive(true);
      setRagComplete(false);
      setRagStages([]);
      setRagDocuments([]);

      try {
        // Execute RAG pipeline via API
        const answer = await executeRAGPipeline(text);

        // Generate response (in production, this goes through ElevenLabs)
        const assistantMessage: ConversationMessage = {
          id: `msg-${Date.now() + 1}`,
          role: "assistant",
          content: answer,
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setCurrentResponse(answer);
        setAvatarStatus("speaking");

        // Simulate speech duration based on answer length
        const speechDuration = Math.min(Math.max(answer.length * 30, 2000), 8000);
        await new Promise((resolve) => setTimeout(resolve, speechDuration));
        setAvatarStatus("idle");
      } catch (error) {
        console.error("Error processing message:", error);
        // Show error message to user
        const errorMessage: ConversationMessage = {
          id: `msg-${Date.now() + 1}`,
          role: "assistant",
          content: "I apologize, but I encountered an error processing your request. Please try again.",
          timestamp: Date.now(),
        };
        setMessages((prev) => [...prev, errorMessage]);
      } finally {
        setIsProcessing(false);
        setRagActive(false);
      }
    },
    []
  );

  // Execute RAG Pipeline via API
  const executeRAGPipeline = async (query: string): Promise<string> => {
    const stages = [
      "orchestration",
      "query_transform",
      "dense_retrieval",
      "sparse_retrieval",
      "fusion",
      "rerank",
      "validation",
      "generation",
      "evaluation",
    ];

    // Show stages progressively while API is processing
    let stageIndex = 0;
    const stageInterval = setInterval(() => {
      if (stageIndex < stages.length - 2) { // Stop before generation/evaluation
        const stage = stages[stageIndex];
        setRagCurrentStage(stage);
        const stageResult: RAGStageResult = {
          stage,
          status: "complete",
          duration_ms: Math.floor(50 + Math.random() * 150),
          items_processed: Math.floor(5 + Math.random() * 20),
        };
        setRagStages((prev) => [...prev, stageResult]);
        stageIndex++;
      }
    }, 150);

    try {
      // Call the real RAG API
      const response = await fetch("/api/salesforce/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          include_mcp_suggestions: true,
        }),
      });

      clearInterval(stageInterval);

      if (!response.ok) {
        throw new Error("RAG API request failed");
      }

      const data = await response.json();

      // Update with real data from API
      if (data.query_analysis) {
        setRagAnalysis({
          original_query: data.query_analysis.original_query || query,
          intent: data.query_analysis.intent || "general_inquiry",
          confidence: data.query_analysis.confidence || 0.85,
          suggested_objects: data.query_analysis.suggested_objects || [],
          is_multi_step: data.query_analysis.is_multi_step || false,
        });
      }

      if (data.documents && data.documents.length > 0) {
        setRagDocuments(data.documents.map((doc: {
          id?: string;
          title?: string;
          content?: string;
          source?: string;
          rerank_score?: number;
          chunk_index?: number;
        }, index: number) => ({
          id: doc.id || `doc-${index}`,
          title: doc.title || "Document",
          content: doc.content || "",
          source: doc.source || "help_docs",
          rerank_score: doc.rerank_score || 0.8,
          chunk_index: doc.chunk_index || 0,
        })));
      }

      // Complete remaining stages
      if (data.stages) {
        setRagStages(data.stages);
      } else {
        // Complete remaining stages manually
        for (let i = stageIndex; i < stages.length; i++) {
          const stageResult: RAGStageResult = {
            stage: stages[i],
            status: "complete",
            duration_ms: Math.floor(100 + Math.random() * 200),
            items_processed: i === stages.length - 2 ? 1 : Math.floor(5 + Math.random() * 10),
          };
          setRagStages((prev) => [...prev, stageResult]);
          setRagCurrentStage(stages[i]);
        }
      }

      if (data.metrics) {
        setRagMetrics({
          faithfulness: data.metrics.faithfulness || 0.9,
          relevancy: data.metrics.relevancy || 0.85,
          completeness: data.metrics.completeness || 0.8,
          composite_score: data.metrics.composite_score || 0.85,
        });
      }

      // Handle MCP execution result - add to operations panel
      if (data.mcp_result) {
        // Map backend action types to frontend types
        const actionTypeMap: Record<string, SalesforceOperation["type"]> = {
          create: "insert",
          query: "query",
          update: "update",
          delete: "delete",
          describe: "describe",
          search: "search",
        };
        const mappedType = actionTypeMap[data.mcp_result.action] || "query";
        const objectName = data.mcp_result.object || "Record";

        // Build description based on action type
        let description = "";
        if (data.mcp_result.action === "query") {
          description = `Query ${objectName}: ${data.mcp_result.totalSize || 0} records`;
        } else if (data.mcp_result.action === "create") {
          description = `Created ${objectName} (ID: ${data.mcp_result.id})`;
        } else if (data.mcp_result.action === "update") {
          description = `Updated ${objectName} (ID: ${data.mcp_result.id})`;
        } else if (data.mcp_result.action === "delete") {
          description = `Deleted ${objectName} (ID: ${data.mcp_result.id})`;
        } else {
          description = `${data.mcp_result.action} on ${objectName}`;
        }

        const mcpOp: SalesforceOperation = {
          id: `mcp-${Date.now()}`,
          type: mappedType,
          description,
          soql: data.mcp_result.soql,
          status: data.mcp_result.success ? "success" : "error",
          timestamp: Date.now(),
          result: data.mcp_result,
          recordCount: data.mcp_result.totalSize || (data.mcp_result.id ? 1 : 0),
          duration_ms: data.latency?.mcp_execution || 0,
        };
        setOperations((prev) => [...prev, mcpOp]);
      }

      setRagComplete(true);

      // Return the answer for the conversation
      return data.answer || `I understand you want to ${query.toLowerCase()}. Let me help you with that based on Salesforce best practices.`;
    } catch (error) {
      clearInterval(stageInterval);
      console.error("RAG API error:", error);

      // Fallback response
      setRagComplete(true);
      return `I understand you're asking about "${query}". While I'm having trouble accessing my full knowledge base right now, I can still help. Could you provide more details about what you're trying to accomplish?`;
    }
  };

  // Handle Salesforce Operation (MCP)
  const executeOperation = useCallback(
    async (operation: Omit<SalesforceOperation, "id" | "status" | "timestamp">) => {
      const newOp: SalesforceOperation = {
        ...operation,
        id: `op-${Date.now()}`,
        status: "pending",
        timestamp: Date.now(),
      };

      setOperations((prev) => [...prev, newOp]);

      // Update to running
      setOperations((prev) =>
        prev.map((op) => (op.id === newOp.id ? { ...op, status: "running" } : op))
      );

      try {
        const response = await fetch("/api/salesforce/mcp", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            operation: operation.type,
            params: operation,
          }),
        });

        const result = await response.json();
        const duration_ms = Date.now() - newOp.timestamp;

        setOperations((prev) =>
          prev.map((op) =>
            op.id === newOp.id
              ? {
                  ...op,
                  status: response.ok ? "success" : "error",
                  result: response.ok ? result : undefined,
                  error: !response.ok ? result.error : undefined,
                  duration_ms,
                  recordCount: result.totalSize,
                }
              : op
          )
        );
      } catch (error) {
        setOperations((prev) =>
          prev.map((op) =>
            op.id === newOp.id
              ? {
                  ...op,
                  status: "error",
                  error: error instanceof Error ? error.message : "Unknown error",
                  duration_ms: Date.now() - newOp.timestamp,
                }
              : op
          )
        );
      }
    },
    []
  );

  // Handle Refresh Screenshot
  const handleRefreshScreenshot = useCallback(async () => {
    // In production, this would capture Salesforce screenshot via Playwright
    setScreenshotUrl(undefined);
  }, []);

  // Handle Greeting Complete
  const handleGreetingComplete = useCallback(() => {
    setShowGreeting(false);
  }, []);

  // ==================== EFFECTS ====================

  // Check Salesforce connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      setConnectionStatus({ connected: false, connecting: true });
      try {
        const response = await fetch("/api/salesforce/status");
        const data = await response.json();

        if (data.connected) {
          setConnectionStatus({
            connected: true,
            connecting: false,
            org: data.org,
          });
        } else {
          setConnectionStatus({
            connected: false,
            connecting: false,
          });
        }
      } catch (error) {
        console.error("Failed to check Salesforce connection:", error);
        setConnectionStatus({
          connected: false,
          connecting: false,
        });
      }
    };

    checkConnection();
  }, []);

  // Note: WebSocket cleanup is handled by useElevenLabsConversation hook

  // ==================== RENDER ====================

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-[#032D60]">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-1/4 -left-20 w-96 h-96 bg-[#0176D3]/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 -right-20 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[#0176D3]/5 rounded-full blur-3xl" />
      </div>

      {/* Greeting Modal */}
      <AnimatePresence>
        {showGreeting && <ConsultantGreeting onComplete={handleGreetingComplete} />}
      </AnimatePresence>

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-40 backdrop-blur-xl bg-slate-950/70 border-b border-white/10">
        <div className="max-w-[1800px] mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="icon" className="text-white/70 hover:text-white">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[#0176D3] to-[#032D60] flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white">Salesforce Virtual Assistant</h1>
                <p className="text-xs text-[#0176D3]">Powered by ElevenLabs + RAG + MCP</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowSettings(!showSettings)}
              className="text-white/70 hover:text-white"
            >
              <Settings className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </header>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            className="fixed top-16 right-0 bottom-0 w-80 z-30 bg-slate-900/95 backdrop-blur-xl border-l border-white/10 overflow-y-auto"
          >
            <SalesforceSettings
              connectionStatus={connectionStatus}
              onConnect={handleSalesforceConnect}
              onDisconnect={handleSalesforceDisconnect}
              selectedVoice={selectedVoice}
              onVoiceChange={setSelectedVoice}
              autoExecute={autoExecute}
              onAutoExecuteChange={setAutoExecute}
              debugMode={debugMode}
              onDebugModeChange={setDebugMode}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="pt-20 pb-8 px-4">
        <div className="max-w-[1800px] mx-auto">
          <div className="grid grid-cols-12 gap-4">
            {/* Left Column - Avatar & Input */}
            <div className="col-span-12 lg:col-span-4 space-y-4">
              {/* Consultant Avatar */}
              <ConsultantAvatar
                videoUrl={avatarVideoUrl}
                status={avatarStatus}
                currentResponse={currentResponse}
              />

              {/* Conversational Input */}
              <ConversationalInput
                isActive={isConversationActive}
                isConnecting={isConnecting}
                isUserSpeaking={conversationState.isUserSpeaking}
                isAgentSpeaking={conversationState.isAgentSpeaking}
                onStart={handleStartConversation}
                onEnd={handleEndConversation}
                onTextSubmit={handleTextSubmit}
                audioLevel={audioLevel}
                currentTranscript={currentTranscript}
              />

              {/* Conversation Panel */}
              <ConversationPanel messages={messages} isProcessing={isProcessing} />
            </div>

            {/* Middle Column - Salesforce View */}
            <div className="col-span-12 lg:col-span-5 space-y-4">
              {/* Salesforce Embed */}
              <SalesforceEmbed
                connectionStatus={connectionStatus}
                screenshotUrl={screenshotUrl}
                onRefresh={handleRefreshScreenshot}
                operations={operations}
              />

              {/* MCP Operations Panel */}
              <SalesforceMCPPanel
                connectionStatus={connectionStatus}
                operations={operations}
              />
            </div>

            {/* Right Column - RAG Context */}
            <div className="col-span-12 lg:col-span-3">
              <RAGContext
                isActive={ragActive}
                currentStage={ragCurrentStage}
                stages={ragStages}
                analysis={ragAnalysis}
                documents={ragDocuments}
                metrics={ragMetrics}
                isComplete={ragComplete}
              />
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-slate-950/70 backdrop-blur-xl border-t border-white/10 py-2 px-4">
        <div className="max-w-[1800px] mx-auto flex items-center justify-between text-xs text-slate-500">
          <span>
            {CONSULTANT_PERSONA.name} • {CONSULTANT_PERSONA.experience}
          </span>
          <span>ElevenLabs Conversational AI + Salesforce MCP + Advanced RAG</span>
        </div>
      </footer>
    </div>
  );
}
