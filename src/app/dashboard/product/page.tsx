"use client";

import { useState, useEffect, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Settings, Sparkles, Users } from "lucide-react";
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
} from "@/components/salesforce-demo";
import { useElevenLabsConversation } from "@/hooks";
import { avatars, getAvatarById, type AvatarProfile } from "@/lib/avatars";

// ElevenLabs Conversational AI Agent ID
const ELEVENLABS_AGENT_ID = "agent_7001kdqegdr4eyct05t0cawfwxtf";

function ProductContent() {
  const searchParams = useSearchParams();
  const avatarId = searchParams.get("avatar");

  // ==================== AVATAR STATE ====================
  const [selectedAvatar, setSelectedAvatar] = useState<AvatarProfile | null>(null);
  const [showAvatarSwitcher, setShowAvatarSwitcher] = useState(false);

  // ==================== UI STATE ====================
  const [showGreeting, setShowGreeting] = useState(true);
  const [showSettings, setShowSettings] = useState(false);

  // ==================== SETTINGS STATE ====================
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [autoExecute, setAutoExecute] = useState(true);
  const [debugMode, setDebugMode] = useState(false);

  // ==================== SALESFORCE CONNECTION STATE ====================
  const [connectionStatus, setConnectionStatus] = useState<SalesforceConnectionStatus>({
    connected: false,
    connecting: false,
  });

  // ==================== OPERATIONS STATE ====================
  const [operations, setOperations] = useState<SalesforceOperation[]>([]);
  const [screenshotUrl, setScreenshotUrl] = useState<string | undefined>();

  // ==================== CONVERSATION STATE ====================
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentTranscript, setCurrentTranscript] = useState("");

  // ==================== AVATAR DISPLAY STATE ====================
  const [avatarVideoUrl, setAvatarVideoUrl] = useState<string | undefined>();
  const [avatarStatus, setAvatarStatus] = useState<"idle" | "listening" | "thinking" | "speaking">("idle");
  const [currentResponse, setCurrentResponse] = useState("");

  // ==================== RAG STATE ====================
  const [ragActive, setRagActive] = useState(false);
  const [ragCurrentStage, setRagCurrentStage] = useState("");
  const [ragStages, setRagStages] = useState<RAGStageResult[]>([]);
  const [ragAnalysis, setRagAnalysis] = useState<SalesforceQueryAnalysis | undefined>();
  const [ragDocuments, setRagDocuments] = useState<SalesforceDocument[]>([]);
  const [ragMetrics, setRagMetrics] = useState<SalesforceRAGMetrics | undefined>();
  const [ragComplete, setRagComplete] = useState(false);

  // ==================== AUDIO LEVEL STATE ====================
  const [audioLevel, setAudioLevel] = useState(0);

  // ==================== LOAD AVATAR ====================
  useEffect(() => {
    if (avatarId) {
      const avatar = getAvatarById(avatarId);
      if (avatar) {
        setSelectedAvatar(avatar);
        setSelectedVoice(avatar.voiceId);
        return;
      }
    }
    // Default to first English avatar
    const defaultAvatar = avatars.find((a) => a.language === "en") || avatars[0];
    setSelectedAvatar(defaultAvatar);
    setSelectedVoice(defaultAvatar.voiceId);
  }, [avatarId]);

  // ==================== ELEVENLABS CONVERSATION HOOK ====================
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
      setMessages((prev) => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg?.role === "assistant" && Date.now() - lastMsg.timestamp < 5000) {
          return prev.map((m, i) =>
            i === prev.length - 1 ? { ...m, content: transcript } : m
          );
        } else {
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

  const isConversationActive = isConnected || isConnecting;

  // Update avatar status based on conversation state
  useEffect(() => {
    if (conversationState.isUserSpeaking) {
      setAvatarStatus("listening");
      setAudioLevel(0.5);
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
      const response = await fetch("/api/salesforce/connect", {
        method: "POST",
      });

      if (response.ok) {
        const data = await response.json();
        if (data.authUrl) {
          window.location.href = data.authUrl;
        }
      } else {
        // Demo mode
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

  // Start Conversation
  const handleStartConversation = useCallback(async () => {
    const systemMessage: ConversationMessage = {
      id: `msg-${Date.now()}`,
      role: "system",
      content: "Starting conversation...",
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, systemMessage]);

    try {
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
        await startConversation(wsUrl, {
          systemPrompt: selectedAvatar?.systemPrompt,
          voiceId: selectedAvatar?.voiceId,
          language: selectedAvatar?.language,
          avatarName: selectedAvatar?.displayName,
        });

        setMessages((prev) => {
          const updated = [...prev];
          const lastSystemIdx = updated.findLastIndex((m) => m.role === "system");
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
  }, [selectedVoice, selectedAvatar, startConversation]);

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
  const handleTextSubmit = useCallback(async (text: string) => {
    if (!selectedAvatar) return;

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
      const answer = await executeRAGPipeline(text, selectedAvatar);

      const assistantMessage: ConversationMessage = {
        id: `msg-${Date.now() + 1}`,
        role: "assistant",
        content: answer,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setCurrentResponse(answer);
      setAvatarStatus("speaking");

      const speechDuration = Math.min(Math.max(answer.length * 30, 2000), 8000);
      await new Promise((resolve) => setTimeout(resolve, speechDuration));
      setAvatarStatus("idle");
    } catch (error) {
      console.error("Error processing message:", error);
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
  }, [selectedAvatar]);

  // Execute RAG Pipeline
  const executeRAGPipeline = async (query: string, avatar: AvatarProfile): Promise<string> => {
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

    let stageIndex = 0;
    const stageInterval = setInterval(() => {
      if (stageIndex < stages.length - 2) {
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
      // Pass avatar context to RAG API for persona-aware responses
      const response = await fetch("/api/salesforce/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          avatar_id: avatar.id,
          avatar_name: avatar.displayName, // Include name for personalized responses
          system_prompt: avatar.systemPrompt,
          expertise_level: avatar.expertiseLevel,
          has_mcp: avatar.hasMCP,
          include_mcp_suggestions: avatar.hasMCP, // Only include MCP suggestions if avatar has capability
          language: avatar.language,
        }),
      });

      clearInterval(stageInterval);

      if (!response.ok) {
        throw new Error("RAG API request failed");
      }

      const data = await response.json();

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
        setRagDocuments(
          data.documents.map(
            (
              doc: {
                id?: string;
                title?: string;
                content?: string;
                source?: string;
                rerank_score?: number;
                chunk_index?: number;
              },
              index: number
            ) => ({
              id: doc.id || `doc-${index}`,
              title: doc.title || "Document",
              content: doc.content || "",
              source: doc.source || "help_docs",
              rerank_score: doc.rerank_score || 0.8,
              chunk_index: doc.chunk_index || 0,
            })
          )
        );
      }

      if (data.stages) {
        setRagStages(data.stages);
      } else {
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

      // Handle MCP execution result
      if (data.mcp_result) {
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

      return (
        data.answer ||
        `I understand you want to ${query.toLowerCase()}. Let me help you with that based on Salesforce best practices.`
      );
    } catch (error) {
      clearInterval(stageInterval);
      console.error("RAG API error:", error);
      setRagComplete(true);
      return `I understand you're asking about "${query}". While I'm having trouble accessing my full knowledge base right now, I can still help. Could you provide more details about what you're trying to accomplish?`;
    }
  };

  // Handle Refresh Screenshot
  const handleRefreshScreenshot = useCallback(async () => {
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

  // ==================== LOADING STATE ====================
  if (!selectedAvatar) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="animate-pulse text-slate-400">Loading consultant...</div>
      </div>
    );
  }

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
        {showGreeting && selectedAvatar && (
          <ConsultantGreeting
            avatar={selectedAvatar}
            onComplete={handleGreetingComplete}
          />
        )}
      </AnimatePresence>

      {/* Top Bar */}
      <div className="fixed top-0 left-0 right-0 z-40 backdrop-blur-xl bg-slate-950/70 border-b border-white/10">
        <div className="max-w-[1800px] mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-9 h-9 rounded-lg bg-gradient-to-br ${selectedAvatar.accentColor} flex items-center justify-center`}>
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-white">{selectedAvatar.name}</h1>
              <p className="text-xs text-slate-400">{selectedAvatar.title}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAvatarSwitcher(!showAvatarSwitcher)}
              className="text-white/70 hover:text-white"
            >
              <Users className="w-4 h-4 mr-2" />
              Switch
            </Button>
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
      </div>

      {/* Avatar Switcher Panel */}
      <AnimatePresence>
        {showAvatarSwitcher && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="fixed top-14 right-20 z-50 w-72 bg-slate-900/95 backdrop-blur-xl border border-white/10 rounded-xl p-3 shadow-2xl"
          >
            <h4 className="text-sm font-medium text-slate-400 mb-2 px-2">Switch Consultant</h4>
            <div className="space-y-1 max-h-80 overflow-y-auto">
              {avatars.map((avatar) => (
                <button
                  key={avatar.id}
                  onClick={() => {
                    setSelectedAvatar(avatar);
                    setSelectedVoice(avatar.voiceId);
                    setMessages([]);
                    setShowAvatarSwitcher(false);
                  }}
                  className={`w-full flex items-center gap-3 p-2 rounded-lg transition-colors ${
                    avatar.id === selectedAvatar.id
                      ? "bg-white/10 border border-white/20"
                      : "hover:bg-white/5"
                  }`}
                >
                  <div
                    className={`w-8 h-8 rounded-lg bg-gradient-to-br ${avatar.accentColor} flex items-center justify-center text-white text-sm font-bold`}
                  >
                    {avatar.name.charAt(0)}
                  </div>
                  <div className="text-left flex-1">
                    <p className="text-white text-sm font-medium">{avatar.name}</p>
                    <p className="text-slate-500 text-xs">{avatar.title}</p>
                  </div>
                  <span className="text-[10px] text-slate-500 uppercase">{avatar.language}</span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            className="fixed top-14 right-0 bottom-0 w-80 z-30 bg-slate-900/95 backdrop-blur-xl border-l border-white/10 overflow-y-auto"
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
      <main className="pt-16 pb-6 px-4">
        <div className="max-w-[1800px] mx-auto">
          <div className="grid grid-cols-10 gap-4">
            {/* Left Column - Avatar & Input (30%) */}
            <div className="col-span-12 lg:col-span-3 space-y-4">
              {/* Consultant Avatar */}
              {selectedAvatar && (
                <ConsultantAvatar
                  avatar={selectedAvatar}
                  videoUrl={avatarVideoUrl}
                  status={avatarStatus}
                  currentResponse={currentResponse}
                />
              )}

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

            {/* Right Column - Salesforce + MCP + RAG (70%) */}
            <div className="col-span-12 lg:col-span-7 space-y-4">
              {/* Salesforce Embed - Full width */}
              <SalesforceEmbed
                connectionStatus={connectionStatus}
                screenshotUrl={screenshotUrl}
                onRefresh={handleRefreshScreenshot}
                operations={operations}
              />

              {/* MCP + RAG Row - Side by side */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* MCP Operations Panel */}
                <SalesforceMCPPanel connectionStatus={connectionStatus} operations={operations} />

                {/* RAG Context */}
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
        </div>
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-slate-950/70 backdrop-blur-xl border-t border-white/10 py-2 px-4">
        <div className="max-w-[1800px] mx-auto flex items-center justify-between text-xs text-slate-500">
          <span>
            {selectedAvatar.name} • {selectedAvatar.title}
          </span>
          <span>ElevenLabs Conversational AI + Salesforce MCP + Advanced RAG</span>
        </div>
      </footer>
    </div>
  );
}

export default function ProductPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[60vh]">
          <div className="animate-pulse text-slate-400">Loading...</div>
        </div>
      }
    >
      <ProductContent />
    </Suspense>
  );
}
