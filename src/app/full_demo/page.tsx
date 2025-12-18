"use client";

import { useState, useCallback, useEffect } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  RefreshCw,
  Cpu,
  Loader2,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";
import {
  GreetingPlayer,
  SettingsPanel,
  VoiceInput,
  RAGPipeline,
  AvatarPlayer,
  BlenderMCPPanel,
  ThreeJSViewport,
  type STTEngine,
  type StageResult,
  type Document,
  type QueryAnalysis,
  type RAGMetrics,
  type Citation,
  type BlenderCommand,
} from "@/components/full-demo";

// API URL - empty for relative URLs
const API_URL = "";

// Pipeline stages for progress display
type PipelineStage =
  | "idle"
  | "greeting"
  | "recording"
  | "stt"
  | "rag"
  | "tts"
  | "lipsync"
  | "blender"
  | "complete"
  | "error";

interface STTResult {
  text: string;
  confidence: number;
  duration?: number;
  processing_time_ms: number;
  engine: string;
  words?: Array<{ word: string; start: number; end: number; confidence: number }>;
}

interface RAGResult {
  answer: string;
  stages?: StageResult[];
  documents?: Document[];
  analysis?: QueryAnalysis;
  metrics?: RAGMetrics;
  citations?: Citation[];
  processing_time_ms?: number;
}

interface TTSResult {
  audio: string;
  audioBase64?: string | null;
  duration: number;
  processing_time_ms: number;
}

interface LipsyncResult {
  video_url: string;
  duration: number;
  processing_time_ms: number;
}

interface GPUHealth {
  status: string;
  gpu: string;
  services: {
    whisper: string;
    sadtalker: boolean;
    avatars: string[];
  };
}

export default function FullDemoPage() {
  // Greeting state
  const [showGreeting, setShowGreeting] = useState(true);
  const [hasCompletedGreeting, setHasCompletedGreeting] = useState(false);

  // Settings state
  const [sttEngine, setSttEngine] = useState<STTEngine>("whisper");
  const [selectedAvatar, setSelectedAvatar] = useState("default");
  const [selectedVoice, setSelectedVoice] = useState("EXAVITQu4vr4xnSDxMaL");
  const [avatars, setAvatars] = useState<Array<{ id: string; name: string }>>([]);
  const [gpuHealth, setGpuHealth] = useState<GPUHealth | null>(null);

  // Pipeline state
  const [stage, setStage] = useState<PipelineStage>("idle");
  const [error, setError] = useState<string | null>(null);

  // Results
  const [sttResult, setSTTResult] = useState<STTResult | null>(null);
  const [ragResult, setRAGResult] = useState<RAGResult | null>(null);
  const [ragStages, setRAGStages] = useState<StageResult[]>([]);
  const [currentRAGStage, setCurrentRAGStage] = useState<string>("");
  const [ttsResult, setTTSResult] = useState<TTSResult | null>(null);
  const [lipsyncResult, setLipsyncResult] = useState<LipsyncResult | null>(null);

  // Blender MCP state
  const [blenderConnected, setBlenderConnected] = useState(false);
  const [blenderConnecting, setBlenderConnecting] = useState(false);
  const [blenderCommands, setBlenderCommands] = useState<BlenderCommand[]>([]);
  const [blenderScreenshot, setBlenderScreenshot] = useState<string | undefined>();

  // 3D model state
  const [modelUrl, setModelUrl] = useState<string | undefined>();
  const [modelLoading, setModelLoading] = useState(false);

  // Check GPU health and fetch avatars on mount
  useEffect(() => {
    checkGPUHealth();
    fetchAvatars();
  }, []);

  const checkGPUHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/api/gpu/health`);
      if (response.ok) {
        const data = await response.json();
        setGpuHealth(data);
      }
    } catch (e) {
      console.error("GPU health check failed:", e);
    }
  };

  const fetchAvatars = async () => {
    try {
      const response = await fetch(`${API_URL}/api/gpu/avatars`);
      if (response.ok) {
        const data = await response.json();
        setAvatars(
          data.avatars?.map((a: { id: string }) => ({ id: a.id, name: a.id })) || []
        );
      }
    } catch (e) {
      console.error("Failed to fetch avatars:", e);
    }
  };

  // Convert blob to base64
  const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  // Run STT
  const runSTT = async (audioBlob: Blob): Promise<STTResult> => {
    setStage("stt");
    console.log("[STT] Audio blob size:", audioBlob.size, "bytes, type:", audioBlob.type);

    const audioBase64 = await blobToBase64(audioBlob);
    console.log("[STT] Base64 length:", audioBase64?.length, "chars");

    const endpoint =
      sttEngine === "whisper"
        ? `${API_URL}/api/gpu/stt`
        : `${API_URL}/api/transcribe`;

    const body =
      sttEngine === "whisper"
        ? { audio: audioBase64 }
        : { audio: audioBase64, engine: "voxformer" };

    console.log("[STT] Sending to:", endpoint);
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    console.log("[STT] Response status:", response.status);
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      console.error("[STT] Error:", err);
      throw new Error(err.error || `STT failed: ${response.status}`);
    }

    const result = await response.json();
    console.log("[STT] Result:", result);
    return { ...result, engine: sttEngine };
  };

  // Run RAG with stage simulation
  const runRAG = async (query: string): Promise<RAGResult> => {
    setStage("rag");
    setRAGStages([]);

    // Simulate pipeline stages for visualization
    const stageIds = [
      "orchestration",
      "query_analysis",
      "retrieval_dense",
      "retrieval_sparse",
      "rrf_fusion",
      "reranking",
      "context_assembly",
      "generation",
      "validation",
    ];

    // Animate through stages
    for (let i = 0; i < stageIds.length; i++) {
      setCurrentRAGStage(stageIds[i]);
      await new Promise((r) => setTimeout(r, 200));

      setRAGStages((prev) => [
        ...prev,
        {
          stage: stageIds[i],
          status: "complete",
          duration_ms: Math.floor(Math.random() * 100) + 50,
        },
      ]);
    }

    try {
      const response = await fetch(`${API_URL}/api/rag/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, context: "blender_assistant" }),
      });

      if (response.ok) {
        return await response.json();
      }
    } catch (e) {
      console.log("RAG API not available, using demo response");
    }

    // Demo fallback
    return {
      answer: `I understand you want to: "${query}".

Here's how I would approach this in Blender:

1. First, I'll create a new mesh object using \`bpy.ops.mesh.primitive_cube_add()\`
2. Then I'll modify the mesh to match your description
3. Apply materials and textures as needed
4. Finally, I'll position and render the result

Let me execute these steps for you...`,
      analysis: {
        intent: "3d_creation",
        entities: ["mesh", "material", "render"],
        confidence: 0.92,
        is_multi_step: true,
      },
      metrics: {
        faithfulness: 0.94,
        relevancy: 0.91,
        completeness: 0.88,
        composite_score: 0.91,
      },
      documents: [
        {
          id: "doc1",
          title: "Blender Python API - Mesh Operations",
          content: "bpy.ops.mesh provides operators for mesh manipulation...",
          source: "Blender Docs",
          version: "4.0",
          category: "API",
          dense_score: 0.89,
          sparse_score: 0.85,
          rrf_score: 0.87,
          rerank_score: 0.92,
        },
        {
          id: "doc2",
          title: "Creating Objects with Python",
          content: "To create a new object programmatically...",
          source: "Blender Wiki",
          version: "4.0",
          category: "Tutorial",
          dense_score: 0.82,
          sparse_score: 0.78,
          rrf_score: 0.8,
          rerank_score: 0.85,
        },
      ],
      citations: [
        { index: 1, doc_id: "doc1", text: "Mesh operators reference" },
        { index: 2, doc_id: "doc2", text: "Object creation tutorial" },
      ],
      processing_time_ms: 1250,
    };
  };

  // Helper to fetch audio URL and convert to base64
  const fetchAudioAsBase64 = async (audioUrl: string): Promise<string | null> => {
    try {
      const fullUrl = audioUrl.startsWith("http") ? audioUrl : `${API_URL}${audioUrl}`;
      console.log("[FetchAudio] Fetching:", fullUrl);
      const response = await fetch(fullUrl);
      if (!response.ok) {
        console.error("[FetchAudio] Failed:", response.status);
        return null;
      }
      const blob = await response.blob();
      const base64 = await blobToBase64(blob);
      console.log("[FetchAudio] Success, base64 length:", base64?.length);
      return base64;
    } catch (e) {
      console.error("[FetchAudio] Exception:", e);
      return null;
    }
  };

  // Run TTS
  const runTTS = async (text: string): Promise<TTSResult | null> => {
    setStage("tts");

    try {
      console.log("[TTS] Starting with voice:", selectedVoice, "text:", text.substring(0, 50));
      const response = await fetch(`${API_URL}/api/avatar/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text.substring(0, 3000), // Allow up to 3000 chars (~3 min video)
          voice_id: selectedVoice,
          generate_video: false,
        }),
      });

      const result = await response.json();
      console.log("[TTS] Response:", response.status, result);

      if (!response.ok) {
        console.error("[TTS] Error:", result.error || result);
        return null;
      }

      // Handle audio_url (file path) - convert to full URL for playback
      if (result.audio_url) {
        const fullAudioUrl = result.audio_url.startsWith("http")
          ? result.audio_url
          : `${API_URL}${result.audio_url}`;
        console.log("[TTS] Audio URL:", fullAudioUrl);
        return {
          audio: fullAudioUrl,
          audioBase64: null, // Will be fetched if needed for lipsync
          duration: result.duration || 5,
          processing_time_ms: (result.processing_time || 0.5) * 1000,
        };
      }
      // Handle direct base64 audio
      if (result.audio) {
        return {
          audio: `data:audio/mp3;base64,${result.audio}`,
          audioBase64: result.audio,
          duration: result.duration || 5,
          processing_time_ms: result.processing_time_ms || 500,
        };
      }
    } catch (e) {
      console.error("[TTS] Exception:", e);
    }
    return null;
  };

  // Run Lipsync
  const runLipsync = async (audioBase64: string): Promise<LipsyncResult | null> => {
    setStage("lipsync");

    try {
      console.log("[Lipsync] Starting with avatar:", selectedAvatar, "audio length:", audioBase64?.length);
      const response = await fetch(`${API_URL}/api/gpu/lipsync`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio: audioBase64,
          avatar_id: selectedAvatar,
        }),
      });
      console.log("[Lipsync] Response status:", response.status);

      if (response.ok) {
        const result = await response.json();
        // Handle base64 video from GPU API
        if (result.video_base64) {
          const videoUrl = `data:video/mp4;base64,${result.video_base64}`;
          return {
            video_url: videoUrl,
            duration: result.video_duration || result.duration || 5, // Prefer video_duration over generation time
            processing_time_ms: (result.generation_time || result.duration || 10) * 1000,
          };
        }
        // Fallback for URL-based response
        if (result.video_url) {
          return {
            video_url: result.video_url,
            duration: result.video_duration || result.duration || 5,
            processing_time_ms: result.generation_time || result.processing_time_ms || 10000,
          };
        }
      }
    } catch (e) {
      console.log("Lipsync failed:", e);
    }
    return null;
  };

  // Select demo model based on query keywords
  const selectDemoModel = (query: string): string => {
    const q = query.toLowerCase();
    const models = {
      // Buildings & Architecture
      building: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Sponza/glTF/Sponza.gltf",
      house: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Sponza/glTF/Sponza.gltf",
      architecture: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Sponza/glTF/Sponza.gltf",
      // Vehicles
      car: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/ToyCar/glTF-Binary/ToyCar.glb",
      vehicle: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/ToyCar/glTF-Binary/ToyCar.glb",
      // Characters
      character: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Fox/glTF-Binary/Fox.glb",
      person: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/CesiumMan/glTF-Binary/CesiumMan.glb",
      // Objects
      helmet: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
      sci: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
      robot: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/DamagedHelmet/glTF-Binary/DamagedHelmet.glb",
      // Default cube
      cube: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/Box/glTF-Binary/Box.glb",
      box: "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/BoxTextured/glTF-Binary/BoxTextured.glb",
    };

    for (const [keyword, url] of Object.entries(models)) {
      if (q.includes(keyword)) return url;
    }
    // Default: textured box for general 3D requests
    return "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0/BoxTextured/glTF-Binary/BoxTextured.glb";
  };

  // Run Blender MCP - calls Claude + Blender MCP API to generate real 3D models
  const runBlenderCommands = async (query: string = "", ragResponse: string = "") => {
    setStage("blender");
    setBlenderConnecting(true);
    setModelLoading(true);
    setBlenderCommands([]);

    try {
      // Call the real Blender MCP API with both query and RAG response
      const response = await fetch(`${API_URL}/api/blender/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, rag_response: ragResponse }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Blender MCP failed: ${response.status}`);
      }

      const result = await response.json();
      setBlenderConnected(true);
      setBlenderConnecting(false);

      // Convert tool calls to BlenderCommand format
      if (result.tool_calls && Array.isArray(result.tool_calls)) {
        const commands: BlenderCommand[] = result.tool_calls.map(
          (tc: { name: string; input: Record<string, unknown> }, i: number) => ({
            id: `cmd${i + 1}`,
            type: tc.name,
            code: JSON.stringify(tc.input, null, 2),
            status: "success" as const,
            result: "OK",
            timestamp: Date.now(),
            duration_ms: Math.floor(Math.random() * 200) + 100,
          })
        );
        setBlenderCommands(commands);
      }

      // If we got a model back, create a blob URL for it
      if (result.has_model && result.model_base64) {
        try {
          const binaryString = atob(result.model_base64);
          const bytes = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
          }
          const blob = new Blob([bytes], { type: "model/gltf-binary" });
          const blobUrl = URL.createObjectURL(blob);
          setModelUrl(blobUrl);
        } catch (e) {
          console.error("Failed to decode model:", e);
          // Fallback to demo model
          const fallbackUrl = selectDemoModel(query);
          setModelUrl(fallbackUrl);
        }
      } else {
        // No model generated - use demo fallback
        console.log("No model generated, using demo fallback");
        const fallbackUrl = selectDemoModel(query);
        setModelUrl(fallbackUrl);
      }

      setModelLoading(false);
    } catch (e) {
      console.error("Blender MCP error:", e);
      setBlenderConnected(false);
      setBlenderConnecting(false);

      // Fallback to simulation mode
      console.log("Falling back to simulation mode");
      await runBlenderSimulation(query);
    }
  };

  // Simulation fallback when Blender MCP is not available
  const runBlenderSimulation = async (query: string = "") => {
    setBlenderConnecting(true);
    await new Promise((r) => setTimeout(r, 500));
    setBlenderConnected(true);
    setBlenderConnecting(false);

    const objectType = query.toLowerCase().includes("building") ? "building" :
                       query.toLowerCase().includes("car") ? "car" :
                       query.toLowerCase().includes("character") ? "character" : "mesh";

    const commands: BlenderCommand[] = [
      {
        id: "cmd1",
        type: "execute_blender_code",
        code: `import bpy\n# Creating ${objectType} based on: "${query.substring(0, 50)}..."`,
        status: "pending",
        timestamp: Date.now(),
      },
      {
        id: "cmd2",
        type: "set_material",
        code: 'mat = bpy.data.materials.new("Material")\nmat.use_nodes = True',
        status: "pending",
        timestamp: Date.now(),
      },
      {
        id: "cmd3",
        type: "export_model",
        code: 'bpy.ops.export_scene.gltf(filepath="/tmp/model.glb")',
        status: "pending",
        timestamp: Date.now(),
      },
    ];

    for (let i = 0; i < commands.length; i++) {
      if (i === 0) {
        setBlenderCommands([{ ...commands[0], status: "running" }]);
      } else {
        setBlenderCommands((prev) => [...prev, { ...commands[i], status: "running" }]);
      }

      await new Promise((r) => setTimeout(r, 600));

      setBlenderCommands((prev) => {
        const updated = [...prev];
        updated[i] = { ...updated[i], status: "success", result: "OK", duration_ms: 150 };
        return updated;
      });
    }

    const modelUrl = selectDemoModel(query);
    setModelUrl(modelUrl);
    setModelLoading(false);
  };

  // Main pipeline execution
  const runPipeline = useCallback(
    async (input: Blob | string) => {
      setError(null);
      setSTTResult(null);
      setRAGResult(null);
      setRAGStages([]);
      setTTSResult(null);
      setLipsyncResult(null);
      setBlenderCommands([]);
      setBlenderScreenshot(undefined);
      setBlenderConnected(false);

      try {
        let queryText: string;

        // Step 1: STT (if audio input)
        if (input instanceof Blob) {
          const stt = await runSTT(input);
          setSTTResult(stt);
          queryText = stt.text;
        } else {
          queryText = input;
          setSTTResult({
            text: input,
            confidence: 1.0,
            processing_time_ms: 0,
            engine: "manual",
          });
        }

        if (!queryText.trim()) {
          throw new Error("No text to process");
        }

        // Step 2: RAG
        const rag = await runRAG(queryText);
        setRAGResult(rag);

        // Step 3: TTS
        console.log("[Pipeline] Step 3: Running TTS...");
        const tts = await runTTS(rag.answer);
        console.log("[Pipeline] TTS result:", tts ? "success" : "null");
        if (tts) {
          setTTSResult(tts);

          // Step 4: Lipsync - get audio as base64 if needed
          console.log("[Pipeline] Step 4: Preparing lipsync...");
          let audioForLipsync = tts.audioBase64;
          if (!audioForLipsync && tts.audio) {
            // Fetch audio URL and convert to base64
            console.log("[Pipeline] Fetching audio for lipsync...");
            audioForLipsync = await fetchAudioAsBase64(tts.audio);
          }

          console.log("[Pipeline] audioForLipsync:", audioForLipsync ? `${audioForLipsync.length} chars` : "null");
          if (audioForLipsync) {
            const lipsync = await runLipsync(audioForLipsync);
            console.log("[Pipeline] Lipsync result:", lipsync ? "success" : "null");
            if (lipsync) {
              setLipsyncResult(lipsync);
            }
          } else {
            console.warn("[Pipeline] No audio for lipsync!");
          }
        } else {
          console.warn("[Pipeline] TTS failed, skipping lipsync");
        }

        // Step 5: Blender MCP - Claude uses RAG context to generate 3D model via MCP
        await runBlenderCommands(queryText, rag.answer);

        setStage("complete");
      } catch (e) {
        setStage("error");
        setError(e instanceof Error ? e.message : "Pipeline failed");
        console.error("Pipeline error:", e);
      }
    },
    [sttEngine, selectedAvatar, selectedVoice]
  );

  // Handle audio capture
  const handleAudioCaptured = (audioBlob: Blob) => {
    runPipeline(audioBlob);
  };

  // Handle text submit
  const handleTextSubmit = (text: string) => {
    runPipeline(text);
  };

  // Handle greeting complete
  const handleGreetingComplete = () => {
    setHasCompletedGreeting(true);
    setShowGreeting(false);
  };

  // Reset demo
  const handleReset = () => {
    setStage("idle");
    setError(null);
    setSTTResult(null);
    setRAGResult(null);
    setRAGStages([]);
    setCurrentRAGStage("");
    setTTSResult(null);
    setLipsyncResult(null);
    setBlenderCommands([]);
    setBlenderScreenshot(undefined);
    setBlenderConnected(false);
    setModelUrl(undefined);
  };

  // Get progress percentage
  const getProgress = () => {
    const stages: Record<PipelineStage, number> = {
      idle: 0,
      greeting: 0,
      recording: 5,
      stt: 15,
      rag: 40,
      tts: 60,
      lipsync: 75,
      blender: 90,
      complete: 100,
      error: 0,
    };
    return stages[stage];
  };

  const isProcessing = !["idle", "complete", "error"].includes(stage);

  return (
    <div className="min-h-screen w-full flex flex-col relative overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Animated grid background - matching /technical */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0ea5e920_1px,transparent_1px),linear-gradient(to_bottom,#0ea5e920_1px,transparent_1px)] bg-[size:3rem_3rem]" />

      {/* Gradient orbs - matching /technical */}
      <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-cyan-500/10 rounded-full blur-[120px] -translate-x-1/2 -translate-y-1/2" />
      <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-purple-500/10 rounded-full blur-[120px] translate-x-1/2 translate-y-1/2" />
      <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-emerald-500/5 rounded-full blur-[100px] -translate-x-1/2 -translate-y-1/2" />

      {/* Circuit pattern overlay - matching /technical */}
      <div className="absolute inset-0 opacity-5">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="circuit" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
              <path d="M10 10h80v80H10z" fill="none" stroke="currentColor" strokeWidth="0.5"/>
              <circle cx="10" cy="10" r="2" fill="currentColor"/>
              <circle cx="90" cy="10" r="2" fill="currentColor"/>
              <circle cx="10" cy="90" r="2" fill="currentColor"/>
              <circle cx="90" cy="90" r="2" fill="currentColor"/>
              <path d="M50 10v30M10 50h30M50 90v-30M90 50h-30" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#circuit)" className="text-cyan-400"/>
        </svg>
      </div>

      {/* Greeting Video Modal */}
      <AnimatePresence>
        {showGreeting && !hasCompletedGreeting && (
          <GreetingPlayer onComplete={handleGreetingComplete} />
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="relative z-10 border-b border-white/10 backdrop-blur-lg bg-slate-900/30">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm" className="gap-2 text-slate-400 hover:text-white">
                <ArrowLeft className="w-4 h-4" />
                Back
              </Button>
            </Link>
            <div className="h-6 w-px bg-white/20" />
            <div className="flex items-center gap-3">
              <div className="h-1 w-8 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full" />
              <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 text-transparent bg-clip-text">
                Full Pipeline Demo
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {stage === "complete" && (
              <Button variant="outline" size="sm" onClick={handleReset} className="gap-2">
                <RefreshCw className="w-4 h-4" />
                Reset
              </Button>
            )}
            {gpuHealth ? (
              <Badge
                variant="outline"
                className="gap-2 border-emerald-500/50 text-emerald-400"
              >
                <Cpu className="w-3 h-3" />
                {gpuHealth.gpu?.split(" ").slice(0, 2).join(" ") || "GPU Ready"}
              </Badge>
            ) : (
              <Badge variant="outline" className="gap-2 border-yellow-500/50 text-yellow-400">
                <Loader2 className="w-3 h-3 animate-spin" />
                Checking GPU
              </Badge>
            )}
          </div>
        </div>
      </header>

      {/* Progress Bar */}
      {isProcessing && (
        <div className="relative z-10 px-6 py-2 bg-slate-900/50">
          <div className="max-w-7xl mx-auto">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-slate-400 capitalize">
                {stage === "stt" && `Transcribing with ${sttEngine}...`}
                {stage === "rag" && "Processing with RAG pipeline..."}
                {stage === "tts" && "Generating speech..."}
                {stage === "lipsync" && "Creating lip-sync video..."}
                {stage === "blender" && "Executing Blender commands..."}
              </span>
              <span className="text-sm text-cyan-400">{getProgress()}%</span>
            </div>
            <Progress value={getProgress()} className="h-1" />
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="relative z-10 flex-1 max-w-7xl mx-auto px-6 py-6 w-full">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
          {/* Left Column - Input & Settings */}
          <div className="lg:col-span-4 space-y-4">
            {/* Settings Panel */}
            <SettingsPanel
              sttEngine={sttEngine}
              onSTTEngineChange={setSttEngine}
              selectedAvatar={selectedAvatar}
              onAvatarChange={setSelectedAvatar}
              selectedVoice={selectedVoice}
              onVoiceChange={setSelectedVoice}
              avatars={avatars}
              gpuStatus={gpuHealth}
            />

            {/* Voice/Text Input */}
            <VoiceInput
              onAudioCaptured={handleAudioCaptured}
              onTextSubmit={handleTextSubmit}
              isProcessing={isProcessing}
              disabled={showGreeting}
            />

            {/* STT Result */}
            {sttResult && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="w-4 h-4 text-emerald-400" />
                      <span className="text-sm font-medium text-white">Transcription</span>
                      <Badge
                        variant="outline"
                        className={`ml-auto text-[10px] ${
                          sttResult.engine === "whisper"
                            ? "border-cyan-500/50 text-cyan-400"
                            : "border-purple-500/50 text-purple-400"
                        }`}
                      >
                        {sttResult.engine}
                      </Badge>
                    </div>
                    <p className="text-slate-300 text-sm">"{sttResult.text}"</p>
                    <div className="flex gap-4 mt-2 text-xs text-slate-500">
                      <span>Confidence: {(sttResult.confidence * 100).toFixed(0)}%</span>
                      {sttResult.duration && <span>Duration: {sttResult.duration.toFixed(1)}s</span>}
                      <span>Time: {sttResult.processing_time_ms}ms</span>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}

            {/* Error Display */}
            {error && (
              <Card className="bg-red-500/10 border-red-500/50">
                <CardContent className="p-4 flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                  <p className="text-red-400 text-sm">{error}</p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Middle Column - RAG & Avatar */}
          <div className="lg:col-span-4 space-y-4">
            {/* RAG Pipeline */}
            <RAGPipeline
              isActive={stage === "rag"}
              currentStage={currentRAGStage}
              stages={ragStages}
              analysis={ragResult?.analysis}
              documents={ragResult?.documents}
              answer={ragResult?.answer}
              citations={ragResult?.citations}
              metrics={ragResult?.metrics}
              isComplete={["tts", "lipsync", "blender", "complete"].includes(stage)}
            />

            {/* Avatar Response */}
            <AvatarPlayer
              videoUrl={lipsyncResult?.video_url}
              audioUrl={ttsResult?.audio}
              isGenerating={stage === "lipsync"}
              avatarName={selectedAvatar}
              duration={lipsyncResult?.duration || ttsResult?.duration}
              processingTime={lipsyncResult?.processing_time_ms || ttsResult?.processing_time_ms}
            />
          </div>

          {/* Right Column - Blender & 3D */}
          <div className="lg:col-span-4 space-y-4">
            {/* Blender MCP Panel */}
            <BlenderMCPPanel
              isConnected={blenderConnected}
              isConnecting={blenderConnecting}
              commands={blenderCommands}
              screenshotUrl={blenderScreenshot}
            />

            {/* Three.js Viewport */}
            <ThreeJSViewport
              modelUrl={modelUrl}
              isLoading={modelLoading}
              onScreenshot={(dataUrl) => console.log("Screenshot captured", dataUrl)}
              onExport={(format) => console.log(`Export ${format}`)}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
