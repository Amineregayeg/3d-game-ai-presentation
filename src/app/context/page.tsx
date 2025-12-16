"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

// =============================================================================
// Comprehensive Project Context Data
// =============================================================================

const PROJECT_CONTEXT = {
  name: "3D Game Generation AI Assistant",
  description: "An AI-powered system for game development featuring custom Speech-to-Text, intelligent RAG-based assistance, TTS with lip-sync avatars, and AI-driven 3D asset generation via Blender MCP.",
  deadline: "2025-01-15",

  infrastructure: {
    vps: {
      host: "5.249.161.66",
      sshPort: 22,
      os: "Debian 13 (Trixie)",
      nodeVersion: "v20.19.6",
      pythonVersion: "3.13.5",
      processManager: "pm2",
      frontendUrl: "http://5.249.161.66:3000",
      backendUrl: "http://5.249.161.66:5000",
      structure: `/home/developer/
├── 3d-game-ai/
│   ├── frontend/          # Next.js app (port 3000)
│   ├── backend/           # Flask API (port 5000)
│   ├── voxformer/         # VoxFormer STT source
│   └── ecosystem.config.js
├── voxformer_checkpoints/ # Trained models (~12GB)
└── voxformer_backup/      # Source backup`
    },
    gpu: {
      provider: "Vast.ai",
      host: "82.141.118.40",
      sshPort: 2674,
      gpu: "NVIDIA RTX 4090 (24GB VRAM)",
      location: "Finland",
      costPerHour: "$0.40",
      status: "OFFLINE - needs recreation",
      structure: `/root/
├── voxformer/           # Training code + checkpoints
├── SadTalker/           # Lip-sync generation
└── workspace/           # HuggingFace cache`
    }
  },

  components: {
    stt: {
      name: "VoxFormer STT",
      description: "Custom Speech-to-Text Transformer with WavLM encoder, Zipformer blocks, RoPE embeddings, and hybrid CTC-Attention loss",
      technologies: ["PyTorch", "WavLM", "Conformer/Zipformer", "RoPE", "CTC", "SwiGLU"],
      status: "Stage 1 training complete, Stage 2 pending GPU",
      docPath: "/docs/technical/STT_ARCHITECTURE_PLAN.md",
      presentation: "/technical",
      demo: "/demo",
      keyFiles: [
        "voxformer/src/model.py - Main VoxFormer model",
        "voxformer/src/encoder.py - Zipformer encoder blocks",
        "voxformer/src/decoder.py - Transformer decoder",
        "voxformer/configs/stage1.yaml - Training config"
      ],
      checkpoints: [
        "voxformer_checkpoints/best_final_stage1.pt (1.6GB)",
        "voxformer_checkpoints/stage2/ - In progress",
        "voxformer_checkpoints/stage3/ - Future"
      ]
    },
    rag: {
      name: "Advanced RAG",
      description: "Hybrid retrieval system with dense BGE-M3 embeddings + sparse BM25, HNSW indexing, RRF fusion, and cross-encoder reranking",
      technologies: ["PostgreSQL", "pgvector", "BGE-M3", "HNSW", "BM25", "MiniLM", "OpenAI GPT-4"],
      status: "Production ready - using GPT-4 for generation",
      docPath: "/docs/technical/RAG_ARCHITECTURE_PLAN.md",
      presentation: "/rag",
      demo: "/rag_demo",
      keyFiles: [
        "backend/rag_api_production.py - Production RAG API",
        "backend/ingest_openai.py - Document ingestion",
        "backend/rag_embeddings.db - Vector database"
      ],
      dataIngested: "Blender 4.4 Manual (~343MB, 2000+ pages)"
    },
    ttsLipsync: {
      name: "TTS + LipSync",
      description: "ElevenLabs Flash v2.5 TTS with SadTalker 3DMM lip-sync for talking avatar generation",
      technologies: ["ElevenLabs", "SadTalker", "3DMM", "GFPGAN", "WebSocket"],
      status: "TTS working, lip-sync needs GPU server",
      docPath: "/docs/technical/TTS_LIPSYNC_ARCHITECTURE_PLAN.md",
      presentation: "/avatar",
      demo: "/avatar_demo",
      keyFiles: [
        "backend/avatar_api.py - Avatar generation API",
        "SadTalker/ - On GPU server"
      ],
      specs: "75ms TTFB, 4.14 MOS quality score"
    },
    mcp: {
      name: "Blender MCP",
      description: "Model Context Protocol server for AI-driven 3D asset creation with Sketchfab, Poly Haven, and Hyper3D Rodin integration",
      technologies: ["MCP", "Blender Python API", "Sketchfab", "Poly Haven", "Hyper3D Rodin"],
      status: "Architecture planned, implementation pending",
      docPath: "/docs/technical/BLENDER_MCP_ARCHITECTURE_PLAN.md",
      presentation: "/mcp",
      keyFiles: [
        "docs/technical/BLENDER_MCP_ARCHITECTURE_PLAN.md"
      ]
    }
  },

  backendAPIs: {
    avatar: {
      base: "/api/avatar",
      endpoints: [
        "POST /speak - Generate TTS audio (+ optional video)",
        "GET /voices - List ElevenLabs voices",
        "GET /avatars - List available avatars",
        "GET /status - Check ElevenLabs/GPU status"
      ]
    },
    rag: {
      base: "/api/rag",
      endpoints: [
        "POST /query - Query RAG system",
        "GET /status - Check RAG status",
        "POST /ingest - Ingest documents (admin)"
      ]
    },
    vault: {
      base: "/api/vault",
      endpoints: [
        "POST /auth - Authenticate (returns JWT)",
        "GET /secrets - List secrets (requires token)",
        "POST /secrets - Create secret",
        "GET /secrets/:id/reveal - Reveal secret value",
        "DELETE /secrets/:id - Delete secret"
      ]
    },
    project: {
      base: "/api",
      endpoints: [
        "GET /health - Health check",
        "GET /context - Project context",
        "GET /tasks - Task list",
        "GET /team - Team members",
        "GET /activity - Activity feed",
        "GET /milestones - Project milestones"
      ]
    }
  },

  frontendRoutes: {
    presentations: [
      "/ - Home/Business overview",
      "/technical - VoxFormer STT deep-dive (25 slides)",
      "/rag - Advanced RAG architecture (20 slides)",
      "/avatar - TTS + LipSync presentation",
      "/mcp - Blender MCP presentation"
    ],
    demos: [
      "/demo - VoxFormer STT demo (needs GPU)",
      "/rag_demo - RAG query interface (working)",
      "/avatar_demo - Avatar TTS demo (TTS working)"
    ],
    management: [
      "/training - Live training dashboard",
      "/implementation - Task tracker",
      "/secrets - Credentials vault",
      "/context - This page (LLM context)"
    ],
    docs: [
      "/docs - Documentation hub",
      "/docs/stt - STT documentation",
      "/docs/rag - RAG documentation",
      "/docs/tts-lipsync - TTS documentation",
      "/docs/mcp - MCP documentation"
    ]
  },

  deployment: {
    workflow: `1. Make changes locally in /mnt/d/3d-game-ai-presentation/
2. Test with: npm run dev
3. Upload to VPS:
   sshpass -p '<VPS_PASSWORD>' scp -o StrictHostKeyChecking=no <file> root@5.249.161.66:/home/developer/3d-game-ai/frontend/<path>
4. Rebuild:
   sshpass -p '<VPS_PASSWORD>' ssh root@5.249.161.66 'cd /home/developer/3d-game-ai/frontend && npm run build'
5. Restart:
   sshpass -p '<VPS_PASSWORD>' ssh root@5.249.161.66 'pm2 restart frontend'`,
    pm2Commands: [
      "pm2 status - Check services",
      "pm2 restart frontend/backend - Restart service",
      "pm2 logs frontend/backend - View logs",
      "pm2 save - Save process list"
    ]
  },

  techStack: {
    frontend: ["Next.js 16", "React 19", "TypeScript", "Tailwind CSS 4", "shadcn/ui", "Framer Motion"],
    backend: ["Flask", "SQLAlchemy", "Gunicorn", "JWT Auth"],
    ai: ["PyTorch", "WavLM", "OpenAI GPT-4", "ElevenLabs", "SadTalker"],
    infra: ["Debian 13", "PM2", "Vast.ai GPU", "SQLite"]
  },

  keyDocumentation: [
    { path: "/docs/technical/STT_ARCHITECTURE_PLAN.md", desc: "VoxFormer complete specification (1,400+ lines)" },
    { path: "/docs/technical/RAG_ARCHITECTURE_PLAN.md", desc: "Advanced RAG system spec (1,500+ lines)" },
    { path: "/docs/technical/TTS_LIPSYNC_ARCHITECTURE_PLAN.md", desc: "TTS + LipSync pipeline" },
    { path: "/docs/technical/BLENDER_MCP_ARCHITECTURE_PLAN.md", desc: "Blender MCP architecture" },
    { path: "/docs/technical/DSP_VOICE_ISOLATION_PLAN.md", desc: "Voice isolation DSP" },
    { path: "/CLAUDE.md", desc: "AI assistant instructions" },
    { path: "/docs/report/", desc: "Academic report (LaTeX)" }
  ]
};

// =============================================================================
// Component
// =============================================================================

export default function ContextPage() {
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  const generateLLMContext = () => {
    const ctx = PROJECT_CONTEXT;

    return `# ${ctx.name}

${ctx.description}

## Quick Reference
- **Frontend**: ${ctx.infrastructure.vps.frontendUrl}
- **Backend API**: ${ctx.infrastructure.vps.backendUrl}
- **VPS Host**: ${ctx.infrastructure.vps.host}
- **Deadline**: ${ctx.deadline}

## Infrastructure

### VPS Server (Hostinger)
- Host: ${ctx.infrastructure.vps.host}
- OS: ${ctx.infrastructure.vps.os}
- Node: ${ctx.infrastructure.vps.nodeVersion}
- Python: ${ctx.infrastructure.vps.pythonVersion}
- Process Manager: ${ctx.infrastructure.vps.processManager}

Directory Structure:
\`\`\`
${ctx.infrastructure.vps.structure}
\`\`\`

### GPU Server (Vast.ai) - ${ctx.infrastructure.gpu.status}
- Host: ${ctx.infrastructure.gpu.host}:${ctx.infrastructure.gpu.sshPort}
- GPU: ${ctx.infrastructure.gpu.gpu}
- Cost: ${ctx.infrastructure.gpu.costPerHour}/hr

## Components

### 1. ${ctx.components.stt.name}
${ctx.components.stt.description}
- **Status**: ${ctx.components.stt.status}
- **Technologies**: ${ctx.components.stt.technologies.join(", ")}
- **Presentation**: ${ctx.components.stt.presentation}
- **Demo**: ${ctx.components.stt.demo}
- **Documentation**: ${ctx.components.stt.docPath}

Key Files:
${ctx.components.stt.keyFiles.map(f => `- ${f}`).join("\n")}

Checkpoints:
${ctx.components.stt.checkpoints.map(c => `- ${c}`).join("\n")}

### 2. ${ctx.components.rag.name}
${ctx.components.rag.description}
- **Status**: ${ctx.components.rag.status}
- **Technologies**: ${ctx.components.rag.technologies.join(", ")}
- **Presentation**: ${ctx.components.rag.presentation}
- **Demo**: ${ctx.components.rag.demo}
- **Documentation**: ${ctx.components.rag.docPath}
- **Data Ingested**: ${ctx.components.rag.dataIngested}

Key Files:
${ctx.components.rag.keyFiles.map(f => `- ${f}`).join("\n")}

### 3. ${ctx.components.ttsLipsync.name}
${ctx.components.ttsLipsync.description}
- **Status**: ${ctx.components.ttsLipsync.status}
- **Technologies**: ${ctx.components.ttsLipsync.technologies.join(", ")}
- **Presentation**: ${ctx.components.ttsLipsync.presentation}
- **Demo**: ${ctx.components.ttsLipsync.demo}
- **Specs**: ${ctx.components.ttsLipsync.specs}

Key Files:
${ctx.components.ttsLipsync.keyFiles.map(f => `- ${f}`).join("\n")}

### 4. ${ctx.components.mcp.name}
${ctx.components.mcp.description}
- **Status**: ${ctx.components.mcp.status}
- **Technologies**: ${ctx.components.mcp.technologies.join(", ")}
- **Presentation**: ${ctx.components.mcp.presentation}
- **Documentation**: ${ctx.components.mcp.docPath}

## Backend API Endpoints

### Avatar API (${ctx.backendAPIs.avatar.base})
${ctx.backendAPIs.avatar.endpoints.map(e => `- ${e}`).join("\n")}

### RAG API (${ctx.backendAPIs.rag.base})
${ctx.backendAPIs.rag.endpoints.map(e => `- ${e}`).join("\n")}

### Vault API (${ctx.backendAPIs.vault.base})
${ctx.backendAPIs.vault.endpoints.map(e => `- ${e}`).join("\n")}

### Project API (${ctx.backendAPIs.project.base})
${ctx.backendAPIs.project.endpoints.map(e => `- ${e}`).join("\n")}

## Frontend Routes

### Presentations
${ctx.frontendRoutes.presentations.map(r => `- ${r}`).join("\n")}

### Demos
${ctx.frontendRoutes.demos.map(r => `- ${r}`).join("\n")}

### Management
${ctx.frontendRoutes.management.map(r => `- ${r}`).join("\n")}

## Deployment Workflow

\`\`\`bash
${ctx.deployment.workflow}
\`\`\`

PM2 Commands:
${ctx.deployment.pm2Commands.map(c => `- ${c}`).join("\n")}

## Tech Stack

- **Frontend**: ${ctx.techStack.frontend.join(", ")}
- **Backend**: ${ctx.techStack.backend.join(", ")}
- **AI/ML**: ${ctx.techStack.ai.join(", ")}
- **Infrastructure**: ${ctx.techStack.infra.join(", ")}

## Key Documentation Files

${ctx.keyDocumentation.map(d => `- **${d.path}**: ${d.desc}`).join("\n")}

## Secrets & Credentials

Access the secrets vault at /secrets (password required).
Contains: API keys (OpenAI, ElevenLabs, Anthropic), VPS credentials, GitHub token.

## Current Priority Tasks

1. Recreate Vast.ai GPU instance for training/inference
2. Complete VoxFormer Stage 2 training
3. Test full avatar pipeline with lip-sync
4. Implement Blender MCP server

---
Generated from /context page - ${new Date().toISOString()}
`;
  };

  const handleCopy = async () => {
    const text = generateLLMContext();

    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.position = "fixed";
      textArea.style.left = "-999999px";
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      textArea.remove();
    }

    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const text = generateLLMContext();
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "project-context.md";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const ctx = PROJECT_CONTEXT;

  return (
    <div className="dark min-h-screen bg-slate-950">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-fuchsia-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-purple-500/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-3xl font-bold text-white">LLM Project Context</h1>
            <Badge className="bg-violet-500/20 text-violet-400 border-violet-500/30">AI Ready</Badge>
          </div>
          <p className="text-slate-400">
            Comprehensive project context for AI assistants to understand and continue development
          </p>
        </div>

        {/* Copy Context Banner */}
        <Card className="bg-gradient-to-r from-violet-500/20 to-fuchsia-500/20 border-violet-500/30 mb-8">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-semibold text-white mb-2">Copy Full Context for LLM</h2>
                <p className="text-slate-300 text-sm">
                  One-click copy of all project info - paste directly into Claude, GPT, or any AI assistant
                </p>
              </div>
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  onClick={handleDownload}
                  className="border-violet-500/50 text-violet-300 hover:bg-violet-500/20"
                >
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download .md
                </Button>
                <Button
                  onClick={handleCopy}
                  className="bg-gradient-to-r from-violet-500 to-fuchsia-600 text-white hover:opacity-90"
                >
                  {copied ? (
                    <>
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy Context
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-slate-800/50 p-1">
            <TabsTrigger value="overview" className="data-[state=active]:bg-slate-700">Overview</TabsTrigger>
            <TabsTrigger value="infrastructure" className="data-[state=active]:bg-slate-700">Infrastructure</TabsTrigger>
            <TabsTrigger value="components" className="data-[state=active]:bg-slate-700">Components</TabsTrigger>
            <TabsTrigger value="apis" className="data-[state=active]:bg-slate-700">APIs</TabsTrigger>
            <TabsTrigger value="preview" className="data-[state=active]:bg-slate-700">Raw Preview</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">{ctx.name}</CardTitle>
                  <CardDescription className="text-slate-400">{ctx.description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 rounded-lg bg-slate-800/50">
                      <p className="text-xs text-slate-500">Frontend</p>
                      <p className="text-sm text-cyan-400 font-mono">{ctx.infrastructure.vps.frontendUrl}</p>
                    </div>
                    <div className="p-3 rounded-lg bg-slate-800/50">
                      <p className="text-xs text-slate-500">Backend API</p>
                      <p className="text-sm text-emerald-400 font-mono">{ctx.infrastructure.vps.backendUrl}</p>
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/50">
                    <p className="text-xs text-slate-500">Deadline</p>
                    <p className="text-white">{ctx.deadline}</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">Tech Stack</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {Object.entries(ctx.techStack).map(([category, techs]) => (
                    <div key={category}>
                      <p className="text-xs text-slate-500 capitalize mb-1">{category}</p>
                      <div className="flex flex-wrap gap-1">
                        {techs.map(tech => (
                          <Badge key={tech} variant="secondary" className="bg-slate-800 text-slate-300 text-xs">
                            {tech}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="lg:col-span-2 bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">Key Documentation</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {ctx.keyDocumentation.map(doc => (
                      <div key={doc.path} className="p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors">
                        <p className="text-sm font-mono text-violet-400">{doc.path}</p>
                        <p className="text-xs text-slate-500">{doc.desc}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Infrastructure Tab */}
          <TabsContent value="infrastructure">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-white">VPS Server</CardTitle>
                    <Badge className="bg-emerald-500/20 text-emerald-400">Online</Badge>
                  </div>
                  <CardDescription className="text-slate-400">Hostinger VPS - Production</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-2 rounded bg-slate-800/50">
                      <p className="text-xs text-slate-500">Host</p>
                      <p className="text-sm text-white font-mono">{ctx.infrastructure.vps.host}</p>
                    </div>
                    <div className="p-2 rounded bg-slate-800/50">
                      <p className="text-xs text-slate-500">SSH Port</p>
                      <p className="text-sm text-white font-mono">{ctx.infrastructure.vps.sshPort}</p>
                    </div>
                    <div className="p-2 rounded bg-slate-800/50">
                      <p className="text-xs text-slate-500">OS</p>
                      <p className="text-sm text-white">{ctx.infrastructure.vps.os}</p>
                    </div>
                    <div className="p-2 rounded bg-slate-800/50">
                      <p className="text-xs text-slate-500">Node.js</p>
                      <p className="text-sm text-white">{ctx.infrastructure.vps.nodeVersion}</p>
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-800/50">
                    <p className="text-xs text-slate-500 mb-2">Directory Structure</p>
                    <pre className="text-xs text-slate-300 font-mono whitespace-pre overflow-x-auto">
                      {ctx.infrastructure.vps.structure}
                    </pre>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-white">GPU Server</CardTitle>
                    <Badge className="bg-amber-500/20 text-amber-400">Offline</Badge>
                  </div>
                  <CardDescription className="text-slate-400">Vast.ai - Training & Inference</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-2 rounded bg-slate-800/50">
                      <p className="text-xs text-slate-500">Host</p>
                      <p className="text-sm text-white font-mono">{ctx.infrastructure.gpu.host}</p>
                    </div>
                    <div className="p-2 rounded bg-slate-800/50">
                      <p className="text-xs text-slate-500">SSH Port</p>
                      <p className="text-sm text-white font-mono">{ctx.infrastructure.gpu.sshPort}</p>
                    </div>
                    <div className="p-2 rounded bg-slate-800/50">
                      <p className="text-xs text-slate-500">GPU</p>
                      <p className="text-sm text-white">{ctx.infrastructure.gpu.gpu}</p>
                    </div>
                    <div className="p-2 rounded bg-slate-800/50">
                      <p className="text-xs text-slate-500">Cost</p>
                      <p className="text-sm text-white">{ctx.infrastructure.gpu.costPerHour}/hr</p>
                    </div>
                  </div>
                  <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
                    <p className="text-sm text-amber-400">{ctx.infrastructure.gpu.status}</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="lg:col-span-2 bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="text-white">Deployment Workflow</CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="text-sm text-slate-300 font-mono whitespace-pre-wrap p-4 bg-slate-800/50 rounded-lg overflow-x-auto">
                    {ctx.deployment.workflow}
                  </pre>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Components Tab */}
          <TabsContent value="components">
            <div className="grid grid-cols-1 gap-6">
              {Object.entries(ctx.components).map(([id, comp]) => {
                const colors: Record<string, string> = {
                  stt: "cyan",
                  rag: "emerald",
                  ttsLipsync: "rose",
                  mcp: "orange"
                };
                const color = colors[id] || "violet";

                return (
                  <Card key={id} className="bg-slate-900/50 border-slate-800">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className={`text-${color}-400`}>{comp.name}</CardTitle>
                        <Badge className={`bg-${color}-500/20 text-${color}-400`}>
                          {comp.status.split(" - ")[0]}
                        </Badge>
                      </div>
                      <CardDescription className="text-slate-400">{comp.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex flex-wrap gap-1">
                        {comp.technologies.map(tech => (
                          <Badge key={tech} variant="secondary" className="bg-slate-800 text-slate-300 text-xs">
                            {tech}
                          </Badge>
                        ))}
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <a href={comp.presentation} className="p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors">
                          <p className="text-xs text-slate-500">Presentation</p>
                          <p className="text-sm text-white">{comp.presentation}</p>
                        </a>
                        {"demo" in comp && (
                          <a href={comp.demo as string} className="p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors">
                            <p className="text-xs text-slate-500">Demo</p>
                            <p className="text-sm text-white">{comp.demo as string}</p>
                          </a>
                        )}
                        <a href={comp.docPath} target="_blank" className="p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors">
                          <p className="text-xs text-slate-500">Documentation</p>
                          <p className="text-sm text-white truncate">{comp.docPath.split("/").pop()}</p>
                        </a>
                      </div>

                      <div className="p-3 rounded-lg bg-slate-800/50">
                        <p className="text-xs text-slate-500 mb-2">Key Files</p>
                        {comp.keyFiles.map(file => (
                          <p key={file} className="text-xs text-slate-400 font-mono">{file}</p>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </TabsContent>

          {/* APIs Tab */}
          <TabsContent value="apis">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(ctx.backendAPIs).map(([name, api]) => (
                <Card key={name} className="bg-slate-900/50 border-slate-800">
                  <CardHeader>
                    <CardTitle className="text-white capitalize">{name} API</CardTitle>
                    <CardDescription className="text-slate-400 font-mono">{api.base}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {api.endpoints.map(endpoint => {
                        const [method, ...rest] = endpoint.split(" ");
                        const path = rest.join(" ");
                        const methodColors: Record<string, string> = {
                          GET: "text-emerald-400",
                          POST: "text-cyan-400",
                          PUT: "text-amber-400",
                          DELETE: "text-red-400"
                        };
                        return (
                          <div key={endpoint} className="p-2 rounded bg-slate-800/50 font-mono text-xs">
                            <span className={methodColors[method] || "text-slate-400"}>{method}</span>
                            <span className="text-slate-300 ml-2">{path}</span>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <Card className="mt-6 bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white">Frontend Routes</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(ctx.frontendRoutes).map(([category, routes]) => (
                    <div key={category}>
                      <p className="text-sm text-slate-500 capitalize mb-2">{category}</p>
                      <div className="space-y-1">
                        {routes.map(route => {
                          const [path, desc] = route.split(" - ");
                          return (
                            <a key={route} href={path} className="block p-2 rounded bg-slate-800/50 hover:bg-slate-800 transition-colors">
                              <p className="text-xs font-mono text-violet-400">{path}</p>
                              {desc && <p className="text-xs text-slate-500">{desc}</p>}
                            </a>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Raw Preview Tab */}
          <TabsContent value="preview">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-white">Raw Context Preview</CardTitle>
                    <CardDescription className="text-slate-400">
                      This is what gets copied to clipboard - ready for any LLM
                    </CardDescription>
                  </div>
                  <Button variant="outline" size="sm" onClick={handleCopy} className="border-slate-700 text-slate-300">
                    {copied ? "Copied!" : "Copy"}
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[600px]">
                  <pre className="text-sm text-slate-300 font-mono whitespace-pre-wrap p-4 bg-slate-800/50 rounded-lg">
                    {generateLLMContext()}
                  </pre>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Navigation Dock */}
      <PresentationDock items={dockItems} />
    </div>
  );
}
