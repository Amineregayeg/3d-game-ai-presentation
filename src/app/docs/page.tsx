"use client";

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

const components = [
  {
    id: "stt",
    name: "VoxFormer STT",
    description: "Custom Speech-to-Text Transformer architecture with Conformer blocks, RoPE embeddings, and CTC loss",
    status: "In Development",
    statusColor: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    phases: 6,
    weeks: "12 weeks",
    technologies: ["PyTorch", "Conformer", "RoPE", "CTC", "SwiGLU"],
    docFile: "STT_ARCHITECTURE_PLAN.md",
    presentationLink: "/technical",
    gradient: "from-cyan-500 to-purple-600",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
      </svg>
    ),
  },
  {
    id: "rag",
    name: "Advanced RAG",
    description: "Hybrid retrieval system with BGE-M3 embeddings, HNSW indexing, BM25, cross-encoder reranking, and RAGAS evaluation",
    status: "In Development",
    statusColor: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    phases: 4,
    weeks: "16 weeks",
    technologies: ["PostgreSQL", "pgvector", "BGE-M3", "HNSW", "BM25", "MiniLM"],
    docFile: "RAG_ARCHITECTURE_PLAN.md",
    presentationLink: "/rag",
    gradient: "from-emerald-500 to-teal-600",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125" />
      </svg>
    ),
  },
  {
    id: "tts-lipsync",
    name: "TTS + LipSync",
    description: "ElevenLabs TTS integration with Wav2Lip/SadTalker lip-sync, viseme mapping, and game engine bridging",
    status: "In Development",
    statusColor: "bg-rose-500/20 text-rose-400 border-rose-500/30",
    phases: 5,
    weeks: "5 weeks",
    technologies: ["ElevenLabs", "Wav2Lip", "SadTalker", "WebSocket", "Unity", "UE5"],
    docFile: "TTS_LIPSYNC_ARCHITECTURE_PLAN.md",
    presentationLink: "/avatar",
    gradient: "from-rose-500 to-pink-600",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
      </svg>
    ),
  },
  {
    id: "mcp",
    name: "Blender MCP",
    description: "Model Context Protocol integration with Blender for AI-driven 3D asset creation, Sketchfab, Poly Haven, and Hyper3D Rodin",
    status: "In Development",
    statusColor: "bg-orange-500/20 text-orange-400 border-orange-500/30",
    phases: 4,
    weeks: "4 weeks",
    technologies: ["MCP", "Blender", "Sketchfab", "Poly Haven", "Hyper3D", "Socket"],
    docFile: "BLENDER_MCP_ARCHITECTURE_PLAN.md",
    presentationLink: "/mcp",
    gradient: "from-orange-500 to-amber-600",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
      </svg>
    ),
  },
];

export default function DocsPage() {
  return (
    <div className="dark min-h-screen bg-slate-950">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-emerald-500/5 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Back to Presentation
              </Link>
              <div className="w-px h-6 bg-slate-700" />
              <h1 className="text-xl font-semibold text-white">Technical Documentation</h1>
            </div>
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="bg-slate-800/50 border-slate-700 text-slate-300">
                4 Components
              </Badge>
              <Badge variant="outline" className="bg-slate-800/50 border-slate-700 text-slate-300">
                ~37 Weeks Total
              </Badge>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Hero Section */}
        <div className="mb-12">
          <h2 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-emerald-400 bg-clip-text text-transparent mb-4">
            3D Game AI Assistant
          </h2>
          <p className="text-lg text-slate-400 max-w-3xl">
            Comprehensive technical documentation for building an AI-powered game development assistant
            with speech recognition, knowledge retrieval, avatar generation, and 3D asset creation capabilities.
          </p>
        </div>

        {/* Component Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {components.map((component) => (
            <Card key={component.id} className="bg-slate-900/50 border-slate-800 hover:border-slate-700 transition-all group">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className={`p-3 rounded-xl bg-gradient-to-br ${component.gradient} bg-opacity-20`}>
                    <div className="text-white">{component.icon}</div>
                  </div>
                  <Badge className={component.statusColor}>
                    {component.status}
                  </Badge>
                </div>
                <CardTitle className="text-white mt-4">{component.name}</CardTitle>
                <CardDescription className="text-slate-400">
                  {component.description}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2 mb-4">
                  {component.technologies.slice(0, 5).map((tech) => (
                    <Badge key={tech} variant="secondary" className="bg-slate-800 text-slate-300 text-xs">
                      {tech}
                    </Badge>
                  ))}
                  {component.technologies.length > 5 && (
                    <Badge variant="secondary" className="bg-slate-800 text-slate-400 text-xs">
                      +{component.technologies.length - 5}
                    </Badge>
                  )}
                </div>
                <div className="flex items-center justify-between text-sm text-slate-500 mb-4">
                  <span>{component.phases} Phases</span>
                  <span>{component.weeks}</span>
                </div>
                <div className="flex gap-2">
                  <Button asChild variant="outline" className="flex-1 bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-700 hover:text-white">
                    <Link href={`/docs/${component.id}`}>
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Documentation
                    </Link>
                  </Button>
                  <Button asChild className={`flex-1 bg-gradient-to-r ${component.gradient} text-white hover:opacity-90`}>
                    <Link href={component.presentationLink}>
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Presentation
                    </Link>
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Components Table */}
        <Card className="bg-slate-900/50 border-slate-800">
          <CardHeader>
            <CardTitle className="text-white">Component Overview</CardTitle>
            <CardDescription className="text-slate-400">
              Detailed comparison of all system components
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="w-full">
              <Table>
                <TableHeader>
                  <TableRow className="border-slate-800 hover:bg-slate-800/50">
                    <TableHead className="text-slate-300">Component</TableHead>
                    <TableHead className="text-slate-300">Key Technologies</TableHead>
                    <TableHead className="text-slate-300">Phases</TableHead>
                    <TableHead className="text-slate-300">Timeline</TableHead>
                    <TableHead className="text-slate-300">Status</TableHead>
                    <TableHead className="text-slate-300 text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {components.map((component) => (
                    <TableRow key={component.id} className="border-slate-800 hover:bg-slate-800/30">
                      <TableCell className="font-medium text-white">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg bg-gradient-to-br ${component.gradient} bg-opacity-20`}>
                            <div className="text-white w-5 h-5 [&>svg]:w-5 [&>svg]:h-5">{component.icon}</div>
                          </div>
                          {component.name}
                        </div>
                      </TableCell>
                      <TableCell className="text-slate-400">
                        <div className="flex flex-wrap gap-1">
                          {component.technologies.slice(0, 3).map((tech) => (
                            <Badge key={tech} variant="secondary" className="bg-slate-800 text-slate-400 text-xs">
                              {tech}
                            </Badge>
                          ))}
                          {component.technologies.length > 3 && (
                            <Badge variant="secondary" className="bg-slate-800 text-slate-500 text-xs">
                              +{component.technologies.length - 3}
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="text-slate-400">{component.phases}</TableCell>
                      <TableCell className="text-slate-400">{component.weeks}</TableCell>
                      <TableCell>
                        <Badge className={component.statusColor}>
                          {component.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-2">
                          <Button asChild variant="ghost" size="sm" className="text-slate-400 hover:text-white">
                            <Link href={`/docs/${component.id}`}>
                              View Docs
                            </Link>
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Quick Links */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <Link href="/implementation" className="group">
            <Card className="bg-slate-900/50 border-slate-800 hover:border-cyan-500/50 transition-all h-full">
              <CardContent className="p-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-white group-hover:text-cyan-400 transition-colors">Implementation</p>
                  <p className="text-xs text-slate-500">Timeline & Backlog</p>
                </div>
              </CardContent>
            </Card>
          </Link>
          <Link href="/" className="group">
            <Card className="bg-slate-900/50 border-slate-800 hover:border-purple-500/50 transition-all h-full">
              <CardContent className="p-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-white group-hover:text-purple-400 transition-colors">Presentation</p>
                  <p className="text-xs text-slate-500">Overview Slides</p>
                </div>
              </CardContent>
            </Card>
          </Link>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="group">
            <Card className="bg-slate-900/50 border-slate-800 hover:border-emerald-500/50 transition-all h-full">
              <CardContent className="p-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-white group-hover:text-emerald-400 transition-colors">GitHub</p>
                  <p className="text-xs text-slate-500">Source Code</p>
                </div>
              </CardContent>
            </Card>
          </a>
          <a href="https://anthropic.com" target="_blank" rel="noopener noreferrer" className="group">
            <Card className="bg-slate-900/50 border-slate-800 hover:border-orange-500/50 transition-all h-full">
              <CardContent className="p-4 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-orange-500/10 text-orange-400">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-white group-hover:text-orange-400 transition-colors">Claude AI</p>
                  <p className="text-xs text-slate-500">Powered by Anthropic</p>
                </div>
              </CardContent>
            </Card>
          </a>
        </div>
      </main>

      {/* Navigation Dock */}
      <PresentationDock items={dockItems} />
    </div>
  );
}
