"use client";

import { MCPSlideWrapper } from "./MCPSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface MCPTitleSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function MCPTitleSlide({ slideNumber, totalSlides }: MCPTitleSlideProps) {
  return (
    <MCPSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Blender MCP">
      <div className="flex flex-col items-center justify-center min-h-[80vh] text-center">
        {/* MCP Protocol Badge */}
        <div className="flex items-center gap-3 mb-6">
          <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/50 px-4 py-1.5 text-sm">
            Model Context Protocol
          </Badge>
          <Badge variant="outline" className="border-amber-500/50 text-amber-400 bg-amber-500/10 px-4 py-1.5 text-sm">
            24 MCP Tools
          </Badge>
        </div>

        {/* Main Title */}
        <h1 className="text-6xl md:text-7xl font-bold mb-4">
          <span className="text-white">Blender</span>
          <span className="bg-gradient-to-r from-orange-400 via-amber-400 to-yellow-400 bg-clip-text text-transparent"> MCP</span>
        </h1>

        {/* Subtitle */}
        <p className="text-xl md:text-2xl text-slate-400 mb-8 max-w-3xl">
          AI-Powered 3D Asset Generation via Model Context Protocol
        </p>

        {/* Architecture Flow Visualization */}
        <div className="relative w-full max-w-4xl mt-8">
          <svg viewBox="0 0 900 200" className="w-full h-auto">
            <defs>
              <linearGradient id="flowGradTitle" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f97316" />
                <stop offset="50%" stopColor="#fbbf24" />
                <stop offset="100%" stopColor="#eab308" />
              </linearGradient>
              <filter id="glowTitle" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <marker id="arrowMCP" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#f97316" />
              </marker>
            </defs>

            {/* Flow line */}
            <path
              d="M50,100 L850,100"
              stroke="url(#flowGradTitle)"
              strokeWidth="3"
              fill="none"
              strokeDasharray="8,4"
              opacity="0.6"
            />

            {/* User Request Node */}
            <g transform="translate(50, 60)">
              <rect x="0" y="0" width="120" height="80" rx="12" fill="#1e293b" stroke="#f97316" strokeWidth="2"/>
              <text x="60" y="35" textAnchor="middle" fill="#f97316" fontSize="11" fontWeight="bold">User Request</text>
              <text x="60" y="55" textAnchor="middle" fill="#94a3b8" fontSize="9">&quot;Create a sword&quot;</text>
            </g>

            {/* Claude AI Node */}
            <g transform="translate(220, 60)">
              <rect x="0" y="0" width="120" height="80" rx="12" fill="#f97316" fillOpacity="0.2" stroke="#f97316" strokeWidth="2"/>
              <text x="60" y="35" textAnchor="middle" fill="#fb923c" fontSize="11" fontWeight="bold">Claude AI</text>
              <text x="60" y="55" textAnchor="middle" fill="#fdba74" fontSize="9">Tool Selection</text>
            </g>

            {/* Arrow */}
            <path d="M175,100 L215,100" stroke="#f97316" strokeWidth="2" markerEnd="url(#arrowMCP)" filter="url(#glowTitle)"/>

            {/* MCP Server Node */}
            <g transform="translate(390, 60)">
              <rect x="0" y="0" width="120" height="80" rx="12" fill="#fbbf24" fillOpacity="0.2" stroke="#fbbf24" strokeWidth="2"/>
              <text x="60" y="35" textAnchor="middle" fill="#fbbf24" fontSize="11" fontWeight="bold">MCP Server</text>
              <text x="60" y="55" textAnchor="middle" fill="#fde68a" fontSize="9">JSON-RPC Bridge</text>
            </g>

            {/* Arrow */}
            <path d="M345,100 L385,100" stroke="#fbbf24" strokeWidth="2" markerEnd="url(#arrowMCP)" filter="url(#glowTitle)"/>

            {/* Blender Node */}
            <g transform="translate(560, 60)">
              <rect x="0" y="0" width="120" height="80" rx="12" fill="#eab308" fillOpacity="0.2" stroke="#eab308" strokeWidth="2"/>
              <text x="60" y="35" textAnchor="middle" fill="#eab308" fontSize="11" fontWeight="bold">Blender</text>
              <text x="60" y="55" textAnchor="middle" fill="#fef08a" fontSize="9">Python bpy API</text>
            </g>

            {/* Arrow */}
            <path d="M515,100 L555,100" stroke="#eab308" strokeWidth="2" markerEnd="url(#arrowMCP)" filter="url(#glowTitle)"/>

            {/* Game Engine Node */}
            <g transform="translate(730, 60)">
              <rect x="0" y="0" width="120" height="80" rx="12" fill="#1e293b" stroke="#84cc16" strokeWidth="2"/>
              <text x="60" y="35" textAnchor="middle" fill="#84cc16" fontSize="11" fontWeight="bold">Game Engine</text>
              <text x="60" y="55" textAnchor="middle" fill="#bef264" fontSize="9">Unity / UE5</text>
            </g>

            {/* Arrow */}
            <path d="M685,100 L725,100" stroke="#84cc16" strokeWidth="2" markerEnd="url(#arrowMCP)" filter="url(#glowTitle)"/>
          </svg>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-4 gap-6 mt-12 max-w-4xl">
          {[
            { label: "Socket Response", value: "<100ms", color: "orange" },
            { label: "Scene Query", value: "<50ms", color: "amber" },
            { label: "Asset Import", value: "<30s", color: "yellow" },
            { label: "AI Generation", value: "30-120s", color: "lime" }
          ].map((metric, i) => (
            <div key={i} className="text-center">
              <div className={`text-3xl font-bold bg-gradient-to-r from-${metric.color}-400 to-${metric.color}-300 bg-clip-text text-transparent`}>
                {metric.value}
              </div>
              <div className="text-xs text-slate-500 mt-1">{metric.label}</div>
            </div>
          ))}
        </div>

        {/* Asset Sources */}
        <div className="flex items-center gap-4 mt-10">
          <span className="text-xs text-slate-600">Asset Sources:</span>
          {["Sketchfab", "Poly Haven", "Hyper3D", "Hunyuan3D"].map((source, i) => (
            <Badge key={i} variant="outline" className="border-slate-700 text-slate-500 text-xs">
              {source}
            </Badge>
          ))}
        </div>
      </div>
    </MCPSlideWrapper>
  );
}
