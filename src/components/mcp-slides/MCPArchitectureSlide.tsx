"use client";

import { MCPSlideWrapper } from "./MCPSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface MCPArchitectureSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function MCPArchitectureSlide({ slideNumber, totalSlides }: MCPArchitectureSlideProps) {
  return (
    <MCPSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="System Architecture">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              System <span className="text-orange-400">Architecture</span>
            </h2>
            <p className="text-slate-400">End-to-end MCP integration with Blender 3D</p>
          </div>
          <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/50">
            Socket-Based Communication
          </Badge>
        </div>

        {/* Full Architecture Diagram */}
        <div className="flex-1 bg-slate-900/50 rounded-xl border border-slate-800/50 p-4">
          <svg viewBox="0 0 1100 500" className="w-full h-full">
            <defs>
              <linearGradient id="claudeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f97316" />
                <stop offset="100%" stopColor="#ea580c" />
              </linearGradient>
              <linearGradient id="mcpGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#fbbf24" />
                <stop offset="100%" stopColor="#f59e0b" />
              </linearGradient>
              <linearGradient id="blenderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#eab308" />
                <stop offset="100%" stopColor="#ca8a04" />
              </linearGradient>
              <linearGradient id="assetGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#84cc16" />
                <stop offset="100%" stopColor="#65a30d" />
              </linearGradient>
              <marker id="arrowArch" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
              </marker>
              <filter id="boxShadow">
                <feDropShadow dx="0" dy="4" stdDeviation="4" floodOpacity="0.3"/>
              </filter>
            </defs>

            {/* User Input */}
            <g transform="translate(50, 50)">
              <rect width="150" height="60" rx="8" fill="#1e293b" stroke="#475569" strokeWidth="1.5" filter="url(#boxShadow)"/>
              <text x="75" y="30" textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="bold">User Request</text>
              <text x="75" y="45" textAnchor="middle" fill="#64748b" fontSize="9">&quot;Create a medieval sword&quot;</text>
            </g>

            {/* Arrow to Claude */}
            <path d="M200,80 L280,80" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowArch)" strokeDasharray="4,2"/>

            {/* Claude AI */}
            <g transform="translate(280, 30)">
              <rect width="200" height="100" rx="12" fill="url(#claudeGrad)" fillOpacity="0.2" stroke="#f97316" strokeWidth="2" filter="url(#boxShadow)"/>
              <text x="100" y="30" textAnchor="middle" fill="#f97316" fontSize="14" fontWeight="bold">Claude AI</text>
              <text x="100" y="50" textAnchor="middle" fill="#fdba74" fontSize="10">Parse intent</text>
              <text x="100" y="65" textAnchor="middle" fill="#fdba74" fontSize="10">Query RAG for patterns</text>
              <text x="100" y="80" textAnchor="middle" fill="#fdba74" fontSize="10">Select MCP tools</text>
            </g>

            {/* Arrow to MCP Server */}
            <path d="M480,80 L560,80" stroke="#f97316" strokeWidth="2" markerEnd="url(#arrowArch)"/>
            <text x="520" y="70" textAnchor="middle" fill="#f97316" fontSize="9">Tool Call</text>

            {/* MCP Server */}
            <g transform="translate(560, 30)">
              <rect width="200" height="100" rx="12" fill="url(#mcpGrad)" fillOpacity="0.2" stroke="#fbbf24" strokeWidth="2" filter="url(#boxShadow)"/>
              <text x="100" y="30" textAnchor="middle" fill="#fbbf24" fontSize="14" fontWeight="bold">MCP Server</text>
              <text x="100" y="50" textAnchor="middle" fill="#fde68a" fontSize="10">Python SDK</text>
              <text x="100" y="65" textAnchor="middle" fill="#fde68a" fontSize="10">JSON-RPC Protocol</text>
              <text x="100" y="80" textAnchor="middle" fill="#fde68a" fontSize="10">Tool dispatch</text>
            </g>

            {/* Arrow to Blender */}
            <path d="M660,130 L660,180" stroke="#fbbf24" strokeWidth="2" markerEnd="url(#arrowArch)"/>
            <rect x="620" y="145" width="80" height="20" rx="4" fill="#1e293b" stroke="#fbbf24" strokeWidth="1"/>
            <text x="660" y="158" textAnchor="middle" fill="#fbbf24" fontSize="8">TCP:9876</text>

            {/* Blender Section */}
            <g transform="translate(480, 200)">
              <rect width="360" height="180" rx="12" fill="url(#blenderGrad)" fillOpacity="0.15" stroke="#eab308" strokeWidth="2" filter="url(#boxShadow)"/>
              <text x="180" y="25" textAnchor="middle" fill="#eab308" fontSize="14" fontWeight="bold">Blender 3D (Addon)</text>

              {/* Socket Server */}
              <rect x="20" y="45" width="100" height="50" rx="6" fill="#1e293b" stroke="#fbbf24" strokeWidth="1"/>
              <text x="70" y="65" textAnchor="middle" fill="#fbbf24" fontSize="9" fontWeight="bold">Socket Server</text>
              <text x="70" y="80" textAnchor="middle" fill="#94a3b8" fontSize="8">Port 9876</text>

              {/* Command Handler */}
              <rect x="130" y="45" width="100" height="50" rx="6" fill="#1e293b" stroke="#eab308" strokeWidth="1"/>
              <text x="180" y="65" textAnchor="middle" fill="#eab308" fontSize="9" fontWeight="bold">Handlers</text>
              <text x="180" y="80" textAnchor="middle" fill="#94a3b8" fontSize="8">Main Thread</text>

              {/* bpy API */}
              <rect x="240" y="45" width="100" height="50" rx="6" fill="#1e293b" stroke="#ca8a04" strokeWidth="1"/>
              <text x="290" y="65" textAnchor="middle" fill="#ca8a04" fontSize="9" fontWeight="bold">bpy API</text>
              <text x="290" y="80" textAnchor="middle" fill="#94a3b8" fontSize="8">Python</text>

              {/* Operations grid */}
              <g transform="translate(20, 110)">
                {["Create Mesh", "Apply Mats", "Add Mods", "Import", "Scene Setup", "Export FBX"].map((op, i) => (
                  <g key={op} transform={`translate(${(i % 3) * 110}, ${Math.floor(i / 3) * 30})`}>
                    <rect width="100" height="24" rx="4" fill="#1e293b" stroke="#64748b" strokeWidth="0.5"/>
                    <text x="50" y="16" textAnchor="middle" fill="#94a3b8" fontSize="8">{op}</text>
                  </g>
                ))}
              </g>
            </g>

            {/* External Asset Sources */}
            <g transform="translate(50, 200)">
              <rect width="180" height="180" rx="12" fill="url(#assetGrad)" fillOpacity="0.15" stroke="#84cc16" strokeWidth="2" filter="url(#boxShadow)"/>
              <text x="90" y="25" textAnchor="middle" fill="#84cc16" fontSize="12" fontWeight="bold">Asset Sources</text>

              {[
                { name: "Sketchfab", desc: "3D Models", color: "#3b82f6" },
                { name: "Poly Haven", desc: "HDRI/Textures", color: "#10b981" },
                { name: "Hyper3D", desc: "AI Gen", color: "#8b5cf6" },
                { name: "Hunyuan3D", desc: "AI Gen", color: "#ec4899" }
              ].map((source, i) => (
                <g key={source.name} transform={`translate(15, ${45 + i * 32})`}>
                  <rect width="150" height="26" rx="4" fill="#1e293b" stroke={source.color} strokeWidth="1"/>
                  <circle cx="15" cy="13" r="6" fill={source.color} fillOpacity="0.3"/>
                  <text x="35" y="14" fill={source.color} fontSize="9" fontWeight="bold">{source.name}</text>
                  <text x="35" y="22" fill="#64748b" fontSize="7">{source.desc}</text>
                </g>
              ))}
            </g>

            {/* Arrow from Assets to Blender */}
            <path d="M230,290 L480,290" stroke="#84cc16" strokeWidth="1.5" markerEnd="url(#arrowArch)" strokeDasharray="4,2"/>
            <text x="355" y="280" textAnchor="middle" fill="#84cc16" fontSize="8">Import Assets</text>

            {/* Game Engine Export */}
            <g transform="translate(880, 200)">
              <rect width="160" height="180" rx="12" fill="#1e293b" stroke="#22c55e" strokeWidth="2" filter="url(#boxShadow)"/>
              <text x="80" y="25" textAnchor="middle" fill="#22c55e" fontSize="12" fontWeight="bold">Game Engines</text>

              {/* Unity */}
              <rect x="15" y="45" width="130" height="55" rx="6" fill="#3b82f6" fillOpacity="0.15" stroke="#3b82f6" strokeWidth="1"/>
              <text x="80" y="65" textAnchor="middle" fill="#3b82f6" fontSize="10" fontWeight="bold">Unity</text>
              <text x="80" y="80" textAnchor="middle" fill="#93c5fd" fontSize="8">FBX/GLTF Import</text>
              <text x="80" y="92" textAnchor="middle" fill="#64748b" fontSize="7">C# Scripts</text>

              {/* UE5 */}
              <rect x="15" y="110" width="130" height="55" rx="6" fill="#8b5cf6" fillOpacity="0.15" stroke="#8b5cf6" strokeWidth="1"/>
              <text x="80" y="130" textAnchor="middle" fill="#8b5cf6" fontSize="10" fontWeight="bold">Unreal Engine 5</text>
              <text x="80" y="145" textAnchor="middle" fill="#c4b5fd" fontSize="8">FBX/USD Import</text>
              <text x="80" y="157" textAnchor="middle" fill="#64748b" fontSize="7">MetaHuman Ready</text>
            </g>

            {/* Arrow from Blender to Game Engine */}
            <path d="M840,290 L880,290" stroke="#22c55e" strokeWidth="2" markerEnd="url(#arrowArch)"/>
            <text x="860" y="280" textAnchor="middle" fill="#22c55e" fontSize="8">Export</text>

            {/* Protocol Labels */}
            <g transform="translate(560, 420)">
              <rect width="200" height="50" rx="8" fill="#f97316" fillOpacity="0.1" stroke="#f97316" strokeWidth="1"/>
              <text x="100" y="22" textAnchor="middle" fill="#f97316" fontSize="10" fontWeight="bold">Communication Protocol</text>
              <text x="100" y="38" textAnchor="middle" fill="#fdba74" fontSize="9">TCP Socket + JSON-RPC 2.0</text>
            </g>
          </svg>
        </div>
      </div>
    </MCPSlideWrapper>
  );
}
