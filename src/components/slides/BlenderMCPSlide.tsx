"use client";

import { SlideWrapper } from "./SlideWrapper";

interface BlenderMCPSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function BlenderMCPSlide({ slideNumber, totalSlides }: BlenderMCPSlideProps) {
  const capabilities = [
    { name: "Create Meshes", desc: "Generate primitives, complex shapes, and custom geometry" },
    { name: "Apply Materials", desc: "Set colors, textures, PBR materials automatically" },
    { name: "Add Modifiers", desc: "Subdivision, bevel, mirror, and more" },
    { name: "Scene Management", desc: "Organize objects, lighting, and camera setup" },
    { name: "Export Assets", desc: "FBX, OBJ, GLTF for game engine import" },
    { name: "Animation Ready", desc: "Rig preparation and basic animations" }
  ];

  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-amber-500/10 border border-amber-500/30 rounded-full">
            <div className="w-2 h-2 bg-amber-400 rounded-full animate-pulse" />
            <span className="text-amber-400 text-sm font-medium">Module 4</span>
          </div>
          <h2 className="text-5xl font-bold text-white">
            Blender <span className="bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">MCP</span> Integration
          </h2>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto">
            Model Context Protocol for direct Blender control and 3D asset generation
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left - How it works */}
          <div className="space-y-6">
            <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-2xl">
              <h3 className="text-xl font-semibold text-white mb-4">How MCP Works</h3>

              <div className="space-y-4">
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-amber-500/20 rounded-lg flex items-center justify-center text-amber-400 font-bold">1</div>
                  <div>
                    <div className="font-medium text-white">User Request Parsing</div>
                    <div className="text-sm text-slate-400">&ldquo;Create a low-poly sword with metallic material&rdquo;</div>
                  </div>
                </div>

                <div className="w-px h-6 bg-slate-700 ml-4" />

                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-amber-500/20 rounded-lg flex items-center justify-center text-amber-400 font-bold">2</div>
                  <div>
                    <div className="font-medium text-white">RAG Context Retrieval</div>
                    <div className="text-sm text-slate-400">Fetches Blender commands and modeling patterns</div>
                  </div>
                </div>

                <div className="w-px h-6 bg-slate-700 ml-4" />

                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-amber-500/20 rounded-lg flex items-center justify-center text-amber-400 font-bold">3</div>
                  <div>
                    <div className="font-medium text-white">MCP Tool Calling</div>
                    <div className="text-sm text-slate-400">LLM generates Blender Python commands</div>
                  </div>
                </div>

                <div className="w-px h-6 bg-slate-700 ml-4" />

                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-amber-500/20 rounded-lg flex items-center justify-center text-amber-400 font-bold">4</div>
                  <div>
                    <div className="font-medium text-white">Blender Execution</div>
                    <div className="text-sm text-slate-400">Commands executed in Blender via MCP bridge</div>
                  </div>
                </div>

                <div className="w-px h-6 bg-slate-700 ml-4" />

                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center text-green-400 font-bold">5</div>
                  <div>
                    <div className="font-medium text-white">Asset Ready</div>
                    <div className="text-sm text-slate-400">3D model created and ready for export</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right - Capabilities */}
          <div className="space-y-6">
            <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-2xl">
              <h3 className="text-xl font-semibold text-white mb-4">MCP Capabilities</h3>

              <div className="grid grid-cols-2 gap-3">
                {capabilities.map((cap) => (
                  <div key={cap.name} className="p-3 bg-slate-900/50 border border-amber-500/20 rounded-xl hover:border-amber-500/40 transition-colors">
                    <div className="font-medium text-amber-400 text-sm">{cap.name}</div>
                    <div className="text-xs text-slate-500 mt-1">{cap.desc}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Code example */}
            <div className="p-6 bg-slate-900/80 border border-slate-700/50 rounded-2xl font-mono text-sm">
              <div className="text-slate-500 mb-2"># MCP Blender Call Example</div>
              <div className="text-green-400">mcp.blender.create_mesh(</div>
              <div className="text-slate-300 pl-4">type=<span className="text-amber-400">&quot;sword&quot;</span>,</div>
              <div className="text-slate-300 pl-4">style=<span className="text-amber-400">&quot;low_poly&quot;</span>,</div>
              <div className="text-slate-300 pl-4">material=<span className="text-amber-400">&quot;metallic&quot;</span>,</div>
              <div className="text-slate-300 pl-4">export_format=<span className="text-amber-400">&quot;fbx&quot;</span></div>
              <div className="text-green-400">)</div>
            </div>
          </div>
        </div>
      </div>
    </SlideWrapper>
  );
}
