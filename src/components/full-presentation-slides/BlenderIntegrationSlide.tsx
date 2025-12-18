"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Box, Monitor, Server, ArrowLeftRight, Download, Eye } from "lucide-react";

interface BlenderIntegrationSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function BlenderIntegrationSlide({ slideNumber, totalSlides }: BlenderIntegrationSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Blender Integration">
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold text-white mb-2">Hybrid 3D Architecture</h2>
          <p className="text-slate-400">Three.js browser preview + Blender GPU execution</p>
        </div>

        {/* Main architecture diagram */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Browser - Three.js */}
          <Card className="bg-slate-900/50 border-cyan-500/30 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
                  <Monitor className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <CardTitle className="text-lg text-white">Browser</CardTitle>
                  <p className="text-sm text-cyan-400">Three.js Preview</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="p-4 bg-slate-800/50 rounded-lg border border-dashed border-cyan-500/30 text-center">
                <Box className="w-12 h-12 text-cyan-400 mx-auto mb-2" />
                <div className="text-xs text-slate-400">Real-time 3D Viewport</div>
              </div>
              <div className="space-y-2 text-sm">
                {[
                  "Wireframe & solid shading",
                  "Orbit controls for navigation",
                  "Progressive updates live",
                  "No installation required"
                ].map((feat) => (
                  <div key={feat} className="flex items-center gap-2 text-slate-300">
                    <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full" />
                    {feat}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Connection */}
          <Card className="bg-slate-900/50 border-amber-500/30 backdrop-blur-sm flex flex-col justify-center">
            <CardContent className="p-4 space-y-4">
              <div className="text-center">
                <ArrowLeftRight className="w-8 h-8 text-amber-400 mx-auto mb-2" />
                <div className="text-sm font-semibold text-white">Scene Sync Protocol</div>
              </div>

              {/* Data flow arrows */}
              <div className="flex flex-col items-center gap-2">
                <div className="flex items-center gap-2 p-2 bg-slate-800/50 rounded w-full">
                  <Badge variant="outline" className="border-cyan-500/30 text-cyan-300 text-xs">Browser</Badge>
                  <span className="text-slate-500 text-xs flex-1 text-center">← Scene Updates</span>
                  <Badge variant="outline" className="border-orange-500/30 text-orange-300 text-xs">GPU</Badge>
                </div>
                <div className="flex items-center gap-2 p-2 bg-slate-800/50 rounded w-full">
                  <Badge variant="outline" className="border-cyan-500/30 text-cyan-300 text-xs">Browser</Badge>
                  <span className="text-slate-500 text-xs flex-1 text-center">← Render Results</span>
                  <Badge variant="outline" className="border-orange-500/30 text-orange-300 text-xs">GPU</Badge>
                </div>
                <div className="flex items-center gap-2 p-2 bg-slate-800/50 rounded w-full">
                  <Badge variant="outline" className="border-cyan-500/30 text-cyan-300 text-xs">Browser</Badge>
                  <span className="text-slate-500 text-xs flex-1 text-center">← .blend File</span>
                  <Badge variant="outline" className="border-orange-500/30 text-orange-300 text-xs">GPU</Badge>
                </div>
              </div>

              {/* Protocol */}
              <div className="text-center text-xs text-slate-500">
                WebSocket / SSE streaming
              </div>
            </CardContent>
          </Card>

          {/* GPU Server - Blender */}
          <Card className="bg-slate-900/50 border-orange-500/30 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-orange-500/10 rounded-lg border border-orange-500/30">
                  <Server className="w-5 h-5 text-orange-400" />
                </div>
                <div>
                  <CardTitle className="text-lg text-white">GPU Server</CardTitle>
                  <p className="text-sm text-orange-400">Blender Headless</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="p-4 bg-slate-800/50 rounded-lg border border-dashed border-orange-500/30 text-center">
                <div className="text-2xl mb-2">🧊</div>
                <div className="text-xs text-slate-400">Blender MCP Server</div>
              </div>
              <div className="space-y-2 text-sm">
                {[
                  "Executes bpy commands",
                  "Generates .blend files",
                  "GPU-accelerated renders",
                  "Consistent results"
                ].map((feat) => (
                  <div key={feat} className="flex items-center gap-2 text-slate-300">
                    <div className="w-1.5 h-1.5 bg-orange-500 rounded-full" />
                    {feat}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Command example */}
        <Card className="bg-slate-900/30 border-slate-700/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="h-1 w-8 bg-gradient-to-r from-orange-500 to-cyan-500 rounded-full" />
              <span className="text-sm font-mono text-orange-400">EXAMPLE COMMAND EXECUTION</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Blender Command */}
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-orange-400 mb-2 flex items-center gap-2">
                  <Server className="w-3 h-3" /> Blender Command (GPU)
                </div>
                <pre className="text-xs text-slate-300 font-mono overflow-x-auto">
{`bpy.ops.mesh.primitive_cylinder_add(
    radius=0.3,
    depth=2.0,
    location=(0, 0, 1)
)
trunk = bpy.context.active_object
trunk.name = "TreeTrunk"`}
                </pre>
              </div>

              {/* Three.js Update */}
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-cyan-400 mb-2 flex items-center gap-2">
                  <Monitor className="w-3 h-3" /> Three.js Update (Browser)
                </div>
                <pre className="text-xs text-slate-300 font-mono overflow-x-auto">
{`const geometry = new CylinderGeometry(
    0.3, 0.3, 2.0, 16
);
const mesh = new Mesh(geometry, material);
mesh.position.set(0, 0, 1);
scene.add(mesh);`}
                </pre>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Benefits */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { icon: Monitor, label: "No Installation", desc: "Works in any browser", color: "cyan" },
            { icon: Server, label: "GPU Power", desc: "Server-side rendering", color: "orange" },
            { icon: Eye, label: "Live Preview", desc: "Real-time updates", color: "purple" },
            { icon: Download, label: "Export Ready", desc: "Download .blend files", color: "emerald" },
          ].map((item) => {
            const Icon = item.icon;
            return (
              <Card key={item.label} className={`bg-slate-900/50 border-${item.color}-500/30 backdrop-blur-sm`}>
                <CardContent className="p-4 text-center">
                  <Icon className={`w-6 h-6 text-${item.color}-400 mx-auto mb-2`} />
                  <div className="text-sm font-semibold text-white">{item.label}</div>
                  <div className="text-xs text-slate-500">{item.desc}</div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
