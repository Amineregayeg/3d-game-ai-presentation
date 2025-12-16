"use client";

import { MCPSlideWrapper } from "./MCPSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface BlenderAddonSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function BlenderAddonSlide({ slideNumber, totalSlides }: BlenderAddonSlideProps) {
  return (
    <MCPSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Blender Addon">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Blender <span className="text-orange-400">Addon</span>
            </h2>
            <p className="text-slate-400">Socket server implementation with Python bpy API</p>
          </div>
          <div className="flex gap-2">
            <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/50">
              addon.py
            </Badge>
            <Badge variant="outline" className="border-yellow-500/50 text-yellow-400 bg-yellow-500/10">
              Blender 3.6+
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 flex-1">
          {/* Socket Server Architecture */}
          <Card className="bg-gradient-to-br from-orange-500/10 to-amber-500/10 border-orange-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 rounded-lg bg-orange-500/20 flex items-center justify-center">
                  <svg className="w-5 h-5 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-bold text-orange-400">Socket Server Architecture</h3>
                  <span className="text-xs text-slate-500">BlenderMCPServer class</span>
                </div>
              </div>

              <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs overflow-hidden">
                <div className="text-slate-500"># Socket server configuration</div>
                <div><span className="text-orange-400">class</span> <span className="text-amber-400">BlenderMCPServer</span>:</div>
                <div className="pl-4 text-slate-400"><span className="text-orange-400">def</span> <span className="text-cyan-400">__init__</span>(self):</div>
                <div className="pl-8 text-slate-400">self.host = <span className="text-green-400">&quot;localhost&quot;</span></div>
                <div className="pl-8 text-slate-400">self.port = <span className="text-amber-300">9876</span></div>
                <div className="pl-8 text-slate-400">self.command_queue = []</div>
                <div className="pl-8 text-slate-400">self.running = <span className="text-orange-400">False</span></div>
                <div className="mt-2 pl-4 text-slate-400"><span className="text-orange-400">def</span> <span className="text-cyan-400">start</span>(self):</div>
                <div className="pl-8 text-slate-500"># Start daemon thread</div>
                <div className="pl-8 text-slate-400">threading.Thread(target=self._listen)</div>
                <div className="pl-8 text-slate-500"># Register main thread timer</div>
                <div className="pl-8 text-slate-400">bpy.app.timers.register(...)</div>
              </div>

              {/* Flow Diagram */}
              <div className="mt-4">
                <svg viewBox="0 0 400 100" className="w-full h-auto">
                  <defs>
                    <marker id="arrowAddon" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
                      <path d="M0,0 L0,6 L8,3 z" fill="#f97316" />
                    </marker>
                  </defs>

                  {/* Daemon Thread */}
                  <rect x="10" y="10" width="90" height="35" rx="4" fill="#f97316" fillOpacity="0.2" stroke="#f97316" strokeWidth="1"/>
                  <text x="55" y="25" textAnchor="middle" fill="#f97316" fontSize="8" fontWeight="bold">Daemon Thread</text>
                  <text x="55" y="38" textAnchor="middle" fill="#fdba74" fontSize="7">_listen_loop()</text>

                  <path d="M105,27 L135,27" stroke="#f97316" strokeWidth="1.5" markerEnd="url(#arrowAddon)"/>

                  {/* Command Queue */}
                  <rect x="140" y="10" width="90" height="35" rx="4" fill="#fbbf24" fillOpacity="0.2" stroke="#fbbf24" strokeWidth="1"/>
                  <text x="185" y="25" textAnchor="middle" fill="#fbbf24" fontSize="8" fontWeight="bold">Command Queue</text>
                  <text x="185" y="38" textAnchor="middle" fill="#fde68a" fontSize="7">Thread-safe</text>

                  <path d="M235,27 L265,27" stroke="#fbbf24" strokeWidth="1.5" markerEnd="url(#arrowAddon)"/>

                  {/* Main Thread */}
                  <rect x="270" y="10" width="90" height="35" rx="4" fill="#eab308" fillOpacity="0.2" stroke="#eab308" strokeWidth="1"/>
                  <text x="315" y="25" textAnchor="middle" fill="#eab308" fontSize="8" fontWeight="bold">Main Thread</text>
                  <text x="315" y="38" textAnchor="middle" fill="#fef08a" fontSize="7">Timer callback</text>

                  {/* bpy execution */}
                  <rect x="270" y="55" width="90" height="35" rx="4" fill="#84cc16" fillOpacity="0.2" stroke="#84cc16" strokeWidth="1"/>
                  <text x="315" y="70" textAnchor="middle" fill="#84cc16" fontSize="8" fontWeight="bold">bpy API</text>
                  <text x="315" y="83" textAnchor="middle" fill="#bef264" fontSize="7">Execute code</text>

                  <path d="M315,45 L315,55" stroke="#84cc16" strokeWidth="1.5" markerEnd="url(#arrowAddon)"/>
                </svg>
              </div>
            </CardContent>
          </Card>

          {/* Command Handlers */}
          <Card className="bg-gradient-to-br from-amber-500/10 to-yellow-500/10 border-amber-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                  <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-bold text-amber-400">Command Handlers</h3>
                  <span className="text-xs text-slate-500">_execute_command() dispatcher</span>
                </div>
              </div>

              <div className="space-y-2">
                {[
                  { handler: "_handle_get_scene_info", returns: "scene_name, object_count, objects[]" },
                  { handler: "_handle_get_object_info", returns: "location, rotation, scale, materials" },
                  { handler: "_handle_execute_code", returns: "output, success" },
                  { handler: "_handle_get_screenshot", returns: "base64 PNG" }
                ].map((h, i) => (
                  <div key={i} className="bg-slate-900/60 rounded-lg p-2 flex items-start justify-between gap-2">
                    <span className="font-mono text-xs text-amber-400">{h.handler}</span>
                    <span className="text-xs text-slate-500 text-right">{h.returns}</span>
                  </div>
                ))}
              </div>

              {/* Code Execution Example */}
              <div className="mt-4 bg-slate-900/80 rounded-lg p-3 font-mono text-xs">
                <div className="text-slate-500"># Code execution handler</div>
                <div><span className="text-orange-400">def</span> <span className="text-cyan-400">_handle_execute_code</span>(self, params):</div>
                <div className="pl-4 text-slate-400">code = params.get(<span className="text-green-400">&quot;code&quot;</span>)</div>
                <div className="pl-4 text-slate-500"># Capture stdout</div>
                <div className="pl-4 text-slate-400">stdout = io.StringIO()</div>
                <div className="pl-4 text-slate-400"><span className="text-orange-400">try</span>:</div>
                <div className="pl-8 text-slate-400"><span className="text-cyan-400">exec</span>(code, {`{`}<span className="text-green-400">&quot;bpy&quot;</span>: bpy{`}`})</div>
                <div className="pl-8 text-slate-400"><span className="text-orange-400">return</span> {`{`}<span className="text-green-400">&quot;output&quot;</span>: stdout{`}`}</div>
              </div>
            </CardContent>
          </Card>

          {/* Material Node Graph */}
          <Card className="bg-gradient-to-br from-yellow-500/10 to-lime-500/10 border-yellow-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                  <svg className="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                  </svg>
                </div>
                <h3 className="text-sm font-bold text-yellow-400">PBR Material Creation</h3>
              </div>

              {/* Node Graph Visualization */}
              <svg viewBox="0 0 380 150" className="w-full h-auto">
                <defs>
                  <linearGradient id="bsdfGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#84cc16" />
                    <stop offset="100%" stopColor="#22c55e" />
                  </linearGradient>
                </defs>

                {/* Texture Nodes */}
                <rect x="10" y="10" width="70" height="30" rx="4" fill="#ef4444" fillOpacity="0.2" stroke="#ef4444" strokeWidth="1"/>
                <text x="45" y="28" textAnchor="middle" fill="#ef4444" fontSize="8">Albedo</text>

                <rect x="10" y="50" width="70" height="30" rx="4" fill="#64748b" fillOpacity="0.2" stroke="#64748b" strokeWidth="1"/>
                <text x="45" y="68" textAnchor="middle" fill="#94a3b8" fontSize="8">Roughness</text>

                <rect x="10" y="90" width="70" height="30" rx="4" fill="#3b82f6" fillOpacity="0.2" stroke="#3b82f6" strokeWidth="1"/>
                <text x="45" y="108" textAnchor="middle" fill="#3b82f6" fontSize="8">Normal</text>

                {/* Normal Map Node */}
                <rect x="120" y="90" width="60" height="30" rx="4" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="1"/>
                <text x="150" y="108" textAnchor="middle" fill="#8b5cf6" fontSize="7">Normal Map</text>

                {/* Connections */}
                <path d="M80,25 L230,35" stroke="#ef4444" strokeWidth="1"/>
                <path d="M80,65 L230,55" stroke="#64748b" strokeWidth="1"/>
                <path d="M80,105 L120,105" stroke="#3b82f6" strokeWidth="1"/>
                <path d="M180,105 L230,75" stroke="#8b5cf6" strokeWidth="1"/>

                {/* Principled BSDF */}
                <rect x="230" y="20" width="90" height="100" rx="6" fill="url(#bsdfGrad)" fillOpacity="0.2" stroke="#84cc16" strokeWidth="1.5"/>
                <text x="275" y="40" textAnchor="middle" fill="#84cc16" fontSize="9" fontWeight="bold">Principled</text>
                <text x="275" y="52" textAnchor="middle" fill="#84cc16" fontSize="9" fontWeight="bold">BSDF</text>
                <text x="275" y="70" textAnchor="middle" fill="#bef264" fontSize="7">Base Color ●</text>
                <text x="275" y="82" textAnchor="middle" fill="#bef264" fontSize="7">Roughness ●</text>
                <text x="275" y="94" textAnchor="middle" fill="#bef264" fontSize="7">Metallic ●</text>
                <text x="275" y="106" textAnchor="middle" fill="#bef264" fontSize="7">Normal ●</text>

                {/* Output */}
                <path d="M320,70 L340,70" stroke="#84cc16" strokeWidth="1.5"/>
                <rect x="340" y="55" width="35" height="30" rx="4" fill="#1e293b" stroke="#84cc16" strokeWidth="1"/>
                <text x="358" y="73" textAnchor="middle" fill="#84cc16" fontSize="7">Output</text>
              </svg>
            </CardContent>
          </Card>

          {/* Asset Import Pipeline */}
          <Card className="bg-gradient-to-br from-lime-500/10 to-green-500/10 border-lime-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-8 h-8 rounded-lg bg-lime-500/20 flex items-center justify-center">
                  <svg className="w-5 h-5 text-lime-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                </div>
                <h3 className="text-sm font-bold text-lime-400">Asset Import Pipeline</h3>
              </div>

              <div className="space-y-2">
                {[
                  { step: "1", action: "Download GLTF/FBX archive", icon: "↓" },
                  { step: "2", action: "Extract to temporary directory", icon: "📁" },
                  { step: "3", action: "Import via bpy.ops.import_scene", icon: "📥" },
                  { step: "4", action: "Clean up temp files", icon: "🧹" },
                  { step: "5", action: "Position in scene (origin)", icon: "📍" }
                ].map((s) => (
                  <div key={s.step} className="flex items-center gap-3 bg-slate-900/50 rounded-lg p-2">
                    <div className="w-6 h-6 rounded-full bg-lime-500/20 flex items-center justify-center text-lime-400 text-xs font-bold">
                      {s.step}
                    </div>
                    <span className="text-xs text-slate-300">{s.action}</span>
                  </div>
                ))}
              </div>

              <div className="mt-3 bg-slate-900/80 rounded-lg p-2 font-mono text-xs">
                <div className="text-lime-400">bpy.ops.import_scene.gltf(filepath=path)</div>
                <div className="text-slate-500"># Returns imported objects list</div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </MCPSlideWrapper>
  );
}
