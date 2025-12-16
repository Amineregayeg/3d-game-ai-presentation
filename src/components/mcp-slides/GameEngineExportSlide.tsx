"use client";

import { MCPSlideWrapper } from "./MCPSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface GameEngineExportSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function GameEngineExportSlide({ slideNumber, totalSlides }: GameEngineExportSlideProps) {
  return (
    <MCPSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Game Engine Export">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Game Engine <span className="text-orange-400">Export</span>
            </h2>
            <p className="text-slate-400">FBX/GLTF export with engine-specific configurations</p>
          </div>
          <div className="flex gap-2">
            <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/50">
              Unity
            </Badge>
            <Badge className="bg-violet-500/20 text-violet-400 border-violet-500/50">
              Unreal Engine 5
            </Badge>
          </div>
        </div>

        {/* Export Formats Table */}
        <Card className="bg-slate-800/30 border-slate-700/50 mb-4">
          <CardContent className="p-4">
            <h3 className="text-sm font-bold text-white mb-3">Supported Export Formats</h3>
            <div className="grid grid-cols-4 gap-2">
              {[
                { format: "FBX", ext: ".fbx", unity: true, ue5: true, best: "General 3D assets" },
                { format: "GLTF", ext: ".gltf/.glb", unity: true, ue5: true, best: "Web-ready, PBR" },
                { format: "OBJ", ext: ".obj", unity: true, ue5: true, best: "Simple meshes" },
                { format: "USD", ext: ".usd/.usda", unity: true, ue5: true, best: "Complex scenes" }
              ].map((f, i) => (
                <div key={i} className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-orange-400">{f.format}</span>
                    <span className="text-xs text-slate-500 font-mono">{f.ext}</span>
                  </div>
                  <div className="flex items-center gap-2 mb-2">
                    <Badge className={`text-xs ${f.unity ? 'bg-blue-500/20 text-blue-400' : 'bg-slate-700 text-slate-500'}`}>
                      Unity {f.unity ? '✓' : '✗'}
                    </Badge>
                    <Badge className={`text-xs ${f.ue5 ? 'bg-violet-500/20 text-violet-400' : 'bg-slate-700 text-slate-500'}`}>
                      UE5 {f.ue5 ? '✓' : '✗'}
                    </Badge>
                  </div>
                  <div className="text-xs text-slate-500">{f.best}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-2 gap-4 flex-1">
          {/* Unity Export Settings */}
          <Card className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <svg className="w-6 h-6 text-blue-400" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-blue-400">Unity Export</h3>
                  <span className="text-xs text-slate-500">FBX with Unity-specific settings</span>
                </div>
              </div>

              <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs overflow-hidden mb-4">
                <div className="text-slate-500"># Unity FBX Export</div>
                <div className="text-blue-400">bpy.ops.export_scene.fbx(</div>
                <div className="pl-4"><span className="text-cyan-400">filepath</span>=output_path,</div>
                <div className="pl-4"><span className="text-cyan-400">use_selection</span>=<span className="text-amber-400">True</span>,</div>
                <div className="pl-4"><span className="text-cyan-400">global_scale</span>=<span className="text-green-400">1.0</span>,</div>
                <div className="pl-4"><span className="text-cyan-400">apply_scale_options</span>=<span className="text-green-400">&apos;FBX_SCALE_ALL&apos;</span>,</div>
                <div className="pl-4"><span className="text-cyan-400">bake_space_transform</span>=<span className="text-amber-400">True</span>,</div>
                <div className="pl-4"><span className="text-cyan-400">use_mesh_modifiers</span>=<span className="text-amber-400">True</span>,</div>
                <div className="pl-4"><span className="text-cyan-400">mesh_smooth_type</span>=<span className="text-green-400">&apos;FACE&apos;</span>,</div>
                <div className="pl-4"><span className="text-cyan-400">primary_bone_axis</span>=<span className="text-green-400">&apos;Y&apos;</span></div>
                <div className="text-blue-400">)</div>
              </div>

              {/* Unity Import Flow */}
              <div className="space-y-2">
                <div className="text-xs text-slate-500 font-semibold mb-2">Import Workflow</div>
                {[
                  "Blender Export (FBX)",
                  "Unity Assets Folder",
                  "Auto-Import (mesh, materials)",
                  "Prefab creation (optional)",
                  "Ready for Scene"
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${i < 4 ? 'bg-blue-400' : 'bg-emerald-400'}`} />
                    <span className={`text-xs ${i < 4 ? 'text-slate-400' : 'text-emerald-400 font-semibold'}`}>{step}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* UE5 Export Settings */}
          <Card className="bg-gradient-to-br from-violet-500/10 to-purple-500/10 border-violet-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-violet-500/20 flex items-center justify-center">
                  <svg className="w-6 h-6 text-violet-400" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-violet-400">Unreal Engine 5</h3>
                  <span className="text-xs text-slate-500">FBX with UE5-specific settings</span>
                </div>
              </div>

              <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs overflow-hidden mb-4">
                <div className="text-slate-500"># UE5 FBX Export</div>
                <div className="text-violet-400">bpy.ops.export_scene.fbx(</div>
                <div className="pl-4"><span className="text-purple-400">filepath</span>=output_path,</div>
                <div className="pl-4"><span className="text-purple-400">use_selection</span>=<span className="text-amber-400">True</span>,</div>
                <div className="pl-4"><span className="text-purple-400">apply_scale_options</span>=<span className="text-green-400">&apos;FBX_SCALE_NONE&apos;</span>,</div>
                <div className="pl-4"><span className="text-purple-400">bake_space_transform</span>=<span className="text-amber-400">False</span>,</div>
                <div className="pl-4"><span className="text-purple-400">add_leaf_bones</span>=<span className="text-amber-400">False</span>,</div>
                <div className="pl-4"><span className="text-purple-400">axis_forward</span>=<span className="text-green-400">&apos;-Z&apos;</span>,</div>
                <div className="pl-4"><span className="text-purple-400">axis_up</span>=<span className="text-green-400">&apos;Y&apos;</span></div>
                <div className="text-violet-400">)</div>
              </div>

              {/* UE5 Import Flow */}
              <div className="space-y-2">
                <div className="text-xs text-slate-500 font-semibold mb-2">Import Workflow</div>
                {[
                  "Blender Export (FBX)",
                  "Content Browser Drag & Drop",
                  "FBX Import Dialog",
                  "Collision/LOD generation",
                  "UStaticMesh Asset Ready"
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${i < 4 ? 'bg-violet-400' : 'bg-emerald-400'}`} />
                    <span className={`text-xs ${i < 4 ? 'text-slate-400' : 'text-emerald-400 font-semibold'}`}>{step}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Material Mapping */}
        <div className="mt-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <h3 className="text-sm font-semibold text-white mb-3">Material Property Mapping</h3>
          <div className="grid grid-cols-3 gap-4">
            {/* Blender */}
            <div className="bg-orange-500/10 rounded-lg p-3 border border-orange-500/30">
              <div className="text-xs font-bold text-orange-400 mb-2 flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-orange-500/30 flex items-center justify-center text-xs">B</div>
                Blender (Principled BSDF)
              </div>
              <div className="space-y-1 text-xs text-slate-400 font-mono">
                <div>Base Color</div>
                <div>Metallic</div>
                <div>Roughness</div>
                <div>Normal</div>
                <div>Emission</div>
              </div>
            </div>

            {/* Unity */}
            <div className="bg-blue-500/10 rounded-lg p-3 border border-blue-500/30">
              <div className="text-xs font-bold text-blue-400 mb-2 flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-blue-500/30 flex items-center justify-center text-xs">U</div>
                Unity (Standard/URP)
              </div>
              <div className="space-y-1 text-xs text-slate-400 font-mono">
                <div>Albedo</div>
                <div>Metallic</div>
                <div className="text-amber-400">1 - Smoothness</div>
                <div>Normal Map</div>
                <div>Emission</div>
              </div>
            </div>

            {/* UE5 */}
            <div className="bg-violet-500/10 rounded-lg p-3 border border-violet-500/30">
              <div className="text-xs font-bold text-violet-400 mb-2 flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-violet-500/30 flex items-center justify-center text-xs">E</div>
                UE5 (Material)
              </div>
              <div className="space-y-1 text-xs text-slate-400 font-mono">
                <div>Base Color</div>
                <div>Metallic</div>
                <div>Roughness</div>
                <div>Normal</div>
                <div>Emissive Color</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </MCPSlideWrapper>
  );
}
