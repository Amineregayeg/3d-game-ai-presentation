"use client";

import { MCPSlideWrapper } from "./MCPSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface MCPToolsSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function MCPToolsSlide({ slideNumber, totalSlides }: MCPToolsSlideProps) {
  const toolCategories = [
    {
      name: "Scene Operations",
      color: "orange",
      tools: [
        { name: "get_scene_info", params: "—", desc: "Scene name, object count, object list" },
        { name: "get_object_info", params: "object_name", desc: "Location, rotation, scale, materials" },
        { name: "get_viewport_screenshot", params: "max_size=800", desc: "Base64 PNG capture" }
      ]
    },
    {
      name: "Code Execution",
      color: "amber",
      tools: [
        { name: "execute_blender_code", params: "code: str", desc: "Run arbitrary Python in Blender" }
      ]
    },
    {
      name: "Asset Search",
      color: "yellow",
      tools: [
        { name: "search_sketchfab_models", params: "query, categories, count", desc: "Search 3D marketplace" },
        { name: "search_polyhaven_assets", params: "asset_type, categories", desc: "Find HDRIs, textures" },
        { name: "get_polyhaven_categories", params: "asset_type", desc: "List available categories" }
      ]
    },
    {
      name: "Asset Download",
      color: "lime",
      tools: [
        { name: "download_sketchfab_model", params: "uid", desc: "Download & import GLTF" },
        { name: "download_polyhaven_asset", params: "asset_id, type, res", desc: "Download textures/models" }
      ]
    },
    {
      name: "AI Generation",
      color: "green",
      tools: [
        { name: "generate_hyper3d_model_via_text", params: "prompt, bbox", desc: "Text-to-3D generation" },
        { name: "generate_hyper3d_model_via_images", params: "urls, bbox", desc: "Image-to-3D generation" },
        { name: "generate_hunyuan3d_model", params: "prompt, image_url", desc: "Tencent AI generation" },
        { name: "poll_rodin_job_status", params: "subscription_key", desc: "Check generation progress" },
        { name: "poll_hunyuan_job_status", params: "job_id", desc: "Check Hunyuan progress" },
        { name: "import_generated_asset", params: "name, task_uuid", desc: "Import generated model" }
      ]
    }
  ];

  const colorMap: Record<string, { bg: string; border: string; text: string; badgeBg: string }> = {
    orange: { bg: "bg-orange-500/10", border: "border-orange-500/30", text: "text-orange-400", badgeBg: "bg-orange-500/20" },
    amber: { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400", badgeBg: "bg-amber-500/20" },
    yellow: { bg: "bg-yellow-500/10", border: "border-yellow-500/30", text: "text-yellow-400", badgeBg: "bg-yellow-500/20" },
    lime: { bg: "bg-lime-500/10", border: "border-lime-500/30", text: "text-lime-400", badgeBg: "bg-lime-500/20" },
    green: { bg: "bg-green-500/10", border: "border-green-500/30", text: "text-green-400", badgeBg: "bg-green-500/20" }
  };

  return (
    <MCPSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="MCP Tools">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              24 MCP <span className="text-orange-400">Tools</span>
            </h2>
            <p className="text-slate-400">Complete Blender control via JSON-RPC</p>
          </div>
          <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/50 text-lg px-4 py-1">
            24 Tools
          </Badge>
        </div>

        <div className="grid grid-cols-5 gap-3 flex-1">
          {toolCategories.map((cat) => (
            <Card key={cat.name} className={`${colorMap[cat.color].bg} ${colorMap[cat.color].border} border`}>
              <CardContent className="p-3">
                <Badge className={`${colorMap[cat.color].badgeBg} ${colorMap[cat.color].text} border-0 mb-3 text-xs`}>
                  {cat.name}
                </Badge>
                <div className="space-y-2">
                  {cat.tools.map((tool) => (
                    <div key={tool.name} className="bg-slate-900/60 rounded-lg p-2">
                      <div className={`font-mono text-xs ${colorMap[cat.color].text} font-bold truncate`}>
                        {tool.name}
                      </div>
                      {tool.params !== "—" && (
                        <div className="text-xs text-slate-500 font-mono truncate mt-0.5">
                          ({tool.params})
                        </div>
                      )}
                      <div className="text-xs text-slate-400 mt-1 line-clamp-2">
                        {tool.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Code Example */}
        <div className="mt-4 grid grid-cols-2 gap-4">
          <Card className="bg-slate-800/50 border-orange-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-orange-500" />
                <span className="text-xs text-orange-400 font-semibold">Command Message</span>
              </div>
              <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs">
                <div className="text-slate-500">{`{`}</div>
                <div className="pl-2"><span className="text-orange-400">&quot;type&quot;</span>: <span className="text-amber-300">&quot;execute_blender_code&quot;</span>,</div>
                <div className="pl-2"><span className="text-orange-400">&quot;params&quot;</span>: {`{`}</div>
                <div className="pl-4"><span className="text-orange-400">&quot;code&quot;</span>: <span className="text-green-400">&quot;bpy.ops.mesh.primitive_cube_add(size=2)&quot;</span></div>
                <div className="pl-2">{`}`}</div>
                <div className="text-slate-500">{`}`}</div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-green-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-xs text-green-400 font-semibold">Response Message</span>
              </div>
              <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs">
                <div className="text-slate-500">{`{`}</div>
                <div className="pl-2"><span className="text-green-400">&quot;status&quot;</span>: <span className="text-emerald-300">&quot;success&quot;</span>,</div>
                <div className="pl-2"><span className="text-green-400">&quot;result&quot;</span>: {`{`}</div>
                <div className="pl-4"><span className="text-green-400">&quot;output&quot;</span>: <span className="text-emerald-300">&quot;Cube created at origin&quot;</span>,</div>
                <div className="pl-4"><span className="text-green-400">&quot;object_name&quot;</span>: <span className="text-emerald-300">&quot;Cube&quot;</span></div>
                <div className="pl-2">{`}`}</div>
                <div className="text-slate-500">{`}`}</div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </MCPSlideWrapper>
  );
}
