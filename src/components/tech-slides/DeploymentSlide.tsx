"use client";

import { TechSlideWrapper } from "./TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface DeploymentSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function DeploymentSlide({ slideNumber, totalSlides }: DeploymentSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Production Deployment">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          <span className="text-emerald-400">Deployment</span> & Optimization
        </h2>
        <p className="text-slate-400 mb-6">ONNX export, INT8 quantization, and gRPC serving</p>

        <div className="flex-1 grid grid-cols-3 gap-6">
          {/* Export Pipeline */}
          <Card className="bg-slate-800/30 border-slate-700/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Badge className="bg-amber-500/20 text-amber-400">Export</Badge>
                Model Optimization
              </CardTitle>
            </CardHeader>
            <CardContent>
              <svg viewBox="0 0 220 320" className="w-full">
                <defs>
                  <linearGradient id="deployGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#f59e0b"/>
                    <stop offset="50%" stopColor="#06b6d4"/>
                    <stop offset="100%" stopColor="#10b981"/>
                  </linearGradient>
                </defs>

                {/* PyTorch Model */}
                <g transform="translate(35, 10)">
                  <rect width="150" height="50" rx="6" fill="#1e293b" stroke="#f59e0b" strokeWidth="2"/>
                  <text x="75" y="25" textAnchor="middle" fill="#f59e0b" fontSize="11" fontWeight="bold">PyTorch Model</text>
                  <text x="75" y="42" textAnchor="middle" fill="#fcd34d" fontSize="9">~285MB (FP32)</text>
                </g>

                <path d="M110 65 L110 85" stroke="#64748b" strokeWidth="2" markerEnd="url(#deployArrow)"/>

                {/* ONNX Export */}
                <g transform="translate(35, 90)">
                  <rect width="150" height="50" rx="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="2"/>
                  <text x="75" y="25" textAnchor="middle" fill="#06b6d4" fontSize="11" fontWeight="bold">ONNX Export</text>
                  <text x="75" y="42" textAnchor="middle" fill="#67e8f9" fontSize="9">~285MB (FP32)</text>
                </g>

                <path d="M110 145 L110 165" stroke="#64748b" strokeWidth="2" markerEnd="url(#deployArrow)"/>

                {/* FP16 Conversion */}
                <g transform="translate(35, 170)">
                  <rect width="150" height="50" rx="6" fill="#1e293b" stroke="#a855f7" strokeWidth="2"/>
                  <text x="75" y="25" textAnchor="middle" fill="#a855f7" fontSize="11" fontWeight="bold">FP16 Conversion</text>
                  <text x="75" y="42" textAnchor="middle" fill="#c4b5fd" fontSize="9">~145MB (2x smaller)</text>
                </g>

                <path d="M110 225 L110 245" stroke="#64748b" strokeWidth="2" markerEnd="url(#deployArrow)"/>

                {/* INT8 Quantization */}
                <g transform="translate(35, 250)">
                  <rect width="150" height="50" rx="6" fill="#10b981" opacity="0.2" stroke="#10b981" strokeWidth="2"/>
                  <text x="75" y="25" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">INT8 Quantized</text>
                  <text x="75" y="42" textAnchor="middle" fill="#6ee7b7" fontSize="9">~75MB (4x smaller)</text>
                </g>

                <defs>
                  <marker id="deployArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
                  </marker>
                </defs>
              </svg>
            </CardContent>
          </Card>

          {/* Size & Performance Comparison */}
          <Card className="bg-slate-800/30 border-slate-700/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Badge className="bg-cyan-500/20 text-cyan-400">Metrics</Badge>
                Performance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Model Size */}
              <div>
                <div className="text-xs text-slate-400 mb-2">Model Size Reduction</div>
                <div className="space-y-2">
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-amber-400">FP32 (Original)</span>
                      <span className="text-slate-400">285MB</span>
                    </div>
                    <Progress value={100} className="h-2" />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-purple-400">FP16</span>
                      <span className="text-slate-400">145MB</span>
                    </div>
                    <Progress value={50} className="h-2" />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-emerald-400">INT8</span>
                      <span className="text-slate-400">75MB</span>
                    </div>
                    <Progress value={26} className="h-2" />
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div>
                <div className="text-xs text-slate-400 mb-2">Runtime Performance</div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 bg-slate-900/50 rounded text-center">
                    <div className="text-xl font-bold text-cyan-400">&lt;0.1</div>
                    <div className="text-xs text-slate-500">RTF (GPU)</div>
                  </div>
                  <div className="p-2 bg-slate-900/50 rounded text-center">
                    <div className="text-xl font-bold text-purple-400">&lt;0.3</div>
                    <div className="text-xs text-slate-500">RTF (CPU)</div>
                  </div>
                  <div className="p-2 bg-slate-900/50 rounded text-center">
                    <div className="text-xl font-bold text-amber-400">&lt;200</div>
                    <div className="text-xs text-slate-500">TTFT (ms)</div>
                  </div>
                  <div className="p-2 bg-slate-900/50 rounded text-center">
                    <div className="text-xl font-bold text-emerald-400">&lt;0.3%</div>
                    <div className="text-xs text-slate-500">WER Loss</div>
                  </div>
                </div>
              </div>

              {/* Quantization Config */}
              <div>
                <div className="text-xs text-slate-400 mb-2">INT8 Quantization</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between p-1.5 bg-slate-900/30 rounded">
                    <span className="text-slate-400">Type</span>
                    <span className="text-emerald-400">Dynamic</span>
                  </div>
                  <div className="flex justify-between p-1.5 bg-slate-900/30 rounded">
                    <span className="text-slate-400">Calibration</span>
                    <span className="text-emerald-400">dev-clean</span>
                  </div>
                  <div className="flex justify-between p-1.5 bg-slate-900/30 rounded">
                    <span className="text-slate-400">Backend</span>
                    <span className="text-emerald-400">ONNX Runtime</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* gRPC Server Architecture */}
          <Card className="bg-slate-800/30 border-slate-700/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg text-white flex items-center gap-2">
                <Badge className="bg-pink-500/20 text-pink-400">Serving</Badge>
                gRPC Server
              </CardTitle>
            </CardHeader>
            <CardContent>
              <svg viewBox="0 0 200 280" className="w-full">
                {/* Clients */}
                <g transform="translate(10, 10)">
                  <rect width="55" height="40" rx="4" fill="#1e293b" stroke="#ec4899" strokeWidth="1.5"/>
                  <text x="27" y="20" textAnchor="middle" fill="#f9a8d4" fontSize="8">Unity</text>
                  <text x="27" y="32" textAnchor="middle" fill="#f472b6" fontSize="7">C# Client</text>
                </g>

                <g transform="translate(75, 10)">
                  <rect width="55" height="40" rx="4" fill="#1e293b" stroke="#ec4899" strokeWidth="1.5"/>
                  <text x="27" y="20" textAnchor="middle" fill="#f9a8d4" fontSize="8">Unreal</text>
                  <text x="27" y="32" textAnchor="middle" fill="#f472b6" fontSize="7">C++ Client</text>
                </g>

                <g transform="translate(140, 10)">
                  <rect width="55" height="40" rx="4" fill="#1e293b" stroke="#ec4899" strokeWidth="1.5"/>
                  <text x="27" y="20" textAnchor="middle" fill="#f9a8d4" fontSize="8">Python</text>
                  <text x="27" y="32" textAnchor="middle" fill="#f472b6" fontSize="7">Client</text>
                </g>

                {/* Connection lines */}
                <path d="M37 55 L100 85" stroke="#ec4899" strokeWidth="1" opacity="0.5"/>
                <path d="M100 55 L100 85" stroke="#ec4899" strokeWidth="1" opacity="0.5"/>
                <path d="M167 55 L100 85" stroke="#ec4899" strokeWidth="1" opacity="0.5"/>

                {/* gRPC Server */}
                <g transform="translate(25, 90)">
                  <rect width="150" height="80" rx="6" fill="#ec4899" opacity="0.2" stroke="#ec4899" strokeWidth="2"/>
                  <text x="75" y="20" textAnchor="middle" fill="#ec4899" fontSize="10" fontWeight="bold">gRPC Server</text>

                  <rect x="10" y="30" width="60" height="25" rx="3" fill="#1e293b"/>
                  <text x="40" y="47" textAnchor="middle" fill="#f9a8d4" fontSize="7">Transcribe</text>

                  <rect x="80" y="30" width="60" height="25" rx="3" fill="#1e293b"/>
                  <text x="110" y="47" textAnchor="middle" fill="#f9a8d4" fontSize="7">StreamASR</text>

                  <text x="75" y="72" textAnchor="middle" fill="#f472b6" fontSize="7">Proto: voxformer.proto</text>
                </g>

                {/* Model Backend */}
                <g transform="translate(25, 185)">
                  <rect width="150" height="80" rx="6" fill="#10b981" opacity="0.2" stroke="#10b981" strokeWidth="2"/>
                  <text x="75" y="20" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="bold">Model Backend</text>

                  <rect x="10" y="30" width="60" height="22" rx="3" fill="#1e293b"/>
                  <text x="40" y="45" textAnchor="middle" fill="#6ee7b7" fontSize="7">ONNX RT</text>

                  <rect x="80" y="30" width="60" height="22" rx="3" fill="#1e293b"/>
                  <text x="110" y="45" textAnchor="middle" fill="#6ee7b7" fontSize="7">TensorRT</text>

                  <text x="75" y="72" textAnchor="middle" fill="#34d399" fontSize="7">GPU: CUDA | CPU: OpenVINO</text>
                </g>

                {/* Connection */}
                <path d="M100 175 L100 183" stroke="#64748b" strokeWidth="2" markerEnd="url(#deployArrow)"/>
              </svg>

              {/* Proto snippet */}
              <div className="mt-2 p-2 bg-slate-900/50 rounded text-xs font-mono">
                <div className="text-pink-400">service VoxFormer {"{"}</div>
                <div className="text-slate-400 ml-2">rpc Transcribe(Audio)</div>
                <div className="text-slate-400 ml-4">returns (Text);</div>
                <div className="text-slate-400 ml-2">rpc StreamASR(stream Chunk)</div>
                <div className="text-slate-400 ml-4">returns (stream Token);</div>
                <div className="text-pink-400">{"}"}</div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Bottom: Deployment Artifacts */}
        <div className="mt-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-semibold text-white">Final Deliverables</span>
              <span className="text-xs text-slate-500 ml-4">Ready for production deployment</span>
            </div>
            <div className="flex gap-6">
              {[
                { name: "voxformer.onnx", size: "145MB", type: "FP16" },
                { name: "voxformer_int8.onnx", size: "75MB", type: "INT8" },
                { name: "voxformer.trt", size: "~60MB", type: "TensorRT" },
                { name: "server.py", size: "-", type: "gRPC" },
              ].map((file) => (
                <div key={file.name} className="text-center">
                  <div className="text-xs font-mono text-cyan-400">{file.name}</div>
                  <div className="text-xs text-slate-500">{file.size} | {file.type}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </TechSlideWrapper>
  );
}
