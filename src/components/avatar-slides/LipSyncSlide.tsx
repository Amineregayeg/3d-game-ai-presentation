"use client";

import { AvatarSlideWrapper } from "./AvatarSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface LipSyncSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function LipSyncSlide({ slideNumber, totalSlides }: LipSyncSlideProps) {
  return (
    <AvatarSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Lip-Synchronization">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Hybrid <span className="text-rose-400">Lip-Sync</span> Strategy
            </h2>
            <p className="text-slate-400">Wav2Lip for real-time + SadTalker for emotional expressions</p>
          </div>
          <Badge variant="outline" className="border-fuchsia-500/50 text-fuchsia-400 bg-fuchsia-500/10">
            Parallel Processing
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-6 flex-1">
          {/* Wav2Lip */}
          <Card className="bg-gradient-to-br from-rose-500/10 to-pink-500/10 border-rose-500/30">
            <CardContent className="p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-white">Wav2Lip</h3>
                <Badge className="bg-rose-500/20 text-rose-400 border-rose-500/50">Standard Dialogue</Badge>
              </div>

              {/* Architecture Diagram */}
              <div className="bg-slate-900/50 rounded-lg p-4 mb-4">
                <svg viewBox="0 0 350 120" className="w-full h-24">
                  <defs>
                    <linearGradient id="w2lGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#f43f5e"/>
                      <stop offset="100%" stopColor="#ec4899"/>
                    </linearGradient>
                  </defs>

                  {/* Audio Input */}
                  <rect x="5" y="40" width="60" height="40" rx="6" fill="#1e293b" stroke="#f43f5e" strokeWidth="1"/>
                  <text x="35" y="58" textAnchor="middle" fill="#f43f5e" fontSize="8">Audio</text>
                  <text x="35" y="70" textAnchor="middle" fill="#64748b" fontSize="7">16kHz PCM</text>

                  {/* Arrow */}
                  <path d="M70 60 L95 60" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

                  {/* MFCC Extraction */}
                  <rect x="100" y="35" width="65" height="50" rx="6" fill="#f43f5e" fillOpacity="0.2" stroke="#f43f5e" strokeWidth="1"/>
                  <text x="132" y="55" textAnchor="middle" fill="#f43f5e" fontSize="8" fontWeight="bold">MFCC</text>
                  <text x="132" y="70" textAnchor="middle" fill="#fda4af" fontSize="7">39 features</text>

                  {/* Arrow */}
                  <path d="M170 60 L195 60" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

                  {/* Neural Network */}
                  <rect x="200" y="30" width="70" height="60" rx="6" fill="#ec4899" fillOpacity="0.3" stroke="#ec4899" strokeWidth="1"/>
                  <text x="235" y="50" textAnchor="middle" fill="#ec4899" fontSize="8" fontWeight="bold">Encoder</text>
                  <text x="235" y="62" textAnchor="middle" fill="#f9a8d4" fontSize="7">Attention</text>
                  <text x="235" y="74" textAnchor="middle" fill="#f9a8d4" fontSize="7">Decoder</text>

                  {/* Arrow */}
                  <path d="M275 60 L300 60" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

                  {/* Output */}
                  <rect x="305" y="40" width="40" height="40" rx="6" fill="url(#w2lGrad)" fillOpacity="0.4" stroke="#ec4899" strokeWidth="1"/>
                  <text x="325" y="58" textAnchor="middle" fill="white" fontSize="8">Lips</text>
                  <text x="325" y="70" textAnchor="middle" fill="#f9a8d4" fontSize="6">Sync</text>
                </svg>
              </div>

              {/* Specs Grid */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                {[
                  { label: "Accuracy", value: "95%+", color: "text-rose-400" },
                  { label: "Latency", value: "100-300ms", color: "text-pink-400" },
                  { label: "GPU VRAM", value: "4-6GB", color: "text-fuchsia-400" },
                  { label: "FPS", value: "25-30", color: "text-orange-400" }
                ].map((spec) => (
                  <div key={spec.label} className="bg-slate-900/50 rounded p-2">
                    <div className={`text-lg font-bold ${spec.color}`}>{spec.value}</div>
                    <div className="text-xs text-slate-500">{spec.label}</div>
                  </div>
                ))}
              </div>

              {/* Pros/Cons */}
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-xs text-slate-400">Production-proven, high accuracy lip-sync</span>
                </div>
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-xs text-slate-400">Speaker-agnostic (works with any voice)</span>
                </div>
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <span className="text-xs text-slate-400">Mouth only - no head movement/expressions</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* SadTalker */}
          <Card className="bg-gradient-to-br from-fuchsia-500/10 to-violet-500/10 border-fuchsia-500/30">
            <CardContent className="p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-white">SadTalker</h3>
                <Badge className="bg-fuchsia-500/20 text-fuchsia-400 border-fuchsia-500/50">Cinematic Moments</Badge>
              </div>

              {/* Architecture Diagram */}
              <div className="bg-slate-900/50 rounded-lg p-4 mb-4">
                <svg viewBox="0 0 350 120" className="w-full h-24">
                  <defs>
                    <linearGradient id="stGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#d946ef"/>
                      <stop offset="100%" stopColor="#a78bfa"/>
                    </linearGradient>
                  </defs>

                  {/* Audio + Image Input */}
                  <g transform="translate(5, 25)">
                    <rect width="55" height="30" rx="4" fill="#1e293b" stroke="#d946ef" strokeWidth="1"/>
                    <text x="27" y="18" textAnchor="middle" fill="#d946ef" fontSize="8">Audio</text>
                  </g>
                  <g transform="translate(5, 60)">
                    <rect width="55" height="30" rx="4" fill="#1e293b" stroke="#a78bfa" strokeWidth="1"/>
                    <text x="27" y="18" textAnchor="middle" fill="#a78bfa" fontSize="8">Image</text>
                  </g>

                  {/* Arrows */}
                  <path d="M65 40 L85 55" stroke="#64748b" strokeWidth="1"/>
                  <path d="M65 75 L85 60" stroke="#64748b" strokeWidth="1"/>

                  {/* 3D Face Model */}
                  <rect x="90" y="35" width="65" height="50" rx="6" fill="#d946ef" fillOpacity="0.2" stroke="#d946ef" strokeWidth="1"/>
                  <text x="122" y="53" textAnchor="middle" fill="#d946ef" fontSize="8" fontWeight="bold">3D Face</text>
                  <text x="122" y="68" textAnchor="middle" fill="#e9d5ff" fontSize="7">Morphable</text>

                  {/* Arrow */}
                  <path d="M160 60 L185 60" stroke="#64748b" strokeWidth="1.5"/>

                  {/* Motion Prediction */}
                  <rect x="190" y="30" width="70" height="60" rx="6" fill="#a78bfa" fillOpacity="0.3" stroke="#a78bfa" strokeWidth="1"/>
                  <text x="225" y="48" textAnchor="middle" fill="#a78bfa" fontSize="8" fontWeight="bold">Motion</text>
                  <text x="225" y="60" textAnchor="middle" fill="#c4b5fd" fontSize="7">Pose +</text>
                  <text x="225" y="72" textAnchor="middle" fill="#c4b5fd" fontSize="7">Emotion</text>

                  {/* Arrow */}
                  <path d="M265 60 L290 60" stroke="#64748b" strokeWidth="1.5"/>

                  {/* Output */}
                  <rect x="295" y="35" width="50" height="50" rx="6" fill="url(#stGrad)" fillOpacity="0.4" stroke="#a78bfa" strokeWidth="1"/>
                  <text x="320" y="55" textAnchor="middle" fill="white" fontSize="8">Full</text>
                  <text x="320" y="67" textAnchor="middle" fill="#e9d5ff" fontSize="7">Animation</text>
                </svg>
              </div>

              {/* Specs Grid */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                {[
                  { label: "Expression", value: "Natural", color: "text-fuchsia-400" },
                  { label: "Latency", value: "200-500ms", color: "text-violet-400" },
                  { label: "GPU VRAM", value: "8-12GB", color: "text-purple-400" },
                  { label: "Emotions", value: "5+", color: "text-pink-400" }
                ].map((spec) => (
                  <div key={spec.label} className="bg-slate-900/50 rounded p-2">
                    <div className={`text-lg font-bold ${spec.color}`}>{spec.value}</div>
                    <div className="text-xs text-slate-500">{spec.label}</div>
                  </div>
                ))}
              </div>

              {/* Features */}
              <div className="space-y-2">
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-xs text-slate-400">Head poses (nodding, tilting)</span>
                </div>
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-xs text-slate-400">Eye blinking + gaze control</span>
                </div>
                <div className="flex items-start gap-2">
                  <svg className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-xs text-slate-400">Emotional expressions from audio</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Hybrid Strategy */}
        <div className="mt-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm text-rose-400 font-semibold">Hybrid Decision Logic</span>
              <div className="flex items-center gap-8 mt-2">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-rose-500" />
                  <span className="text-sm text-slate-300">Standard NPC dialogue → Wav2Lip (fast)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-fuchsia-500" />
                  <span className="text-sm text-slate-300">Story cutscenes / emotional → SadTalker (expressive)</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-center">
                <div className="text-xl font-bold text-rose-400">80%</div>
                <div className="text-xs text-slate-500">Wav2Lip Usage</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-fuchsia-400">20%</div>
                <div className="text-xs text-slate-500">SadTalker Usage</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </AvatarSlideWrapper>
  );
}
