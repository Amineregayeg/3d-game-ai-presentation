"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MessageSquare, Volume2, Video, Smile, Clock, Zap } from "lucide-react";

interface AvatarTTSSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function AvatarTTSSlide({ slideNumber, totalSlides }: AvatarTTSSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Avatar & TTS System">
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold text-white mb-2">AI Avatar with Lip-Synchronized Speech</h2>
          <p className="text-slate-400">ElevenLabs TTS + MuseTalk real-time lip-sync</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* TTS System */}
          <Card className="bg-slate-900/50 border-emerald-500/30 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-500/10 rounded-lg border border-emerald-500/30">
                  <Volume2 className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <CardTitle className="text-xl text-white">ElevenLabs TTS</CardTitle>
                  <p className="text-sm text-emerald-400">Neural Text-to-Speech</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Features */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <div className="text-xs text-slate-500">VOICES</div>
                  <div className="text-lg font-bold text-emerald-400">29+</div>
                  <div className="text-xs text-slate-500">Premium voices</div>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <div className="text-xs text-slate-500">LATENCY</div>
                  <div className="text-lg font-bold text-emerald-400">&lt;500ms</div>
                  <div className="text-xs text-slate-500">TTFB</div>
                </div>
              </div>

              {/* Voice options */}
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-500 mb-2">AVAILABLE VOICES</div>
                <div className="flex flex-wrap gap-2">
                  {["Rachel", "Adam", "Antoni", "Bella", "Domi", "Elli"].map((voice) => (
                    <Badge
                      key={voice}
                      variant="outline"
                      className="border-emerald-500/30 text-emerald-300 bg-emerald-500/10"
                    >
                      {voice}
                    </Badge>
                  ))}
                  <Badge variant="outline" className="border-slate-600 text-slate-400">+23 more</Badge>
                </div>
              </div>

              {/* Settings */}
              <div className="space-y-2">
                <div className="flex justify-between p-2 bg-slate-800/50 rounded text-sm">
                  <span className="text-slate-400">Stability</span>
                  <span className="text-emerald-400">0.5</span>
                </div>
                <div className="flex justify-between p-2 bg-slate-800/50 rounded text-sm">
                  <span className="text-slate-400">Similarity Boost</span>
                  <span className="text-emerald-400">0.75</span>
                </div>
                <div className="flex justify-between p-2 bg-slate-800/50 rounded text-sm">
                  <span className="text-slate-400">Model</span>
                  <span className="text-emerald-400">eleven_multilingual_v2</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Lip Sync System */}
          <Card className="bg-slate-900/50 border-pink-500/30 backdrop-blur-sm">
            <CardHeader className="pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-pink-500/10 rounded-lg border border-pink-500/30">
                  <Video className="w-5 h-5 text-pink-400" />
                </div>
                <div>
                  <CardTitle className="text-xl text-white">MuseTalk Lip-Sync</CardTitle>
                  <p className="text-sm text-pink-400">Real-time Face Animation</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Features */}
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <div className="text-xs text-slate-500">OUTPUT</div>
                  <div className="text-lg font-bold text-pink-400">25 FPS</div>
                  <div className="text-xs text-slate-500">Video generation</div>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <div className="text-xs text-slate-500">PROCESSING</div>
                  <div className="text-lg font-bold text-pink-400">GPU</div>
                  <div className="text-xs text-slate-500">Accelerated</div>
                </div>
              </div>

              {/* Pipeline */}
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-500 mb-2">PROCESSING PIPELINE</div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <Badge variant="outline" className="border-pink-500/30 text-pink-300">Audio Input</Badge>
                  <span className="text-slate-500">→</span>
                  <Badge variant="outline" className="border-pink-500/30 text-pink-300">Phoneme Extract</Badge>
                  <span className="text-slate-500">→</span>
                  <Badge variant="outline" className="border-pink-500/30 text-pink-300">Face Landmarks</Badge>
                  <span className="text-slate-500">→</span>
                  <Badge variant="outline" className="border-pink-500/30 text-pink-300">Video Render</Badge>
                </div>
              </div>

              {/* Avatar options */}
              <div className="p-3 bg-slate-800/50 rounded-lg">
                <div className="text-xs text-slate-500 mb-2">AVATAR STYLES</div>
                <div className="flex gap-3">
                  {[
                    { name: "Default", color: "bg-cyan-500" },
                    { name: "Professional", color: "bg-purple-500" },
                    { name: "Casual", color: "bg-emerald-500" },
                  ].map((avatar) => (
                    <div key={avatar.name} className="flex flex-col items-center">
                      <div className={`w-10 h-10 ${avatar.color} rounded-full mb-1 flex items-center justify-center`}>
                        <Smile className="w-5 h-5 text-white" />
                      </div>
                      <span className="text-xs text-slate-400">{avatar.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Auto-greeting behavior */}
        <Card className="bg-slate-900/30 border-cyan-500/30">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <MessageSquare className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-semibold text-white">Automatic Greeting Behavior</h3>
              <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10 text-xs ml-2">
                On Page Load
              </Badge>
            </div>
            <div className="p-3 bg-slate-800/50 rounded-lg border-l-2 border-cyan-500/50">
              <p className="text-slate-300 italic text-sm">
                &quot;Hello! I&apos;m your 3D Game AI Assistant. I can help you create 3D models, materials,
                and animations in Blender. Just click the microphone and tell me what you&apos;d like to create!&quot;
              </p>
            </div>
            <div className="flex items-center gap-6 mt-3 text-xs text-slate-400">
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                <span>~8s greeting duration</span>
              </div>
              <div className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                <span>Pre-generated for instant playback</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </TechSlideWrapper>
  );
}
