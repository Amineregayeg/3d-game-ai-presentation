"use client";

import { AvatarSlideWrapper } from "./AvatarSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface GameEngineSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function GameEngineSlide({ slideNumber, totalSlides }: GameEngineSlideProps) {
  return (
    <AvatarSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Game Engine Integration">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              Game Engine <span className="text-rose-400">Integration</span>
            </h2>
            <p className="text-slate-400">Unity C# and Unreal Engine 5 C++ implementations</p>
          </div>
          <div className="flex gap-2">
            <Badge variant="outline" className="border-blue-500/50 text-blue-400 bg-blue-500/10">
              Unity
            </Badge>
            <Badge variant="outline" className="border-violet-500/50 text-violet-400 bg-violet-500/10">
              Unreal Engine 5
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-6 flex-1">
          {/* Unity Implementation */}
          <Card className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <svg className="w-6 h-6 text-blue-400" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">Unity C#</h3>
                  <span className="text-xs text-slate-500">ElevenLabsTTSManager.cs</span>
                </div>
              </div>

              {/* Code Preview */}
              <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs overflow-hidden mb-4">
                <div className="text-slate-500">// PlayDialogue method</div>
                <div className="text-blue-400">public void <span className="text-cyan-400">PlayDialogue</span>(string text)</div>
                <div className="text-slate-400">{`{`}</div>
                <div className="pl-4 text-slate-400">
                  <span className="text-rose-400">if</span> (cache.TryGetValue(text, <span className="text-blue-400">out</span> clip))
                </div>
                <div className="pl-8 text-slate-400">audioSource.<span className="text-cyan-400">PlayOneShot</span>(clip);</div>
                <div className="pl-4 text-slate-400">
                  <span className="text-rose-400">else</span>
                </div>
                <div className="pl-8 text-slate-400">
                  <span className="text-cyan-400">StartCoroutine</span>(<span className="text-amber-400">RequestTTS</span>(text));
                </div>
                <div className="text-slate-400">{`}`}</div>
              </div>

              {/* Features */}
              <div className="space-y-2">
                {[
                  "UnityWebRequest for ElevenLabs API",
                  "DownloadHandlerAudioClip for streaming",
                  "Dialogue cache with Dictionary<string, AudioClip>",
                  "AudioSource component for playback"
                ].map((feature, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                    <span className="text-slate-400">{feature}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Unreal Engine 5 Implementation */}
          <Card className="bg-gradient-to-br from-violet-500/10 to-purple-500/10 border-violet-500/30">
            <CardContent className="p-4">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-violet-500/20 flex items-center justify-center">
                  <svg className="w-6 h-6 text-violet-400" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">Unreal Engine 5 C++</h3>
                  <span className="text-xs text-slate-500">ATTSManager.h / .cpp</span>
                </div>
              </div>

              {/* Code Preview */}
              <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs overflow-hidden mb-4">
                <div className="text-slate-500">// PlayDialogue declaration</div>
                <div><span className="text-violet-400">UFUNCTION</span>(BlueprintCallable)</div>
                <div><span className="text-blue-400">void</span> <span className="text-cyan-400">PlayDialogue</span>(</div>
                <div className="pl-4"><span className="text-blue-400">const</span> FString& Text,</div>
                <div className="pl-4"><span className="text-blue-400">const</span> FString& CharacterName</div>
                <div>);</div>
                <div className="mt-2 text-slate-500">// FHttpModule request</div>
                <div className="text-purple-300">Request-&gt;<span className="text-cyan-400">SetURL</span>(...)</div>
              </div>

              {/* Features */}
              <div className="space-y-2">
                {[
                  "FHttpModule for REST/WebSocket",
                  "UAudioComponent for playback",
                  "MetaHuman audio-driven lip-sync",
                  "Animation Blueprint integration"
                ].map((feature, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <div className="w-1.5 h-1.5 rounded-full bg-violet-400" />
                    <span className="text-slate-400">{feature}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* MetaHuman Integration */}
        <div className="mt-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center gap-4 mb-3">
            <h3 className="text-sm font-semibold text-white">MetaHuman Audio-Driven Lip-Sync (UE5)</h3>
            <Badge className="bg-violet-500/20 text-violet-400 border-violet-500/50 text-xs">Native Support</Badge>
          </div>
          <div className="relative h-24">
            <svg viewBox="0 0 800 80" className="w-full h-full">
              <defs>
                <linearGradient id="mhGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#8b5cf6"/>
                  <stop offset="100%" stopColor="#a78bfa"/>
                </linearGradient>
              </defs>

              {/* Audio Input */}
              <rect x="20" y="20" width="100" height="40" rx="6" fill="#1e293b" stroke="#8b5cf6" strokeWidth="1.5"/>
              <text x="70" y="44" textAnchor="middle" fill="#8b5cf6" fontSize="10">Audio Waveform</text>

              {/* Arrow */}
              <path d="M125 40 L165 40" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

              {/* Audio Analysis Node */}
              <rect x="170" y="15" width="120" height="50" rx="6" fill="#8b5cf6" fillOpacity="0.2" stroke="#8b5cf6" strokeWidth="1"/>
              <text x="230" y="35" textAnchor="middle" fill="#8b5cf6" fontSize="9" fontWeight="bold">Audio Analysis</text>
              <text x="230" y="50" textAnchor="middle" fill="#c4b5fd" fontSize="8">Spectrum + Phonemes</text>

              {/* Arrow */}
              <path d="M295 40 L335 40" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

              {/* Viseme Mapping */}
              <rect x="340" y="15" width="120" height="50" rx="6" fill="#a78bfa" fillOpacity="0.2" stroke="#a78bfa" strokeWidth="1"/>
              <text x="400" y="35" textAnchor="middle" fill="#a78bfa" fontSize="9" fontWeight="bold">Viseme Mapping</text>
              <text x="400" y="50" textAnchor="middle" fill="#ddd6fe" fontSize="8">ARKit Compatible</text>

              {/* Arrow */}
              <path d="M465 40 L505 40" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

              {/* Pose Modifier */}
              <rect x="510" y="15" width="110" height="50" rx="6" fill="#c4b5fd" fillOpacity="0.2" stroke="#c4b5fd" strokeWidth="1"/>
              <text x="565" y="35" textAnchor="middle" fill="#c4b5fd" fontSize="9" fontWeight="bold">Pose Modifier</text>
              <text x="565" y="50" textAnchor="middle" fill="#e9d5ff" fontSize="8">Bone Controls</text>

              {/* Arrow */}
              <path d="M625 40 L665 40" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#arrowAvatar)"/>

              {/* MetaHuman */}
              <rect x="670" y="15" width="110" height="50" rx="6" fill="url(#mhGrad)" fillOpacity="0.4" stroke="#a78bfa" strokeWidth="1.5"/>
              <text x="725" y="35" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">MetaHuman</text>
              <text x="725" y="50" textAnchor="middle" fill="#e9d5ff" fontSize="8">Animated Face</text>
            </svg>
          </div>
        </div>
      </div>
    </AvatarSlideWrapper>
  );
}
