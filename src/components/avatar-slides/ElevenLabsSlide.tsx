"use client";

import { AvatarSlideWrapper } from "./AvatarSlideWrapper";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

interface ElevenLabsSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function ElevenLabsSlide({ slideNumber, totalSlides }: ElevenLabsSlideProps) {
  const voiceProfiles = [
    { name: "Rachel", id: "21m00Tcm4TlvDq8ikWAM", type: "Protagonist", traits: "Warm, Professional" },
    { name: "Antoni", id: "MF3mGyEYCHltNTjLimPt", type: "Narrator", traits: "Deep, Authoritative" },
    { name: "Bella", id: "EXAVITQu4emQHoruIezw", type: "NPC", traits: "Young, Cheerful" },
    { name: "Custom", id: "voice_clone_xyz", type: "Antagonist", traits: "Cloned Voice" }
  ];

  return (
    <AvatarSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="TTS Engine">
      <div className="flex flex-col h-full">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-4xl font-bold text-white">
              ElevenLabs <span className="text-rose-400">Flash 2.5</span>
            </h2>
            <p className="text-slate-400">Industry-leading voice synthesis with 75ms latency</p>
          </div>
          <Badge variant="outline" className="border-rose-500/50 text-rose-400 bg-rose-500/10">
            Primary TTS Provider
          </Badge>
        </div>

        <div className="grid grid-cols-2 gap-6 flex-1">
          {/* Left Column - API Architecture */}
          <div className="space-y-4">
            {/* WebSocket Streaming Diagram */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-rose-500" />
                  WebSocket Streaming
                </h3>
                <div className="relative h-32">
                  <svg viewBox="0 0 400 100" className="w-full h-full">
                    <defs>
                      <linearGradient id="wsGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#f43f5e"/>
                        <stop offset="100%" stopColor="#ec4899"/>
                      </linearGradient>
                    </defs>

                    {/* Client */}
                    <rect x="10" y="30" width="80" height="40" rx="6" fill="#1e293b" stroke="#f43f5e" strokeWidth="1.5"/>
                    <text x="50" y="54" textAnchor="middle" fill="#f43f5e" fontSize="10">Game Client</text>

                    {/* WebSocket Connection */}
                    <g transform="translate(100, 45)">
                      <path d="M0 5 L180 5" stroke="url(#wsGrad)" strokeWidth="2" strokeDasharray="8,4"/>
                      <text x="90" y="0" textAnchor="middle" fill="#ec4899" fontSize="8">wss://api.elevenlabs.io</text>

                      {/* Bidirectional arrows */}
                      <polygon points="175,0 185,5 175,10" fill="#ec4899"/>
                      <polygon points="5,10 -5,5 5,0" fill="#f43f5e"/>
                    </g>

                    {/* Audio chunks flowing back */}
                    <g transform="translate(115, 60)">
                      {[0,1,2,3,4].map((i) => (
                        <rect key={i} x={i * 32} y="0" width="25" height="10" rx="2" fill="#ec4899" opacity={0.8 - i * 0.12}/>
                      ))}
                      <text x="80" y="22" textAnchor="middle" fill="#64748b" fontSize="7">Audio chunks (streaming)</text>
                    </g>

                    {/* ElevenLabs Server */}
                    <rect x="290" y="25" width="100" height="50" rx="6" fill="#ec4899" fillOpacity="0.2" stroke="#ec4899" strokeWidth="1.5"/>
                    <text x="340" y="48" textAnchor="middle" fill="#ec4899" fontSize="9" fontWeight="bold">ElevenLabs</text>
                    <text x="340" y="62" textAnchor="middle" fill="#f9a8d4" fontSize="8">Flash 2.5</text>
                  </svg>
                </div>
              </CardContent>
            </Card>

            {/* API Configuration */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-pink-500" />
                  Voice Configuration
                </h3>
                <div className="bg-slate-900/80 rounded-lg p-3 font-mono text-xs">
                  <div className="text-slate-500">// Request payload</div>
                  <div className="text-rose-400">{`{`}</div>
                  <div className="pl-4 text-pink-300">&quot;model_id&quot;: <span className="text-emerald-400">&quot;eleven_flash_v2_5&quot;</span>,</div>
                  <div className="pl-4 text-pink-300">&quot;voice_settings&quot;: {`{`}</div>
                  <div className="pl-8 text-pink-300">&quot;stability&quot;: <span className="text-amber-400">0.5</span>,</div>
                  <div className="pl-8 text-pink-300">&quot;similarity_boost&quot;: <span className="text-amber-400">0.75</span></div>
                  <div className="pl-4 text-pink-300">{`}`},</div>
                  <div className="pl-4 text-pink-300">&quot;text&quot;: <span className="text-emerald-400">&quot;&lt;speak&gt;...&lt;/speak&gt;&quot;</span></div>
                  <div className="text-rose-400">{`}`}</div>
                </div>
              </CardContent>
            </Card>

            {/* SSML Support */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-fuchsia-500" />
                  SSML Emotion Control
                </h3>
                <div className="space-y-2">
                  {[
                    { tag: "<emotion name=\"friendly\">", desc: "Warm greeting tone" },
                    { tag: "<emotion name=\"serious\">", desc: "Dramatic moments" },
                    { tag: "<prosody rate=\"slow\">", desc: "Important lines" },
                    { tag: "<break time=\"500ms\"/>", desc: "Dramatic pauses" }
                  ].map((item) => (
                    <div key={item.tag} className="flex items-center justify-between text-xs">
                      <code className="text-rose-400 font-mono">{item.tag}</code>
                      <span className="text-slate-500">{item.desc}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Voice Profiles & Specs */}
          <div className="space-y-4">
            {/* Performance Specs */}
            <Card className="bg-gradient-to-br from-rose-500/10 to-pink-500/10 border-rose-500/30">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-4">Performance Benchmarks</h3>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: "Time to First Byte", value: "75ms", icon: "lightning" },
                    { label: "Voice Quality (MOS)", value: "4.14", icon: "star" },
                    { label: "Languages Supported", value: "32", icon: "globe" },
                    { label: "Concurrent Streams", value: "1000+", icon: "stream" }
                  ].map((spec) => (
                    <div key={spec.label} className="bg-slate-900/50 rounded-lg p-3">
                      <div className="text-2xl font-bold text-rose-400">{spec.value}</div>
                      <div className="text-xs text-slate-500">{spec.label}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Voice Profiles */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-orange-500" />
                  Character Voice Library
                </h3>
                <div className="space-y-2">
                  {voiceProfiles.map((voice) => (
                    <div key={voice.name} className="flex items-center justify-between bg-slate-900/50 rounded-lg p-2">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-rose-500 to-pink-500 flex items-center justify-center text-white text-xs font-bold">
                          {voice.name[0]}
                        </div>
                        <div>
                          <div className="text-sm font-medium text-white">{voice.name}</div>
                          <div className="text-xs text-slate-500">{voice.traits}</div>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-xs border-slate-600 text-slate-400">
                        {voice.type}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Voice Cloning */}
            <Card className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-4">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-violet-500" />
                  Voice Cloning Workflow
                </h3>
                <div className="flex items-center gap-2">
                  {[
                    { step: "1", label: "Record", desc: "1-min sample" },
                    { step: "2", label: "Upload", desc: "API call" },
                    { step: "3", label: "Process", desc: "~3 hours" },
                    { step: "4", label: "Deploy", desc: "voice_id" }
                  ].map((item, i) => (
                    <div key={item.step} className="flex items-center">
                      <div className="text-center">
                        <div className="w-8 h-8 rounded-full bg-violet-500/20 border border-violet-500/50 flex items-center justify-center text-violet-400 text-xs font-bold mb-1">
                          {item.step}
                        </div>
                        <div className="text-xs text-white">{item.label}</div>
                        <div className="text-xs text-slate-500">{item.desc}</div>
                      </div>
                      {i < 3 && (
                        <svg className="w-6 h-6 text-slate-600 mx-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Bottom: Pricing Info */}
        <div className="mt-4 p-3 bg-slate-800/30 rounded-xl border border-slate-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div>
                <span className="text-xs text-slate-500 uppercase tracking-wider">Pricing</span>
                <div className="text-sm text-slate-300">$5-$330/mo (subscription) | $0.30/1M chars (pay-as-you-go)</div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <div className="text-lg font-bold text-emerald-400">99.9%</div>
                <div className="text-xs text-slate-500">Uptime SLA</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-rose-400">24kHz</div>
                <div className="text-xs text-slate-500">Audio Quality</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </AvatarSlideWrapper>
  );
}
