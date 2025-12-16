"use client";

import { AvatarSlideWrapper } from "./AvatarSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";

interface AvatarArchitectureSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function AvatarArchitectureSlide({ slideNumber, totalSlides }: AvatarArchitectureSlideProps) {
  return (
    <AvatarSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="System Architecture">
      <div className="flex flex-col h-full">
        <h2 className="text-4xl font-bold text-white mb-2">
          TTS + LipSync <span className="text-rose-400">Pipeline Architecture</span>
        </h2>
        <p className="text-slate-400 mb-4">Real-time voice synthesis with synchronized avatar animation</p>

        {/* Main Architecture Diagram */}
        <div className="flex-1 relative">
          <svg viewBox="0 0 1000 380" className="w-full h-full">
            <defs>
              <linearGradient id="gradRose" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#e11d48" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradPink" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ec4899" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#db2777" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradFuchsia" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#d946ef" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#c026d3" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradOrange" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#fb923c" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#ea580c" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="gradViolet" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.8"/>
              </linearGradient>
              <filter id="glowAvatar">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
              <marker id="arrowAvatar" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
            </defs>

            {/* Text Input from RAG */}
            <g transform="translate(20, 140)">
              <rect width="90" height="80" rx="8" fill="#1e293b" stroke="#10b981" strokeWidth="2"/>
              <text x="45" y="25" textAnchor="middle" fill="#10b981" fontSize="10" fontFamily="monospace">Component 2</text>
              <text x="45" y="42" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold">RAG Output</text>
              <rect x="10" y="52" width="70" height="20" rx="4" fill="#10b981" fillOpacity="0.2"/>
              <text x="45" y="66" textAnchor="middle" fill="#a7f3d0" fontSize="8">&quot;Hello there!&quot;</text>
            </g>

            {/* Arrow 1 */}
            <path d="M115 180 L155 180" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowAvatar)"/>

            {/* Text Preprocessing */}
            <g transform="translate(160, 130)">
              <rect width="120" height="100" rx="8" fill="url(#gradRose)" filter="url(#glowAvatar)"/>
              <text x="60" y="25" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Text Processing</text>
              <rect x="10" y="35" width="100" height="20" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="49" textAnchor="middle" fill="#fda4af" fontSize="8">SSML Annotation</text>
              <rect x="10" y="58" width="100" height="20" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="72" textAnchor="middle" fill="#fda4af" fontSize="8">Chunking (1000 chars)</text>
              <rect x="10" y="81" width="100" height="12" rx="2" fill="#0f172a" opacity="0.3"/>
              <text x="60" y="90" textAnchor="middle" fill="#fda4af" fontSize="7">Emotion Tags</text>
            </g>

            {/* Arrow 2 */}
            <path d="M285 180 L325 180" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowAvatar)"/>

            {/* ElevenLabs TTS Engine */}
            <g transform="translate(330, 110)">
              <rect width="150" height="140" rx="10" fill="url(#gradPink)" filter="url(#glowAvatar)"/>
              <text x="75" y="25" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">ElevenLabs</text>
              <text x="75" y="42" textAnchor="middle" fill="#f9a8d4" fontSize="9">Flash 2.5 TTS</text>

              {/* WebSocket indicator */}
              <rect x="15" y="52" width="120" height="25" rx="4" fill="#0f172a" opacity="0.5"/>
              <text x="75" y="68" textAnchor="middle" fill="#f9a8d4" fontSize="9">WebSocket Streaming</text>

              {/* Performance specs */}
              <rect x="15" y="82" width="55" height="45" rx="4" fill="#0f172a" opacity="0.4"/>
              <text x="42" y="98" textAnchor="middle" fill="#fbcfe8" fontSize="8">TTFB</text>
              <text x="42" y="115" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">75ms</text>

              <rect x="80" y="82" width="55" height="45" rx="4" fill="#0f172a" opacity="0.4"/>
              <text x="107" y="98" textAnchor="middle" fill="#fbcfe8" fontSize="8">MOS</text>
              <text x="107" y="115" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold">4.14</text>
            </g>

            {/* Arrow 3 */}
            <path d="M485 180 L525 180" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowAvatar)"/>

            {/* Audio Stream */}
            <g transform="translate(530, 150)">
              <rect width="100" height="60" rx="8" fill="#1e293b" stroke="#ec4899" strokeWidth="2"/>
              <text x="50" y="22" textAnchor="middle" fill="#ec4899" fontSize="10" fontWeight="bold">Audio Stream</text>
              <text x="50" y="38" textAnchor="middle" fill="#64748b" fontSize="8">PCM 24kHz</text>
              {/* Waveform */}
              <g transform="translate(15, 42)">
                {[0,1,2,3,4,5,6,7,8,9,10,11,12,13].map((i) => (
                  <rect
                    key={i}
                    x={i * 5}
                    y={8 - Math.sin(i * 0.6) * 6}
                    width="3"
                    height={Math.abs(Math.sin(i * 0.6) * 12) + 2}
                    fill="#ec4899"
                    opacity={0.6}
                    rx="1"
                  />
                ))}
              </g>
            </g>

            {/* Split arrows to parallel processing */}
            <path d="M635 165 L680 100" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowAvatar)"/>
            <path d="M635 195 L680 260" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowAvatar)"/>

            {/* Audio Playback Branch */}
            <g transform="translate(685, 55)">
              <rect width="120" height="80" rx="8" fill="url(#gradOrange)" filter="url(#glowAvatar)"/>
              <text x="60" y="22" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Audio Playback</text>
              <rect x="10" y="32" width="100" height="16" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="43" textAnchor="middle" fill="#fed7aa" fontSize="7">Circular Buffer</text>
              <rect x="10" y="52" width="100" height="16" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="63" textAnchor="middle" fill="#fed7aa" fontSize="7">AudioSource Play</text>
            </g>

            {/* LipSync Branch */}
            <g transform="translate(685, 225)">
              <rect width="120" height="100" rx="8" fill="url(#gradFuchsia)" filter="url(#glowAvatar)"/>
              <text x="60" y="22" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Lip-Sync Engine</text>

              {/* Wav2Lip */}
              <rect x="10" y="32" width="100" height="28" rx="4" fill="#0f172a" opacity="0.5"/>
              <text x="60" y="45" textAnchor="middle" fill="#f5d0fe" fontSize="8" fontWeight="bold">Wav2Lip</text>
              <text x="60" y="56" textAnchor="middle" fill="#e9d5ff" fontSize="7">Real-time (95%+)</text>

              {/* SadTalker */}
              <rect x="10" y="64" width="100" height="28" rx="4" fill="#0f172a" opacity="0.5"/>
              <text x="60" y="77" textAnchor="middle" fill="#f5d0fe" fontSize="8" fontWeight="bold">SadTalker</text>
              <text x="60" y="88" textAnchor="middle" fill="#e9d5ff" fontSize="7">Emotional Expressions</text>
            </g>

            {/* Merge arrows */}
            <path d="M810 95 L855 160" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowAvatar)"/>
            <path d="M810 275 L855 210" stroke="#64748b" strokeWidth="2" markerEnd="url(#arrowAvatar)"/>

            {/* Avatar Animation Output */}
            <g transform="translate(860, 140)">
              <rect width="120" height="100" rx="10" fill="url(#gradViolet)" filter="url(#glowAvatar)"/>
              <text x="60" y="25" textAnchor="middle" fill="white" fontSize="11" fontWeight="bold">Game Engine</text>
              <rect x="10" y="35" width="100" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="47" textAnchor="middle" fill="#ddd6fe" fontSize="8">Blend Shapes</text>
              <rect x="10" y="56" width="100" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="68" textAnchor="middle" fill="#ddd6fe" fontSize="8">MetaHuman Sync</text>
              <rect x="10" y="77" width="100" height="18" rx="3" fill="#0f172a" opacity="0.4"/>
              <text x="60" y="89" textAnchor="middle" fill="#ddd6fe" fontSize="8">Avatar Animation</text>
            </g>

            {/* Latency indicators */}
            <g transform="translate(160, 255)">
              <text x="0" y="0" fill="#64748b" fontSize="8" fontFamily="monospace">5ms</text>
              <path d="M0 5 L120 5" stroke="#f43f5e" strokeWidth="1" strokeDasharray="3"/>
            </g>
            <g transform="translate(330, 275)">
              <text x="0" y="0" fill="#64748b" fontSize="8" fontFamily="monospace">75ms TTFB</text>
              <path d="M0 5 L150 5" stroke="#ec4899" strokeWidth="1" strokeDasharray="3"/>
            </g>
            <g transform="translate(685, 340)">
              <text x="0" y="0" fill="#64748b" fontSize="8" fontFamily="monospace">100-150ms parallel</text>
              <path d="M0 5 L120 5" stroke="#d946ef" strokeWidth="1" strokeDasharray="3"/>
            </g>

            {/* Cache indicator */}
            <g transform="translate(330, 15)">
              <rect width="150" height="40" rx="6" fill="#374151" stroke="#4b5563" strokeWidth="1" strokeDasharray="4"/>
              <text x="75" y="18" textAnchor="middle" fill="#9ca3af" fontSize="9" fontWeight="bold">Dialogue Cache</text>
              <text x="75" y="32" textAnchor="middle" fill="#6b7280" fontSize="8">80-90% hit rate</text>
            </g>
            <path d="M405 55 L405 108" stroke="#64748b" strokeWidth="1" strokeDasharray="4"/>
          </svg>
        </div>

        {/* Performance Metrics */}
        <div className="flex justify-center gap-4 mt-2">
          {[
            { metric: "75ms", label: "TTS Latency", color: "text-rose-400" },
            { metric: "4.14", label: "MOS Quality", color: "text-pink-400" },
            { metric: "95%+", label: "Lip-Sync Accuracy", color: "text-fuchsia-400" },
            { metric: "<300ms", label: "End-to-End", color: "text-orange-400" }
          ].map((item) => (
            <Card key={item.label} className="bg-slate-800/50 border-slate-700/50">
              <CardContent className="p-3 text-center">
                <div className={`text-lg font-bold ${item.color}`}>{item.metric}</div>
                <div className="text-xs text-slate-500">{item.label}</div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </AvatarSlideWrapper>
  );
}
