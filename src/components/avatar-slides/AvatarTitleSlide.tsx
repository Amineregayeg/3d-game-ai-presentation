"use client";

import Image from "next/image";
import { AvatarSlideWrapper } from "./AvatarSlideWrapper";
import { Badge } from "@/components/ui/badge";

interface AvatarTitleSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function AvatarTitleSlide({ slideNumber, totalSlides }: AvatarTitleSlideProps) {
  return (
    <AvatarSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="flex flex-col items-center justify-center h-full text-center space-y-8">
        {/* Logo */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-rose-500 to-pink-500 rounded-full blur-2xl opacity-30 scale-150" />
          <Image
            src="/medtech-logo.png"
            alt="MedTech Logo"
            width={100}
            height={100}
            className="relative drop-shadow-2xl"
          />
        </div>

        {/* Title */}
        <div className="space-y-4">
          <div className="flex items-center justify-center gap-3">
            <Badge variant="outline" className="border-rose-500/50 text-rose-400 bg-rose-500/10">
              Component 3
            </Badge>
            <Badge variant="outline" className="border-pink-500/50 text-pink-400 bg-pink-500/10">
              Production Architecture
            </Badge>
          </div>

          <h1 className="text-6xl md:text-7xl font-bold">
            <span className="bg-gradient-to-r from-rose-400 via-pink-500 to-fuchsia-500 bg-clip-text text-transparent">
              TTS + LipSync
            </span>
          </h1>

          <p className="text-2xl text-slate-400 font-light">
            Voice Synthesis & Avatar Animation for 3D Game AI
          </p>
        </div>

        {/* Model specs preview */}
        <div className="flex flex-wrap justify-center gap-6 mt-8">
          {[
            { label: "ElevenLabs", desc: "Flash 2.5" },
            { label: "75ms", desc: "TTFB" },
            { label: "4.14", desc: "MOS Score" },
            { label: "Wav2Lip", desc: "Lip-Sync" },
            { label: "SadTalker", desc: "Emotions" }
          ].map((item) => (
            <div key={item.label} className="text-center">
              <div className="text-xl font-bold text-white">{item.label}</div>
              <div className="text-xs text-slate-500 uppercase tracking-wider">{item.desc}</div>
            </div>
          ))}
        </div>

        {/* Data flow visualization */}
        <div className="w-full max-w-4xl h-24 relative mt-8">
          <svg viewBox="0 0 600 80" className="w-full h-full">
            <defs>
              <linearGradient id="avatarGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#f43f5e" />
                <stop offset="50%" stopColor="#ec4899" />
                <stop offset="100%" stopColor="#d946ef" />
              </linearGradient>
              <marker id="avatarArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="url(#avatarGradient)"/>
              </marker>
              <filter id="avatarGlow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Text Input box */}
            <rect x="10" y="20" width="80" height="40" rx="8" fill="#1e293b" stroke="#f43f5e" strokeWidth="1.5"/>
            <text x="50" y="38" textAnchor="middle" fill="#f43f5e" fontSize="10" fontFamily="monospace">Text</text>
            <text x="50" y="52" textAnchor="middle" fill="#64748b" fontSize="8">(from RAG)</text>

            {/* Arrow 1 */}
            <path d="M95 40 L130 40" stroke="url(#avatarGradient)" strokeWidth="2" markerEnd="url(#avatarArrow)"/>

            {/* TTS box */}
            <rect x="135" y="15" width="100" height="50" rx="8" fill="#f43f5e" fillOpacity="0.2" stroke="#f43f5e" strokeWidth="1.5" filter="url(#avatarGlow)"/>
            <text x="185" y="38" textAnchor="middle" fill="#f43f5e" fontSize="11" fontWeight="bold">ElevenLabs</text>
            <text x="185" y="52" textAnchor="middle" fill="#fda4af" fontSize="8">TTS Engine</text>

            {/* Arrow 2 */}
            <path d="M240 40 L275 40" stroke="url(#avatarGradient)" strokeWidth="2" markerEnd="url(#avatarArrow)"/>

            {/* Audio Stream box */}
            <rect x="280" y="15" width="90" height="50" rx="8" fill="#ec4899" fillOpacity="0.2" stroke="#ec4899" strokeWidth="1.5" filter="url(#avatarGlow)"/>
            <text x="325" y="35" textAnchor="middle" fill="#ec4899" fontSize="10" fontWeight="bold">Audio</text>
            <text x="325" y="50" textAnchor="middle" fill="#f9a8d4" fontSize="9">Stream</text>

            {/* Split to parallel */}
            <path d="M375 30 L410 20" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#avatarArrow)"/>
            <path d="M375 50 L410 60" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#avatarArrow)"/>

            {/* Playback box */}
            <rect x="415" y="5" width="75" height="30" rx="6" fill="#d946ef" fillOpacity="0.2" stroke="#d946ef" strokeWidth="1"/>
            <text x="452" y="24" textAnchor="middle" fill="#d946ef" fontSize="9">Playback</text>

            {/* LipSync box */}
            <rect x="415" y="45" width="75" height="30" rx="6" fill="#d946ef" fillOpacity="0.2" stroke="#d946ef" strokeWidth="1"/>
            <text x="452" y="64" textAnchor="middle" fill="#d946ef" fontSize="9">Lip-Sync</text>

            {/* Merge arrows */}
            <path d="M495 20 L530 35" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#avatarArrow)"/>
            <path d="M495 60 L530 45" stroke="#64748b" strokeWidth="1.5" markerEnd="url(#avatarArrow)"/>

            {/* Avatar Output */}
            <rect x="535" y="20" width="55" height="40" rx="8" fill="url(#avatarGradient)" fillOpacity="0.3" stroke="url(#avatarGradient)" strokeWidth="2"/>
            <text x="562" y="44" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">Avatar</text>

            {/* Waveform decoration */}
            <g transform="translate(290, 65)">
              {[0,1,2,3,4,5,6,7,8,9].map((i) => (
                <rect
                  key={i}
                  x={i * 7}
                  y={-3 - Math.sin(i * 0.8) * 6}
                  width="4"
                  height={6 + Math.sin(i * 0.8) * 6}
                  fill="#ec4899"
                  opacity={0.4}
                  rx="1"
                />
              ))}
            </g>
          </svg>
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="text-xs text-slate-500 font-mono bg-slate-900/80 px-3 py-1 rounded mt-16">
              Text-to-Speech + Avatar Animation Pipeline
            </span>
          </div>
        </div>

        {/* Performance targets */}
        <div className="flex gap-8 mt-4">
          {[
            { value: "<100ms", label: "TTS Latency", achieved: true },
            { value: ">4.0", label: "Voice Quality", achieved: true },
            { value: "95%+", label: "Lip-Sync", achieved: true },
            { value: "32", label: "Languages", achieved: true }
          ].map((item) => (
            <div key={item.label} className="text-center">
              <div className="flex items-center justify-center gap-1">
                <span className="text-lg font-bold text-rose-400">{item.value}</span>
                {item.achieved && (
                  <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </div>
              <div className="text-xs text-slate-500">{item.label}</div>
            </div>
          ))}
        </div>

        {/* Version and date */}
        <div className="text-xs text-slate-600 font-mono">
          Architecture v1.0 | December 2025
        </div>
      </div>
    </AvatarSlideWrapper>
  );
}
