"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";

interface ProblemHookSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function ProblemHookSlide({ slideNumber, totalSlides }: ProblemHookSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="The Problem">
      <div className="flex flex-col items-center justify-center h-full">
        {/* Header */}
        <h2 className="text-4xl font-bold text-white mb-2 text-center">
          Why This Matters: Blender Is{" "}
          <span className="text-cyan-400">Powerful</span>, But{" "}
          <span className="text-purple-400">Hard to Use</span>
        </h2>

        {/* Visual: Funnel Diagram */}
        <div className="flex-1 flex items-center justify-center w-full max-w-4xl">
          <svg viewBox="0 0 800 400" className="w-full h-auto">
            <defs>
              <linearGradient id="funnelTop" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.8"/>
              </linearGradient>
              <linearGradient id="funnelBottom" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#a855f7" stopOpacity="0.8"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.8"/>
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Funnel shape */}
            <path
              d="M 100 60 L 700 60 L 550 340 L 250 340 Z"
              fill="none"
              stroke="url(#funnelTop)"
              strokeWidth="3"
              filter="url(#glow)"
            />

            {/* Top section - Downloads */}
            <rect x="130" y="80" width="540" height="80" rx="8" fill="#06b6d4" fillOpacity="0.15" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.5"/>
            <text x="400" y="115" textAnchor="middle" fill="white" fontSize="32" fontWeight="bold">14+ Million</text>
            <text x="400" y="145" textAnchor="middle" fill="#94a3b8" fontSize="16">Downloads (Blender, 2020)</text>

            {/* Arrow down */}
            <path d="M 400 175 L 400 200" stroke="#64748b" strokeWidth="2" strokeDasharray="4"/>
            <polygon points="400,215 390,200 410,200" fill="#64748b"/>

            {/* Middle section - Active Users */}
            <rect x="200" y="220" width="400" height="80" rx="8" fill="#a855f7" fillOpacity="0.15" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.5"/>
            <text x="400" y="255" textAnchor="middle" fill="white" fontSize="28" fontWeight="bold">~1-3 Million</text>
            <text x="400" y="285" textAnchor="middle" fill="#94a3b8" fontSize="14">Active Users (estimated)</text>

            {/* Drop-off annotation */}
            <g transform="translate(620, 150)">
              <rect x="0" y="0" width="160" height="60" rx="6" fill="#ef4444" fillOpacity="0.1" stroke="#ef4444" strokeWidth="1" strokeOpacity="0.5"/>
              <text x="80" y="25" textAnchor="middle" fill="#ef4444" fontSize="12" fontWeight="bold">STEEP LEARNING CURVE</text>
              <text x="80" y="45" textAnchor="middle" fill="#fca5a5" fontSize="11">High drop-off rate</text>
            </g>

            {/* Arrow pointing to annotation */}
            <path d="M 600 180 L 620 165" stroke="#ef4444" strokeWidth="1.5" strokeOpacity="0.6"/>
          </svg>
        </div>

        {/* Key Points */}
        <div className="grid grid-cols-3 gap-6 max-w-4xl w-full mt-4">
          {[
            { icon: "cube", text: "Powerful open-source 3D tool" },
            { icon: "layers", text: "Complex UI, tools & workflows" },
            { icon: "trending-down", text: "Beginners struggle to be productive" }
          ].map((item, i) => (
            <div
              key={i}
              className="flex items-center gap-3 p-4 bg-slate-800/30 border border-slate-700/30 rounded-lg"
            >
              <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
                {item.icon === "cube" && (
                  <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                  </svg>
                )}
                {item.icon === "layers" && (
                  <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                )}
                {item.icon === "trending-down" && (
                  <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                  </svg>
                )}
              </div>
              <span className="text-slate-300 text-sm">{item.text}</span>
            </div>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
