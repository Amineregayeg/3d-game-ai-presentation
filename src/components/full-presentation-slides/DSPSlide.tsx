"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";

interface DSPSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function DSPSlide({ slideNumber, totalSlides }: DSPSlideProps) {
  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="Audio Preprocessing">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center mb-6">
          <h2 className="text-4xl font-bold text-white mb-2">
            DSP & Input Processing <span className="text-slate-500">Before the Network</span>
          </h2>
        </div>

        {/* Pipeline Visualization */}
        <div className="flex-1 flex items-center justify-center">
          <svg viewBox="0 0 1000 320" className="w-full max-w-5xl h-auto">
            <defs>
              <linearGradient id="dspGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#0891b2" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="dspGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="dspGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#ea580c" stopOpacity="0.9"/>
              </linearGradient>
              <linearGradient id="dspGrad4" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.9"/>
                <stop offset="100%" stopColor="#059669" stopOpacity="0.9"/>
              </linearGradient>
              <marker id="dspArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>
              </marker>
            </defs>

            {/* Mic Input */}
            <g transform="translate(30, 100)">
              <rect width="130" height="120" rx="12" fill="#1e293b" stroke="#64748b" strokeWidth="2"/>
              <text x="65" y="35" textAnchor="middle" fill="#94a3b8" fontSize="14" fontWeight="bold">Microphone</text>
              <circle cx="65" cy="70" r="25" fill="#0f172a" stroke="#64748b" strokeWidth="1"/>
              <text x="65" y="77" textAnchor="middle" fill="#64748b" fontSize="24">🎤</text>
              <text x="65" y="110" textAnchor="middle" fill="#64748b" fontSize="10">Raw capture</text>
            </g>

            {/* Arrow */}
            <path d="M 170 160 L 200 160" stroke="#64748b" strokeWidth="2" markerEnd="url(#dspArrow)"/>

            {/* Resampling */}
            <g transform="translate(210, 100)">
              <rect width="150" height="120" rx="12" fill="url(#dspGrad1)"/>
              <text x="75" y="30" textAnchor="middle" fill="white" fontSize="13" fontWeight="bold">Resampling</text>
              <rect x="15" y="45" width="120" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="75" y="64" textAnchor="middle" fill="#67e8f9" fontSize="10">→ 16kHz mono</text>
              <rect x="15" y="80" width="120" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="75" y="99" textAnchor="middle" fill="#67e8f9" fontSize="10">torchaudio.resample</text>
            </g>

            {/* Arrow */}
            <path d="M 370 160 L 400 160" stroke="#64748b" strokeWidth="2" markerEnd="url(#dspArrow)"/>

            {/* Normalization */}
            <g transform="translate(410, 100)">
              <rect width="150" height="120" rx="12" fill="url(#dspGrad2)"/>
              <text x="75" y="30" textAnchor="middle" fill="white" fontSize="13" fontWeight="bold">Normalization</text>
              <rect x="15" y="45" width="120" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="75" y="64" textAnchor="middle" fill="#c4b5fd" fontSize="10">Peak normalize</text>
              <rect x="15" y="80" width="120" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="75" y="99" textAnchor="middle" fill="#c4b5fd" fontSize="10">[-1, 1] range</text>
            </g>

            {/* Arrow */}
            <path d="M 570 160 L 600 160" stroke="#64748b" strokeWidth="2" markerEnd="url(#dspArrow)"/>

            {/* Silence Segmentation */}
            <g transform="translate(610, 100)">
              <rect width="150" height="120" rx="12" fill="url(#dspGrad3)"/>
              <text x="75" y="25" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Silence</text>
              <text x="75" y="42" textAnchor="middle" fill="white" fontSize="12" fontWeight="bold">Segmentation</text>
              <rect x="15" y="52" width="120" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="75" y="71" textAnchor="middle" fill="#fdba74" fontSize="10">VAD trimming</text>
              <rect x="15" y="85" width="120" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="75" y="104" textAnchor="middle" fill="#fdba74" fontSize="10">Remove silence</text>
            </g>

            {/* Arrow */}
            <path d="M 770 160 L 800 160" stroke="#64748b" strokeWidth="2" markerEnd="url(#dspArrow)"/>

            {/* Model Input */}
            <g transform="translate(810, 100)">
              <rect width="150" height="120" rx="12" fill="url(#dspGrad4)"/>
              <text x="75" y="30" textAnchor="middle" fill="white" fontSize="13" fontWeight="bold">Model Input</text>
              <rect x="15" y="45" width="120" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="75" y="64" textAnchor="middle" fill="#6ee7b7" fontSize="10">Clean tensor</text>
              <rect x="15" y="80" width="120" height="28" rx="4" fill="rgba(0,0,0,0.3)"/>
              <text x="75" y="99" textAnchor="middle" fill="#6ee7b7" fontSize="10">→ WavLM</text>
            </g>
          </svg>
        </div>

        {/* Key Benefits */}
        <div className="grid grid-cols-3 gap-6 max-w-4xl mx-auto mt-4">
          {[
            {
              icon: "sparkles",
              title: "Cleaner Input",
              desc: "Improves convergence",
              color: "cyan"
            },
            {
              icon: "cpu",
              title: "Reduces Burden",
              desc: "Less model work",
              color: "purple"
            },
            {
              icon: "check",
              title: "Stable Alignment",
              desc: "Better decoding",
              color: "emerald"
            }
          ].map((item) => (
            <div
              key={item.title}
              className={`p-4 bg-${item.color}-500/10 border border-${item.color}-500/30 rounded-xl text-center`}
            >
              <div className={`w-10 h-10 mx-auto mb-2 rounded-lg bg-${item.color}-500/20 flex items-center justify-center`}>
                {item.icon === "sparkles" && (
                  <svg className={`w-5 h-5 text-${item.color}-400`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                  </svg>
                )}
                {item.icon === "cpu" && (
                  <svg className={`w-5 h-5 text-${item.color}-400`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                  </svg>
                )}
                {item.icon === "check" && (
                  <svg className={`w-5 h-5 text-${item.color}-400`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                )}
              </div>
              <h3 className="text-white font-semibold mb-1">{item.title}</h3>
              <p className="text-slate-400 text-sm">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </TechSlideWrapper>
  );
}
