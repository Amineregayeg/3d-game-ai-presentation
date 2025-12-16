"use client";

import { SlideWrapper } from "./SlideWrapper";

interface STTSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function STTSlide({ slideNumber, totalSlides }: STTSlideProps) {
  const features = [
    {
      title: "WavLM Backbone",
      description: "Pretrained audio features from 94,000 hours of speech data",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
        </svg>
      )
    },
    {
      title: "Custom Transformer",
      description: "6-layer Zipformer encoder + 4-layer decoder built from scratch",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      )
    },
    {
      title: "Real-time Streaming",
      description: "Sub-200ms latency with chunked processing and KV-cache",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      title: "Gaming Optimized",
      description: "Fine-tuned on gaming vocabulary and voice commands",
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 00-1-1H4a2 2 0 110-4h1a1 1 0 001-1V7a1 1 0 011-1h3a1 1 0 001-1V4z" />
        </svg>
      )
    }
  ];

  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
        {/* Left content */}
        <div className="space-y-8">
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full">
              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
              <span className="text-cyan-400 text-sm font-medium">Module 1</span>
            </div>
            <h2 className="text-5xl font-bold text-white">
              <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">VoxFormer</span> STT
            </h2>
            <p className="text-xl text-slate-400">
              WavLM backbone + custom Zipformer transformer for elite-level speech recognition
            </p>
          </div>

          {/* Features */}
          <div className="space-y-4">
            {features.map((feature) => (
              <div key={feature.title} className="flex items-start gap-4 p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl hover:border-cyan-500/30 transition-colors">
                <div className="flex-shrink-0 w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center text-cyan-400">
                  {feature.icon}
                </div>
                <div>
                  <h3 className="font-semibold text-white">{feature.title}</h3>
                  <p className="text-sm text-slate-400">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right visualization */}
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 rounded-3xl blur-3xl" />
          <div className="relative bg-slate-800/50 border border-slate-700/50 rounded-3xl p-8 backdrop-blur-sm">
            {/* Architecture diagram */}
            <div className="space-y-6">
              <h3 className="text-lg font-semibold text-white text-center">Architecture Overview</h3>

              <div className="space-y-3">
                {/* WavLM */}
                <div className="p-3 bg-slate-900/50 border border-purple-500/30 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-purple-400 rounded-full" />
                      <span className="font-medium text-purple-400">WavLM-Base</span>
                    </div>
                    <span className="text-xs text-slate-500">95M params (frozen)</span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500 pl-6">
                    Pretrained audio feature extractor → 768-dim @ 50fps
                  </p>
                </div>

                {/* Arrow */}
                <div className="flex justify-center">
                  <svg className="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                  </svg>
                </div>

                {/* Zipformer Encoder */}
                <div className="p-3 bg-slate-900/50 border border-cyan-500/30 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-cyan-400 rounded-full" />
                      <span className="font-medium text-cyan-400">Zipformer Encoder</span>
                    </div>
                    <span className="text-xs text-slate-500">25M params (custom)</span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500 pl-6">
                    6 Conformer blocks with U-Net downsampling
                  </p>
                </div>

                {/* Arrow */}
                <div className="flex justify-center">
                  <svg className="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                  </svg>
                </div>

                {/* Transformer Decoder */}
                <div className="p-3 bg-slate-900/50 border border-amber-500/30 rounded-xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-amber-400 rounded-full" />
                      <span className="font-medium text-amber-400">Transformer Decoder</span>
                    </div>
                    <span className="text-xs text-slate-500">20M params (custom)</span>
                  </div>
                  <p className="mt-1 text-xs text-slate-500 pl-6">
                    4 layers with cross-attention + BPE 2K vocab
                  </p>
                </div>
              </div>

              {/* Performance metrics */}
              <div className="grid grid-cols-4 gap-3 pt-4 border-t border-slate-700/50">
                <div className="text-center">
                  <div className="text-xl font-bold text-cyan-400">&lt;3.5%</div>
                  <div className="text-xs text-slate-500">WER</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-purple-400">&lt;200ms</div>
                  <div className="text-xs text-slate-500">Latency</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-amber-400">$20</div>
                  <div className="text-xs text-slate-500">Training</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-emerald-400">7 days</div>
                  <div className="text-xs text-slate-500">Dev Time</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </SlideWrapper>
  );
}
