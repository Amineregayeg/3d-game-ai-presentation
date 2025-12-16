"use client";

import { SlideWrapper } from "./SlideWrapper";

interface TTSLipsyncSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function TTSLipsyncSlide({ slideNumber, totalSlides }: TTSLipsyncSlideProps) {
  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
        {/* Left - TTS Section */}
        <div className="space-y-6">
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-orange-500/10 border border-orange-500/30 rounded-full">
              <div className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" />
              <span className="text-orange-400 text-sm font-medium">Module 3</span>
            </div>
            <h2 className="text-5xl font-bold text-white">
              TTS + <span className="bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">LipSync</span>
            </h2>
            <p className="text-xl text-slate-400">
              Premium voice synthesis with real-time avatar animation
            </p>
          </div>

          {/* TTS Options */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Text-to-Speech Providers</h3>

            <div className="p-4 bg-slate-800/30 border border-green-500/30 rounded-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                    <svg className="w-5 h-5 text-green-400" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M22.5 12c0 5.8-4.7 10.5-10.5 10.5S1.5 17.8 1.5 12 6.2 1.5 12 1.5 22.5 6.2 22.5 12z"/>
                    </svg>
                  </div>
                  <div>
                    <div className="font-medium text-white">OpenAI TTS</div>
                    <div className="text-xs text-slate-500">HD quality, multiple voices</div>
                  </div>
                </div>
                <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">Recommended</span>
              </div>
            </div>

            <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <div>
                  <div className="font-medium text-white">ElevenLabs</div>
                  <div className="text-xs text-slate-500">Voice cloning, emotional range</div>
                </div>
              </div>
            </div>

            <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                  </svg>
                </div>
                <div>
                  <div className="font-medium text-white">Azure Neural TTS</div>
                  <div className="text-xs text-slate-500">Enterprise-grade, SSML support</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right - LipSync & Avatar */}
        <div className="space-y-6">
          {/* Avatar visualization */}
          <div className="relative p-8 bg-slate-800/30 border border-slate-700/50 rounded-3xl">
            <div className="absolute inset-0 bg-gradient-to-br from-orange-500/10 to-red-500/10 rounded-3xl" />

            <div className="relative flex flex-col items-center space-y-6">
              {/* Avatar placeholder */}
              <div className="w-40 h-40 bg-gradient-to-br from-orange-500 to-red-600 rounded-full flex items-center justify-center shadow-2xl shadow-orange-500/30">
                <svg className="w-20 h-20 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>

              {/* Audio waveform visualization */}
              <div className="flex items-end gap-1 h-8">
                {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((i) => (
                  <div
                    key={i}
                    className="w-2 bg-gradient-to-t from-orange-500 to-red-500 rounded-full animate-pulse"
                    style={{
                      height: `${Math.random() * 24 + 8}px`,
                      animationDelay: `${i * 0.1}s`
                    }}
                  />
                ))}
              </div>

              <p className="text-slate-400 text-sm text-center">
                Real-time lip-sync avatar responds naturally
              </p>
            </div>
          </div>

          {/* LipSync tech */}
          <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl">
            <h4 className="font-semibold text-white mb-3">LipSync Technology</h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 bg-slate-900/50 rounded-lg">
                <div className="text-orange-400 font-medium text-sm">Viseme Mapping</div>
                <div className="text-xs text-slate-500">Audio-to-mouth shape sync</div>
              </div>
              <div className="p-3 bg-slate-900/50 rounded-lg">
                <div className="text-orange-400 font-medium text-sm">Blend Shapes</div>
                <div className="text-xs text-slate-500">Smooth facial animations</div>
              </div>
              <div className="p-3 bg-slate-900/50 rounded-lg">
                <div className="text-orange-400 font-medium text-sm">Emotion Detection</div>
                <div className="text-xs text-slate-500">Expressive responses</div>
              </div>
              <div className="p-3 bg-slate-900/50 rounded-lg">
                <div className="text-orange-400 font-medium text-sm">3D Ready</div>
                <div className="text-xs text-slate-500">Unity/Unreal compatible</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </SlideWrapper>
  );
}
