"use client";

interface CoverSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function CoverSlide({ slideNumber, totalSlides }: CoverSlideProps) {
  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center p-8 relative overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Animated grid background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0ea5e910_1px,transparent_1px),linear-gradient(to_bottom,#0ea5e910_1px,transparent_1px)] bg-[size:4rem_4rem]" />

      {/* Gradient orbs */}
      <div className="absolute top-0 left-1/4 w-[800px] h-[800px] bg-cyan-500/8 rounded-full blur-[150px]" />
      <div className="absolute bottom-0 right-1/4 w-[800px] h-[800px] bg-purple-500/8 rounded-full blur-[150px]" />

      {/* Main content */}
      <div className="relative z-10 text-center max-w-4xl">
        {/* Logos */}
        <div className="flex items-center justify-center gap-12 mb-12">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/3dgameassistant/logo.png"
            alt="Platform Logo"
            width={80}
            height={80}
            className="opacity-90"
          />
          <div className="h-16 w-px bg-gradient-to-b from-transparent via-slate-600 to-transparent" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/3dgameassistant/medtech-logo.png"
            alt="MedTech Logo"
            width={120}
            height={60}
            className="opacity-90"
          />
        </div>

        {/* Title */}
        <h1 className="text-6xl font-bold text-white mb-4 tracking-tight">
          3D Game Generation
          <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
            AI Assistant
          </span>
        </h1>

        {/* Subtitle */}
        <div className="inline-flex items-center gap-3 px-6 py-2 bg-slate-800/50 border border-slate-700/50 rounded-full mb-16">
          <span className="text-slate-400 text-lg">Deep Learning</span>
          <span className="text-slate-600">|</span>
          <span className="text-cyan-400 text-lg font-medium">Final Project</span>
        </div>

        {/* Academic Info */}
        <div className="space-y-6 text-slate-400">
          {/* Professor */}
          <div className="flex items-center justify-center gap-2">
            <span className="text-slate-500">Professor:</span>
            <span className="text-white font-medium">Hichem Kallel</span>
          </div>

          {/* Lab Instructor */}
          <div className="flex items-center justify-center gap-2">
            <span className="text-slate-500">Lab Instructor:</span>
            <span className="text-white font-medium">Med Iheb Hergli</span>
          </div>

          {/* Divider */}
          <div className="flex items-center justify-center gap-4 py-4">
            <div className="h-px w-24 bg-gradient-to-r from-transparent to-slate-700" />
            <span className="text-xs text-slate-600 uppercase tracking-wider">Team</span>
            <div className="h-px w-24 bg-gradient-to-l from-transparent to-slate-700" />
          </div>

          {/* Team Members */}
          <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
            {[
              "Amine Regaieg",
              "Firas Bajjar",
              "Med Salim Soussi",
              "Ons Ouenniche"
            ].map((name) => (
              <div
                key={name}
                className="px-4 py-2 bg-slate-800/30 border border-slate-700/30 rounded-lg text-white/90"
              >
                {name}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Slide number */}
      <div className="absolute bottom-4 right-8 z-20 text-slate-400 text-base font-mono font-bold">
        {String(slideNumber).padStart(2, '0')} / {String(totalSlides).padStart(2, '0')}
      </div>
    </div>
  );
}
