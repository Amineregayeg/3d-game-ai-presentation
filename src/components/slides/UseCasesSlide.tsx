"use client";

import { SlideWrapper } from "./SlideWrapper";

interface UseCasesSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function UseCasesSlide({ slideNumber, totalSlides }: UseCasesSlideProps) {
  const useCases = [
    {
      title: "Indie Game Development",
      description: "Solo developers and small teams can rapidly prototype and create game assets without dedicated 3D artists",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      color: "from-green-500 to-emerald-600",
      examples: ["Quick prototyping", "Asset iteration", "Style exploration"]
    },
    {
      title: "Game Jams",
      description: "Create game assets on-the-fly during time-constrained game jams and hackathons",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      color: "from-purple-500 to-pink-600",
      examples: ["48-hour events", "Rapid iteration", "Last-minute assets"]
    },
    {
      title: "Educational Tools",
      description: "Teaching 3D modeling and game development concepts through natural language interaction",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      ),
      color: "from-blue-500 to-cyan-600",
      examples: ["Learning Blender", "3D concepts", "Interactive tutorials"]
    },
    {
      title: "Accessibility",
      description: "Enable creators with disabilities or those unfamiliar with complex 3D software to create assets",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
      ),
      color: "from-amber-500 to-orange-600",
      examples: ["Voice-first design", "No 3D skills needed", "Inclusive creation"]
    },
    {
      title: "Professional Studios",
      description: "Speed up pre-production and concept art phases with rapid 3D previsualization",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
        </svg>
      ),
      color: "from-red-500 to-rose-600",
      examples: ["Concept visualization", "Client presentations", "Team collaboration"]
    },
    {
      title: "VR/AR Content",
      description: "Generate 3D assets optimized for virtual and augmented reality applications",
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      ),
      color: "from-indigo-500 to-violet-600",
      examples: ["VR environments", "AR objects", "Immersive content"]
    }
  ];

  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <h2 className="text-5xl font-bold text-white">
            Use <span className="bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">Cases</span>
          </h2>
          <p className="text-xl text-slate-400">
            Empowering creators across industries and skill levels
          </p>
        </div>

        {/* Use cases grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {useCases.map((useCase) => (
            <div
              key={useCase.title}
              className="group p-6 bg-slate-800/30 border border-slate-700/50 rounded-2xl hover:border-slate-600/50 transition-all duration-300"
            >
              <div className={`w-14 h-14 bg-gradient-to-br ${useCase.color} rounded-xl flex items-center justify-center text-white mb-4 shadow-lg group-hover:scale-110 transition-transform`}>
                {useCase.icon}
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">{useCase.title}</h3>
              <p className="text-sm text-slate-400 mb-4">{useCase.description}</p>
              <div className="flex flex-wrap gap-2">
                {useCase.examples.map((example) => (
                  <span key={example} className="px-2 py-1 bg-slate-900/50 border border-slate-700/50 rounded text-xs text-slate-400">
                    {example}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </SlideWrapper>
  );
}
