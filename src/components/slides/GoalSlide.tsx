"use client";

import { SlideWrapper } from "./SlideWrapper";

interface GoalSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function GoalSlide({ slideNumber, totalSlides }: GoalSlideProps) {
  const goals = [
    {
      icon: "01",
      title: "Voice-Driven Creation",
      description: "Enable users to create 3D game assets through natural speech, eliminating the need for complex 3D modeling skills"
    },
    {
      icon: "02",
      title: "Intelligent Understanding",
      description: "Leverage advanced RAG to understand context, game development concepts, and user intent accurately"
    },
    {
      icon: "03",
      title: "Interactive Avatar Assistant",
      description: "Provide a human-like avatar with lip-synced responses for an engaging, personalized experience"
    },
    {
      icon: "04",
      title: "Seamless 3D Generation",
      description: "Generate production-ready 3D components directly in Blender through the MCP integration"
    }
  ];

  return (
    <SlideWrapper slideNumber={slideNumber} totalSlides={totalSlides}>
      <div className="space-y-12">
        {/* Header */}
        <div className="text-center space-y-4">
          <h2 className="text-5xl font-bold text-white">
            Project <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">Goal</span>
          </h2>
          <p className="text-xl text-slate-400">
            Democratizing 3D game asset creation through AI
          </p>
        </div>

        {/* Goals grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {goals.map((goal) => (
            <div
              key={goal.title}
              className="group p-6 bg-slate-800/30 border border-slate-700/50 rounded-2xl backdrop-blur-sm hover:border-cyan-500/50 transition-all duration-300"
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center text-white font-bold text-lg">
                  {goal.icon}
                </div>
                <div className="space-y-2">
                  <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 transition-colors">
                    {goal.title}
                  </h3>
                  <p className="text-slate-400 leading-relaxed">
                    {goal.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Vision statement */}
        <div className="text-center p-6 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 border border-slate-700/50 rounded-2xl">
          <p className="text-lg text-slate-300 italic">
            &ldquo;Transform how game developers and creators bring their ideas to life -
            from concept to 3D asset with just your voice.&rdquo;
          </p>
        </div>
      </div>
    </SlideWrapper>
  );
}
