"use client";

import { TechSlideWrapper } from "@/components/tech-slides/TechSlideWrapper";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Play, Mic, Cpu, MessageSquare, CheckCircle } from "lucide-react";

interface SystemOverviewSlideProps {
  slideNumber: number;
  totalSlides: number;
}

export function SystemOverviewSlide({ slideNumber, totalSlides }: SystemOverviewSlideProps) {
  const phases = [
    {
      phase: 1,
      title: "Welcome",
      subtitle: "Automatic",
      icon: Play,
      color: "cyan",
      description: "Avatar greets user and explains capabilities",
      details: ["Page loads → Avatar appears", "Greeting animation plays", "3D viewport shows empty scene"],
    },
    {
      phase: 2,
      title: "User Input",
      subtitle: "Recording",
      icon: Mic,
      color: "purple",
      description: "User speaks their request via microphone",
      details: ["Click microphone to record", "Real-time waveform display", "Auto-stop on silence"],
    },
    {
      phase: 3,
      title: "Processing",
      subtitle: "Parallel",
      icon: Cpu,
      color: "amber",
      description: "STT transcription + RAG query in parallel",
      details: ["STT converts speech to text", "RAG retrieves knowledge", "Progress shown for each stage"],
    },
    {
      phase: 4,
      title: "Response",
      subtitle: "Simultaneous",
      icon: MessageSquare,
      color: "emerald",
      description: "Avatar speaks while Blender executes",
      details: ["Avatar explains the action", "Blender commands execute", "3D model builds in real-time"],
    },
    {
      phase: 5,
      title: "Complete",
      subtitle: "Interactive",
      icon: CheckCircle,
      color: "pink",
      description: "Result ready for interaction or download",
      details: ["Avatar asks follow-up", "3D viewport shows result", "Download .blend option"],
    },
  ];

  return (
    <TechSlideWrapper slideNumber={slideNumber} totalSlides={totalSlides} title="User Experience Flow">
      <div className="space-y-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">5-Phase Interaction Model</h2>
          <p className="text-slate-400">From voice input to 3D creation in seconds</p>
        </div>

        {/* Timeline visualization */}
        <div className="relative">
          {/* Connection line */}
          <div className="absolute top-16 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 hidden lg:block" />

          {/* Phase cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {phases.map((phase, index) => {
              const Icon = phase.icon;
              return (
                <Card
                  key={phase.phase}
                  className={`bg-slate-900/50 border-${phase.color}-500/30 backdrop-blur-sm relative overflow-hidden group hover:border-${phase.color}-500/60 transition-all`}
                >
                  {/* Glow effect on hover */}
                  <div className={`absolute inset-0 bg-${phase.color}-500/5 opacity-0 group-hover:opacity-100 transition-opacity`} />

                  <CardContent className="p-4 relative">
                    {/* Phase number */}
                    <div className="flex items-center justify-between mb-3">
                      <Badge
                        variant="outline"
                        className={`border-${phase.color}-500/50 text-${phase.color}-400 bg-${phase.color}-500/10`}
                      >
                        Phase {phase.phase}
                      </Badge>
                      <div className={`p-2 rounded-lg bg-${phase.color}-500/10 border border-${phase.color}-500/30`}>
                        <Icon className={`w-4 h-4 text-${phase.color}-400`} />
                      </div>
                    </div>

                    {/* Title */}
                    <h3 className="text-lg font-semibold text-white mb-1">{phase.title}</h3>
                    <p className={`text-xs text-${phase.color}-400 mb-2`}>{phase.subtitle}</p>

                    {/* Description */}
                    <p className="text-sm text-slate-400 mb-3">{phase.description}</p>

                    {/* Details */}
                    <ul className="space-y-1">
                      {phase.details.map((detail, i) => (
                        <li key={i} className="text-xs text-slate-500 flex items-start gap-2">
                          <span className={`text-${phase.color}-500 mt-1`}>•</span>
                          {detail}
                        </li>
                      ))}
                    </ul>

                    {/* Arrow to next */}
                    {index < phases.length - 1 && (
                      <div className="absolute -right-2 top-1/2 -translate-y-1/2 text-slate-600 hidden lg:block z-10">
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/>
                        </svg>
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Example flow */}
        <Card className="bg-slate-900/30 border-slate-700/50 mt-6">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="h-1 w-8 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full" />
              <span className="text-sm font-mono text-cyan-400">EXAMPLE INTERACTION</span>
            </div>

            <div className="flex flex-wrap items-center gap-3 text-sm">
              <Badge variant="outline" className="border-slate-600 text-slate-300 bg-slate-800/50">
                User: &quot;Create a low-poly tree with green leaves&quot;
              </Badge>
              <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
              <Badge variant="outline" className="border-purple-500/30 text-purple-300 bg-purple-500/10">
                RAG: Retrieves Blender mesh primitives docs
              </Badge>
              <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
              <Badge variant="outline" className="border-emerald-500/30 text-emerald-300 bg-emerald-500/10">
                Avatar: Explains while building
              </Badge>
              <svg className="w-4 h-4 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
              <Badge variant="outline" className="border-orange-500/30 text-orange-300 bg-orange-500/10">
                Result: 3D tree in viewport
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </TechSlideWrapper>
  );
}
