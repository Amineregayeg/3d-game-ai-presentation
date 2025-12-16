"use client";

import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface AvatarSlideWrapperProps {
  children: ReactNode;
  className?: string;
  slideNumber?: number;
  totalSlides?: number;
  title?: string;
}

export function AvatarSlideWrapper({
  children,
  className,
  slideNumber,
  totalSlides,
  title
}: AvatarSlideWrapperProps) {
  return (
    <div className={cn(
      "min-h-screen w-full flex flex-col p-8 relative overflow-hidden",
      "bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950",
      className
    )}>
      {/* Animated grid background with voice wave pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#f4366320_1px,transparent_1px),linear-gradient(to_bottom,#f4366320_1px,transparent_1px)] bg-[size:3rem_3rem]" />

      {/* Sound wave decoration */}
      <div className="absolute inset-0 opacity-[0.03]">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
          <defs>
            <pattern id="soundWave" x="0" y="0" width="200" height="100" patternUnits="userSpaceOnUse">
              <path
                d="M0 50 Q25 20, 50 50 T100 50 T150 50 T200 50"
                fill="none"
                stroke="currentColor"
                strokeWidth="1"
              />
              <path
                d="M0 50 Q25 80, 50 50 T100 50 T150 50 T200 50"
                fill="none"
                stroke="currentColor"
                strokeWidth="0.5"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#soundWave)" className="text-rose-400"/>
        </svg>
      </div>

      {/* Gradient orbs - Rose/Pink theme */}
      <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-rose-500/10 rounded-full blur-[120px] -translate-x-1/2 -translate-y-1/2" />
      <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-pink-500/10 rounded-full blur-[120px] translate-x-1/2 translate-y-1/2" />
      <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-orange-500/5 rounded-full blur-[100px] -translate-x-1/2 -translate-y-1/2" />
      <div className="absolute top-1/4 right-1/4 w-[300px] h-[300px] bg-fuchsia-500/5 rounded-full blur-[80px]" />

      {/* Voice/Audio circuit pattern overlay */}
      <div className="absolute inset-0 opacity-5">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="voiceCircuit" x="0" y="0" width="120" height="120" patternUnits="userSpaceOnUse">
              {/* Microphone symbol */}
              <circle cx="60" cy="30" r="12" fill="none" stroke="currentColor" strokeWidth="0.5"/>
              <rect x="54" y="42" width="12" height="20" rx="2" fill="none" stroke="currentColor" strokeWidth="0.5"/>
              <path d="M48 55 Q48 70, 60 70 Q72 70, 72 55" fill="none" stroke="currentColor" strokeWidth="0.5"/>
              <line x1="60" y1="70" x2="60" y2="85" stroke="currentColor" strokeWidth="0.5"/>
              <line x1="50" y1="85" x2="70" y2="85" stroke="currentColor" strokeWidth="0.5"/>

              {/* Sound waves */}
              <path d="M80 30 Q90 30, 90 40" fill="none" stroke="currentColor" strokeWidth="0.3"/>
              <path d="M85 25 Q100 30, 100 45" fill="none" stroke="currentColor" strokeWidth="0.3"/>
              <path d="M40 30 Q30 30, 30 40" fill="none" stroke="currentColor" strokeWidth="0.3"/>
              <path d="M35 25 Q20 30, 20 45" fill="none" stroke="currentColor" strokeWidth="0.3"/>

              {/* Connection nodes */}
              <circle cx="10" cy="60" r="2" fill="currentColor"/>
              <circle cx="110" cy="60" r="2" fill="currentColor"/>
              <circle cx="60" cy="110" r="2" fill="currentColor"/>

              {/* Connection lines */}
              <line x1="10" y1="60" x2="48" y2="60" stroke="currentColor" strokeWidth="0.3"/>
              <line x1="72" y1="60" x2="110" y2="60" stroke="currentColor" strokeWidth="0.3"/>
              <line x1="60" y1="85" x2="60" y2="110" stroke="currentColor" strokeWidth="0.3"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#voiceCircuit)" className="text-rose-400"/>
        </svg>
      </div>

      {/* Title bar */}
      {title && (
        <div className="relative z-10 mb-6">
          <div className="flex items-center gap-4">
            <div className="h-1 w-12 bg-gradient-to-r from-rose-500 to-pink-500 rounded-full" />
            <h1 className="text-sm font-mono text-rose-400 uppercase tracking-wider">{title}</h1>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="relative z-10 flex-1 w-full max-w-7xl mx-auto">
        {children}
      </div>

      {/* Slide number */}
      {slideNumber && totalSlides && (
        <div className="absolute bottom-4 right-8 z-20 flex items-center gap-3">
          <div className="flex gap-1">
            {Array.from({ length: totalSlides }, (_, i) => (
              <div
                key={i}
                className={cn(
                  "w-1.5 h-1.5 rounded-full transition-all",
                  i + 1 === slideNumber ? "bg-rose-500 w-4" : "bg-slate-600"
                )}
              />
            ))}
          </div>
          <span className="text-slate-500 text-sm font-mono">
            {String(slideNumber).padStart(2, '0')} / {String(totalSlides).padStart(2, '0')}
          </span>
        </div>
      )}
    </div>
  );
}
