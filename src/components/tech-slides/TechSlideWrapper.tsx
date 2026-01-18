"use client";

import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface TechSlideWrapperProps {
  children: ReactNode;
  className?: string;
  slideNumber?: number;
  totalSlides?: number;
  title?: string;
}

export function TechSlideWrapper({
  children,
  className,
  slideNumber,
  totalSlides,
  title
}: TechSlideWrapperProps) {
  return (
    <div className={cn(
      "min-h-screen w-full flex flex-col p-8 relative overflow-hidden",
      "bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950",
      className
    )}>
      {/* Animated grid background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0ea5e920_1px,transparent_1px),linear-gradient(to_bottom,#0ea5e920_1px,transparent_1px)] bg-[size:3rem_3rem]" />

      {/* Gradient orbs */}
      <div className="absolute top-0 left-0 w-[600px] h-[600px] bg-cyan-500/10 rounded-full blur-[120px] -translate-x-1/2 -translate-y-1/2" />
      <div className="absolute bottom-0 right-0 w-[600px] h-[600px] bg-purple-500/10 rounded-full blur-[120px] translate-x-1/2 translate-y-1/2" />
      <div className="absolute top-1/2 left-1/2 w-[400px] h-[400px] bg-emerald-500/5 rounded-full blur-[100px] -translate-x-1/2 -translate-y-1/2" />

      {/* Circuit pattern overlay */}
      <div className="absolute inset-0 opacity-5">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="circuit" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
              <path d="M10 10h80v80H10z" fill="none" stroke="currentColor" strokeWidth="0.5"/>
              <circle cx="10" cy="10" r="2" fill="currentColor"/>
              <circle cx="90" cy="10" r="2" fill="currentColor"/>
              <circle cx="10" cy="90" r="2" fill="currentColor"/>
              <circle cx="90" cy="90" r="2" fill="currentColor"/>
              <path d="M50 10v30M10 50h30M50 90v-30M90 50h-30" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#circuit)" className="text-cyan-400"/>
        </svg>
      </div>

      {/* Title bar */}
      {title && (
        <div className="relative z-10 mb-6">
          <div className="flex items-center gap-4">
            <div className="h-1 w-12 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full" />
            <h1 className="text-sm font-mono text-cyan-400 uppercase tracking-wider">{title}</h1>
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
                  i + 1 === slideNumber ? "bg-cyan-500 w-4" : "bg-slate-600"
                )}
              />
            ))}
          </div>
          <span className="text-slate-400 text-base font-mono font-bold">
            {String(slideNumber).padStart(2, '0')} / {String(totalSlides).padStart(2, '0')}
          </span>
        </div>
      )}
    </div>
  );
}
