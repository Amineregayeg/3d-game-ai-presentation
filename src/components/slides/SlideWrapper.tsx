"use client";

import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface SlideWrapperProps {
  children: ReactNode;
  className?: string;
  slideNumber?: number;
  totalSlides?: number;
}

export function SlideWrapper({
  children,
  className,
  slideNumber,
  totalSlides
}: SlideWrapperProps) {
  return (
    <div className={cn(
      "min-h-screen w-full flex flex-col items-center justify-center p-8 relative",
      "bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950",
      className
    )}>
      {/* Background grid pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b20_1px,transparent_1px),linear-gradient(to_bottom,#1e293b20_1px,transparent_1px)] bg-[size:4rem_4rem]" />

      {/* Gradient orbs */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />

      {/* Content */}
      <div className="relative z-10 w-full max-w-6xl">
        {children}
      </div>

      {/* Slide number */}
      {slideNumber && totalSlides && (
        <div className="absolute bottom-4 right-8 text-slate-500 text-sm font-mono">
          {slideNumber} / {totalSlides}
        </div>
      )}
    </div>
  );
}
