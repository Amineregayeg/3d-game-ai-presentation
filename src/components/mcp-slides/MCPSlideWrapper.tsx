"use client";

import { ReactNode } from "react";

interface MCPSlideWrapperProps {
  children: ReactNode;
  slideNumber: number;
  totalSlides: number;
  title?: string;
}

export function MCPSlideWrapper({ children, slideNumber, totalSlides, title }: MCPSlideWrapperProps) {
  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden flex items-center justify-center">
      {/* Animated gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />

      {/* Orange/Amber glowing orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-orange-500/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-amber-500/15 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
      <div className="absolute top-1/2 right-1/3 w-64 h-64 bg-yellow-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />

      {/* 3D Grid pattern */}
      <div className="absolute inset-0 opacity-20">
        <svg width="100%" height="100%" className="absolute inset-0">
          <defs>
            <pattern id="mcpGrid" width="60" height="60" patternUnits="userSpaceOnUse">
              <path d="M 60 0 L 0 0 0 60" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-orange-500/30" />
            </pattern>
            <linearGradient id="mcpGridFade" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="white" stopOpacity="0.3" />
              <stop offset="50%" stopColor="white" stopOpacity="0.1" />
              <stop offset="100%" stopColor="white" stopOpacity="0" />
            </linearGradient>
          </defs>
          <rect width="100%" height="100%" fill="url(#mcpGrid)" mask="url(#mcpGridFade)" />
        </svg>
      </div>

      {/* 3D Cube/Connection node pattern */}
      <svg className="absolute inset-0 w-full h-full opacity-10" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="nodeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#f97316" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#fbbf24" stopOpacity="0.3" />
          </linearGradient>
        </defs>
        {/* Connection lines */}
        <path d="M100,200 Q400,100 700,300" stroke="url(#nodeGrad)" strokeWidth="1" fill="none" opacity="0.4" />
        <path d="M1200,150 Q1500,400 1800,200" stroke="url(#nodeGrad)" strokeWidth="1" fill="none" opacity="0.4" />
        <path d="M200,800 Q600,600 1000,900" stroke="url(#nodeGrad)" strokeWidth="1" fill="none" opacity="0.3" />
        <path d="M1400,700 Q1600,500 1850,800" stroke="url(#nodeGrad)" strokeWidth="1" fill="none" opacity="0.3" />

        {/* 3D Cube nodes */}
        <g transform="translate(150, 180)">
          <polygon points="0,10 15,0 30,10 15,20" fill="#f97316" fillOpacity="0.3" />
          <polygon points="0,10 15,20 15,35 0,25" fill="#ea580c" fillOpacity="0.4" />
          <polygon points="15,20 30,10 30,25 15,35" fill="#fb923c" fillOpacity="0.2" />
        </g>
        <g transform="translate(1750, 180)">
          <polygon points="0,10 15,0 30,10 15,20" fill="#f97316" fillOpacity="0.3" />
          <polygon points="0,10 15,20 15,35 0,25" fill="#ea580c" fillOpacity="0.4" />
          <polygon points="15,20 30,10 30,25 15,35" fill="#fb923c" fillOpacity="0.2" />
        </g>
        <g transform="translate(250, 780)">
          <polygon points="0,10 15,0 30,10 15,20" fill="#fbbf24" fillOpacity="0.3" />
          <polygon points="0,10 15,20 15,35 0,25" fill="#f59e0b" fillOpacity="0.4" />
          <polygon points="15,20 30,10 30,25 15,35" fill="#fcd34d" fillOpacity="0.2" />
        </g>
        <g transform="translate(1600, 750)">
          <polygon points="0,10 15,0 30,10 15,20" fill="#fbbf24" fillOpacity="0.3" />
          <polygon points="0,10 15,20 15,35 0,25" fill="#f59e0b" fillOpacity="0.4" />
          <polygon points="15,20 30,10 30,25 15,35" fill="#fcd34d" fillOpacity="0.2" />
        </g>

        {/* Blender logo shape abstraction */}
        <g transform="translate(960, 540)" opacity="0.15">
          <circle cx="0" cy="0" r="150" stroke="#f97316" strokeWidth="2" fill="none" />
          <circle cx="0" cy="0" r="100" stroke="#fbbf24" strokeWidth="1.5" fill="none" />
          <circle cx="0" cy="0" r="50" stroke="#fcd34d" strokeWidth="1" fill="none" />
          <ellipse cx="0" cy="0" rx="180" ry="60" stroke="#f97316" strokeWidth="1" fill="none" transform="rotate(45)" />
        </g>
      </svg>

      {/* Content container */}
      <div className="relative z-10 w-full max-w-7xl mx-auto px-8 py-6">
        {children}
      </div>

      {/* Slide counter */}
      <div className="absolute bottom-6 right-8 z-20 flex items-center gap-2">
        <span className="text-orange-400 font-mono text-sm font-bold">{String(slideNumber).padStart(2, '0')}</span>
        <div className="flex gap-1">
          {Array.from({ length: totalSlides }, (_, i) => (
            <div
              key={i}
              className={`w-1.5 h-1.5 rounded-full transition-all ${
                i < slideNumber ? 'bg-orange-500' : 'bg-slate-700'
              }`}
            />
          ))}
        </div>
        <span className="text-slate-600 font-mono text-sm">{String(totalSlides).padStart(2, '0')}</span>
      </div>

      {/* Title bar with MCP branding */}
      {title && (
        <div className="absolute top-6 left-8 z-20">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-500 to-amber-500 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <polyline points="7.5 4.21 12 6.81 16.5 4.21" />
                <polyline points="7.5 19.79 7.5 14.6 3 12" />
                <polyline points="21 12 16.5 14.6 16.5 19.79" />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
            </div>
            <div>
              <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">MCP Integration</span>
              <div className="h-0.5 w-full bg-gradient-to-r from-orange-500 to-amber-500 rounded-full mt-1" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
