"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface SlideInfo {
  title: string;
  subtitle?: string;
}

interface SlideNavigationProps {
  currentSlide: number;
  totalSlides: number;
  slides: SlideInfo[];
  onSlideChange: (index: number) => void;
  onPrev: () => void;
  onNext: () => void;
  isFirstSlide: boolean;
  isLastSlide: boolean;
  presentationTitle: string;
  accentGradient?: string;
}

export function SlideNavigation({
  currentSlide,
  totalSlides,
  slides,
  onSlideChange,
  onPrev,
  onNext,
  isFirstSlide,
  isLastSlide,
  presentationTitle,
  accentGradient = "from-cyan-500 to-purple-500",
}: SlideNavigationProps) {
  const copySlideLink = () => {
    const url = `${window.location.origin}${window.location.pathname}#slide-${currentSlide + 1}`;
    navigator.clipboard.writeText(url);
  };

  return (
    <>
      {/* Slide Navigation Dots with Dropdown */}
      <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex gap-2 px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg backdrop-blur-sm hover:bg-slate-700/80 transition-all">
              {slides.map((_, index) => (
                <span
                  key={index}
                  className={`w-2 h-2 rounded-full transition-all duration-300 ${
                    index === currentSlide
                      ? `w-6 bg-gradient-to-r ${accentGradient}`
                      : "bg-slate-600"
                  }`}
                />
              ))}
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="bg-slate-900 border-slate-700 max-h-80 overflow-y-auto">
            {slides.map((slide, index) => (
              <DropdownMenuItem
                key={index}
                onClick={() => onSlideChange(index)}
                className={`cursor-pointer ${
                  index === currentSlide
                    ? "bg-slate-800 text-white"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                <span className="w-6 text-xs text-slate-500">{index + 1}.</span>
                <span className="truncate">{slide.title}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Copy Link Button */}
        <button
          onClick={copySlideLink}
          className="p-2 bg-slate-800/80 border border-slate-700 rounded-lg backdrop-blur-sm hover:bg-slate-700/80 transition-all text-slate-400 hover:text-white"
          title="Copy link to this slide"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
        </button>
      </div>

      {/* Slide Title */}
      <div className="fixed top-4 right-4 z-50 px-3 py-1.5 bg-slate-800/80 border border-slate-700 rounded-lg backdrop-blur-sm">
        <span className="text-xs text-slate-500 font-mono">{presentationTitle}</span>
      </div>
    </>
  );
}
