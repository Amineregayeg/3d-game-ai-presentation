"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";

interface PresenterNotesProps {
  notes: string[];
  slideNumber: number;
  totalSlides: number;
  slideTitle?: string;
}

export function PresenterNotes({ notes, slideNumber, totalSlides, slideTitle }: PresenterNotesProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-4 left-4 z-50 px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-all backdrop-blur-sm flex items-center gap-2"
        title={isOpen ? "Hide presenter notes" : "Show presenter notes"}
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <span className="text-xs hidden sm:inline">Notes</span>
        {notes.length > 0 && (
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
        )}
      </button>

      {/* Notes Panel */}
      {isOpen && (
        <div className="fixed left-4 bottom-16 z-50 w-80 max-h-[60vh] animate-in slide-in-from-bottom-2 duration-200">
          <Card className="bg-slate-900/95 border-slate-700 backdrop-blur-xl shadow-2xl">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm text-white flex items-center gap-2">
                  <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Presenter Notes
                </CardTitle>
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-slate-500 hover:text-white transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="secondary" className="bg-slate-800 text-slate-400 text-xs">
                  Slide {slideNumber}/{totalSlides}
                </Badge>
                {slideTitle && (
                  <span className="text-xs text-slate-500 truncate">{slideTitle}</span>
                )}
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[200px] pr-2">
                {notes.length > 0 ? (
                  <ul className="space-y-2">
                    {notes.map((note, index) => (
                      <li key={index} className="flex items-start gap-2 text-sm text-slate-300">
                        <span className="text-cyan-400 mt-1">•</span>
                        <span>{note}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-slate-500 italic">No notes for this slide</p>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      )}
    </>
  );
}
