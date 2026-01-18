"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Volume2, VolumeX, SkipForward, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface GreetingPlayerProps {
  onComplete: () => void;
  videoUrl?: string;
  autoPlay?: boolean;
}

export function GreetingPlayer({
  onComplete,
  videoUrl = "/3dgameassistant/videos/greeting.mp4",
  autoPlay = true
}: GreetingPlayerProps) {
  const [isVisible, setIsVisible] = useState(true);
  const [isMuted, setIsMuted] = useState(true); // Start muted for autoplay to work
  const [hasError, setHasError] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Greeting plays every time page loads
  useEffect(() => {
    // Reset visibility on mount
    setIsVisible(true);
  }, []);

  const handleSkip = () => {
    setIsVisible(false);
    onComplete();
  };

  const handleVideoEnd = () => {
    setTimeout(() => {
      setIsVisible(false);
      onComplete();
    }, 500);
  };

  const handleVideoError = () => {
    setHasError(true);
    // Show fallback greeting for 3 seconds then continue
    setTimeout(() => {
      handleSkip();
    }, 3000);
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleReload = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play().catch(() => {});
    }
    setHasError(false);
  };

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-md"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="relative w-full max-w-2xl mx-4"
        >
          {/* Video Container */}
          <div className="relative rounded-2xl overflow-hidden bg-slate-900 border border-white/10 shadow-2xl">
            {/* Header */}
            <div className="absolute top-0 left-0 right-0 z-10 p-4 bg-gradient-to-b from-black/80 to-transparent">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-sm text-white/80">AI Assistant</span>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleReload}
                    className="h-8 w-8 text-white/70 hover:text-white hover:bg-white/10"
                    title="Replay"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={toggleMute}
                    className="h-8 w-8 text-white/70 hover:text-white hover:bg-white/10"
                    title={isMuted ? "Unmute" : "Mute"}
                  >
                    {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleSkip}
                    className="h-8 w-8 text-white/70 hover:text-white hover:bg-white/10"
                    title="Close"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Video or Fallback */}
            {hasError ? (
              <div className="aspect-video flex flex-col items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900 p-8">
                <div className="w-24 h-24 rounded-full bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center mb-6">
                  <span className="text-4xl">🤖</span>
                </div>
                <h2 className="text-2xl font-bold text-white mb-4 text-center">
                  Welcome to 3D Game AI Assistant!
                </h2>
                <p className="text-slate-400 text-center max-w-md">
                  I can help you create 3D assets, generate game objects in Blender,
                  and answer questions about 3D modeling. Click below to get started!
                </p>
              </div>
            ) : (
              <video
                ref={videoRef}
                src={videoUrl}
                autoPlay
                muted
                playsInline
                onEnded={handleVideoEnd}
                onError={handleVideoError}
                onLoadedData={() => setIsLoaded(true)}
                onCanPlay={() => {
                  // Try to play when ready
                  videoRef.current?.play().catch(() => {});
                }}
                className="w-full aspect-video object-cover"
              />
            )}

            {/* Footer with Skip Button */}
            <div className="absolute bottom-0 left-0 right-0 z-10 p-4 bg-gradient-to-t from-black/80 to-transparent">
              <div className="flex items-center justify-between">
                <p className="text-xs text-white/50">
                  {hasError ? "Loading greeting..." : isMuted ? "Click speaker icon to unmute" : "Playing greeting..."}
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSkip}
                  className="gap-2 border-white/20 text-white hover:bg-white/10"
                >
                  <SkipForward className="w-4 h-4" />
                  Skip Intro
                </Button>
              </div>
            </div>
          </div>

          {/* Decorative Elements */}
          <div className="absolute -top-20 -left-20 w-40 h-40 bg-cyan-500/20 rounded-full blur-3xl" />
          <div className="absolute -bottom-20 -right-20 w-40 h-40 bg-purple-500/20 rounded-full blur-3xl" />
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
