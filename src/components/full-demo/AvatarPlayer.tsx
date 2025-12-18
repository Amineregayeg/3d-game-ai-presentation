"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Video, Volume2, VolumeX, Loader2, Play, Pause, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface AvatarPlayerProps {
  videoUrl?: string;
  audioUrl?: string;
  isGenerating: boolean;
  generationProgress?: number;
  avatarName?: string;
  duration?: number;
  processingTime?: number;
}

export function AvatarPlayer({
  videoUrl,
  audioUrl,
  isGenerating,
  generationProgress = 0,
  avatarName,
  duration,
  processingTime,
}: AvatarPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Auto-play when video is ready
  useEffect(() => {
    if (videoUrl && videoRef.current) {
      videoRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [videoUrl]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    } else if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
    }
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
    }
    setIsMuted(!isMuted);
  };

  const restart = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play();
      setIsPlaying(true);
    }
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-base text-white flex items-center gap-2">
          <Video className="w-4 h-4 text-pink-400" />
          Avatar Response
          {avatarName && (
            <span className="text-xs text-slate-400 font-normal ml-auto">
              {avatarName}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        {/* Video/Audio Container */}
        <div className="relative aspect-video bg-slate-900">
          {isGenerating ? (
            // Loading State
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="relative">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-pink-500/20 to-purple-500/20 animate-pulse" />
                <Loader2 className="w-10 h-10 text-pink-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-spin" />
              </div>
              <p className="mt-4 text-sm text-slate-400">Generating lip-sync video...</p>
              {generationProgress > 0 && (
                <div className="mt-2 w-48 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-pink-500 to-purple-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${generationProgress}%` }}
                  />
                </div>
              )}
              <p className="mt-1 text-xs text-slate-500">This may take 30-60 seconds</p>
            </div>
          ) : videoUrl ? (
            // Video Player
            <video
              ref={videoRef}
              src={videoUrl}
              loop
              muted={isMuted}
              playsInline
              className="w-full h-full object-cover"
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            />
          ) : audioUrl ? (
            // Audio Only (with avatar placeholder)
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900">
              <div className="w-24 h-24 rounded-full bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center mb-4">
                <span className="text-4xl">🤖</span>
              </div>
              <p className="text-slate-400 text-sm">Audio Response</p>
              <audio
                ref={audioRef}
                src={audioUrl}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
              />
            </div>
          ) : (
            // Empty State
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center mb-3">
                <Video className="w-8 h-8 text-slate-600" />
              </div>
              <p className="text-slate-500 text-sm">Video will appear here</p>
            </div>
          )}

          {/* Controls Overlay */}
          {(videoUrl || audioUrl) && !isGenerating && (
            <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={togglePlay}
                    className="h-8 w-8 text-white hover:bg-white/20"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={restart}
                    className="h-8 w-8 text-white hover:bg-white/20"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={toggleMute}
                    className="h-8 w-8 text-white hover:bg-white/20"
                  >
                    {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                  </Button>
                </div>
                <div className="text-xs text-white/70">
                  {duration && <span>{duration.toFixed(1)}s</span>}
                  {processingTime && (
                    <span className="ml-2 text-slate-400">
                      Generated in {(processingTime / 1000).toFixed(1)}s
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
