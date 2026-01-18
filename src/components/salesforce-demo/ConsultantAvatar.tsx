"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import {
  User,
  Volume2,
  VolumeX,
  Loader2,
  MessageCircle,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { avatars, type AvatarProfile } from "@/lib/avatars";

// Default avatar for backward compatibility
const DEFAULT_AVATAR = avatars.find(a => a.id === 'jordan-en') || avatars[0];

interface ConsultantAvatarProps {
  avatar?: AvatarProfile;
  videoUrl?: string;
  status: "idle" | "listening" | "thinking" | "speaking";
  currentResponse?: string;
}

export function ConsultantAvatar({
  avatar: avatarProp,
  videoUrl,
  status,
  currentResponse,
}: ConsultantAvatarProps) {
  const avatar = avatarProp || DEFAULT_AVATAR;
  // Derive conversation state from simple status prop
  const conversationState = {
    status: status === "thinking" ? "processing" : status === "idle" ? "idle" : status === "speaking" ? "speaking" : "listening",
    isUserSpeaking: status === "listening",
    isAgentSpeaking: status === "speaking",
    transcript: "",
    agentText: currentResponse || "",
  };
  const [isMuted, setIsMuted] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Auto-play when video is ready
  useEffect(() => {
    if (videoUrl && videoRef.current) {
      videoRef.current.play().catch(() => {});
    }
  }, [videoUrl]);

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
    }
    setIsMuted(!isMuted);
  };

  const getStatusBadge = () => {
    switch (conversationState.status) {
      case "speaking":
        return (
          <Badge className="bg-[#0176D3] text-white animate-pulse">
            <Volume2 className="w-3 h-3 mr-1" />
            Speaking
          </Badge>
        );
      case "listening":
        return (
          <Badge className="bg-emerald-500 text-white">
            <MessageCircle className="w-3 h-3 mr-1" />
            Listening
          </Badge>
        );
      case "processing":
        return (
          <Badge className="bg-amber-500 text-white animate-pulse">
            <Sparkles className="w-3 h-3 mr-1" />
            Thinking
          </Badge>
        );
      case "connected":
        return (
          <Badge className="bg-emerald-500/80 text-white">
            Ready
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="text-slate-400 border-slate-600">
            Offline
          </Badge>
        );
    }
  };

  return (
    <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-base text-white flex items-center gap-2">
          <User className="w-4 h-4 text-[#0176D3]" />
          {avatar.displayName}
          <div className="ml-auto">
            {getStatusBadge()}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        {/* Avatar Container */}
        <div className="relative aspect-video bg-gradient-to-br from-[#032D60] to-slate-900">
          {videoUrl ? (
            // Video Player (lip-sync)
            <video
              ref={videoRef}
              src={videoUrl}
              loop
              muted={isMuted}
              playsInline
              className="w-full h-full object-cover"
            />
          ) : (
            // Animated Avatar Placeholder
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              {/* Avatar Circle with Animation */}
              <motion.div
                className="relative"
                animate={
                  conversationState.isAgentSpeaking
                    ? { scale: [1, 1.05, 1] }
                    : {}
                }
                transition={{ duration: 0.5, repeat: Infinity }}
              >
                {/* Outer glow when speaking */}
                {conversationState.isAgentSpeaking && (
                  <motion.div
                    className="absolute inset-0 rounded-full bg-[#0176D3]/30"
                    animate={{ scale: [1, 1.3, 1], opacity: [0.5, 0, 0.5] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    style={{ margin: "-20px" }}
                  />
                )}

                {/* Avatar Image */}
                <div className="w-32 h-32 rounded-full bg-gradient-to-br from-[#0176D3] to-[#032D60] flex items-center justify-center border-4 border-white/20 shadow-2xl overflow-hidden">
                  <Image
                    src={avatar.avatar}
                    alt={avatar.displayName}
                    width={128}
                    height={128}
                    className="w-full h-full object-cover"
                  />
                </div>

                {/* Status Indicator */}
                <motion.div
                  className={`absolute bottom-2 right-2 w-4 h-4 rounded-full border-2 border-white ${
                    conversationState.status === "idle"
                      ? "bg-slate-500"
                      : conversationState.isAgentSpeaking
                      ? "bg-[#0176D3]"
                      : "bg-emerald-500"
                  }`}
                  animate={
                    conversationState.isAgentSpeaking
                      ? { scale: [1, 1.2, 1] }
                      : {}
                  }
                  transition={{ duration: 0.3, repeat: Infinity }}
                />
              </motion.div>

              {/* Name and Role */}
              <div className="mt-4 text-center">
                <h3 className="text-white font-semibold text-lg">{avatar.displayName}</h3>
                <p className="text-slate-400 text-sm">
                  {avatar.title}
                </p>
              </div>

              {/* Loading State */}
              {conversationState.status === "connecting" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="absolute inset-0 bg-black/50 flex items-center justify-center"
                >
                  <div className="text-center">
                    <Loader2 className="w-8 h-8 text-[#0176D3] animate-spin mx-auto mb-2" />
                    <p className="text-white text-sm">Connecting...</p>
                  </div>
                </motion.div>
              )}
            </div>
          )}

          {/* Audio element for voice-only mode (future) */}

          {/* Controls Overlay */}
          <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent">
            <div className="flex items-center justify-between">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleMute}
                className="h-8 w-8 text-white hover:bg-white/20"
              >
                {isMuted ? (
                  <VolumeX className="w-4 h-4" />
                ) : (
                  <Volume2 className="w-4 h-4" />
                )}
              </Button>
              <div className="text-xs text-white/70">
                {conversationState.isAgentSpeaking && "Speaking..."}
              </div>
            </div>
          </div>
        </div>

        {/* Current Response Text */}
        <AnimatePresence>
          {currentResponse && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="border-t border-white/10"
            >
              <div className="p-3 max-h-32 overflow-y-auto">
                <p className="text-slate-300 text-sm leading-relaxed">
                  {currentResponse}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}
