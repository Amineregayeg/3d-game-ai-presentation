"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Volume2, VolumeX, Loader2 } from "lucide-react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { avatars, type AvatarProfile } from "@/lib/avatars";

// Default avatar for backward compatibility
const DEFAULT_AVATAR = avatars.find(a => a.id === 'jordan-en') || avatars[0];

// Generate greeting text based on avatar
function getGreetingText(avatar: AvatarProfile): string {
  if (avatar.language === 'fr') {
    return `Bonjour ! Je suis ${avatar.displayName}, votre ${avatar.title}. Je suis là pour vous aider avec tout ce qui concerne Salesforce - de la création de champs personnalisés à la construction d'automatisations avec Flow. Dites-moi ce dont vous avez besoin !`;
  }
  return `Hello! I'm ${avatar.displayName}, your ${avatar.title}. I'm here to help you with anything Salesforce - from creating custom fields and objects, to building automation with Flows, to troubleshooting issues. Just tell me what you need, and let's get started!`;
}

interface ConsultantGreetingProps {
  onComplete: () => void;
  avatar?: AvatarProfile;
  videoUrl?: string;
  audioUrl?: string;
}

export function ConsultantGreeting({
  onComplete,
  avatar: avatarProp,
  videoUrl,
  audioUrl: providedAudioUrl,
}: ConsultantGreetingProps) {
  const avatar = avatarProp || DEFAULT_AVATAR;
  const greetingText = getGreetingText(avatar);
  const [isMuted, setIsMuted] = useState(false);
  const [isVisible, setIsVisible] = useState(true);
  const [isLoadingAudio, setIsLoadingAudio] = useState(true);
  const [audioUrl, setAudioUrl] = useState<string | null>(providedAudioUrl || null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Generate TTS audio on mount
  useEffect(() => {
    const generateGreetingAudio = async () => {
      try {
        const response = await fetch("/api/elevenlabs/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: greetingText,
            voiceId: avatar.voiceId,
          }),
        });

        const data = await response.json();
        if (data.audioUrl) {
          setAudioUrl(data.audioUrl);
        }
        // If quota exceeded, audio will be null but that's okay
      } catch (error) {
        console.error("Failed to generate greeting audio:", error);
      } finally {
        setIsLoadingAudio(false);
      }
    };

    if (!providedAudioUrl) {
      generateGreetingAudio();
    } else {
      setIsLoadingAudio(false);
    }
  }, [providedAudioUrl, avatar.voiceId, greetingText]);

  // Play audio when available
  useEffect(() => {
    if (audioUrl && audioRef.current && !isMuted) {
      audioRef.current.play().then(() => {
        setIsSpeaking(true);
      }).catch((e) => {
        console.warn("Auto-play blocked:", e);
      });
    }
  }, [audioUrl, isMuted]);

  // Auto-dismiss after audio ends or timeout
  useEffect(() => {
    const timer = setTimeout(() => {
      handleDismiss();
    }, 20000); // 20 second timeout

    return () => clearTimeout(timer);
  }, []);

  const handleDismiss = () => {
    // Stop audio if playing
    if (audioRef.current) {
      audioRef.current.pause();
    }
    setIsVisible(false);
    setTimeout(onComplete, 300);
  };

  const handleAudioEnded = () => {
    setIsSpeaking(false);
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
    }
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={handleDismiss}
        >
          {/* Hidden Audio Element */}
          {audioUrl && (
            <audio
              ref={audioRef}
              src={audioUrl}
              muted={isMuted}
              onEnded={handleAudioEnded}
              onPlay={() => setIsSpeaking(true)}
              onPause={() => setIsSpeaking(false)}
            />
          )}

          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="relative w-full max-w-2xl mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={handleDismiss}
              className="absolute -top-12 right-0 text-white/70 hover:text-white hover:bg-white/10 z-10"
            >
              <X className="w-6 h-6" />
            </Button>

            {/* Video/Greeting Container */}
            <div className="relative rounded-2xl overflow-hidden bg-gradient-to-br from-[#032D60] to-slate-900 shadow-2xl border border-white/10">
              {videoUrl ? (
                <video
                  src={videoUrl}
                  autoPlay
                  muted={isMuted}
                  playsInline
                  onEnded={handleDismiss}
                  className="w-full aspect-video object-cover"
                />
              ) : (
                // Animated Greeting without video
                <div className="aspect-video flex flex-col items-center justify-center p-8">
                  {/* Animated Avatar */}
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", delay: 0.2 }}
                    className="relative mb-6"
                  >
                    {/* Speaking pulse animation */}
                    {isSpeaking && (
                      <>
                        <motion.div
                          className="absolute inset-0 rounded-full bg-[#0176D3]/40"
                          animate={{ scale: [1, 1.3, 1], opacity: [0.6, 0, 0.6] }}
                          transition={{ duration: 1, repeat: Infinity }}
                          style={{ margin: "-15px" }}
                        />
                        <motion.div
                          className="absolute inset-0 rounded-full bg-[#0176D3]/30"
                          animate={{ scale: [1, 1.5, 1], opacity: [0.4, 0, 0.4] }}
                          transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
                          style={{ margin: "-20px" }}
                        />
                      </>
                    )}
                    {!isSpeaking && (
                      <motion.div
                        className="absolute inset-0 rounded-full bg-[#0176D3]/30"
                        animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        style={{ margin: "-10px" }}
                      />
                    )}
                    <motion.div
                      className="w-24 h-24 rounded-full bg-gradient-to-br from-[#0176D3] to-[#032D60] flex items-center justify-center border-4 border-white/20 overflow-hidden"
                      animate={isSpeaking ? { scale: [1, 1.05, 1] } : {}}
                      transition={{ duration: 0.3, repeat: isSpeaking ? Infinity : 0 }}
                    >
                      {isLoadingAudio ? (
                        <Loader2 className="w-10 h-10 text-white animate-spin" />
                      ) : (
                        <Image
                          src={avatar.avatar}
                          alt={avatar.displayName}
                          width={96}
                          height={96}
                          className="w-full h-full object-cover"
                        />
                      )}
                    </motion.div>

                    {/* Speaking indicator */}
                    {isSpeaking && (
                      <motion.div
                        className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-emerald-500 border-2 border-white flex items-center justify-center"
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 0.5, repeat: Infinity }}
                      >
                        <Volume2 className="w-3 h-3 text-white" />
                      </motion.div>
                    )}
                  </motion.div>

                  {/* Greeting Text */}
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="text-center"
                  >
                    <h2 className="text-2xl font-bold text-white mb-2">
                      {avatar.language === 'fr' ? 'Bonjour ! Je suis' : "Hello! I'm"} {avatar.displayName}
                    </h2>
                    <p className="text-[#0176D3] font-medium mb-4">
                      {avatar.language === 'fr' ? 'Votre' : 'Your'} {avatar.title}
                    </p>
                    <motion.p
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 1 }}
                      className="text-slate-300 max-w-md mx-auto leading-relaxed"
                    >
                      {avatar.description}
                    </motion.p>
                  </motion.div>

                  {/* CTA */}
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1.5 }}
                    className="mt-6"
                  >
                    <Button
                      onClick={handleDismiss}
                      className="bg-[#0176D3] hover:bg-[#0176D3]/90 text-white px-8"
                    >
                      {avatar.language === 'fr' ? 'Commençons' : "Let's Get Started"}
                    </Button>
                  </motion.div>
                </div>
              )}

              {/* Audio Controls */}
              <div className="absolute bottom-4 right-4">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleMute}
                  className="text-white/70 hover:text-white hover:bg-white/10"
                >
                  {isMuted ? (
                    <VolumeX className="w-5 h-5" />
                  ) : (
                    <Volume2 className="w-5 h-5" />
                  )}
                </Button>
              </div>

              {/* Skip hint */}
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 2 }}
                className="absolute bottom-4 left-4 text-xs text-white/50"
              >
                Click anywhere or press Esc to skip
              </motion.p>
            </div>

            {/* Salesforce Branding */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="mt-4 text-center"
            >
              <p className="text-slate-500 text-xs">
                Powered by ElevenLabs Conversational AI + Salesforce MCP
              </p>
            </motion.div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
