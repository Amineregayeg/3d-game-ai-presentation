"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, Square, Loader2, Send, Keyboard } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface VoiceInputProps {
  onAudioCaptured: (audioBlob: Blob) => void;
  onTextSubmit: (text: string) => void;
  isProcessing: boolean;
  disabled?: boolean;
}

export function VoiceInput({
  onAudioCaptured,
  onTextSubmit,
  isProcessing,
  disabled = false,
}: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [textInput, setTextInput] = useState("");
  const [audioLevel, setAudioLevel] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      // Set up audio analysis for visualization
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Animate audio level
      const updateLevel = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          setAudioLevel(average / 255);
        }
        animationFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();

      // Set up media recorder - prefer Opus as it's more widely supported
      let mimeType = "audio/webm;codecs=opus";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        // Fallback to basic webm if opus not supported
        mimeType = "audio/webm";
      }

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordingTime(0);

      // Timer
      timerRef.current = setInterval(() => {
        setRecordingTime((t) => t + 1);
      }, 1000);
    } catch (error) {
      console.error("Failed to start recording:", error);
    }
  }, []);

  const stopRecording = useCallback(async () => {
    if (!mediaRecorderRef.current) return;

    return new Promise<Blob>((resolve) => {
      mediaRecorderRef.current!.onstop = () => {
        const mimeType = mediaRecorderRef.current?.mimeType || "audio/webm";
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        resolve(blob);
      };

      mediaRecorderRef.current!.stop();
      mediaRecorderRef.current!.stream.getTracks().forEach((t) => t.stop());
      setIsRecording(false);

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      setAudioLevel(0);
    });
  }, []);

  const handleRecordClick = async () => {
    if (isRecording) {
      const blob = await stopRecording();
      if (blob && blob.size > 0) {
        onAudioCaptured(blob);
      }
    } else {
      await startRecording();
    }
  };

  const handleTextSubmit = () => {
    if (textInput.trim()) {
      onTextSubmit(textInput.trim());
      setTextInput("");
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="bg-slate-800/50 border border-white/10 rounded-xl overflow-hidden backdrop-blur-sm">
      <Tabs defaultValue="voice" className="w-full">
        <TabsList className="w-full grid grid-cols-2 bg-slate-900/50 rounded-none border-b border-white/10">
          <TabsTrigger
            value="voice"
            className="gap-2 data-[state=active]:bg-cyan-500/20 data-[state=active]:text-cyan-400"
          >
            <Mic className="w-4 h-4" />
            Voice
          </TabsTrigger>
          <TabsTrigger
            value="text"
            className="gap-2 data-[state=active]:bg-purple-500/20 data-[state=active]:text-purple-400"
          >
            <Keyboard className="w-4 h-4" />
            Text
          </TabsTrigger>
        </TabsList>

        <TabsContent value="voice" className="p-6 m-0">
          <div className="flex flex-col items-center">
            {/* Recording Button with Animation */}
            <div className="relative">
              {/* Pulse rings when recording */}
              <AnimatePresence>
                {isRecording && (
                  <>
                    <motion.div
                      initial={{ scale: 1, opacity: 0.5 }}
                      animate={{ scale: 1.5, opacity: 0 }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                      className="absolute inset-0 rounded-full bg-red-500"
                    />
                    <motion.div
                      initial={{ scale: 1, opacity: 0.3 }}
                      animate={{ scale: 1.8, opacity: 0 }}
                      transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
                      className="absolute inset-0 rounded-full bg-red-500"
                    />
                  </>
                )}
              </AnimatePresence>

              {/* Audio level ring */}
              {isRecording && (
                <motion.div
                  animate={{ scale: 1 + audioLevel * 0.3 }}
                  className="absolute inset-0 rounded-full bg-red-500/30"
                />
              )}

              <Button
                size="lg"
                onClick={handleRecordClick}
                disabled={disabled || isProcessing}
                className={`relative w-24 h-24 rounded-full transition-all z-10 ${
                  isRecording
                    ? "bg-red-500 hover:bg-red-600"
                    : "bg-gradient-to-br from-cyan-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500"
                }`}
              >
                {isProcessing ? (
                  <Loader2 className="w-8 h-8 animate-spin" />
                ) : isRecording ? (
                  <Square className="w-8 h-8" />
                ) : (
                  <Mic className="w-8 h-8" />
                )}
              </Button>
            </div>

            {/* Status Text */}
            <div className="mt-4 text-center">
              {isProcessing ? (
                <p className="text-cyan-400 animate-pulse">Processing audio...</p>
              ) : isRecording ? (
                <div className="space-y-1">
                  <p className="text-red-400 font-medium">Recording...</p>
                  <p className="text-2xl font-mono text-white">{formatTime(recordingTime)}</p>
                  <p className="text-xs text-slate-500">Click to stop</p>
                </div>
              ) : (
                <div className="space-y-1">
                  <p className="text-slate-400">Click to start recording</p>
                  <p className="text-xs text-slate-500">
                    Speak clearly into your microphone
                  </p>
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="text" className="p-6 m-0">
          <div className="space-y-4">
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Type your message here... e.g., 'Create a low-poly sword in Blender'"
              disabled={disabled || isProcessing}
              className="w-full h-32 px-4 py-3 bg-slate-700/50 border border-white/10 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 resize-none"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleTextSubmit();
                }
              }}
            />
            <Button
              onClick={handleTextSubmit}
              disabled={!textInput.trim() || disabled || isProcessing}
              className="w-full bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500"
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Send className="w-4 h-4 mr-2" />
              )}
              Send Message
            </Button>
            <p className="text-xs text-center text-slate-500">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
