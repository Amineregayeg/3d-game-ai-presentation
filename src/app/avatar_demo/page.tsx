"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Send,
  Volume2,
  Video,
  RefreshCw,
  Loader2,
  Settings2,
  Play,
  Pause,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  Clock,
  Zap,
  User,
  AudioWaveform,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

// =============================================================================
// Types
// =============================================================================

interface Voice {
  id: string;
  name: string;
  category?: string;
  preview_url?: string;
  labels?: Record<string, string>;
}

interface AvatarOption {
  id: string;
  name: string;
  url: string;
}

interface AvatarResponse {
  audio_url: string;
  video_url?: string;
  duration: number;
  processing_time: number;
  request_id: string;
  has_video: boolean;
}

interface SystemStatus {
  elevenlabs: boolean;
  musetalk: boolean;
  gpu_server: string;
}

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://5.249.161.66:5000";

// =============================================================================
// Sample texts for quick testing
// =============================================================================

const sampleTexts = [
  {
    label: "Greeting",
    text: "Hello! Welcome to the 3D Game AI Assistant. How can I help you today?",
  },
  {
    label: "Quest",
    text: "Your quest awaits, brave adventurer. Journey to the ancient ruins and retrieve the Crystal of Eternity.",
  },
  {
    label: "Combat",
    text: "Watch out! Enemies approaching from the north. Ready your weapons and prepare for battle!",
  },
  {
    label: "Tutorial",
    text: "To craft a sword, you'll need three iron ingots and two leather strips. Visit the blacksmith in the village.",
  },
];

// =============================================================================
// Component
// =============================================================================

export default function AvatarDemoPage() {
  // State
  const [text, setText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [response, setResponse] = useState<AvatarResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Voice & Avatar selection
  const [voices, setVoices] = useState<Voice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState("21m00Tcm4TlvDq8ikWAM"); // Rachel
  const [avatars, setAvatars] = useState<AvatarOption[]>([]);
  const [selectedAvatar, setSelectedAvatar] = useState("default");

  // Settings
  const [generateVideo, setGenerateVideo] = useState(false);
  const [stability, setStability] = useState([0.5]);
  const [similarityBoost, setSimilarityBoost] = useState([0.75]);

  // System status
  const [status, setStatus] = useState<SystemStatus | null>(null);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // =============================================================================
  // Effects
  // =============================================================================

  // Fetch voices, avatars, and status on mount
  useEffect(() => {
    fetchVoices();
    fetchAvatars();
    fetchStatus();
  }, []);

  // =============================================================================
  // API Functions
  // =============================================================================

  const fetchVoices = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/avatar/voices`);
      if (res.ok) {
        const data = await res.json();
        setVoices(data.voices || []);
      }
    } catch (err) {
      console.error("Failed to fetch voices:", err);
    }
  };

  const fetchAvatars = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/avatar/avatars`);
      if (res.ok) {
        const data = await res.json();
        setAvatars(data.avatars || []);
      }
    } catch (err) {
      console.error("Failed to fetch avatars:", err);
      // Default avatar
      setAvatars([{ id: "default", name: "Default Avatar", url: "/avatars/default.png" }]);
    }
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/avatar/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.error("Failed to fetch status:", err);
    }
  };

  const handleSubmit = async () => {
    if (!text.trim()) return;

    setIsLoading(true);
    setProgress(10);
    setProgressMessage("Initializing...");
    setError(null);
    setResponse(null);

    try {
      // Stage 1: Sending request
      setProgress(20);
      setProgressMessage("Generating speech with ElevenLabs...");

      const res = await fetch(`${API_BASE}/api/avatar/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text.trim(),
          voice_id: selectedVoice,
          avatar_id: selectedAvatar,
          generate_video: generateVideo,
          voice_settings: {
            stability: stability[0],
            similarity_boost: similarityBoost[0],
          },
        }),
      });

      setProgress(60);
      setProgressMessage(generateVideo ? "Running lip-sync on GPU..." : "Processing audio...");

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || "Failed to generate avatar speech");
      }

      const data: AvatarResponse = await res.json();
      setProgress(90);
      setProgressMessage("Finalizing...");

      // Prepend API base to URLs
      data.audio_url = `${API_BASE}${data.audio_url}`;
      if (data.video_url) {
        data.video_url = `${API_BASE}${data.video_url}`;
      }

      setResponse(data);
      setProgress(100);
      setProgressMessage("Complete!");

      // Auto-play
      setTimeout(() => {
        if (data.has_video && videoRef.current) {
          videoRef.current.load();
          videoRef.current.play().catch(() => {});
        } else if (audioRef.current) {
          audioRef.current.load();
          audioRef.current.play().catch(() => {});
        }
      }, 100);

    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleText = (sampleText: string) => {
    setText(sampleText);
  };

  // =============================================================================
  // Render
  // =============================================================================

  return (
    <TooltipProvider>
      <div className="dark min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        {/* Background effects */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 -left-32 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-rose-500/5 rounded-full blur-3xl" />
        </div>

        <div className="relative z-10 container mx-auto px-4 py-8 max-w-6xl">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-rose-400 bg-clip-text text-transparent mb-3">
              Avatar Demo
            </h1>
            <p className="text-slate-400 text-lg">
              ElevenLabs Flash v2.5 TTS + MuseTalk 1.5 Lip-Sync
            </p>
            <div className="flex justify-center gap-2 mt-4">
              <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 bg-cyan-500/10">
                <Zap className="w-3 h-3 mr-1" />
                75ms TTFB
              </Badge>
              <Badge variant="outline" className="border-purple-500/50 text-purple-400 bg-purple-500/10">
                <AudioWaveform className="w-3 h-3 mr-1" />
                4.14 MOS
              </Badge>
              {status?.musetalk && (
                <Badge variant="outline" className="border-rose-500/50 text-rose-400 bg-rose-500/10">
                  <Video className="w-3 h-3 mr-1" />
                  30fps+
                </Badge>
              )}
            </div>
          </motion.div>

          {/* System Status */}
          {status && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-center gap-4 mb-6"
            >
              <div className="flex items-center gap-2 text-sm">
                {status.elevenlabs ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-red-400" />
                )}
                <span className="text-slate-400">ElevenLabs</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                {status.musetalk ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-amber-400" />
                )}
                <span className="text-slate-400">MuseTalk</span>
              </div>
            </motion.div>
          )}

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Left Column: Input */}
            <div className="space-y-6">
              {/* Text Input Card */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-cyan-400">
                      <Mic className="w-5 h-5" />
                      Text Input
                    </CardTitle>
                    <CardDescription>
                      Enter text for the avatar to speak
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Textarea
                      placeholder="Hello! Welcome to the 3D Game AI Assistant..."
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      className="min-h-[120px] bg-slate-800/50 border-slate-700 text-slate-100 placeholder:text-slate-500 resize-none"
                      maxLength={1000}
                    />
                    <div className="flex justify-between items-center text-sm">
                      <span className="text-slate-500">
                        {text.length} / 1000 characters
                      </span>
                      <Button
                        onClick={handleSubmit}
                        disabled={isLoading || !text.trim()}
                        className="bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 text-white"
                      >
                        {isLoading ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <Send className="w-4 h-4 mr-2" />
                            Generate
                          </>
                        )}
                      </Button>
                    </div>

                    {/* Sample Texts */}
                    <div className="pt-2 border-t border-slate-700/50">
                      <p className="text-xs text-slate-500 mb-2">Quick samples:</p>
                      <div className="flex flex-wrap gap-2">
                        {sampleTexts.map((sample) => (
                          <Button
                            key={sample.label}
                            variant="outline"
                            size="sm"
                            onClick={() => handleSampleText(sample.text)}
                            className="text-xs border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800"
                          >
                            {sample.label}
                          </Button>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Settings Card */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-purple-400">
                      <Settings2 className="w-5 h-5" />
                      Settings
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Voice Selection */}
                    <div className="space-y-2">
                      <Label className="text-slate-300">Voice</Label>
                      <Select value={selectedVoice} onValueChange={setSelectedVoice}>
                        <SelectTrigger className="bg-slate-800/50 border-slate-700">
                          <SelectValue placeholder="Select voice" />
                        </SelectTrigger>
                        <SelectContent className="bg-slate-800 border-slate-700">
                          {voices.length > 0 ? (
                            voices.map((voice) => (
                              <SelectItem key={voice.id} value={voice.id}>
                                {voice.name}
                              </SelectItem>
                            ))
                          ) : (
                            <SelectItem value="21m00Tcm4TlvDq8ikWAM">
                              Rachel (Default)
                            </SelectItem>
                          )}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Avatar Selection */}
                    <div className="space-y-2">
                      <Label className="text-slate-300">Avatar</Label>
                      <div className="flex gap-2 flex-wrap">
                        {avatars.map((avatar) => (
                          <Tooltip key={avatar.id}>
                            <TooltipTrigger asChild>
                              <button
                                onClick={() => setSelectedAvatar(avatar.id)}
                                className={`p-1 rounded-lg border-2 transition-all ${
                                  selectedAvatar === avatar.id
                                    ? "border-purple-500 bg-purple-500/20"
                                    : "border-slate-700 hover:border-slate-500"
                                }`}
                              >
                                <Avatar className="w-12 h-12">
                                  <AvatarImage src={avatar.url} alt={avatar.name} />
                                  <AvatarFallback className="bg-slate-700">
                                    <User className="w-6 h-6 text-slate-400" />
                                  </AvatarFallback>
                                </Avatar>
                              </button>
                            </TooltipTrigger>
                            <TooltipContent>{avatar.name}</TooltipContent>
                          </Tooltip>
                        ))}
                      </div>
                    </div>

                    {/* Voice Settings */}
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label className="text-slate-300">Stability</Label>
                          <span className="text-sm text-slate-500">{stability[0].toFixed(2)}</span>
                        </div>
                        <Slider
                          value={stability}
                          onValueChange={setStability}
                          min={0}
                          max={1}
                          step={0.05}
                          className="py-2"
                        />
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <Label className="text-slate-300">Similarity</Label>
                          <span className="text-sm text-slate-500">{similarityBoost[0].toFixed(2)}</span>
                        </div>
                        <Slider
                          value={similarityBoost}
                          onValueChange={setSimilarityBoost}
                          min={0}
                          max={1}
                          step={0.05}
                          className="py-2"
                        />
                      </div>
                    </div>

                    {/* Video Toggle */}
                    <div className="flex items-center justify-between py-2 border-t border-slate-700/50">
                      <div className="space-y-0.5">
                        <Label className="text-slate-300">Generate Video</Label>
                        <p className="text-xs text-slate-500">
                          {status?.musetalk ? "MuseTalk lip-sync available" : "MuseTalk not available"}
                        </p>
                      </div>
                      <Switch
                        checked={generateVideo}
                        onCheckedChange={setGenerateVideo}
                        disabled={!status?.musetalk}
                      />
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Right Column: Output */}
            <div className="space-y-6">
              {/* Progress */}
              <AnimatePresence>
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                  >
                    <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                      <CardContent className="py-4">
                        <div className="flex justify-between text-sm text-slate-400 mb-2">
                          <span>{progressMessage}</span>
                          <span>{progress}%</span>
                        </div>
                        <Progress value={progress} className="h-2" />
                      </CardContent>
                    </Card>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Output Card */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-rose-400">
                      {response?.has_video ? (
                        <Video className="w-5 h-5" />
                      ) : (
                        <Volume2 className="w-5 h-5" />
                      )}
                      Avatar Output
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {/* Loading State */}
                    {isLoading && !response && (
                      <Skeleton className="w-full aspect-video bg-slate-800 rounded-lg" />
                    )}

                    {/* Video Output */}
                    {response?.has_video && response.video_url && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="space-y-4"
                      >
                        <video
                          ref={videoRef}
                          className="w-full rounded-lg border border-slate-700"
                          controls
                          src={response.video_url}
                        />
                        <div className="flex gap-4 text-sm text-slate-400">
                          <div className="flex items-center gap-1">
                            <Clock className="w-4 h-4" />
                            Duration: {response.duration.toFixed(1)}s
                          </div>
                          <div className="flex items-center gap-1">
                            <Zap className="w-4 h-4" />
                            Processing: {response.processing_time.toFixed(1)}s
                          </div>
                        </div>
                      </motion.div>
                    )}

                    {/* Audio Only Output */}
                    {response && !response.has_video && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="space-y-4"
                      >
                        <div className="w-full aspect-video bg-slate-800/50 rounded-lg flex flex-col items-center justify-center border border-slate-700">
                          <Avatar className="w-24 h-24 mb-4">
                            <AvatarImage
                              src={avatars.find(a => a.id === selectedAvatar)?.url}
                              alt="Avatar"
                            />
                            <AvatarFallback className="bg-slate-700 text-4xl">
                              <User className="w-12 h-12 text-slate-400" />
                            </AvatarFallback>
                          </Avatar>
                          <audio
                            ref={audioRef}
                            controls
                            src={response.audio_url}
                            className="w-full max-w-sm"
                          />
                        </div>
                        <div className="flex gap-4 text-sm text-slate-400">
                          <div className="flex items-center gap-1">
                            <Clock className="w-4 h-4" />
                            Duration: {response.duration.toFixed(1)}s
                          </div>
                          <div className="flex items-center gap-1">
                            <Zap className="w-4 h-4" />
                            Processing: {response.processing_time.toFixed(1)}s
                          </div>
                        </div>
                        <p className="text-xs text-slate-500 text-center">
                          Video generation disabled. Enable &quot;Generate Video&quot; for lip-sync.
                        </p>
                      </motion.div>
                    )}

                    {/* Empty State */}
                    {!isLoading && !response && (
                      <div className="w-full aspect-video bg-slate-800/30 rounded-lg flex flex-col items-center justify-center border border-dashed border-slate-700">
                        <Sparkles className="w-12 h-12 text-slate-600 mb-3" />
                        <p className="text-slate-500">
                          Avatar output will appear here
                        </p>
                        <p className="text-xs text-slate-600 mt-1">
                          Enter text and click Generate
                        </p>
                      </div>
                    )}

                    {/* Error State */}
                    {error && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="p-4 bg-red-500/10 border border-red-500/50 rounded-lg"
                      >
                        <div className="flex items-center gap-2 text-red-400">
                          <AlertCircle className="w-5 h-5" />
                          <span className="font-medium">Error</span>
                        </div>
                        <p className="text-red-300 text-sm mt-1">{error}</p>
                      </motion.div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>

              {/* Tech Info Card */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-sm">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-emerald-400 text-sm">
                      <Sparkles className="w-4 h-4" />
                      Technology Stack
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="space-y-1">
                        <p className="text-slate-500">TTS Engine</p>
                        <p className="text-slate-300">ElevenLabs Flash v2.5</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-slate-500">Lip-Sync</p>
                        <p className="text-slate-300">MuseTalk 1.5</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-slate-500">Voice Quality</p>
                        <p className="text-slate-300">4.14 MOS Score</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-slate-500">Latency</p>
                        <p className="text-slate-300">75ms TTFB</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>
          </div>
        </div>

        {/* Navigation Dock */}
        <PresentationDock items={dockItems} />
      </div>
    </TooltipProvider>
  );
}
