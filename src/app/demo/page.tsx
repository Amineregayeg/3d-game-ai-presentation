"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Upload,
  Play,
  Square,
  Volume2,
  Loader2,
  Activity,
  Zap,
  Clock,
  Target,
  Download,
  RefreshCw,
  ChevronRight,
  Sparkles,
  AudioWaveform,
  Music,
  Brain,
  Gauge,
  FileAudio,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";

// Types
interface Word {
  word: string;
  confidence: number;
  start: number;
  end: number;
}

interface TranscriptionResult {
  text: string;
  words: Word[];
  duration: number;
  rtf: number;
  // Stage 1 specific fields
  ctcText?: string;
  decoderText?: string;
  stage?: number;
  note?: string;
}

interface AudioState {
  isRecording: boolean;
  isPlaying: boolean;
  isProcessing: boolean;
  hasAudio: boolean;
  duration: number;
  currentTime: number;
}

// Mock transcription for demo
const mockTranscription: TranscriptionResult = {
  text: "The quick brown fox jumps over the lazy dog near the riverbank",
  words: [
    { word: "The", confidence: 0.98, start: 0.0, end: 0.12 },
    { word: "quick", confidence: 0.95, start: 0.12, end: 0.38 },
    { word: "brown", confidence: 0.97, start: 0.38, end: 0.62 },
    { word: "fox", confidence: 0.99, start: 0.62, end: 0.85 },
    { word: "jumps", confidence: 0.94, start: 0.85, end: 1.15 },
    { word: "over", confidence: 0.96, start: 1.15, end: 1.38 },
    { word: "the", confidence: 0.98, start: 1.38, end: 1.48 },
    { word: "lazy", confidence: 0.93, start: 1.48, end: 1.78 },
    { word: "dog", confidence: 0.97, start: 1.78, end: 2.02 },
    { word: "near", confidence: 0.91, start: 2.02, end: 2.28 },
    { word: "the", confidence: 0.98, start: 2.28, end: 2.38 },
    { word: "riverbank", confidence: 0.89, start: 2.38, end: 2.95 },
  ],
  duration: 0.42,
  rtf: 0.14,
};

// Sample audio options
const sampleAudios = [
  { id: "gaming", label: "Gaming Command", duration: "2.1s" },
  { id: "dialogue", label: "NPC Dialogue", duration: "4.5s" },
  { id: "ambient", label: "Ambient Speech", duration: "3.2s" },
];

export default function DemoPage() {
  const [audioState, setAudioState] = useState<AudioState>({
    isRecording: false,
    isPlaying: false,
    isProcessing: false,
    hasAudio: false,
    duration: 0,
    currentTime: 0,
  });
  const [transcription, setTranscription] = useState<TranscriptionResult | null>(null);
  const [activeTab, setActiveTab] = useState("record");
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const [spectrogramData, setSpectrogramData] = useState<number[][]>([]);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const spectrogramRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [recordedAudioBase64, setRecordedAudioBase64] = useState<string | null>(null);

  // Generate mock waveform data
  useEffect(() => {
    const data = Array.from({ length: 100 }, () => Math.random() * 0.8 + 0.1);
    setWaveformData(data);

    // Generate mock spectrogram (mel bands x time frames)
    const specData = Array.from({ length: 40 }, () =>
      Array.from({ length: 100 }, () => Math.random())
    );
    setSpectrogramData(specData);
  }, []);

  // Draw waveform
  useEffect(() => {
    if (!canvasRef.current || waveformData.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const barWidth = width / waveformData.length;

    ctx.clearRect(0, 0, width, height);

    // Gradient
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, "#06b6d4");
    gradient.addColorStop(0.5, "#8b5cf6");
    gradient.addColorStop(1, "#06b6d4");

    waveformData.forEach((value, i) => {
      const barHeight = value * height * 0.8;
      const x = i * barWidth;
      const y = (height - barHeight) / 2;

      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barWidth - 1, barHeight);
    });

    // Playhead
    if (audioState.hasAudio && audioState.duration > 0) {
      const playheadX = (audioState.currentTime / audioState.duration) * width;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
    }
  }, [waveformData, audioState.currentTime, audioState.duration, audioState.hasAudio]);

  // Draw spectrogram
  useEffect(() => {
    if (!spectrogramRef.current || spectrogramData.length === 0) return;
    const canvas = spectrogramRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const cellWidth = width / spectrogramData[0].length;
    const cellHeight = height / spectrogramData.length;

    ctx.clearRect(0, 0, width, height);

    spectrogramData.forEach((row, y) => {
      row.forEach((value, x) => {
        const intensity = Math.floor(value * 255);
        ctx.fillStyle = `rgb(${intensity * 0.2}, ${intensity * 0.8}, ${intensity})`;
        ctx.fillRect(x * cellWidth, (spectrogramData.length - 1 - y) * cellHeight, cellWidth, cellHeight);
      });
    });
  }, [spectrogramData]);

  const handleRecord = useCallback(async () => {
    if (audioState.isRecording) {
      // Stop recording
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      setAudioState(prev => ({
        ...prev,
        isRecording: false,
      }));
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          }
        });

        audioChunksRef.current = [];
        setRecordedAudioBase64(null);

        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4'
        });

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          // Stop all tracks
          stream.getTracks().forEach(track => track.stop());

          // Combine chunks into a single blob
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

          // Convert to base64
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64 = reader.result as string;
            // Remove data URL prefix to get raw base64
            const base64Data = base64.split(',')[1];
            setRecordedAudioBase64(base64Data);

            // Calculate duration (approximate from blob size)
            const durationSec = audioBlob.size / 16000; // rough estimate
            setAudioState(prev => ({
              ...prev,
              hasAudio: true,
              duration: Math.max(1, durationSec),
            }));
          };
          reader.readAsDataURL(audioBlob);
        };

        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.start(100); // Collect data every 100ms

        setAudioState(prev => ({
          ...prev,
          isRecording: true,
          hasAudio: false,
        }));
        setTranscription(null);

      } catch (error) {
        console.error("Microphone access denied:", error);
        alert("Please allow microphone access to record audio");
      }
    }
  }, [audioState.isRecording]);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Read the file as base64
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        // Remove data URL prefix to get raw base64
        const base64Data = base64.split(',')[1];
        setRecordedAudioBase64(base64Data);

        // Get audio duration using Audio element
        const audio = new Audio(base64);
        audio.onloadedmetadata = () => {
          setAudioState(prev => ({
            ...prev,
            hasAudio: true,
            duration: audio.duration || 5,
          }));
        };
        // Fallback if metadata doesn't load
        setAudioState(prev => ({
          ...prev,
          hasAudio: true,
          duration: 5,
        }));
      };
      reader.readAsDataURL(file);
      setTranscription(null);
    }
  }, []);

  const handleSelectSample = useCallback((sampleId: string) => {
    setSelectedSample(sampleId);
    setAudioState(prev => ({
      ...prev,
      hasAudio: true,
      duration: parseFloat(sampleAudios.find(s => s.id === sampleId)?.duration || "0"),
    }));
    setTranscription(null);
  }, []);

  const handleTranscribe = useCallback(async () => {
    setAudioState(prev => ({ ...prev, isProcessing: true }));

    try {
      // First, check if the inference server is available
      const healthRes = await fetch("/api/transcribe", { method: "GET" });

      if (!healthRes.ok) {
        console.warn("Inference server not available, using mock");
        setTranscription({
          ...mockTranscription,
          text: "[Server unavailable - showing mock data]",
        });
        return;
      }

      // Determine what audio to transcribe
      let requestBody: { audio?: string; useTestAudio?: boolean };

      if (recordedAudioBase64) {
        // Use the recorded audio from microphone
        requestBody = { audio: recordedAudioBase64 };
      } else {
        // Fall back to test audio from GPU
        requestBody = { useTestAudio: true };
      }

      const transcribeRes = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (transcribeRes.ok) {
        const result = await transcribeRes.json();
        // Map API words format to component format
        const words = (result.words || []).map((w: { word: string; confidence: number }, i: number) => ({
          word: w.word,
          confidence: w.confidence / 100, // API returns 0-100, component expects 0-1
          start: i * 0.1, // Approximate timing
          end: (i + 1) * 0.1,
        }));
        setTranscription({
          text: result.transcription || "[No transcription]",
          words: words,
          duration: result.processing_time_sec || result.audio_duration_sec || 0,
          rtf: result.real_time_factor || 0.1,
          ctcText: result.decoder_transcription,
          decoderText: result.decoder_transcription,
          stage: result.stage || 1,
          note: result.note,
        });
      } else {
        const error = await transcribeRes.json();
        console.error("Transcription error:", error);
        setTranscription({
          ...mockTranscription,
          text: `[Error: ${error.error || "Unknown error"}]`,
        });
      }
    } catch (error) {
      console.error("Transcription failed:", error);
      setTranscription({
        ...mockTranscription,
        text: "[Connection failed - showing mock data]",
      });
    } finally {
      setAudioState(prev => ({ ...prev, isProcessing: false }));
    }
  }, [recordedAudioBase64]);

  const handlePlayPause = useCallback(() => {
    setAudioState(prev => ({ ...prev, isPlaying: !prev.isPlaying }));
  }, []);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.95) return "text-emerald-400";
    if (confidence >= 0.90) return "text-cyan-400";
    if (confidence >= 0.85) return "text-yellow-400";
    return "text-orange-400";
  };

  const getConfidenceBarColor = (confidence: number) => {
    if (confidence >= 0.95) return "bg-emerald-500";
    if (confidence >= 0.90) return "bg-cyan-500";
    if (confidence >= 0.85) return "bg-yellow-500";
    return "bg-orange-500";
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white overflow-hidden relative">
      {/* Background Effects */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-emerald-500/10 rounded-full blur-3xl" />
      </div>

      {/* Grid Pattern */}
      <div
        className="fixed inset-0 opacity-[0.02] pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(to right, white 1px, transparent 1px),
            linear-gradient(to bottom, white 1px, transparent 1px)
          `,
          backgroundSize: "50px 50px",
        }}
      />

      {/* Main Content */}
      <div className="relative z-10 container mx-auto px-6 py-8 pb-32">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 rounded-2xl bg-gradient-to-br from-cyan-500 to-purple-600 shadow-lg shadow-cyan-500/25">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-emerald-400 bg-clip-text text-transparent">
              VoxFormer Live Demo
            </h1>
          </div>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Experience real-time speech-to-text transcription powered by our custom VoxFormer architecture.
            Upload audio, record your voice, or try our sample clips.
          </p>

          {/* Model Status Badge */}
          <div className="flex items-center justify-center gap-2 mt-4">
            <Badge variant="outline" className="border-yellow-500/50 text-yellow-400 px-3 py-1">
              <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse mr-2" />
              Stage 1 Complete
            </Badge>
            <Badge variant="outline" className="border-cyan-500/50 text-cyan-400 px-3 py-1">
              v1.0 • 110.4M trainable • Final Loss: 0.87
            </Badge>
          </div>
        </motion.div>

        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Input */}
          <div className="col-span-5 space-y-6">
            {/* Audio Input Card */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl overflow-hidden">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2 text-cyan-400">
                    <Music className="w-5 h-5" />
                    Audio Input
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                    <TabsList className="grid w-full grid-cols-3 bg-slate-800/50">
                      <TabsTrigger value="record" className="data-[state=active]:bg-cyan-600">
                        <Mic className="w-4 h-4 mr-2" />
                        Record
                      </TabsTrigger>
                      <TabsTrigger value="upload" className="data-[state=active]:bg-purple-600">
                        <Upload className="w-4 h-4 mr-2" />
                        Upload
                      </TabsTrigger>
                      <TabsTrigger value="samples" className="data-[state=active]:bg-emerald-600">
                        <FileAudio className="w-4 h-4 mr-2" />
                        Samples
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent value="record" className="mt-4">
                      <div className="flex flex-col items-center gap-4">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={handleRecord}
                          className={`w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 ${
                            audioState.isRecording
                              ? "bg-red-600 shadow-lg shadow-red-500/50 animate-pulse"
                              : "bg-gradient-to-br from-cyan-500 to-purple-600 shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/50"
                          }`}
                        >
                          {audioState.isRecording ? (
                            <Square className="w-8 h-8 text-white" />
                          ) : (
                            <Mic className="w-10 h-10 text-white" />
                          )}
                        </motion.button>
                        <p className="text-sm text-slate-400">
                          {audioState.isRecording ? "Recording... Click to stop" : "Click to start recording"}
                        </p>
                      </div>
                    </TabsContent>

                    <TabsContent value="upload" className="mt-4">
                      <div
                        onClick={() => fileInputRef.current?.click()}
                        className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center cursor-pointer hover:border-purple-500 hover:bg-purple-500/5 transition-all duration-300"
                      >
                        <Upload className="w-12 h-12 mx-auto mb-3 text-slate-500" />
                        <p className="text-slate-400 mb-1">Drop audio file here or click to browse</p>
                        <p className="text-xs text-slate-500">Supports WAV, MP3, WEBM (max 10MB)</p>
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="audio/*"
                          onChange={handleFileUpload}
                          className="hidden"
                        />
                      </div>
                    </TabsContent>

                    <TabsContent value="samples" className="mt-4">
                      <div className="space-y-2">
                        {sampleAudios.map((sample) => (
                          <motion.button
                            key={sample.id}
                            whileHover={{ x: 4 }}
                            onClick={() => handleSelectSample(sample.id)}
                            className={`w-full flex items-center justify-between p-3 rounded-lg transition-all duration-200 ${
                              selectedSample === sample.id
                                ? "bg-emerald-600/20 border border-emerald-500/50"
                                : "bg-slate-800/50 hover:bg-slate-800 border border-transparent"
                            }`}
                          >
                            <div className="flex items-center gap-3">
                              <div className={`p-2 rounded-lg ${
                                selectedSample === sample.id ? "bg-emerald-600" : "bg-slate-700"
                              }`}>
                                <FileAudio className="w-4 h-4" />
                              </div>
                              <span className="font-medium">{sample.label}</span>
                            </div>
                            <Badge variant="outline" className="text-xs">
                              {sample.duration}
                            </Badge>
                          </motion.button>
                        ))}
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </motion.div>

            {/* Waveform Display */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center justify-between">
                    <span className="flex items-center gap-2 text-purple-400">
                      <Activity className="w-5 h-5" />
                      Waveform
                    </span>
                    {audioState.hasAudio && (
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={handlePlayPause}
                          className="h-8 w-8 p-0"
                        >
                          {audioState.isPlaying ? (
                            <Square className="w-4 h-4" />
                          ) : (
                            <Play className="w-4 h-4" />
                          )}
                        </Button>
                        <span className="text-xs text-slate-400 font-mono">
                          {audioState.currentTime.toFixed(1)}s / {audioState.duration.toFixed(1)}s
                        </span>
                      </div>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="relative h-24 bg-slate-800/50 rounded-lg overflow-hidden">
                    {audioState.hasAudio ? (
                      <canvas
                        ref={canvasRef}
                        width={400}
                        height={96}
                        className="w-full h-full"
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-slate-500">
                        <span className="text-sm">No audio loaded</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Spectrogram */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2 text-emerald-400">
                    <Gauge className="w-5 h-5" />
                    Mel Spectrogram
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="relative h-32 bg-slate-800/50 rounded-lg overflow-hidden">
                    {audioState.hasAudio ? (
                      <canvas
                        ref={spectrogramRef}
                        width={400}
                        height={128}
                        className="w-full h-full"
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-slate-500">
                        <span className="text-sm">Spectrogram will appear here</span>
                      </div>
                    )}
                  </div>
                  <div className="flex justify-between mt-2 text-xs text-slate-500">
                    <span>0 Hz</span>
                    <span>Frequency (Mel Scale)</span>
                    <span>8 kHz</span>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Transcribe Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Button
                onClick={handleTranscribe}
                disabled={!audioState.hasAudio || audioState.isProcessing}
                className="w-full h-14 text-lg font-semibold bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {audioState.isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5 mr-2" />
                    Transcribe with VoxFormer
                  </>
                )}
              </Button>
            </motion.div>
          </div>

          {/* Right Column - Output */}
          <div className="col-span-7 space-y-6">
            {/* Transcription Result */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center justify-between">
                    <span className="flex items-center gap-2 text-cyan-400">
                      <Sparkles className="w-5 h-5" />
                      Transcription
                    </span>
                    {transcription && (
                      <div className="flex items-center gap-2">
                        <Badge className="bg-emerald-600/20 text-emerald-400 border-emerald-500/50">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Complete
                        </Badge>
                      </div>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <AnimatePresence mode="wait">
                    {audioState.isProcessing ? (
                      <motion.div
                        key="loading"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="space-y-3"
                      >
                        <Skeleton className="h-6 w-3/4 bg-slate-700" />
                        <Skeleton className="h-6 w-1/2 bg-slate-700" />
                        <div className="flex gap-2 mt-4">
                          {[...Array(8)].map((_, i) => (
                            <Skeleton key={i} className="h-8 w-16 bg-slate-700" />
                          ))}
                        </div>
                      </motion.div>
                    ) : transcription ? (
                      <motion.div
                        key="result"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="space-y-4"
                      >
                        {/* Main text */}
                        <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50">
                          <p className="text-lg leading-relaxed">
                            &ldquo;{transcription.text}&rdquo;
                          </p>
                        </div>

                        {/* Word-level confidence */}
                        <div>
                          <p className="text-sm text-slate-400 mb-2">Word Confidence Scores</p>
                          <div className="flex flex-wrap gap-2">
                            {transcription.words.map((word, i) => (
                              <motion.div
                                key={i}
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: i * 0.05 }}
                                className="group relative"
                              >
                                <div className={`px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 hover:border-cyan-500/50 transition-colors cursor-default`}>
                                  <span className="font-medium">{word.word}</span>
                                  <div className="flex items-center gap-1 mt-1">
                                    <div className="h-1 w-12 bg-slate-700 rounded-full overflow-hidden">
                                      <div
                                        className={`h-full ${getConfidenceBarColor(word.confidence)} transition-all`}
                                        style={{ width: `${word.confidence * 100}%` }}
                                      />
                                    </div>
                                    <span className={`text-xs ${getConfidenceColor(word.confidence)}`}>
                                      {(word.confidence * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                </div>
                                {/* Tooltip */}
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-700 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                                  {word.start.toFixed(2)}s - {word.end.toFixed(2)}s
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-center py-12 text-slate-500"
                      >
                        <Brain className="w-16 h-16 mx-auto mb-4 opacity-30" />
                        <p>Transcription will appear here</p>
                        <p className="text-sm mt-1">Record, upload, or select a sample to get started</p>
                        <p className="text-xs mt-3 text-yellow-500/60">Stage 1 Demo Mode - Full accuracy available after Stage 2 training</p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </CardContent>
              </Card>
            </motion.div>

            {/* Metrics Row */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="grid grid-cols-4 gap-4"
            >
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl">
                <CardContent className="p-4 text-center">
                  <Clock className="w-6 h-6 mx-auto mb-2 text-cyan-400" />
                  <div className="text-2xl font-bold text-white">
                    {transcription ? `${transcription.duration.toFixed(2)}s` : "—"}
                  </div>
                  <div className="text-xs text-slate-400">Processing Time</div>
                </CardContent>
              </Card>

              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl">
                <CardContent className="p-4 text-center">
                  <Gauge className="w-6 h-6 mx-auto mb-2 text-purple-400" />
                  <div className="text-2xl font-bold text-white">
                    {transcription ? `${transcription.rtf.toFixed(2)}x` : "—"}
                  </div>
                  <div className="text-xs text-slate-400">Real-Time Factor</div>
                </CardContent>
              </Card>

              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl">
                <CardContent className="p-4 text-center">
                  <Target className="w-6 h-6 mx-auto mb-2 text-emerald-400" />
                  <div className="text-2xl font-bold text-white">
                    {transcription
                      ? `${(transcription.words.reduce((a, w) => a + w.confidence, 0) / transcription.words.length * 100).toFixed(1)}%`
                      : "—"}
                  </div>
                  <div className="text-xs text-slate-400">Avg Confidence</div>
                </CardContent>
              </Card>

              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl">
                <CardContent className="p-4 text-center">
                  <AudioWaveform className="w-6 h-6 mx-auto mb-2 text-yellow-400" />
                  <div className="text-2xl font-bold text-white">
                    {transcription ? transcription.words.length : "—"}
                  </div>
                  <div className="text-xs text-slate-400">Words Detected</div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Model Info */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <Card className="bg-slate-900/80 border-slate-700/50 backdrop-blur-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center justify-between">
                    <span className="flex items-center gap-2 text-purple-400">
                      <Brain className="w-5 h-5" />
                      Model Information
                    </span>
                    <div className="flex items-center gap-2">
                      <Switch
                        checked={showComparison}
                        onCheckedChange={setShowComparison}
                        className="data-[state=checked]:bg-purple-600"
                      />
                      <span className="text-xs text-slate-400">Show Stage Comparison</span>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-2 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Architecture</span>
                        <span className="font-mono text-cyan-400">VoxFormer v1.0</span>
                      </div>
                      <div className="flex justify-between items-center p-2 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Parameters</span>
                        <span className="font-mono text-white">110.4M trainable</span>
                      </div>
                      <div className="flex justify-between items-center p-2 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Training Data</span>
                        <span className="font-mono text-white">LibriSpeech 100h</span>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-2 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Stage</span>
                        <span className="font-mono text-yellow-400">1 of 3 (WavLM frozen)</span>
                      </div>
                      <div className="flex justify-between items-center p-2 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Epochs Trained</span>
                        <span className="font-mono text-white">19/20 (~8 hours)</span>
                      </div>
                      <div className="flex justify-between items-center p-2 bg-slate-800/50 rounded-lg">
                        <span className="text-slate-400">Final Loss</span>
                        <span className="font-mono text-emerald-400">0.87 (88% reduction)</span>
                      </div>
                    </div>
                  </div>

                  {/* Comparison Section */}
                  <AnimatePresence>
                    {showComparison && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 overflow-hidden"
                      >
                        <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700/50">
                          <p className="text-sm text-slate-400 mb-3">Training Stage Comparison</p>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="text-center p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/30">
                              <p className="font-semibold text-yellow-400">Stage 1 (Current)</p>
                              <p className="text-2xl font-bold mt-1">0.87 Loss</p>
                              <p className="text-xs text-slate-400 mt-1">WavLM frozen • 19 epochs</p>
                            </div>
                            <div className="text-center p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/30">
                              <p className="font-semibold text-emerald-400">Stage 2 (Next)</p>
                              <p className="text-2xl font-bold mt-1">~0.3 Loss</p>
                              <p className="text-xs text-slate-400 mt-1">Unfreeze top 3 WavLM layers</p>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Action Buttons */}
                  <div className="flex gap-3 mt-4">
                    <Button variant="outline" className="flex-1 border-slate-600 hover:bg-slate-800">
                      <Download className="w-4 h-4 mr-2" />
                      Download Model
                    </Button>
                    <Button variant="outline" className="flex-1 border-slate-600 hover:bg-slate-800" asChild>
                      <a href="/training">
                        <RefreshCw className="w-4 h-4 mr-2" />
                        View Training
                      </a>
                    </Button>
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
  );
}
