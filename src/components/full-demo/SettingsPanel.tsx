"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings, ChevronDown, Mic, User, Volume2, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

export type STTEngine = "whisper" | "voxformer";

interface Avatar {
  id: string;
  name: string;
  thumbnail?: string;
}

interface Voice {
  id: string;
  name: string;
  gender: "male" | "female";
}

interface SettingsPanelProps {
  sttEngine: STTEngine;
  onSTTEngineChange: (engine: STTEngine) => void;
  selectedAvatar: string;
  onAvatarChange: (avatar: string) => void;
  selectedVoice: string;
  onVoiceChange: (voice: string) => void;
  avatars: Avatar[];
  gpuStatus?: {
    status: string;
    gpu: string;
  } | null;
}

const VOICES: Voice[] = [
  { id: "EXAVITQu4vr4xnSDxMaL", name: "Rachel", gender: "female" },
  { id: "21m00Tcm4TlvDq8ikWAM", name: "Rachel", gender: "female" },
  { id: "AZnzlk1XvdvUeBnXmlld", name: "Domi", gender: "female" },
  { id: "MF3mGyEYCl7XYWbV9V6O", name: "Elli", gender: "female" },
  { id: "TxGEqnHWrfWFTfGW9XjX", name: "Josh", gender: "male" },
  { id: "VR6AewLTigWG4xSOukaG", name: "Arnold", gender: "male" },
];

export function SettingsPanel({
  sttEngine,
  onSTTEngineChange,
  selectedAvatar,
  onAvatarChange,
  selectedVoice,
  onVoiceChange,
  avatars,
  gpuStatus,
}: SettingsPanelProps) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="bg-slate-800/50 border border-white/10 rounded-xl overflow-hidden backdrop-blur-sm">
        <CollapsibleTrigger asChild>
          <Button
            variant="ghost"
            className="w-full flex items-center justify-between p-4 hover:bg-white/5"
          >
            <div className="flex items-center gap-3">
              <Settings className="w-5 h-5 text-purple-400" />
              <span className="font-medium text-white">Settings</span>
            </div>
            <div className="flex items-center gap-2">
              {gpuStatus && (
                <Badge
                  variant="outline"
                  className={`text-xs ${
                    gpuStatus.status === "ok"
                      ? "border-emerald-500/50 text-emerald-400"
                      : "border-yellow-500/50 text-yellow-400"
                  }`}
                >
                  <Cpu className="w-3 h-3 mr-1" />
                  {gpuStatus.gpu?.split(" ")[0] || "GPU"}
                </Badge>
              )}
              <motion.div
                animate={{ rotate: isOpen ? 180 : 0 }}
                transition={{ duration: 0.2 }}
              >
                <ChevronDown className="w-4 h-4 text-slate-400" />
              </motion.div>
            </div>
          </Button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <AnimatePresence>
            {isOpen && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="px-4 pb-4 space-y-4"
              >
                {/* STT Engine Selection */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm text-slate-400">
                    <Mic className="w-4 h-4" />
                    Speech-to-Text Engine
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => onSTTEngineChange("whisper")}
                      className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                        sttEngine === "whisper"
                          ? "bg-cyan-500 text-white shadow-lg shadow-cyan-500/25"
                          : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50 hover:text-white"
                      }`}
                    >
                      <div className="flex items-center justify-center gap-2">
                        <span>Whisper</span>
                        <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                          GPU
                        </Badge>
                      </div>
                      <span className="text-[10px] opacity-70 block mt-0.5">
                        OpenAI large-v3
                      </span>
                    </button>
                    <button
                      onClick={() => onSTTEngineChange("voxformer")}
                      className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
                        sttEngine === "voxformer"
                          ? "bg-purple-500 text-white shadow-lg shadow-purple-500/25"
                          : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50 hover:text-white"
                      }`}
                    >
                      <div className="flex items-center justify-center gap-2">
                        <span>VoxFormer</span>
                        <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                          Custom
                        </Badge>
                      </div>
                      <span className="text-[10px] opacity-70 block mt-0.5">
                        Our STT Model
                      </span>
                    </button>
                  </div>
                </div>

                {/* Avatar Selection */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm text-slate-400">
                    <User className="w-4 h-4" />
                    Avatar
                  </label>
                  <div className="flex gap-2 flex-wrap">
                    {avatars.length > 0 ? (
                      avatars.map((avatar) => (
                        <button
                          key={avatar.id}
                          onClick={() => onAvatarChange(avatar.id)}
                          className={`px-3 py-2 rounded-lg text-sm transition-all ${
                            selectedAvatar === avatar.id
                              ? "bg-cyan-500 text-white"
                              : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50 hover:text-white"
                          }`}
                        >
                          {avatar.name || avatar.id}
                        </button>
                      ))
                    ) : (
                      <>
                        {["default", "assistant", "tech"].map((id) => (
                          <button
                            key={id}
                            onClick={() => onAvatarChange(id)}
                            className={`px-3 py-2 rounded-lg text-sm transition-all capitalize ${
                              selectedAvatar === id
                                ? "bg-cyan-500 text-white"
                                : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50 hover:text-white"
                            }`}
                          >
                            {id}
                          </button>
                        ))}
                      </>
                    )}
                  </div>
                </div>

                {/* Voice Selection */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm text-slate-400">
                    <Volume2 className="w-4 h-4" />
                    Voice (ElevenLabs)
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {VOICES.slice(0, 6).map((voice) => (
                      <button
                        key={voice.id}
                        onClick={() => onVoiceChange(voice.id)}
                        className={`px-3 py-2 rounded-lg text-sm transition-all ${
                          selectedVoice === voice.id
                            ? "bg-purple-500 text-white"
                            : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50 hover:text-white"
                        }`}
                      >
                        <span>{voice.name}</span>
                        <span className="text-[10px] opacity-70 block">
                          {voice.gender}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}
