"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Settings,
  ChevronDown,
  Volume2,
  Cloud,
  Shield,
  Zap,
  User,
  Loader2,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import {
  SalesforceConnectionStatus,
  SalesforceSettings as SettingsType,
  CONSULTANT_VOICES,
} from "./types";

interface SalesforceSettingsPanelProps {
  connectionStatus: SalesforceConnectionStatus;
  onConnect: () => void;
  onDisconnect: () => void;
  selectedVoice: string;
  onVoiceChange: (voiceId: string) => void;
  autoExecute: boolean;
  onAutoExecuteChange: (value: boolean) => void;
  debugMode: boolean;
  onDebugModeChange: (value: boolean) => void;
}

export function SalesforceSettingsPanel({
  connectionStatus,
  onConnect,
  onDisconnect,
  selectedVoice,
  onVoiceChange,
  autoExecute,
  onAutoExecuteChange,
  debugMode,
  onDebugModeChange,
}: SalesforceSettingsPanelProps) {
  const [isOpen, setIsOpen] = useState(true);

  const handleVoiceChange = (voiceId: string) => {
    onVoiceChange(voiceId);
  };

  const settings = { voiceId: selectedVoice, autoExecute, debugMode };

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div className="bg-slate-800/50 border border-white/10 rounded-xl overflow-hidden backdrop-blur-sm">
        <CollapsibleTrigger asChild>
          <Button
            variant="ghost"
            className="w-full flex items-center justify-between p-4 hover:bg-white/5"
          >
            <div className="flex items-center gap-3">
              <Settings className="w-5 h-5 text-[#0176D3]" />
              <span className="font-medium text-white">Settings</span>
            </div>
            <div className="flex items-center gap-2">
              {/* Connection Status */}
              {connectionStatus.connecting ? (
                <Badge
                  variant="outline"
                  className="text-xs border-cyan-500/50 text-cyan-400"
                >
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  Connecting
                </Badge>
              ) : connectionStatus.connected ? (
                <Badge
                  variant="outline"
                  className="text-xs border-emerald-500/50 text-emerald-400"
                >
                  <CheckCircle className="w-3 h-3 mr-1" />
                  {connectionStatus.org?.name || "Connected"}
                </Badge>
              ) : (
                <Badge
                  variant="outline"
                  className="text-xs border-slate-500/50 text-slate-400"
                >
                  <XCircle className="w-3 h-3 mr-1" />
                  Not Connected
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
                {/* Salesforce Connection */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm text-slate-400">
                    <Cloud className="w-4 h-4 text-[#0176D3]" />
                    Salesforce Connection
                  </label>
                  <div className="bg-slate-900/50 rounded-lg p-3">
                    {connectionStatus.connected && connectionStatus.org ? (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-white font-medium text-sm">
                              {connectionStatus.org.name}
                            </p>
                            <p className="text-slate-400 text-xs">
                              {connectionStatus.org.username}
                            </p>
                          </div>
                          <Badge
                            variant="outline"
                            className="text-[10px] border-[#0176D3]/50 text-[#0176D3]"
                          >
                            {connectionStatus.org.orgType}
                          </Badge>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={onDisconnect}
                          className="w-full text-xs border-red-500/50 text-red-400 hover:bg-red-500/10"
                        >
                          Disconnect
                        </Button>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p className="text-slate-400 text-xs text-center mb-2">
                          Connect to your Salesforce org to enable MCP operations
                        </p>
                        <Button
                          onClick={onConnect}
                          disabled={connectionStatus.connecting}
                          className="w-full bg-[#0176D3] hover:bg-[#0176D3]/90 text-white"
                        >
                          {connectionStatus.connecting ? (
                            <>
                              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                              Connecting...
                            </>
                          ) : (
                            <>
                              <Cloud className="w-4 h-4 mr-2" />
                              Connect Salesforce
                            </>
                          )}
                        </Button>
                        {connectionStatus.error && (
                          <p className="text-red-400 text-xs text-center">
                            {connectionStatus.error}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Voice Selection */}
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm text-slate-400">
                    <Volume2 className="w-4 h-4" />
                    Consultant Voice
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    {CONSULTANT_VOICES.map((voice) => (
                      <button
                        key={voice.id}
                        onClick={() => handleVoiceChange(voice.id)}
                        className={`px-3 py-2 rounded-lg text-sm transition-all ${
                          settings.voiceId === voice.id
                            ? "bg-[#0176D3] text-white shadow-lg shadow-[#0176D3]/25"
                            : "bg-slate-700/50 text-slate-400 hover:bg-slate-600/50 hover:text-white"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <User className="w-3 h-3" />
                          <span>{voice.name}</span>
                        </div>
                        <span className="text-[10px] opacity-70 block mt-0.5">
                          {voice.style}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Options */}
                <div className="space-y-3">
                  <label className="flex items-center gap-2 text-sm text-slate-400">
                    <Zap className="w-4 h-4" />
                    Options
                  </label>

                  {/* Auto Execute */}
                  <div className="flex items-center justify-between bg-slate-900/50 rounded-lg p-3">
                    <div>
                      <p className="text-white text-sm">Auto-Execute</p>
                      <p className="text-slate-500 text-xs">
                        Run Salesforce operations automatically
                      </p>
                    </div>
                    <Switch
                      checked={autoExecute}
                      onCheckedChange={onAutoExecuteChange}
                    />
                  </div>

                  {/* Debug Info */}
                  <div className="flex items-center justify-between bg-slate-900/50 rounded-lg p-3">
                    <div>
                      <p className="text-white text-sm">Debug Info</p>
                      <p className="text-slate-500 text-xs">
                        Show detailed operation logs
                      </p>
                    </div>
                    <Switch
                      checked={debugMode}
                      onCheckedChange={onDebugModeChange}
                    />
                  </div>
                </div>

                {/* Consultant Info */}
                <div className="bg-gradient-to-r from-[#0176D3]/10 to-[#032D60]/10 rounded-lg p-3 border border-[#0176D3]/20">
                  <div className="flex items-center gap-2 mb-2">
                    <Shield className="w-4 h-4 text-[#0176D3]" />
                    <span className="text-white text-sm font-medium">
                      Consultant: Alex
                    </span>
                  </div>
                  <p className="text-slate-400 text-xs">
                    Senior Salesforce Consultant with 15+ years experience.
                    Specializes in Sales Cloud, Service Cloud, and integrations.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}
