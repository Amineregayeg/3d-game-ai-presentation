"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Send,
  Keyboard,
  PhoneCall,
  PhoneOff,
  Loader2,
  Volume2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ConversationalInputProps {
  isActive: boolean;
  isConnecting?: boolean;
  isUserSpeaking?: boolean;
  isAgentSpeaking?: boolean;
  onStart: () => void;
  onEnd: () => void;
  onTextSubmit: (text: string) => void;
  audioLevel: number;
  currentTranscript: string;
  disabled?: boolean;
}

export function ConversationalInput({
  isActive,
  isConnecting = false,
  isUserSpeaking = false,
  isAgentSpeaking = false,
  onStart,
  onEnd,
  onTextSubmit,
  audioLevel,
  currentTranscript,
  disabled = false,
}: ConversationalInputProps) {
  const [textInput, setTextInput] = useState("");

  // Derive conversation state from props
  const conversationState = {
    status: isConnecting ? "connecting" : isActive ? "connected" : "idle",
    isUserSpeaking: isUserSpeaking || (isActive && audioLevel > 0.1),
    isAgentSpeaking: isAgentSpeaking,
    transcript: currentTranscript,
    agentText: "",
  };

  const handleTextSubmit = () => {
    if (textInput.trim()) {
      onTextSubmit(textInput.trim());
      setTextInput("");
    }
  };

  const getStatusText = () => {
    switch (conversationState.status) {
      case "idle":
        return "Start a conversation with your Salesforce consultant";
      case "connecting":
        return "Connecting to consultant...";
      case "connected":
        return "Connected - Ready to help";
      case "listening":
        return "Listening...";
      case "speaking":
        return "Alex is speaking...";
      case "processing":
        return "Thinking...";
      case "error":
        return "Connection error - Try again";
      default:
        return "";
    }
  };

  const getStatusColor = () => {
    switch (conversationState.status) {
      case "connected":
      case "listening":
        return "text-emerald-400";
      case "speaking":
        return "text-[#0176D3]";
      case "processing":
        return "text-amber-400";
      case "error":
        return "text-red-400";
      default:
        return "text-slate-400";
    }
  };

  const isConversationActive = isActive;

  return (
    <div className="bg-slate-800/50 border border-white/10 rounded-xl overflow-hidden backdrop-blur-sm">
      <Tabs defaultValue="voice" className="w-full">
        <TabsList className="w-full grid grid-cols-2 bg-slate-900/50 rounded-none border-b border-white/10">
          <TabsTrigger
            value="voice"
            className="gap-2 data-[state=active]:bg-[#0176D3]/20 data-[state=active]:text-[#0176D3]"
          >
            <PhoneCall className="w-4 h-4" />
            Voice
          </TabsTrigger>
          <TabsTrigger
            value="text"
            className="gap-2 data-[state=active]:bg-[#032D60]/20 data-[state=active]:text-[#0176D3]"
          >
            <Keyboard className="w-4 h-4" />
            Text
          </TabsTrigger>
        </TabsList>

        <TabsContent value="voice" className="p-6 m-0">
          <div className="flex flex-col items-center">
            {/* Conversation Button with Animation */}
            <div className="relative">
              {/* Pulse rings when active */}
              <AnimatePresence>
                {isConversationActive && (
                  <>
                    <motion.div
                      initial={{ scale: 1, opacity: 0.5 }}
                      animate={{ scale: 1.5, opacity: 0 }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="absolute inset-0 rounded-full bg-[#0176D3]"
                    />
                    <motion.div
                      initial={{ scale: 1, opacity: 0.3 }}
                      animate={{ scale: 1.8, opacity: 0 }}
                      transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                      className="absolute inset-0 rounded-full bg-[#0176D3]"
                    />
                  </>
                )}
              </AnimatePresence>

              {/* Audio level ring */}
              {conversationState.isUserSpeaking && (
                <motion.div
                  animate={{ scale: 1 + audioLevel * 0.4 }}
                  className="absolute inset-0 rounded-full bg-emerald-500/30"
                />
              )}

              {/* Agent speaking indicator */}
              {conversationState.isAgentSpeaking && (
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 0.5, repeat: Infinity }}
                  className="absolute inset-0 rounded-full bg-[#0176D3]/30"
                />
              )}

              <Button
                size="lg"
                onClick={
                  isConversationActive
                    ? onEnd
                    : onStart
                }
                disabled={disabled || conversationState.status === "connecting"}
                className={`relative w-28 h-28 rounded-full transition-all z-10 ${
                  isConversationActive
                    ? conversationState.isUserSpeaking
                      ? "bg-emerald-500 hover:bg-emerald-600"
                      : conversationState.isAgentSpeaking
                      ? "bg-[#0176D3] hover:bg-[#0176D3]/90"
                      : "bg-[#032D60] hover:bg-[#032D60]/90"
                    : "bg-gradient-to-br from-[#0176D3] to-[#032D60] hover:from-[#0176D3]/90 hover:to-[#032D60]/90"
                }`}
              >
                {conversationState.status === "connecting" ? (
                  <Loader2 className="w-10 h-10 animate-spin" />
                ) : isConversationActive ? (
                  conversationState.isUserSpeaking ? (
                    <Mic className="w-10 h-10" />
                  ) : conversationState.isAgentSpeaking ? (
                    <Volume2 className="w-10 h-10" />
                  ) : (
                    <PhoneOff className="w-10 h-10" />
                  )
                ) : (
                  <PhoneCall className="w-10 h-10" />
                )}
              </Button>
            </div>

            {/* Status Text */}
            <div className="mt-4 text-center">
              <p className={`font-medium ${getStatusColor()}`}>
                {getStatusText()}
              </p>

              {/* Live transcript */}
              {conversationState.transcript && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-3 bg-slate-900/50 rounded-lg p-3 max-w-xs"
                >
                  <p className="text-xs text-slate-500 mb-1">You said:</p>
                  <p className="text-white text-sm">
                    {conversationState.transcript}
                  </p>
                </motion.div>
              )}

              {/* Instructions */}
              {!isConversationActive && (
                <div className="mt-3 space-y-1">
                  <p className="text-xs text-slate-500">
                    Click to start a real-time conversation
                  </p>
                  <p className="text-xs text-slate-600">
                    Alex will help with your Salesforce questions
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
              placeholder="Ask Alex anything about Salesforce... e.g., 'How do I create a custom field on Account?'"
              disabled={disabled}
              className="w-full h-32 px-4 py-3 bg-slate-700/50 border border-white/10 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0176D3]/50 resize-none"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleTextSubmit();
                }
              }}
            />
            <Button
              onClick={handleTextSubmit}
              disabled={!textInput.trim() || disabled}
              className="w-full bg-gradient-to-r from-[#0176D3] to-[#032D60] hover:from-[#0176D3]/90 hover:to-[#032D60]/90"
            >
              <Send className="w-4 h-4 mr-2" />
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
