"use client";

import { useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageCircle,
  User,
  Bot,
  Clock,
  CheckCircle,
  Loader2,
  Terminal,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ConversationMessage, ToolCall } from "./types";

interface ConversationPanelProps {
  messages: ConversationMessage[];
  isProcessing: boolean;
}

function ToolCallDisplay({ toolCall }: { toolCall: ToolCall }) {
  return (
    <div className="bg-slate-900/50 rounded p-2 mt-2 border border-slate-700/50">
      <div className="flex items-center gap-2 text-xs">
        <Terminal className="w-3 h-3 text-[#0176D3]" />
        <span className="text-slate-400">Tool:</span>
        <Badge variant="outline" className="text-[9px] border-[#0176D3]/50 text-[#0176D3]">
          {toolCall.name}
        </Badge>
        {toolCall.status === "running" && (
          <Loader2 className="w-3 h-3 text-[#0176D3] animate-spin ml-auto" />
        )}
        {toolCall.status === "success" && (
          <CheckCircle className="w-3 h-3 text-emerald-400 ml-auto" />
        )}
      </div>
      {(() => {
        if (toolCall.output === undefined || toolCall.output === null) return null;
        const outputStr = typeof toolCall.output === "string"
          ? toolCall.output
          : JSON.stringify(toolCall.output, null, 2).slice(0, 200);
        return (
          <pre className="text-[10px] text-slate-400 mt-1 overflow-x-auto">
            {outputStr}
          </pre>
        );
      })()}
    </div>
  );
}

function MessageBubble({ message }: { message: ConversationMessage }) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  if (isSystem) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex justify-center my-2"
      >
        <Badge variant="outline" className="text-[10px] border-slate-600 text-slate-400">
          {message.content}
        </Badge>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex gap-2 ${isUser ? "flex-row-reverse" : ""}`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser
            ? "bg-emerald-500/20 text-emerald-400"
            : "bg-[#0176D3]/20 text-[#0176D3]"
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Message Content */}
      <div
        className={`flex-1 max-w-[80%] ${isUser ? "text-right" : "text-left"}`}
      >
        <div
          className={`inline-block rounded-lg px-3 py-2 ${
            isUser
              ? "bg-emerald-500/20 text-white"
              : "bg-slate-700/50 text-slate-200"
          }`}
        >
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>

          {/* Tool Calls */}
          {message.toolCalls && message.toolCalls.length > 0 && (
            <div className="mt-2 space-y-1">
              {message.toolCalls.map((tc) => (
                <ToolCallDisplay key={tc.id} toolCall={tc} />
              ))}
            </div>
          )}
        </div>

        {/* Timestamp */}
        <div
          className={`text-[10px] text-slate-500 mt-1 flex items-center gap-1 ${
            isUser ? "justify-end" : "justify-start"
          }`}
        >
          <Clock className="w-2.5 h-2.5" />
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </div>
      </div>
    </motion.div>
  );
}

export function ConversationPanel({
  messages,
  isProcessing,
}: ConversationPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-base text-white flex items-center gap-2">
          <MessageCircle className="w-4 h-4 text-[#0176D3]" />
          Conversation
          {isProcessing && (
            <Badge
              variant="outline"
              className="ml-auto text-[10px] border-[#0176D3]/50 text-[#0176D3] animate-pulse"
            >
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
              Processing
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div
          ref={scrollRef}
          className="h-64 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent"
        >
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-500">
              <MessageCircle className="w-8 h-8 mb-2 opacity-50" />
              <p className="text-sm">No messages yet</p>
              <p className="text-xs mt-1">Start a conversation with Alex</p>
            </div>
          ) : (
            <AnimatePresence>
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </AnimatePresence>
          )}

          {/* Typing Indicator */}
          {isProcessing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-2"
            >
              <div className="w-8 h-8 rounded-full bg-[#0176D3]/20 flex items-center justify-center">
                <Bot className="w-4 h-4 text-[#0176D3]" />
              </div>
              <div className="bg-slate-700/50 rounded-lg px-4 py-2">
                <div className="flex gap-1">
                  <motion.div
                    animate={{ opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0 }}
                    className="w-2 h-2 rounded-full bg-[#0176D3]"
                  />
                  <motion.div
                    animate={{ opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                    className="w-2 h-2 rounded-full bg-[#0176D3]"
                  />
                  <motion.div
                    animate={{ opacity: [0.4, 1, 0.4] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                    className="w-2 h-2 rounded-full bg-[#0176D3]"
                  />
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
