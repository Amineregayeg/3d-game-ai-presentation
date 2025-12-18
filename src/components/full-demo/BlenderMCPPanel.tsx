"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Box,
  Terminal,
  CheckCircle,
  XCircle,
  Loader2,
  ChevronDown,
  Code,
  Clock,
  Wifi,
  WifiOff,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export interface BlenderCommand {
  id: string;
  type: string;
  code?: string;
  params?: Record<string, unknown>;
  status: "pending" | "running" | "success" | "error";
  result?: string;
  error?: string;
  duration_ms?: number;
  timestamp: number;
}

interface BlenderMCPPanelProps {
  isConnected: boolean;
  isConnecting: boolean;
  commands: BlenderCommand[];
  onConnect?: () => void;
  onDisconnect?: () => void;
  screenshotUrl?: string;
}

export function BlenderMCPPanel({
  isConnected,
  isConnecting,
  commands,
  onConnect,
  onDisconnect,
  screenshotUrl,
}: BlenderMCPPanelProps) {
  const [expandedCommand, setExpandedCommand] = useState<string | null>(null);

  const getStatusIcon = (status: BlenderCommand["status"]) => {
    switch (status) {
      case "pending":
        return <div className="w-3 h-3 rounded-full bg-slate-500" />;
      case "running":
        return <Loader2 className="w-3 h-3 text-cyan-400 animate-spin" />;
      case "success":
        return <CheckCircle className="w-3 h-3 text-emerald-400" />;
      case "error":
        return <XCircle className="w-3 h-3 text-red-400" />;
    }
  };

  const getStatusColor = (status: BlenderCommand["status"]) => {
    switch (status) {
      case "pending":
        return "border-slate-500/50 text-slate-400";
      case "running":
        return "border-cyan-500/50 text-cyan-400";
      case "success":
        return "border-emerald-500/50 text-emerald-400";
      case "error":
        return "border-red-500/50 text-red-400";
    }
  };

  return (
    <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-base text-white flex items-center gap-2">
          <Box className="w-4 h-4 text-orange-400" />
          Blender MCP
          <div className="ml-auto flex items-center gap-2">
            {isConnecting ? (
              <Badge variant="outline" className="text-[10px] border-cyan-500/50 text-cyan-400">
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                Connecting
              </Badge>
            ) : isConnected ? (
              <Badge variant="outline" className="text-[10px] border-emerald-500/50 text-emerald-400">
                <Wifi className="w-3 h-3 mr-1" />
                Connected
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px] border-slate-500/50 text-slate-400">
                <WifiOff className="w-3 h-3 mr-1" />
                Disconnected
              </Badge>
            )}
            {!isConnected && onConnect && (
              <Button
                variant="outline"
                size="sm"
                onClick={onConnect}
                disabled={isConnecting}
                className="h-6 text-xs"
              >
                Connect
              </Button>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Screenshot Preview */}
        {screenshotUrl && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative rounded-lg overflow-hidden border border-white/10"
          >
            <img
              src={screenshotUrl}
              alt="Blender viewport"
              className="w-full h-auto"
            />
            <div className="absolute bottom-2 right-2">
              <Badge className="bg-black/50 text-white text-[10px]">
                Viewport Screenshot
              </Badge>
            </div>
          </motion.div>
        )}

        {/* Command Log */}
        <div className="space-y-2">
          {commands.length === 0 ? (
            <div className="text-center py-6 text-slate-500 text-sm">
              <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No commands executed yet</p>
              <p className="text-xs mt-1">Commands will appear here during 3D generation</p>
            </div>
          ) : (
            <AnimatePresence>
              {commands.map((cmd, index) => (
                <motion.div
                  key={cmd.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`rounded-lg border overflow-hidden ${getStatusColor(cmd.status)}`}
                >
                  {/* Command Header */}
                  <div
                    className="flex items-center gap-2 p-2 cursor-pointer hover:bg-white/5"
                    onClick={() =>
                      setExpandedCommand(expandedCommand === cmd.id ? null : cmd.id)
                    }
                  >
                    {getStatusIcon(cmd.status)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-white truncate">
                          {cmd.type}
                        </span>
                        {cmd.duration_ms && (
                          <span className="text-[10px] text-slate-500 flex items-center gap-1">
                            <Clock className="w-2.5 h-2.5" />
                            {cmd.duration_ms}ms
                          </span>
                        )}
                      </div>
                    </div>
                    <motion.div
                      animate={{ rotate: expandedCommand === cmd.id ? 180 : 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <ChevronDown className="w-3 h-3 text-slate-400" />
                    </motion.div>
                  </div>

                  {/* Expanded Content */}
                  <AnimatePresence>
                    {expandedCommand === cmd.id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-t border-white/10"
                      >
                        <div className="p-2 space-y-2">
                          {/* Code */}
                          {cmd.code && (
                            <div className="bg-slate-900/50 rounded p-2">
                              <div className="flex items-center gap-1 text-[10px] text-slate-500 mb-1">
                                <Code className="w-3 h-3" />
                                Python Code
                              </div>
                              <pre className="text-[10px] text-slate-300 overflow-x-auto whitespace-pre-wrap">
                                {cmd.code}
                              </pre>
                            </div>
                          )}

                          {/* Result */}
                          {cmd.result && (
                            <div className="bg-emerald-500/10 rounded p-2">
                              <div className="text-[10px] text-emerald-400 mb-1">Result</div>
                              <pre className="text-[10px] text-slate-300 overflow-x-auto">
                                {cmd.result}
                              </pre>
                            </div>
                          )}

                          {/* Error */}
                          {cmd.error && (
                            <div className="bg-red-500/10 rounded p-2">
                              <div className="text-[10px] text-red-400 mb-1">Error</div>
                              <pre className="text-[10px] text-red-300 overflow-x-auto">
                                {cmd.error}
                              </pre>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
