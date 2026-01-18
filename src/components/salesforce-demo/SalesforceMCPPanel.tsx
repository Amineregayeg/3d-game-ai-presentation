"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Cloud,
  Terminal,
  CheckCircle,
  XCircle,
  Loader2,
  ChevronDown,
  Database,
  Code,
  Clock,
  Wifi,
  WifiOff,
  Search,
  PenTool,
  Trash,
  FileCode,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SalesforceOperation, SalesforceConnectionStatus } from "./types";

interface SalesforceMCPPanelProps {
  connectionStatus: SalesforceConnectionStatus;
  operations: SalesforceOperation[];
}

export function SalesforceMCPPanel({
  connectionStatus,
  operations,
}: SalesforceMCPPanelProps) {
  const [expandedOperation, setExpandedOperation] = useState<string | null>(null);

  const getOperationIcon = (type: SalesforceOperation["type"]) => {
    switch (type) {
      case "query":
        return <Search className="w-3 h-3" />;
      case "insert":
      case "update":
        return <PenTool className="w-3 h-3" />;
      case "delete":
        return <Trash className="w-3 h-3" />;
      case "describe":
        return <Database className="w-3 h-3" />;
      case "apex":
        return <FileCode className="w-3 h-3" />;
      case "search":
        return <Search className="w-3 h-3" />;
      default:
        return <Code className="w-3 h-3" />;
    }
  };

  const getStatusIcon = (status: SalesforceOperation["status"]) => {
    switch (status) {
      case "pending":
        return <div className="w-3 h-3 rounded-full bg-slate-500" />;
      case "running":
        return <Loader2 className="w-3 h-3 text-[#0176D3] animate-spin" />;
      case "success":
        return <CheckCircle className="w-3 h-3 text-emerald-400" />;
      case "error":
        return <XCircle className="w-3 h-3 text-red-400" />;
    }
  };

  const getStatusColor = (status: SalesforceOperation["status"]) => {
    switch (status) {
      case "pending":
        return "border-slate-500/50 text-slate-400";
      case "running":
        return "border-[#0176D3]/50 text-[#0176D3]";
      case "success":
        return "border-emerald-500/50 text-emerald-400";
      case "error":
        return "border-red-500/50 text-red-400";
    }
  };

  const getTypeColor = (type: SalesforceOperation["type"]) => {
    switch (type) {
      case "query":
        return "bg-purple-500/20 text-purple-400";
      case "insert":
        return "bg-emerald-500/20 text-emerald-400";
      case "update":
        return "bg-amber-500/20 text-amber-400";
      case "delete":
        return "bg-red-500/20 text-red-400";
      case "describe":
        return "bg-cyan-500/20 text-cyan-400";
      case "apex":
        return "bg-orange-500/20 text-orange-400";
      case "search":
        return "bg-blue-500/20 text-blue-400";
      default:
        return "bg-slate-500/20 text-slate-400";
    }
  };

  return (
    <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-base text-white flex items-center gap-2">
          <Cloud className="w-4 h-4 text-[#0176D3]" />
          Salesforce MCP
          <div className="ml-auto flex items-center gap-2">
            {connectionStatus.connecting ? (
              <Badge
                variant="outline"
                className="text-[10px] border-[#0176D3]/50 text-[#0176D3]"
              >
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                Connecting
              </Badge>
            ) : connectionStatus.connected ? (
              <Badge
                variant="outline"
                className="text-[10px] border-emerald-500/50 text-emerald-400"
              >
                <Wifi className="w-3 h-3 mr-1" />
                Connected
              </Badge>
            ) : (
              <Badge
                variant="outline"
                className="text-[10px] border-slate-500/50 text-slate-400"
              >
                <WifiOff className="w-3 h-3 mr-1" />
                Disconnected
              </Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Org Info */}
        {connectionStatus.connected && connectionStatus.org && (
          <div className="bg-slate-900/50 rounded-lg p-2 border border-[#0176D3]/20">
            <div className="flex items-center gap-2 text-xs">
              <Cloud className="w-3 h-3 text-[#0176D3]" />
              <span className="text-white font-medium">
                {connectionStatus.org.name}
              </span>
              <Badge
                variant="outline"
                className="text-[9px] px-1 py-0 border-slate-600"
              >
                {connectionStatus.org.orgType}
              </Badge>
            </div>
          </div>
        )}

        {/* Operations Log */}
        <div className="space-y-2">
          {operations.length === 0 ? (
            <div className="text-center py-6 text-slate-500 text-sm">
              <Terminal className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No operations executed yet</p>
              <p className="text-xs mt-1">
                Operations will appear here as Alex works
              </p>
            </div>
          ) : (
            <AnimatePresence>
              {operations.map((op, index) => (
                <motion.div
                  key={op.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className={`rounded-lg border overflow-hidden ${getStatusColor(
                    op.status
                  )}`}
                >
                  {/* Operation Header */}
                  <div
                    className="flex items-center gap-2 p-2 cursor-pointer hover:bg-white/5"
                    onClick={() =>
                      setExpandedOperation(
                        expandedOperation === op.id ? null : op.id
                      )
                    }
                  >
                    {getStatusIcon(op.status)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <Badge
                          className={`text-[9px] px-1.5 py-0 ${getTypeColor(
                            op.type
                          )}`}
                        >
                          {getOperationIcon(op.type)}
                          <span className="ml-1 uppercase">{op.type}</span>
                        </Badge>
                        <span className="text-xs text-white truncate">
                          {op.description}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 mt-0.5">
                        {op.recordCount !== undefined && (
                          <span className="text-[10px] text-slate-500">
                            {op.recordCount} records
                          </span>
                        )}
                        {op.duration_ms && (
                          <span className="text-[10px] text-slate-500 flex items-center gap-1">
                            <Clock className="w-2.5 h-2.5" />
                            {op.duration_ms}ms
                          </span>
                        )}
                      </div>
                    </div>
                    <motion.div
                      animate={{
                        rotate: expandedOperation === op.id ? 180 : 0,
                      }}
                      transition={{ duration: 0.2 }}
                    >
                      <ChevronDown className="w-3 h-3 text-slate-400" />
                    </motion.div>
                  </div>

                  {/* Expanded Content */}
                  <AnimatePresence>
                    {expandedOperation === op.id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-t border-white/10"
                      >
                        <div className="p-2 space-y-2">
                          {/* SOQL Query */}
                          {op.soql !== undefined && op.soql !== null && op.soql !== "" && (
                            <div className="bg-slate-900/50 rounded p-2">
                              <div className="flex items-center gap-1 text-[10px] text-slate-500 mb-1">
                                <Database className="w-3 h-3" />
                                SOQL Query
                              </div>
                              <pre className="text-[10px] text-cyan-400 overflow-x-auto whitespace-pre-wrap font-mono">
                                {op.soql}
                              </pre>
                            </div>
                          )}

                          {/* Result */}
                          {(() => {
                            if (op.result === undefined || op.result === null) return null;
                            const resultStr = typeof op.result === "string"
                              ? op.result
                              : JSON.stringify(op.result, null, 2);
                            return (
                              <div className="bg-emerald-500/10 rounded p-2">
                                <div className="text-[10px] text-emerald-400 mb-1">
                                  Result
                                </div>
                                <pre className="text-[10px] text-slate-300 overflow-x-auto max-h-32">
                                  {resultStr}
                                </pre>
                              </div>
                            );
                          })()}

                          {/* Error */}
                          {op.error !== undefined && op.error !== null && op.error !== "" && (
                            <div className="bg-red-500/10 rounded p-2">
                              <div className="text-[10px] text-red-400 mb-1">
                                Error
                              </div>
                              <pre className="text-[10px] text-red-300 overflow-x-auto">
                                {op.error}
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
