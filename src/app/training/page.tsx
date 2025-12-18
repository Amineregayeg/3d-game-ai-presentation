"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { PresentationDock } from "@/components/presentation-dock";
import { dockItems } from "@/lib/dock-items";
import {
  Cpu,
  Zap,
  TrendingDown,
  Clock,
  Server,
  Activity,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  RefreshCw,
  Gauge,
  BarChart3,
  Target,
  Flame,
  Database,
  Shield,
  TrendingUp,
} from "lucide-react";

type TrainingStatus = "running" | "completed" | "error" | "idle" | "initializing" | "stopped";

interface TrainingMetrics {
  epoch: number;
  step: number;
  totalSteps: number;
  totalEpochs: number;
  loss: number;
  ctcLoss: number;
  ceLoss: number;
  wer: number | null;
  learningRate: string;
  speed: number;
  gpuMemory: number;
  gpuUtil: number;
  eta: string;
  status: TrainingStatus;
  lastUpdate: string;
  stage: string;
  backup: { last_backup: string | null; status: string } | null;
}

interface EpochHistory {
  epoch: number;
  avgLoss: number;
  ctcLoss: number;
  ceLoss: number;
  wer: number | null;
  time: string;
}

interface BackupEntry {
  time: string;
  type: string;
  status: "success" | "failed";
}

interface WERHistoryEntry {
  step: number;
  wer: number;
  timestamp: string;
}

export default function TrainingDashboard() {
  const [metrics, setMetrics] = useState<TrainingMetrics>({
    epoch: 0,
    step: 0,
    totalSteps: 3568,
    totalEpochs: 5,
    loss: 0,
    ctcLoss: 0,
    ceLoss: 0,
    wer: null,
    learningRate: "0",
    speed: 0,
    gpuMemory: 0,
    gpuUtil: 0,
    eta: "Calculating...",
    status: "idle",
    lastUpdate: new Date().toISOString(),
    stage: "stage4",
    backup: null,
  });

  const [history, setHistory] = useState<EpochHistory[]>([]);
  const [lossHistory, setLossHistory] = useState<{ step: number; loss: number }[]>([]);
  const [backupHistory, setBackupHistory] = useState<BackupEntry[]>([]);
  const [werHistory, setWerHistory] = useState<WERHistoryEntry[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<"loss" | "ctc" | "ce">("loss");

  const fetchMetrics = useCallback(async () => {
    try {
      // Use basePath for API route
      const res = await fetch("/3dgameassistant/api/training-status");
      if (!res.ok) throw new Error("Failed to fetch");
      const data = await res.json();

      setMetrics(data.metrics);
      if (data.history) setHistory(data.history);
      if (data.lossHistory) setLossHistory(data.lossHistory);
      if (data.backupHistory) setBackupHistory(data.backupHistory);
      if (data.werHistory) setWerHistory(data.werHistory);
      setError(null);
    } catch (err) {
      setError("Failed to fetch training status");
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    if (autoRefresh) {
      const interval = setInterval(fetchMetrics, 5000);
      return () => clearInterval(interval);
    }
  }, [fetchMetrics, autoRefresh]);

  const epochProgress = (metrics.step / metrics.totalSteps) * 100;
  const totalProgress = ((metrics.epoch * metrics.totalSteps + metrics.step) / (metrics.totalEpochs * metrics.totalSteps)) * 100;
  const maxLoss = Math.max(10, ...lossHistory.map(h => h.loss));

  // Calculate stats
  const startLoss = 5.0; // Stage 4 starts from Stage 2 checkpoint (~5.0 loss)
  const lossReduction = startLoss > 0 ? ((startLoss - metrics.loss) / startLoss * 100) : 0;
  const epochsCompleted = history.length;
  const epochsRemaining = metrics.totalEpochs - metrics.epoch;

  // Animated ring progress component
  const CircularProgress = ({ value, max, size = 120, strokeWidth = 8, color, label, sublabel }: {
    value: number;
    max: number;
    size?: number;
    strokeWidth?: number;
    color: string;
    label: string;
    sublabel?: string;
  }) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const progress = Math.min(value / max, 1);
    const offset = circumference - progress * circumference;

    return (
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="transform -rotate-90" width={size} height={size}>
          <circle
            className="text-slate-700"
            strokeWidth={strokeWidth}
            stroke="currentColor"
            fill="transparent"
            r={radius}
            cx={size / 2}
            cy={size / 2}
          />
          <circle
            className={`transition-all duration-1000 ease-out ${color}`}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            stroke="currentColor"
            fill="transparent"
            r={radius}
            cx={size / 2}
            cy={size / 2}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold text-white">{label}</span>
          {sublabel && <span className="text-xs text-slate-400">{sublabel}</span>}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white p-4 md:p-6">
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />
      </div>

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              VoxFormer {metrics.stage ? metrics.stage.replace('stage', 'Stage ') : 'Training'} Dashboard
            </h1>
            <p className="text-slate-400 mt-1">
              {metrics.stage === 'stage4' ? 'CTC-Only Recovery Training' : 'Live training metrics'} from RTX 4090 GPU
            </p>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
                autoRefresh
                  ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 shadow-lg shadow-cyan-500/20"
                  : "bg-slate-800 text-slate-400 border border-slate-700"
              }`}
            >
              <RefreshCw className={`h-4 w-4 ${autoRefresh ? "animate-spin" : ""}`} />
              {autoRefresh ? "Live" : "Paused"}
            </button>
            <div className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              metrics.status === "running"
                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                : metrics.status === "completed"
                ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30"
                : metrics.status === "error"
                ? "bg-red-500/20 text-red-400 border border-red-500/30"
                : "bg-slate-800 text-slate-400 border border-slate-700"
            }`}>
              {metrics.status === "running" ? (
                <>
                  <Activity className="h-4 w-4 animate-pulse" />
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  Training
                </>
              ) : metrics.status === "completed" ? (
                <>
                  <CheckCircle2 className="h-4 w-4" />
                  Completed
                </>
              ) : metrics.status === "error" ? (
                <>
                  <AlertCircle className="h-4 w-4" />
                  Error
                </>
              ) : (
                <>
                  <Clock className="h-4 w-4" />
                  Idle
                </>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6 text-red-400 flex items-center gap-2">
            <AlertCircle className="h-5 w-5" />
            {error}
          </div>
        )}

        {/* Key Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <Card className="bg-gradient-to-br from-cyan-500/20 to-cyan-500/5 border-cyan-500/30 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-cyan-400 text-sm font-medium">Current Loss</p>
                  <p className="text-3xl font-bold text-white mt-1">{metrics.loss.toFixed(2)}</p>
                </div>
                <div className="h-12 w-12 rounded-full bg-cyan-500/20 flex items-center justify-center">
                  <TrendingDown className="h-6 w-6 text-cyan-400" />
                </div>
              </div>
              <div className="mt-2 flex items-center gap-1 text-xs">
                <span className="text-emerald-400">↓ {lossReduction.toFixed(0)}%</span>
                <span className="text-slate-500">from start</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500/20 to-purple-500/5 border-purple-500/30 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-400 text-sm font-medium">Epoch</p>
                  <p className="text-3xl font-bold text-white mt-1">{metrics.epoch}<span className="text-lg text-slate-500">/{metrics.totalEpochs}</span></p>
                </div>
                <div className="h-12 w-12 rounded-full bg-purple-500/20 flex items-center justify-center">
                  <BarChart3 className="h-6 w-6 text-purple-400" />
                </div>
              </div>
              <div className="mt-2">
                <Progress value={totalProgress} className="h-1.5" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-emerald-500/20 to-emerald-500/5 border-emerald-500/30 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-emerald-400 text-sm font-medium">Speed</p>
                  <p className="text-3xl font-bold text-white mt-1">{metrics.speed.toFixed(1)}<span className="text-lg text-slate-500">it/s</span></p>
                </div>
                <div className="h-12 w-12 rounded-full bg-emerald-500/20 flex items-center justify-center">
                  <Zap className="h-6 w-6 text-emerald-400" />
                </div>
              </div>
              <div className="mt-2 text-xs text-slate-500">
                ~{Math.round(metrics.totalSteps / (metrics.speed || 1) / 60)} min/epoch
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-orange-500/20 to-orange-500/5 border-orange-500/30 backdrop-blur-sm">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-orange-400 text-sm font-medium">ETA</p>
                  <p className="text-2xl font-bold text-white mt-1">{metrics.eta}</p>
                </div>
                <div className="h-12 w-12 rounded-full bg-orange-500/20 flex items-center justify-center">
                  <Clock className="h-6 w-6 text-orange-400" />
                </div>
              </div>
              <div className="mt-2 text-xs text-slate-500">
                {epochsRemaining} epochs remaining
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column */}
          <div className="col-span-12 lg:col-span-4 space-y-4">
            {/* Circular Progress Cards */}
            <Card className="bg-slate-800/60 border-slate-700/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2 text-white">
                  <Gauge className="h-5 w-5 text-cyan-400" />
                  Progress Rings
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex justify-around items-center py-4">
                  <CircularProgress
                    value={metrics.epoch}
                    max={metrics.totalEpochs}
                    color="text-cyan-400"
                    label={`${metrics.epoch}`}
                    sublabel="Epoch"
                  />
                  <CircularProgress
                    value={epochProgress}
                    max={100}
                    color="text-purple-400"
                    label={`${epochProgress.toFixed(0)}%`}
                    sublabel="Step"
                  />
                  <CircularProgress
                    value={metrics.gpuUtil}
                    max={100}
                    color="text-emerald-400"
                    label={`${metrics.gpuUtil}%`}
                    sublabel="GPU"
                  />
                </div>
              </CardContent>
            </Card>

            {/* GPU Status */}
            <Card className="bg-slate-800/60 border-slate-700/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2 text-white">
                  <Server className="h-5 w-5 text-cyan-400" />
                  GPU Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400 flex items-center gap-2">
                      <Database className="h-4 w-4" />
                      VRAM Usage
                    </span>
                    <span className="text-white font-mono">{metrics.gpuMemory.toFixed(1)} / 24 GB</span>
                  </div>
                  <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-500"
                      style={{ width: `${(metrics.gpuMemory / 24) * 100}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400 flex items-center gap-2">
                      <Cpu className="h-4 w-4" />
                      GPU Utilization
                    </span>
                    <span className="text-white font-mono">{metrics.gpuUtil}%</span>
                  </div>
                  <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-500 to-green-400 transition-all duration-500"
                      style={{ width: `${metrics.gpuUtil}%` }}
                    />
                  </div>
                </div>
                <div className="pt-2 border-t border-slate-700">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-slate-500">Device</span>
                      <p className="text-white font-mono">RTX 4090</p>
                    </div>
                    <div>
                      <span className="text-slate-500">Provider</span>
                      <p className="text-white font-mono">Vast.ai</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Loss Components */}
            <Card className="bg-slate-800/60 border-slate-700/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2 text-white">
                  <Target className="h-5 w-5 text-purple-400" />
                  Loss Breakdown
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="relative h-4 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="absolute left-0 h-full bg-purple-500 transition-all duration-500"
                    style={{ width: `${(metrics.ctcLoss / (metrics.ctcLoss + metrics.ceLoss || 1)) * 100}%` }}
                  />
                  <div
                    className="absolute right-0 h-full bg-emerald-500 transition-all duration-500"
                    style={{ width: `${(metrics.ceLoss / (metrics.ctcLoss + metrics.ceLoss || 1)) * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-purple-500" />
                    <span className="text-slate-400">CTC</span>
                    <span className="text-purple-400 font-mono">{metrics.ctcLoss.toFixed(3)}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-emerald-500" />
                    <span className="text-slate-400">CE</span>
                    <span className="text-emerald-400 font-mono">{metrics.ceLoss.toFixed(3)}</span>
                  </div>
                </div>
                <div className="pt-2 border-t border-slate-700">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Learning Rate</span>
                    <span className="text-yellow-400 font-mono">{metrics.learningRate}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Backup Status */}
            <Card className={`bg-slate-800/60 backdrop-blur-sm ${metrics.backup?.status === 'success' ? 'border-emerald-500/30' : 'border-yellow-500/30'}`}>
              <CardHeader className="pb-2">
                <CardTitle className={`text-lg flex items-center gap-2 ${metrics.backup?.status === 'success' ? 'text-emerald-400' : 'text-yellow-400'}`}>
                  <Shield className="h-5 w-5" />
                  Backup Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-3">
                  <div className={`h-10 w-10 rounded-full flex items-center justify-center ${metrics.backup?.status === 'success' ? 'bg-emerald-500/20' : 'bg-yellow-500/20'}`}>
                    {metrics.backup?.status === 'success' ? (
                      <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                    ) : (
                      <Clock className="h-5 w-5 text-yellow-400" />
                    )}
                  </div>
                  <div>
                    <p className="text-white font-medium">
                      {metrics.backup?.status === 'success' ? 'Auto-backup Active' : 'Backup Pending'}
                    </p>
                    <p className="text-sm text-slate-400">
                      {metrics.backup?.last_backup
                        ? `Last: ${new Date(metrics.backup.last_backup).toLocaleTimeString()}`
                        : 'Syncing to VPS every 5 min'}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Backup History Table */}
            <Card className="bg-slate-800/60 border-slate-700/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2 text-white">
                  <Database className="h-5 w-5 text-cyan-400" />
                  Backup History
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-y-auto max-h-48">
                  {backupHistory.length > 0 ? (
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 bg-slate-800/95">
                        <tr className="border-b border-slate-700">
                          <th className="text-left py-1.5 px-1 text-slate-400 font-medium">Time</th>
                          <th className="text-left py-1.5 px-1 text-slate-400 font-medium">Type</th>
                          <th className="text-right py-1.5 px-1 text-slate-400 font-medium">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {backupHistory.slice().reverse().slice(0, 20).map((entry, idx) => (
                          <tr key={idx} className="border-b border-slate-800/50 hover:bg-slate-700/30">
                            <td className="py-1.5 px-1 font-mono text-slate-300">
                              {entry.time.split(' ')[1] || entry.time}
                            </td>
                            <td className="py-1.5 px-1">
                              <span className={`px-1.5 py-0.5 rounded text-xs ${
                                entry.type === 'checkpoint' ? 'bg-purple-500/20 text-purple-400' :
                                entry.type === 'best_checkpoint' ? 'bg-emerald-500/20 text-emerald-400' :
                                entry.type === 'metrics' ? 'bg-cyan-500/20 text-cyan-400' :
                                entry.type === 'complete' ? 'bg-green-500/20 text-green-400' :
                                entry.type === 'start' ? 'bg-yellow-500/20 text-yellow-400' :
                                'bg-slate-600/50 text-slate-400'
                              }`}>
                                {entry.type}
                              </span>
                            </td>
                            <td className="py-1.5 px-1 text-right">
                              {entry.status === 'success' ? (
                                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 inline" />
                              ) : (
                                <AlertCircle className="h-3.5 w-3.5 text-red-400 inline" />
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="py-6 text-center text-slate-500">
                      <Database className="h-6 w-6 mx-auto mb-2 opacity-50" />
                      <p className="text-xs">No backup history yet</p>
                      <p className="text-xs text-slate-600 mt-1">Backups sync every 5 min</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* WER Tracking Card - PROMINENT */}
            <Card className={`backdrop-blur-sm ${
              metrics.wer === null ? 'bg-slate-800/60 border-slate-700/50' :
              metrics.wer > 100 ? 'bg-gradient-to-br from-red-500/30 to-red-500/5 border-red-500/50 ring-2 ring-red-500/30' :
              metrics.wer > 50 ? 'bg-gradient-to-br from-orange-500/20 to-orange-500/5 border-orange-500/30' :
              metrics.wer > 25 ? 'bg-gradient-to-br from-yellow-500/20 to-yellow-500/5 border-yellow-500/30' :
              metrics.wer > 15 ? 'bg-gradient-to-br from-cyan-500/20 to-cyan-500/5 border-cyan-500/30' :
              'bg-gradient-to-br from-emerald-500/20 to-emerald-500/5 border-emerald-500/30'
            }`}>
              <CardHeader className="pb-2">
                <CardTitle className={`text-lg flex items-center gap-2 ${
                  metrics.wer === null ? 'text-slate-400' :
                  metrics.wer > 100 ? 'text-red-400' :
                  metrics.wer > 50 ? 'text-orange-400' :
                  metrics.wer > 25 ? 'text-yellow-400' :
                  metrics.wer > 15 ? 'text-cyan-400' :
                  'text-emerald-400'
                }`}>
                  {metrics.wer !== null && metrics.wer > 100 ? (
                    <AlertTriangle className="h-5 w-5 animate-pulse" />
                  ) : (
                    <Target className="h-5 w-5" />
                  )}
                  Word Error Rate (WER)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-2">
                  <p className={`text-4xl font-bold ${
                    metrics.wer === null ? 'text-slate-500' :
                    metrics.wer > 100 ? 'text-red-400' :
                    metrics.wer > 50 ? 'text-orange-400' :
                    metrics.wer > 25 ? 'text-yellow-400' :
                    metrics.wer > 15 ? 'text-cyan-400' :
                    'text-emerald-400'
                  }`}>
                    {metrics.wer !== null ? `${metrics.wer.toFixed(1)}%` : 'Waiting...'}
                  </p>
                  <p className="text-sm text-slate-400 mt-1">
                    Target: &lt; 15% | Good: &lt; 25%
                  </p>

                  {/* WER Status indicator */}
                  {metrics.wer !== null && (
                    <div className={`mt-3 px-3 py-1.5 rounded-lg text-xs font-medium inline-flex items-center gap-2 ${
                      metrics.wer > 100 ? 'bg-red-500/20 text-red-400' :
                      metrics.wer > 50 ? 'bg-orange-500/20 text-orange-400' :
                      metrics.wer > 25 ? 'bg-yellow-500/20 text-yellow-400' :
                      metrics.wer > 15 ? 'bg-cyan-500/20 text-cyan-400' :
                      'bg-emerald-500/20 text-emerald-400'
                    }`}>
                      {metrics.wer > 100 ? (
                        <><AlertTriangle className="h-3 w-3" /> CRITICAL - Notify Claude!</>
                      ) : metrics.wer > 50 ? (
                        <><AlertCircle className="h-3 w-3" /> High - Monitor closely</>
                      ) : metrics.wer > 25 ? (
                        <><TrendingDown className="h-3 w-3" /> Improving - Keep training</>
                      ) : metrics.wer > 15 ? (
                        <><TrendingDown className="h-3 w-3" /> Good progress!</>
                      ) : (
                        <><CheckCircle2 className="h-3 w-3" /> Excellent!</>
                      )}
                    </div>
                  )}

                  {/* Progress bar to target */}
                  <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        metrics.wer === null ? 'bg-slate-600' :
                        metrics.wer > 100 ? 'bg-red-500' :
                        metrics.wer > 50 ? 'bg-orange-500' :
                        metrics.wer > 25 ? 'bg-yellow-500' :
                        metrics.wer > 15 ? 'bg-cyan-500' :
                        'bg-emerald-500'
                      }`}
                      style={{ width: `${metrics.wer !== null ? Math.max(5, Math.min(100, 100 - metrics.wer)) : 0}%` }}
                    />
                  </div>
                </div>

                {/* WER History mini chart */}
                {werHistory.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-700">
                    <p className="text-xs text-slate-400 mb-2">WER History</p>
                    <div className="h-20 relative">
                      <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                        <defs>
                          <linearGradient id="werGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="#ec4899" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="#ec4899" stopOpacity="0" />
                          </linearGradient>
                        </defs>
                        {/* Area */}
                        <polygon
                          fill="url(#werGradient)"
                          points={`0,100 ${werHistory.map((w, i) => {
                            const x = (i / Math.max(1, werHistory.length - 1)) * 100;
                            const maxWer = Math.max(100, ...werHistory.map(h => h.wer));
                            const y = 100 - (w.wer / maxWer) * 90;
                            return `${x},${y}`;
                          }).join(" ")} 100,100`}
                        />
                        {/* Line */}
                        <polyline
                          fill="none"
                          stroke="#ec4899"
                          strokeWidth="2"
                          vectorEffect="non-scaling-stroke"
                          points={werHistory.map((w, i) => {
                            const x = (i / Math.max(1, werHistory.length - 1)) * 100;
                            const maxWer = Math.max(100, ...werHistory.map(h => h.wer));
                            const y = 100 - (w.wer / maxWer) * 90;
                            return `${x},${y}`;
                          }).join(" ")}
                        />
                        {/* Target line at 15% */}
                        <line
                          x1="0" y1={100 - (15 / Math.max(100, ...werHistory.map(h => h.wer))) * 90}
                          x2="100" y2={100 - (15 / Math.max(100, ...werHistory.map(h => h.wer))) * 90}
                          stroke="#10b981"
                          strokeWidth="1"
                          strokeDasharray="3,3"
                          vectorEffect="non-scaling-stroke"
                        />
                      </svg>
                      {/* Labels */}
                      <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-slate-500">
                        <span>Step {werHistory[0]?.step}</span>
                        <span>Step {werHistory[werHistory.length - 1]?.step}</span>
                      </div>
                    </div>
                    {/* WER History Table */}
                    <div className="mt-2 max-h-24 overflow-y-auto">
                      <table className="w-full text-xs">
                        <tbody>
                          {werHistory.slice().reverse().slice(0, 5).map((w, i) => (
                            <tr key={i} className="border-b border-slate-800/50">
                              <td className="py-1 text-slate-400">Step {w.step}</td>
                              <td className={`py-1 text-right font-mono ${
                                w.wer > 100 ? 'text-red-400' :
                                w.wer > 50 ? 'text-orange-400' :
                                w.wer > 25 ? 'text-yellow-400' :
                                'text-cyan-400'
                              }`}>{w.wer.toFixed(1)}%</td>
                              <td className="py-1 text-right text-slate-500">
                                {i > 0 ? (
                                  werHistory.slice().reverse()[i - 1].wer > w.wer ? (
                                    <span className="text-red-400">↑</span>
                                  ) : (
                                    <span className="text-emerald-400">↓</span>
                                  )
                                ) : '-'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Thresholds Guide */}
                <div className="mt-4 pt-3 border-t border-slate-700">
                  <p className="text-xs text-slate-400 mb-2">WER Thresholds</p>
                  <div className="space-y-1 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-red-500" />
                      <span className="text-slate-400">&gt;100%: <span className="text-red-400">Critical - Contact Claude</span></span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-orange-500" />
                      <span className="text-slate-400">50-100%: <span className="text-orange-400">High - Watch closely</span></span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-yellow-500" />
                      <span className="text-slate-400">25-50%: <span className="text-yellow-400">Moderate - Keep training</span></span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-cyan-500" />
                      <span className="text-slate-400">15-25%: <span className="text-cyan-400">Good progress</span></span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded bg-emerald-500" />
                      <span className="text-slate-400">&lt;15%: <span className="text-emerald-400">Target achieved!</span></span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Charts */}
          <div className="col-span-12 lg:col-span-8 space-y-4">
            {/* Interactive Loss Chart */}
            <Card className="bg-slate-800/60 border-slate-700/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg flex items-center gap-2 text-white">
                    <TrendingDown className="h-5 w-5 text-cyan-400" />
                    Loss Over Time
                  </CardTitle>
                  <div className="flex gap-2">
                    {(["loss", "ctc", "ce"] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => setSelectedMetric(m)}
                        className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                          selectedMetric === m
                            ? "bg-cyan-500 text-white"
                            : "bg-slate-700 text-slate-400 hover:bg-slate-600"
                        }`}
                      >
                        {m.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-72 relative">
                  {/* Y-axis labels */}
                  <div className="absolute left-0 top-0 bottom-4 w-12 flex flex-col justify-between text-xs text-slate-500">
                    <span>{maxLoss.toFixed(1)}</span>
                    <span>{(maxLoss * 0.75).toFixed(1)}</span>
                    <span>{(maxLoss * 0.5).toFixed(1)}</span>
                    <span>{(maxLoss * 0.25).toFixed(1)}</span>
                    <span>0</span>
                  </div>
                  {/* Chart area */}
                  <div className="ml-14 h-full relative bg-slate-900/50 rounded-lg overflow-hidden border border-slate-700/50">
                    {/* Grid lines */}
                    <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
                      {[0, 1, 2, 3, 4].map(i => (
                        <div key={i} className="border-t border-slate-700/30" />
                      ))}
                    </div>
                    {/* Vertical grid lines */}
                    <div className="absolute inset-0 flex justify-between pointer-events-none">
                      {[0, 1, 2, 3, 4, 5].map(i => (
                        <div key={i} className="border-l border-slate-700/30" />
                      ))}
                    </div>

                    {/* SVG Chart */}
                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                      <defs>
                        <linearGradient id="lossGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                          <stop offset="0%" stopColor="#06b6d4" />
                          <stop offset="100%" stopColor="#8b5cf6" />
                        </linearGradient>
                        <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3" />
                          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0" />
                        </linearGradient>
                      </defs>

                      {/* Area fill */}
                      {lossHistory.length > 1 && (
                        <polygon
                          fill="url(#areaGradient)"
                          points={`0,100 ${lossHistory.map((point, i) => {
                            const x = (i / (lossHistory.length - 1)) * 100;
                            const y = 100 - (point.loss / maxLoss) * 100;
                            return `${x},${y}`;
                          }).join(" ")} 100,100`}
                        />
                      )}

                      {/* Loss line */}
                      {lossHistory.length > 1 && (
                        <polyline
                          fill="none"
                          stroke="url(#lossGradient)"
                          strokeWidth="0.5"
                          vectorEffect="non-scaling-stroke"
                          points={lossHistory.map((point, i) => {
                            const x = (i / (lossHistory.length - 1)) * 100;
                            const y = 100 - (point.loss / maxLoss) * 100;
                            return `${x},${y}`;
                          }).join(" ")}
                        />
                      )}
                    </svg>

                    {/* Current value indicator */}
                    {metrics.loss > 0 && (
                      <div
                        className="absolute right-3 transform -translate-y-1/2 flex items-center gap-2"
                        style={{ top: `${100 - (metrics.loss / maxLoss) * 100}%` }}
                      >
                        <div className="w-3 h-3 bg-cyan-400 rounded-full shadow-lg shadow-cyan-400/50 animate-pulse" />
                        <span className="text-xs font-mono text-cyan-400 bg-slate-900/90 px-2 py-0.5 rounded">
                          {metrics.loss.toFixed(3)}
                        </span>
                      </div>
                    )}
                  </div>
                  {/* X-axis labels */}
                  <div className="ml-14 flex justify-between text-xs text-slate-500 mt-2">
                    <span>Start</span>
                    <span>Epoch {Math.floor(metrics.epoch / 2)}</span>
                    <span>Current</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Epoch History with Visual Bars */}
            <Card className="bg-slate-800/60 border-slate-700/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2 text-white">
                  <BarChart3 className="h-5 w-5 text-purple-400" />
                  Epoch Performance
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {/* Start reference */}
                  <div className="flex items-center gap-3">
                    <span className="w-16 text-xs text-slate-500">Start</span>
                    <div className="flex-1 h-8 bg-slate-700/50 rounded-lg overflow-hidden relative">
                      <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-red-500 to-orange-500" style={{ width: "100%" }} />
                      <div className="absolute inset-0 flex items-center justify-end pr-3">
                        <span className="text-xs font-mono text-white/80">{startLoss.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>

                  {/* Epoch bars */}
                  {history.slice(-8).map((h, i) => {
                    const widthPercent = (h.avgLoss / startLoss) * 100;
                    const isLatest = i === history.slice(-8).length - 1;
                    return (
                      <div key={h.epoch} className={`flex items-center gap-3 ${isLatest ? 'scale-105 transform' : ''}`}>
                        <span className={`w-16 text-xs ${isLatest ? 'text-cyan-400 font-medium' : 'text-slate-400'}`}>
                          Epoch {h.epoch}
                        </span>
                        <div className="flex-1 h-8 bg-slate-700/50 rounded-lg overflow-hidden relative group cursor-pointer transition-all hover:bg-slate-700">
                          <div
                            className={`absolute inset-y-0 left-0 transition-all duration-500 ${
                              isLatest
                                ? 'bg-gradient-to-r from-cyan-500 to-purple-500'
                                : 'bg-gradient-to-r from-cyan-600/80 to-purple-600/80'
                            }`}
                            style={{ width: `${widthPercent}%` }}
                          />
                          <div className="absolute inset-0 flex items-center justify-between px-3">
                            <span className="text-xs text-white/60 opacity-0 group-hover:opacity-100 transition-opacity">
                              CTC: {h.ctcLoss.toFixed(2)} | CE: {h.ceLoss.toFixed(2)}
                            </span>
                            <span className={`text-xs font-mono ${isLatest ? 'text-white' : 'text-white/80'}`}>
                              {h.avgLoss.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}

                  {/* Current progress */}
                  {metrics.loss > 0 && (
                    <div className="flex items-center gap-3 pt-2 border-t border-slate-700">
                      <span className="w-16 text-xs text-emerald-400 font-medium flex items-center gap-1">
                        <Flame className="h-3 w-3" />
                        Live
                      </span>
                      <div className="flex-1 h-8 bg-slate-700/50 rounded-lg overflow-hidden relative">
                        <div
                          className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-500 to-cyan-400 animate-pulse"
                          style={{ width: `${Math.min(100, (metrics.loss / startLoss) * 100)}%` }}
                        />
                        <div className="absolute inset-0 flex items-center justify-end pr-3">
                          <span className="text-xs font-mono text-white font-bold">{metrics.loss.toFixed(2)}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Detailed Epoch Table */}
            <Card className="bg-slate-800/60 border-slate-700/50 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2 text-white">
                  <Activity className="h-5 w-5 text-emerald-400" />
                  Detailed History
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto max-h-48 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-slate-800/95 backdrop-blur-sm">
                      <tr className="border-b border-slate-700">
                        <th className="text-left py-2 px-2 text-slate-400 font-medium">Epoch</th>
                        <th className="text-right py-2 px-2 text-slate-400 font-medium">Loss</th>
                        <th className="text-right py-2 px-2 text-slate-400 font-medium">WER</th>
                        <th className="text-right py-2 px-2 text-slate-400 font-medium">Reduction</th>
                        <th className="text-right py-2 px-2 text-slate-400 font-medium">Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.length > 0 ? history.slice().reverse().map((h) => {
                        const reduction = ((startLoss - h.avgLoss) / startLoss * 100);
                        return (
                          <tr key={h.epoch} className="border-b border-slate-800 hover:bg-slate-700/30 transition-colors">
                            <td className="py-2 px-2 text-white font-mono">{h.epoch}</td>
                            <td className="py-2 px-2 text-right font-mono text-cyan-400">{h.avgLoss.toFixed(4)}</td>
                            <td className="py-2 px-2 text-right font-mono text-pink-400">
                              {h.wer !== null ? `${h.wer.toFixed(1)}%` : '-'}
                            </td>
                            <td className="py-2 px-2 text-right">
                              <span className={`text-xs px-2 py-0.5 rounded ${reduction > 0 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                                {reduction > 0 ? '-' : '+'}{Math.abs(reduction).toFixed(0)}%
                              </span>
                            </td>
                            <td className="py-2 px-2 text-right font-mono text-slate-400">{h.time}</td>
                          </tr>
                        );
                      }) : (
                        <tr>
                          <td colSpan={5} className="py-8 text-center text-slate-500">
                            <Activity className="h-8 w-8 mx-auto mb-2 animate-pulse" />
                            Waiting for first epoch to complete...
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 mb-24 text-center text-slate-500 text-xs border-t border-slate-800 pt-4">
          <p className="flex items-center justify-center gap-4 flex-wrap">
            <span>Last update: {new Date(metrics.lastUpdate).toLocaleString()}</span>
            <span className="hidden md:inline">|</span>
            <span className="font-mono">{metrics.stage ? metrics.stage.toUpperCase() : 'TRAINING'}</span>
            <span className="hidden md:inline">|</span>
            <span>RTX 4090 24GB (Vast.ai)</span>
            <span className="hidden md:inline">|</span>
            <span>Backup: VPS 5.249.161.66</span>
          </p>
        </div>
      </div>

      {/* Navigation Dock */}
      <PresentationDock items={dockItems} />
    </div>
  );
}
