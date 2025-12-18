"use client";

import { useState, useEffect, useCallback } from "react";
import { Activity, Cpu, Clock, Zap, TrendingUp, AlertCircle, CheckCircle } from "lucide-react";

interface GPUMetrics {
  name: string;
  host: string;
  status: "running" | "stopped" | "error" | "unknown";
  epoch: number;
  step: number;
  totalSteps: number;
  totalEpochs: number;
  loss: number;
  ctcLoss: number;
  ceLoss: number;
  speed: number;
  gpuMemory: number;
  gpuMemoryTotal: number;
  gpuUtil: number;
  eta: string;
  lastUpdate: string;
  totalStepsCompleted: number;
  checkpoint: string;
}

interface DualGPUResponse {
  gpus: GPUMetrics[];
  comparison: {
    speedRatio: string;
    progressDiff: number;
    leader: string;
    h100Advantage: string;
  };
  timestamp: string;
}

export default function GPUComparisonPage() {
  const [data, setData] = useState<DualGPUResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch("/3dgameassistant/api/training-status-dual");
      if (!res.ok) throw new Error("Failed to fetch");
      const json = await res.json();
      setData(json);
      setError(null);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [fetchData]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running": return "text-emerald-400";
      case "stopped": return "text-amber-400";
      case "error": return "text-red-400";
      default: return "text-slate-400";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running": return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case "stopped": return <AlertCircle className="w-5 h-5 text-amber-400" />;
      case "error": return <AlertCircle className="w-5 h-5 text-red-400" />;
      default: return <Clock className="w-5 h-5 text-slate-400" />;
    }
  };

  const formatMemory = (mb: number) => {
    return (mb / 1024).toFixed(1) + " GB";
  };

  const getProgressPercent = (gpu: GPUMetrics) => {
    const total = gpu.totalEpochs * gpu.totalSteps;
    return ((gpu.totalStepsCompleted / total) * 100).toFixed(1);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-4 border-cyan-500 border-t-transparent mx-auto mb-4" />
          <p className="text-slate-400">Connecting to GPUs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Cpu className="w-8 h-8 text-cyan-400" />
              GPU Training Comparison
            </h1>
            <p className="text-slate-400 mt-1">Real-time comparison: RTX 4090 vs H100</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-slate-500">Last refresh</p>
            <p className="text-cyan-400 font-mono">{lastRefresh.toLocaleTimeString()}</p>
          </div>
        </div>
      </div>

      {error && (
        <div className="max-w-7xl mx-auto mb-6">
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 text-red-300">
            Error: {error}
          </div>
        </div>
      )}

      {/* Comparison Summary */}
      {data && (
        <div className="max-w-7xl mx-auto mb-8">
          <div className="bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-emerald-500/10 rounded-2xl border border-cyan-500/30 p-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-center">
              <div>
                <p className="text-slate-400 text-sm mb-1">Speed Ratio</p>
                <p className="text-3xl font-bold text-cyan-400">{data.comparison.speedRatio}x</p>
                <p className="text-xs text-slate-500">H100 vs RTX 4090</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm mb-1">Leader</p>
                <p className="text-3xl font-bold text-emerald-400">{data.comparison.leader}</p>
                <p className="text-xs text-slate-500">More steps completed</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm mb-1">Step Difference</p>
                <p className="text-3xl font-bold text-purple-400">{Math.abs(data.comparison.progressDiff).toLocaleString()}</p>
                <p className="text-xs text-slate-500">Steps ahead</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm mb-1">H100 Advantage</p>
                <p className="text-3xl font-bold text-amber-400">{data.comparison.h100Advantage}</p>
                <p className="text-xs text-slate-500">More progress</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* GPU Cards */}
      {data && (
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
          {data.gpus.map((gpu, idx) => (
            <div
              key={gpu.name}
              className={`rounded-2xl border p-6 ${
                idx === 0
                  ? "bg-gradient-to-br from-green-500/5 to-slate-900 border-green-500/30"
                  : "bg-gradient-to-br from-blue-500/5 to-slate-900 border-blue-500/30"
              }`}
            >
              {/* GPU Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className={`p-3 rounded-xl ${idx === 0 ? "bg-green-500/20" : "bg-blue-500/20"}`}>
                    <Cpu className={`w-6 h-6 ${idx === 0 ? "text-green-400" : "text-blue-400"}`} />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white">{gpu.name}</h2>
                    <p className="text-sm text-slate-500">{gpu.host}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusIcon(gpu.status)}
                  <span className={`font-medium capitalize ${getStatusColor(gpu.status)}`}>
                    {gpu.status}
                  </span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mb-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-400">Overall Progress</span>
                  <span className={idx === 0 ? "text-green-400" : "text-blue-400"}>
                    {getProgressPercent(gpu)}%
                  </span>
                </div>
                <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      idx === 0 ? "bg-gradient-to-r from-green-500 to-emerald-400" : "bg-gradient-to-r from-blue-500 to-cyan-400"
                    }`}
                    style={{ width: `${getProgressPercent(gpu)}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>Epoch {gpu.epoch} / {gpu.totalEpochs}</span>
                  <span>Step {gpu.step.toLocaleString()} / {gpu.totalSteps.toLocaleString()}</span>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-4">
                {/* Loss */}
                <div className="bg-slate-800/50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingUp className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm text-slate-400">Loss</span>
                  </div>
                  <p className="text-2xl font-bold text-white">{gpu.loss.toFixed(4)}</p>
                  <div className="text-xs text-slate-500 mt-1">
                    CTC: {gpu.ctcLoss.toFixed(3)} | CE: {gpu.ceLoss.toFixed(3)}
                  </div>
                </div>

                {/* Speed */}
                <div className="bg-slate-800/50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Zap className="w-4 h-4 text-amber-400" />
                    <span className="text-sm text-slate-400">Speed</span>
                  </div>
                  <p className="text-2xl font-bold text-white">{gpu.speed.toFixed(2)}</p>
                  <div className="text-xs text-slate-500 mt-1">iterations/sec</div>
                </div>

                {/* GPU Utilization */}
                <div className="bg-slate-800/50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Activity className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm text-slate-400">GPU Util</span>
                  </div>
                  <p className="text-2xl font-bold text-white">{gpu.gpuUtil}%</p>
                  <div className="h-1.5 bg-slate-700 rounded-full mt-2">
                    <div
                      className="h-full bg-emerald-500 rounded-full"
                      style={{ width: `${gpu.gpuUtil}%` }}
                    />
                  </div>
                </div>

                {/* VRAM */}
                <div className="bg-slate-800/50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Cpu className="w-4 h-4 text-purple-400" />
                    <span className="text-sm text-slate-400">VRAM</span>
                  </div>
                  <p className="text-2xl font-bold text-white">{formatMemory(gpu.gpuMemory)}</p>
                  <div className="text-xs text-slate-500 mt-1">
                    / {formatMemory(gpu.gpuMemoryTotal)} ({((gpu.gpuMemory / gpu.gpuMemoryTotal) * 100).toFixed(0)}%)
                  </div>
                </div>

                {/* ETA */}
                <div className="bg-slate-800/50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Clock className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm text-slate-400">ETA</span>
                  </div>
                  <p className="text-xl font-bold text-white">{gpu.eta}</p>
                </div>

                {/* Checkpoint */}
                <div className="bg-slate-800/50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-sm text-slate-400">Checkpoint</span>
                  </div>
                  <p className="text-sm font-mono text-white truncate" title={gpu.checkpoint}>
                    {gpu.checkpoint}
                  </p>
                </div>
              </div>

              {/* Total Steps */}
              <div className="mt-4 pt-4 border-t border-slate-700/50 flex justify-between items-center">
                <span className="text-slate-400">Total Steps Completed</span>
                <span className={`text-2xl font-bold ${idx === 0 ? "text-green-400" : "text-blue-400"}`}>
                  {gpu.totalStepsCompleted.toLocaleString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="max-w-7xl mx-auto mt-8 text-center text-slate-500 text-sm">
        <p>Auto-refreshes every 10 seconds • VoxFormer Hybrid CTC-Attention Training</p>
      </div>
    </div>
  );
}
