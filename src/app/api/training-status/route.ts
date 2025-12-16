import { NextResponse } from "next/server";
import { readFile } from "fs/promises";

type TrainingStatus = "running" | "completed" | "error" | "idle" | "initializing" | "stopped";

interface VPSMetrics {
  stage: string;
  total_epochs: number;
  total_steps_per_epoch: number;
  status: TrainingStatus;
  epoch: number;
  step: number;
  loss: number;
  ctc_loss: number;
  ce_loss: number;
  wer: number | null;
  learning_rate: string;
  gpu_memory: number;
  gpu_util: number;
  speed: number;
  eta: string;
  last_update: string;
  backup?: {
    last_backup: string | null;
    status: string;
  };
}

// Cache to store historical data
let epochHistory: Array<{
  epoch: number;
  avgLoss: number;
  ctcLoss: number;
  ceLoss: number;
  wer: number | null;
  time: string;
}> = [];

let lossHistory: Array<{ step: number; loss: number }> = [];

export async function GET() {
  try {
    // Read metrics from local file (backed up from GPU every 5 min)
    const metricsPath = "/home/developer/voxformer_checkpoints/stage4/metrics.json";
    let metricsJson = "";
    try {
      metricsJson = await readFile(metricsPath, "utf-8");
    } catch {
      // File doesn't exist yet
    }

    let metrics: VPSMetrics;
    let usingVPSMetrics = false;

    try {
      if (metricsJson && metricsJson.trim()) {
        metrics = JSON.parse(metricsJson);
        usingVPSMetrics = true;
      } else {
        throw new Error("Empty metrics");
      }
    } catch {
      // Fallback: default idle state
      metrics = {
        stage: "stage4",
        total_epochs: 5,
        total_steps_per_epoch: 3568,
        status: "idle",
        epoch: 0,
        step: 0,
        loss: 0,
        ctc_loss: 0,
        ce_loss: 0,
        wer: null,
        learning_rate: "5e-6",
        gpu_memory: 0,
        gpu_util: 0,
        speed: 0,
        eta: "Not started",
        last_update: new Date().toISOString(),
      };
    }

    // Read epoch history from local file
    const epochLogPath = "/home/developer/voxformer_checkpoints/stage4/epoch_history.log";
    let epochLog = "";
    try {
      epochLog = await readFile(epochLogPath, "utf-8");
    } catch {
      // File doesn't exist yet
    }

    // Parse epoch history: "epoch:0,loss:5.13,wer:25.4,time:2025-12-16 10:18:00"
    if (epochLog && epochLog.trim()) {
      const lines = epochLog.trim().split("\n");
      for (const line of lines) {
        // Support both formats: with and without WER
        const matchWithWer = line.match(/epoch:(\d+),loss:(\d+\.?\d*),wer:(\d+\.?\d*),time:(.+)/);
        const matchWithoutWer = line.match(/epoch:(\d+),loss:(\d+\.?\d*),time:(.+)/);

        const match = matchWithWer || matchWithoutWer;
        if (match) {
          const epoch = parseInt(match[1]);
          const avgLoss = parseFloat(match[2]);
          const wer = matchWithWer ? parseFloat(match[3]) : null;
          const timeStr = matchWithWer ? match[4] : match[3];
          const time = timeStr.split(" ")[1] || timeStr;

          if (!epochHistory.some((h) => h.epoch === epoch)) {
            epochHistory.push({
              epoch,
              avgLoss,
              ctcLoss: avgLoss, // CTC-only in Stage 4
              ceLoss: 0,
              wer,
              time: time.substring(0, 5),
            });
          }
        }
      }
      epochHistory.sort((a, b) => a.epoch - b.epoch);
    }

    // Build loss history from current metrics
    if (metrics.loss > 0) {
      const currentStep = metrics.epoch * metrics.total_steps_per_epoch + metrics.step;
      const lastEntry = lossHistory[lossHistory.length - 1];

      // Only add if step has advanced
      if (!lastEntry || currentStep > lastEntry.step) {
        lossHistory.push({ step: currentStep, loss: metrics.loss });
        if (lossHistory.length > 200) lossHistory.shift();
      }
    }

    // Also build from epoch history for historical data
    for (const h of epochHistory) {
      const endStep = (h.epoch + 1) * metrics.total_steps_per_epoch;
      if (!lossHistory.some((l) => Math.abs(l.step - endStep) < 100)) {
        lossHistory.push({ step: endStep, loss: h.avgLoss });
      }
    }
    lossHistory.sort((a, b) => a.step - b.step);

    // Calculate ETA if not provided
    let eta = metrics.eta || "Calculating...";
    if (metrics.speed > 0 && metrics.status === "running" && (!eta || eta === "Calculating...")) {
      const totalSteps = metrics.total_epochs * metrics.total_steps_per_epoch;
      const currentStep = metrics.epoch * metrics.total_steps_per_epoch + metrics.step;
      const remainingSteps = totalSteps - currentStep;
      const remainingSeconds = remainingSteps / metrics.speed;
      const hours = Math.floor(remainingSeconds / 3600);
      const minutes = Math.floor((remainingSeconds % 3600) / 60);
      eta = `${hours}h ${minutes}m`;
    }

    return NextResponse.json({
      metrics: {
        epoch: metrics.epoch,
        step: metrics.step,
        totalSteps: metrics.total_steps_per_epoch,
        totalEpochs: metrics.total_epochs,
        loss: metrics.loss,
        ctcLoss: metrics.ctc_loss,
        ceLoss: metrics.ce_loss,
        wer: metrics.wer,
        learningRate: metrics.learning_rate,
        speed: metrics.speed,
        gpuMemory: metrics.gpu_memory,
        gpuUtil: metrics.gpu_util,
        eta,
        status: metrics.status,
        lastUpdate: metrics.last_update,
        stage: metrics.stage,
        backup: metrics.backup || null,
      },
      history: epochHistory,
      lossHistory,
      source: usingVPSMetrics ? "vps" : "default",
    });
  } catch (error) {
    console.error("Error fetching training status:", error);
    return NextResponse.json(
      {
        metrics: {
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
          eta: "Unknown",
          status: "error" as const,
          lastUpdate: new Date().toISOString(),
          stage: "stage4",
          backup: null,
        },
        history: epochHistory,
        lossHistory,
        error: "Failed to fetch training status",
        source: "error",
      },
      { status: 500 }
    );
  }
}
