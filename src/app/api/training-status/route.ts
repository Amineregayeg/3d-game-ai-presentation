import { NextResponse } from "next/server";
import { readFile, writeFile } from "fs/promises";

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

// Track epoch completions from metrics changes
let lastSeenEpoch = -1;
let lastEpochLoss = 0;

interface BackupEntry {
  time: string;
  type: string;
  status: "success" | "failed";
}

interface EpochHistoryEntry {
  epoch: number;
  avgLoss: number;
  ctcLoss: number;
  ceLoss: number;
  wer: number | null;
  time: string;
}

// Support multiple stages - check fixed first, then hybrid, then stage4
const STAGE4_FIXED_PATH = "/home/developer/voxformer_checkpoints/stage4_fixed";
const STAGE4_HYBRID_PATH = "/home/developer/voxformer_checkpoints/stage4_hybrid";
const STAGE4_PATH = "/home/developer/voxformer_checkpoints/stage4";

async function getActiveStagePath(): Promise<string> {
  const { stat } = await import("fs/promises");

  // ALWAYS prefer stage4_fixed (H100 training with fixes) when it exists
  try {
    await stat(`${STAGE4_FIXED_PATH}/metrics.json`);
    return STAGE4_FIXED_PATH;
  } catch {
    // stage4_fixed doesn't exist, try hybrid
  }

  try {
    await stat(`${STAGE4_HYBRID_PATH}/metrics.json`);
    return STAGE4_HYBRID_PATH;
  } catch {
    // Fall back to stage4
  }

  return STAGE4_PATH;
}

// WER History entry
interface WERHistoryEntry {
  step: number;
  wer: number;
  timestamp: string;
}

// Parse WER history from training.log
async function parseWERHistory(stagePath: string): Promise<WERHistoryEntry[]> {
  const werHistory: WERHistoryEntry[] = [];
  try {
    const logPath = `${stagePath}/training.log`;
    const logContent = await readFile(logPath, "utf-8");

    // Parse lines like: 2025-12-17 18:19:04,737 - __main__ - INFO - Step 500 - Eval WER: 98.85%
    const werRegex = /(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - .+ - INFO - Step (\d+) - Eval WER: ([\d.]+)%/g;
    let match;
    const seenSteps = new Set<number>();

    while ((match = werRegex.exec(logContent)) !== null) {
      const step = parseInt(match[2]);
      // Only keep the latest WER for each step (avoid duplicates from restarts)
      if (!seenSteps.has(step)) {
        seenSteps.add(step);
        werHistory.push({
          step,
          wer: parseFloat(match[3]),
          timestamp: match[1],
        });
      } else {
        // Update to latest value for this step
        const idx = werHistory.findIndex(w => w.step === step);
        if (idx >= 0) {
          werHistory[idx] = {
            step,
            wer: parseFloat(match[3]),
            timestamp: match[1],
          };
        }
      }
    }

    werHistory.sort((a, b) => a.step - b.step);
  } catch {
    // Log doesn't exist or can't be read
  }
  return werHistory;
}

const EPOCH_HISTORY_PATH = "/home/developer/voxformer_checkpoints/stage4_hybrid/epoch_history.json";

// Load epoch history from file
async function loadEpochHistory(): Promise<EpochHistoryEntry[]> {
  try {
    const data = await readFile(EPOCH_HISTORY_PATH, "utf-8");
    return JSON.parse(data);
  } catch {
    return [];
  }
}

// Save epoch history to file
async function saveEpochHistory(history: EpochHistoryEntry[]): Promise<void> {
  try {
    await writeFile(EPOCH_HISTORY_PATH, JSON.stringify(history, null, 2));
  } catch (err) {
    console.error("Failed to save epoch history:", err);
  }
}

export async function GET() {
  try {
    // Determine active stage (hybrid or stage4)
    const stagePath = await getActiveStagePath();

    // Load persisted epoch history
    epochHistory = await loadEpochHistory();

    // Read metrics from local file (backed up from GPU every 5 min)
    const metricsPath = `${stagePath}/metrics.json`;
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
        stage: "stage4_hybrid",
        total_epochs: 10,
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

    // Track epoch completions from metrics (when epoch number increases)
    if (metrics.epoch > lastSeenEpoch && lastSeenEpoch >= 0) {
      // Epoch completed - record the last loss before epoch changed
      const completedEpoch = lastSeenEpoch;
      const now = new Date();
      const timeStr = now.toTimeString().substring(0, 5);

      if (!epochHistory.some((h) => h.epoch === completedEpoch)) {
        epochHistory.push({
          epoch: completedEpoch,
          avgLoss: lastEpochLoss,
          ctcLoss: lastEpochLoss,
          ceLoss: 0,
          wer: null,
          time: timeStr,
        });
        epochHistory.sort((a, b) => a.epoch - b.epoch);
        // Persist to file
        await saveEpochHistory(epochHistory);
      }
    }
    lastSeenEpoch = metrics.epoch;
    lastEpochLoss = metrics.loss;

    // Read backup log to get backup history
    const backupLogPath = `${stagePath}/backup.log`;
    let backupLog = "";
    try {
      backupLog = await readFile(backupLogPath, "utf-8");
    } catch {
      // File doesn't exist yet
    }

    // Parse backup log entries
    const backupHistory: BackupEntry[] = [];
    if (backupLog && backupLog.trim()) {
      const lines = backupLog.trim().split("\n").slice(-50); // Last 50 lines
      for (const line of lines) {
        // Format: [2025-12-16 12:11:32] metrics.json backed up
        const match = line.match(/\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (.+)/);
        if (match) {
          const time = match[1];
          const message = match[2];

          let type = "info";
          let status: "success" | "failed" = "success";

          if (message.includes("metrics.json backed up")) {
            type = "metrics";
          } else if (message.includes("checkpoint backed up") || message.includes("latest checkpoint")) {
            type = "checkpoint";
          } else if (message.includes("best.pt")) {
            type = "best_checkpoint";
          } else if (message.includes("Backup complete")) {
            type = "complete";
          } else if (message.includes("Starting backup")) {
            type = "start";
          } else if (message.includes("failed") || message.includes("error")) {
            status = "failed";
          }

          backupHistory.push({ time, type, status });
        }
      }
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

    // Parse WER history from training log
    const werHistory = await parseWERHistory(stagePath);

    // Get current WER from history if not in metrics
    const currentWER = metrics.wer ?? (werHistory.length > 0 ? werHistory[werHistory.length - 1].wer : null);

    return NextResponse.json({
      metrics: {
        epoch: metrics.epoch,
        step: metrics.step,
        totalSteps: metrics.total_steps_per_epoch,
        totalEpochs: metrics.total_epochs,
        loss: metrics.loss,
        ctcLoss: metrics.ctc_loss,
        ceLoss: metrics.ce_loss,
        wer: currentWER,
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
      backupHistory,
      werHistory,
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
        backupHistory: [],
        error: "Failed to fetch training status",
        source: "error",
      },
      { status: 500 }
    );
  }
}
