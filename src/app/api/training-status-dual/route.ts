import { NextResponse } from "next/server";

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

// GPU server configurations
const GPU_CONFIGS = [
  { name: "RTX 4090", host: "145.236.166.111", port: "17757", vram: 24 },
  { name: "H100", host: "80.188.223.202", port: "17757", vram: 80 },
];

async function fetchGPUMetrics(config: typeof GPU_CONFIGS[0]): Promise<GPUMetrics> {
  const defaultMetrics: GPUMetrics = {
    name: config.name,
    host: config.host,
    status: "unknown",
    epoch: 0,
    step: 0,
    totalSteps: 3568,
    totalEpochs: 10,
    loss: 0,
    ctcLoss: 0,
    ceLoss: 0,
    speed: 0,
    gpuMemory: 0,
    gpuMemoryTotal: config.vram * 1024,
    gpuUtil: 0,
    eta: "Unknown",
    lastUpdate: new Date().toISOString(),
    totalStepsCompleted: 0,
    checkpoint: "None",
  };

  try {
    // SSH directly from VPS to GPU server (this API runs on VPS)
    const sshCmd = `ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -o BatchMode=yes -p ${config.port} root@${config.host} '
      cd /root/voxformer 2>/dev/null || cd /workspace/voxformer 2>/dev/null
      # Get latest training log line
      LOG=$(find checkpoints -name "*.log" -type f -exec stat --format="%Y %n" {} \\; 2>/dev/null | sort -rn | head -1 | cut -d" " -f2)
      if [ -n "$LOG" ]; then
        tail -3 "$LOG" 2>/dev/null | grep -E "Epoch|loss" | tail -1
      fi
      echo SEPARATOR
      # Get GPU stats
      nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
      echo SEPARATOR
      # Get latest checkpoint
      ls -t checkpoints/stage4_fixed/*.pt 2>/dev/null | head -1 || ls -t checkpoints/stage4_hybrid/*.pt 2>/dev/null | head -1 || ls -t checkpoints/stage4/*.pt 2>/dev/null | head -1
      echo SEPARATOR
      # Check if training is running
      pgrep -f train.py > /dev/null && echo RUNNING || echo STOPPED
    '`;

    const { exec } = await import("child_process");
    const { promisify } = await import("util");
    const execAsync = promisify(exec);

    const { stdout } = await execAsync(sshCmd, { timeout: 15000 });
    const parts = stdout.split("SEPARATOR");

    // Parse training log
    const logLine = parts[0]?.trim() || "";
    // Example: Epoch 2:  31%|███       | 1120/3568 [06:48<06:17,  6.48it/s, loss=3.6524, ctc=1.2865, ce=4.6664, lr=7.89e-06]
    const epochMatch = logLine.match(/Epoch\s+(\d+):\s+(\d+)%.*?(\d+)\/(\d+)/);
    const lossMatch = logLine.match(/loss=(\d+\.?\d*)/);
    const ctcMatch = logLine.match(/ctc=(\d+\.?\d*)/);
    const ceMatch = logLine.match(/ce=(\d+\.?\d*)/);
    const speedMatch = logLine.match(/(\d+\.?\d*)\s*it\/s/);

    if (epochMatch) {
      defaultMetrics.epoch = parseInt(epochMatch[1]);
      defaultMetrics.step = parseInt(epochMatch[3]);
      defaultMetrics.totalSteps = parseInt(epochMatch[4]);
      defaultMetrics.totalStepsCompleted = defaultMetrics.epoch * defaultMetrics.totalSteps + defaultMetrics.step;
    }
    if (lossMatch) defaultMetrics.loss = parseFloat(lossMatch[1]);
    if (ctcMatch) defaultMetrics.ctcLoss = parseFloat(ctcMatch[1]);
    if (ceMatch) defaultMetrics.ceLoss = parseFloat(ceMatch[1]);
    if (speedMatch) defaultMetrics.speed = parseFloat(speedMatch[1]);

    // Parse GPU stats
    const gpuLine = parts[1]?.trim() || "";
    const gpuMatch = gpuLine.match(/(\d+)\s*%?,\s*(\d+)\s*MiB,\s*(\d+)\s*MiB/);
    if (gpuMatch) {
      defaultMetrics.gpuUtil = parseInt(gpuMatch[1]);
      defaultMetrics.gpuMemory = parseInt(gpuMatch[2]);
      defaultMetrics.gpuMemoryTotal = parseInt(gpuMatch[3]);
    }

    // Parse checkpoint
    const checkpointLine = parts[2]?.trim() || "";
    if (checkpointLine) {
      defaultMetrics.checkpoint = checkpointLine.split("/").pop() || "None";
    }

    // Parse running status
    const statusLine = parts[3]?.trim() || "";
    defaultMetrics.status = statusLine.includes("RUNNING") ? "running" : "stopped";

    // Calculate ETA
    if (defaultMetrics.speed > 0 && defaultMetrics.status === "running") {
      const totalSteps = defaultMetrics.totalEpochs * defaultMetrics.totalSteps;
      const remainingSteps = totalSteps - defaultMetrics.totalStepsCompleted;
      const remainingSeconds = remainingSteps / defaultMetrics.speed;
      const hours = Math.floor(remainingSeconds / 3600);
      const minutes = Math.floor((remainingSeconds % 3600) / 60);
      defaultMetrics.eta = `${hours}h ${minutes}m`;
    }

    defaultMetrics.lastUpdate = new Date().toISOString();
    return defaultMetrics;
  } catch (error) {
    console.error(`Error fetching ${config.name} metrics:`, error);
    defaultMetrics.status = "error";
    return defaultMetrics;
  }
}

export async function GET() {
  try {
    // Fetch metrics from both GPUs in parallel
    const [rtx4090, h100] = await Promise.all(
      GPU_CONFIGS.map(fetchGPUMetrics)
    );

    // Calculate comparison stats
    const speedRatio = h100.speed > 0 && rtx4090.speed > 0
      ? (h100.speed / rtx4090.speed).toFixed(2)
      : "N/A";

    const progressDiff = h100.totalStepsCompleted - rtx4090.totalStepsCompleted;

    return NextResponse.json({
      gpus: [rtx4090, h100],
      comparison: {
        speedRatio,
        progressDiff,
        leader: h100.totalStepsCompleted > rtx4090.totalStepsCompleted ? "H100" : "RTX 4090",
        h100Advantage: `${((h100.totalStepsCompleted / Math.max(rtx4090.totalStepsCompleted, 1) - 1) * 100).toFixed(0)}%`,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Error fetching dual GPU status:", error);
    return NextResponse.json(
      { error: "Failed to fetch GPU metrics" },
      { status: 500 }
    );
  }
}
