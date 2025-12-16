import { NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import { writeFile, unlink } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

const execAsync = promisify(exec);

// GPU connection details (accessed via VPS jump)
const GPU_HOST = "82.141.118.40";
const GPU_PORT = "2674";
const GPU_USER = "root";
const SSH_KEY = "/home/developer/.ssh/id_rsa";
const SSH_OPTS = `-i ${SSH_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`;

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { audio, useTestAudio } = body;

    // If useTestAudio flag is set, run inference on a LibriSpeech sample
    if (useTestAudio) {
      // Run a Python test script on GPU that loads a sample and transcribes it
      const testCommand = `cd /root/voxformer && python3 test_inference.py`;

      const { stdout, stderr } = await execAsync(
        `ssh ${SSH_OPTS} -p ${GPU_PORT} -o ConnectTimeout=60 -o BatchMode=yes ${GPU_USER}@${GPU_HOST} '${testCommand}'`,
        { timeout: 120000, maxBuffer: 10 * 1024 * 1024 }
      );

      if (stderr && !stdout) {
        console.error("SSH error:", stderr);
        return NextResponse.json(
          { error: "Failed to run test transcription", details: stderr },
          { status: 500 }
        );
      }

      // Parse the JSON output
      const result = JSON.parse(stdout.trim());
      return NextResponse.json(result);
    }

    // Normal transcription with provided audio (base64 webm)
    if (!audio) {
      return NextResponse.json({ error: "No audio provided" }, { status: 400 });
    }

    // Write audio base64 to local temp file, then SCP to GPU
    const timestamp = Date.now();
    const localFilePath = join(tmpdir(), `audio_${timestamp}.b64`);
    const remoteFilePath = `/tmp/audio_${timestamp}.b64`;

    // Step 1: Write base64 to local temp file
    await writeFile(localFilePath, audio, "utf-8");
    console.log(`Wrote ${audio.length} bytes to ${localFilePath}`);

    try {
      // Step 2: SCP the file to GPU
      await execAsync(
        `scp ${SSH_OPTS} -P ${GPU_PORT} -o ConnectTimeout=30 ${localFilePath} ${GPU_USER}@${GPU_HOST}:${remoteFilePath}`,
        { timeout: 60000 }
      );
      console.log(`Uploaded to GPU: ${remoteFilePath}`);

      // Step 3: Run the inference script with the audio file path
      const inferenceCommand = `cd /root/voxformer && python3 transcribe_audio.py ${remoteFilePath}`;

      const { stdout, stderr } = await execAsync(
        `ssh ${SSH_OPTS} -p ${GPU_PORT} -o ConnectTimeout=120 -o BatchMode=yes ${GPU_USER}@${GPU_HOST} '${inferenceCommand}'`,
        { timeout: 180000, maxBuffer: 10 * 1024 * 1024 }
      );

      if (stderr && !stdout) {
        console.error("SSH error:", stderr);
        return NextResponse.json(
          { error: "Failed to process audio", details: stderr },
          { status: 500 }
        );
      }

      // Find the JSON output (last line)
      const lines = stdout.trim().split('\n');
      const jsonLine = lines[lines.length - 1];
      const result = JSON.parse(jsonLine);
      return NextResponse.json(result);
    } finally {
      // Cleanup local temp file
      try {
        await unlink(localFilePath);
      } catch {
        // Ignore cleanup errors
      }
    }

  } catch (error) {
    console.error("Transcription error:", error);

    return NextResponse.json(
      {
        error: "Transcription failed",
        details: error instanceof Error ? error.message : "Unknown error",
        note: "The inference server may not be running. Check GPU status.",
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  // Health check endpoint
  try {
    const { stdout } = await execAsync(
      `ssh ${SSH_OPTS} -p ${GPU_PORT} -o ConnectTimeout=10 -o BatchMode=yes ${GPU_USER}@${GPU_HOST} "curl -s http://localhost:5001/health"`,
      { timeout: 15000 }
    );

    const health = JSON.parse(stdout);
    return NextResponse.json({
      status: "ok",
      inference_server: health,
      connection: "GPU via SSH",
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "error",
        error: "Cannot reach inference server",
        details: error instanceof Error ? error.message : "Unknown",
      },
      { status: 503 }
    );
  }
}
