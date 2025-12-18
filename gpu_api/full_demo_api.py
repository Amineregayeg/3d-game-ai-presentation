#!/usr/bin/env python3
"""
Full Demo GPU API Service
Handles: Whisper STT, SadTalker Lip-sync
"""

import os
import sys
import json
import base64
import tempfile
import time
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# Configuration
WHISPER_MODEL = "large-v3"
SADTALKER_PATH = "/root/SadTalker"
AVATARS_PATH = "/root/avatars"
RESULTS_PATH = "/root/results"

os.makedirs(RESULTS_PATH, exist_ok=True)

whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        import whisper
        print(f"Loading Whisper {WHISPER_MODEL}...")
        whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded!")
    return whisper_model

def convert_to_wav(input_data, output_path):
    """Convert any audio to 16kHz mono WAV"""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(input_data)
        input_path = f.name

    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            output_path
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise Exception(result.stderr or "FFmpeg conversion failed")

        return True
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

@app.route("/health", methods=["GET"])
def health():
    gpu_info = "No GPU"
    try:
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
    except:
        pass
    return jsonify({
        "status": "healthy",
        "gpu": gpu_info,
        "whisper": WHISPER_MODEL,
        "sadtalker": os.path.exists(SADTALKER_PATH),
        "avatars": os.listdir(AVATARS_PATH) if os.path.exists(AVATARS_PATH) else []
    })

@app.route("/api/avatars", methods=["GET"])
def list_avatars():
    avatars = []
    if os.path.exists(AVATARS_PATH):
        for f in os.listdir(AVATARS_PATH):
            if f.endswith((".png", ".jpg", ".jpeg")):
                avatars.append({"id": os.path.splitext(f)[0], "filename": f})
    return jsonify({"avatars": avatars})

@app.route("/api/stt/whisper", methods=["POST"])
def transcribe_whisper():
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or "audio" not in data:
            return jsonify({"error": "No audio provided"}), 400

        audio_data = base64.b64decode(data["audio"])
        language = data.get("language", None)

        # Convert to WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        try:
            convert_to_wav(audio_data, audio_path)
        except Exception as e:
            return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 400

        try:
            model = get_whisper_model()
            result = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                task="transcribe"
            )

            processing_time = time.time() - start_time

            return jsonify({
                "text": result["text"].strip(),
                "language": result.get("language", "en"),
                "confidence": 0.95,
                "processing_time": round(processing_time, 2)
            })
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/sadtalker/generate", methods=["POST"])
def generate_lipsync():
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or "audio" not in data:
            return jsonify({"error": "No audio provided"}), 400

        # Get avatar
        avatar_id = data.get("avatar_id", "default")
        avatar_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            p = os.path.join(AVATARS_PATH, f"{avatar_id}{ext}")
            if os.path.exists(p):
                avatar_path = p
                break

        if not avatar_path:
            # Try exact filename
            for f in os.listdir(AVATARS_PATH):
                if f.startswith(avatar_id):
                    avatar_path = os.path.join(AVATARS_PATH, f)
                    break

        if not avatar_path:
            return jsonify({"error": f"Avatar '{avatar_id}' not found. Available: {os.listdir(AVATARS_PATH)}"}), 404

        # Decode and convert audio
        audio_data = base64.b64decode(data["audio"])
        timestamp = int(time.time() * 1000)
        audio_path = f"/tmp/audio_{timestamp}.wav"

        try:
            convert_to_wav(audio_data, audio_path)
        except Exception as e:
            return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 400

        # Run SadTalker
        result_dir = os.path.join(RESULTS_PATH, f"result_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        preprocess = data.get("preprocess", "crop")

        cmd = [
            "python3", os.path.join(SADTALKER_PATH, "inference.py"),
            "--driven_audio", audio_path,
            "--source_image", avatar_path,
            "--result_dir", result_dir,
            "--preprocess", preprocess,
            "--still",
            "--enhancer", "gfpgan"
        ]

        print(f"Running SadTalker: {' '.join(cmd)}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        process = subprocess.run(
            cmd,
            cwd=SADTALKER_PATH,
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )

        if process.returncode != 0:
            print(f"SadTalker stdout: {process.stdout}")
            print(f"SadTalker stderr: {process.stderr}")
            return jsonify({
                "error": "SadTalker generation failed",
                "details": process.stderr[-500:] if process.stderr else process.stdout[-500:] if process.stdout else "Unknown error"
            }), 500

        # Find result video
        video_path = None
        for f in os.listdir(result_dir):
            if f.endswith(".mp4"):
                video_path = os.path.join(result_dir, f)
                break

        if not video_path:
            return jsonify({"error": "No video generated", "dir_contents": os.listdir(result_dir)}), 500

        # Return video as base64
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        duration = time.time() - start_time

        # Cleanup
        os.unlink(audio_path)

        return jsonify({
            "video_base64": video_b64,
            "duration": round(duration, 2)
        })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "SadTalker timed out (>5min)"}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/result/<path:filename>", methods=["GET"])
def serve_result(filename):
    return send_file(os.path.join(RESULTS_PATH, filename))

if __name__ == "__main__":
    print("=" * 50)
    print("Full Demo GPU API Server")
    print("=" * 50)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"SadTalker: {SADTALKER_PATH}")
    print(f"Avatars: {AVATARS_PATH}")
    print("=" * 50)

    # Preload Whisper
    print("Preloading Whisper model...")
    get_whisper_model()

    app.run(host="0.0.0.0", port=5001, debug=False)
