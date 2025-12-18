"""
GPU Proxy API for VPS Backend
Proxies requests from frontend to GPU server (Whisper STT, SadTalker)
"""

import os
import json
import base64
import requests
from flask import Blueprint, request, jsonify, Response
import time

gpu_proxy = Blueprint('gpu_proxy', __name__)

# GPU Server Configuration
GPU_HOST = os.environ.get('GPU_HOST', '80.188.223.202')
GPU_PORT = os.environ.get('GPU_PORT', '17757')
GPU_API_PORT = os.environ.get('GPU_API_PORT', '5001')

# For internal Docker network (when using SSH tunnel)
GPU_API_URL = os.environ.get('GPU_API_URL', 'http://localhost:5001')

# Timeout configurations
STT_TIMEOUT = 60  # seconds
LIPSYNC_TIMEOUT = 300  # 5 minutes for video generation
DEFAULT_TIMEOUT = 30

def make_gpu_request(endpoint, method='GET', data=None, timeout=DEFAULT_TIMEOUT):
    """Make request to GPU API"""
    url = f"{GPU_API_URL}{endpoint}"
    try:
        if method == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return None, f"Unsupported method: {method}"

        return response, None
    except requests.Timeout:
        return None, f"GPU API timeout after {timeout}s"
    except requests.ConnectionError:
        return None, "GPU API connection failed - server may be down"
    except Exception as e:
        return None, str(e)

@gpu_proxy.route('/api/gpu/health', methods=['GET'])
def gpu_health():
    """Check GPU API health"""
    response, error = make_gpu_request('/health', timeout=10)

    if error:
        return jsonify({
            "status": "error",
            "error": error,
            "gpu_url": GPU_API_URL
        }), 503

    try:
        data = response.json()
        data['proxy'] = 'vps'
        return jsonify(data)
    except:
        return jsonify({
            "status": "error",
            "error": "Invalid response from GPU",
            "raw": response.text[:200]
        }), 500

@gpu_proxy.route('/api/gpu/stt', methods=['POST'])
def transcribe_audio():
    """
    Transcribe audio using Whisper on GPU

    Request: { "audio": "base64_audio_data", "language": "en" (optional) }
    Response: { "text": "...", "segments": [...], "confidence": 0.95, ... }
    """
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({"error": "No audio data provided"}), 400

        # Forward to GPU
        response, error = make_gpu_request(
            '/api/stt/whisper',
            method='POST',
            data=data,
            timeout=STT_TIMEOUT
        )

        if error:
            return jsonify({"error": error}), 503

        result = response.json()
        result['proxy_time_ms'] = round((time.time() - start_time) * 1000, 0)

        if response.status_code != 200:
            return jsonify(result), response.status_code

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@gpu_proxy.route('/api/gpu/avatars', methods=['GET'])
def list_avatars():
    """List available avatars on GPU server"""
    response, error = make_gpu_request('/api/avatars', timeout=10)

    if error:
        return jsonify({"error": error, "avatars": []}), 503

    try:
        return jsonify(response.json())
    except:
        return jsonify({"error": "Invalid response", "avatars": []}), 500

@gpu_proxy.route('/api/gpu/lipsync', methods=['POST'])
def generate_lipsync():
    """
    Generate lip-synced video using SadTalker on GPU

    Request: {
        "audio": "base64_audio_data",
        "avatar_id": "default" (optional),
        "preprocess": "crop" (optional)
    }
    Response: { "video_url": "...", "duration": 5.2 }
    """
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or 'audio' not in data:
            return jsonify({"error": "No audio data provided"}), 400

        # Forward to GPU
        response, error = make_gpu_request(
            '/api/sadtalker/generate',
            method='POST',
            data=data,
            timeout=LIPSYNC_TIMEOUT
        )

        if error:
            return jsonify({"error": error}), 503

        result = response.json()
        result['proxy_time_ms'] = round((time.time() - start_time) * 1000, 0)

        if response.status_code != 200:
            return jsonify(result), response.status_code

        # Rewrite video URL to go through proxy
        if 'video_url' in result:
            original_url = result['video_url']
            result['video_url'] = f"/api/gpu/video{original_url.replace('/api/result', '')}"
            result['original_url'] = original_url

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@gpu_proxy.route('/api/gpu/video/<path:filepath>', methods=['GET'])
def serve_video(filepath):
    """Proxy video files from GPU server"""
    response, error = make_gpu_request(f'/api/result/{filepath}', timeout=60)

    if error:
        return jsonify({"error": error}), 503

    if response.status_code != 200:
        return jsonify({"error": "Video not found"}), 404

    return Response(
        response.content,
        mimetype='video/mp4',
        headers={'Content-Disposition': f'inline; filename="{filepath.split("/")[-1]}"'}
    )

@gpu_proxy.route('/api/gpu/tts', methods=['POST'])
def generate_tts():
    """
    Generate TTS audio using ElevenLabs (via GPU)

    Request: { "text": "...", "voice_id": "..." }
    Response: { "audio": "base64_audio" }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        # Add API key from environment if not provided
        if 'api_key' not in data:
            api_key = os.environ.get('ELEVENLABS_API_KEY')
            if api_key:
                data['api_key'] = api_key
            else:
                return jsonify({"error": "ElevenLabs API key required"}), 400

        response, error = make_gpu_request(
            '/api/tts/elevenlabs',
            method='POST',
            data=data,
            timeout=30
        )

        if error:
            return jsonify({"error": error}), 503

        return jsonify(response.json()), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@gpu_proxy.route('/api/gpu/pipeline', methods=['POST'])
def full_pipeline():
    """
    Full pipeline: Audio -> STT -> (RAG on VPS) -> TTS -> Lip-sync

    Request: {
        "audio": "base64_input_audio",        # User's voice input
        "skip_stt": false,                    # Optional: skip STT if text provided
        "text": "...",                        # Optional: direct text input
        "avatar_id": "default",               # Optional: avatar selection
        "response_audio": "base64_audio"      # Optional: pre-generated TTS audio
    }

    Response: {
        "stt": { "text": "...", ... },       # STT results
        "lipsync": { "video_url": "..." }    # Lip-sync video
    }
    """
    start_time = time.time()

    try:
        data = request.get_json()
        results = {
            "pipeline_time_ms": 0,
            "stages": {}
        }

        # Stage 1: STT (if audio provided and not skipped)
        if 'audio' in data and not data.get('skip_stt', False):
            stt_start = time.time()
            stt_response, error = make_gpu_request(
                '/api/stt/whisper',
                method='POST',
                data={"audio": data['audio'], "language": data.get('language')},
                timeout=STT_TIMEOUT
            )

            if error:
                results['stages']['stt'] = {"error": error}
            else:
                stt_result = stt_response.json()
                stt_result['stage_time_ms'] = round((time.time() - stt_start) * 1000, 0)
                results['stages']['stt'] = stt_result
                results['transcribed_text'] = stt_result.get('text', '')

        # Stage 2: Lip-sync (if response_audio provided)
        if 'response_audio' in data:
            lipsync_start = time.time()
            lipsync_response, error = make_gpu_request(
                '/api/sadtalker/generate',
                method='POST',
                data={
                    "audio": data['response_audio'],
                    "avatar_id": data.get('avatar_id', 'default'),
                    "preprocess": data.get('preprocess', 'crop')
                },
                timeout=LIPSYNC_TIMEOUT
            )

            if error:
                results['stages']['lipsync'] = {"error": error}
            else:
                lipsync_result = lipsync_response.json()
                lipsync_result['stage_time_ms'] = round((time.time() - lipsync_start) * 1000, 0)

                # Rewrite video URL
                if 'video_url' in lipsync_result:
                    original_url = lipsync_result['video_url']
                    lipsync_result['video_url'] = f"/api/gpu/video{original_url.replace('/api/result', '')}"

                results['stages']['lipsync'] = lipsync_result

        results['pipeline_time_ms'] = round((time.time() - start_time) * 1000, 0)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket support for real-time updates (optional)
# Can be extended with Flask-SocketIO for progress updates
