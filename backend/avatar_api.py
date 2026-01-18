"""
Avatar API Module - TTS + Lip-Sync Integration
ElevenLabs Flash v2.5 + MuseTalk 1.5
"""

import os
import subprocess
import requests
import time
import json
from pathlib import Path
from flask import Blueprint, request, jsonify, send_from_directory, current_app

avatar_bp = Blueprint('avatar', __name__)

# =============================================================================
# Configuration
# =============================================================================

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.environ.get(
    'ELEVENLABS_API_KEY',
    'sk_ded55dd000176fdef60da05787b2dabd496a7e154b86d733'
)
ELEVENLABS_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel

# GPU Server Configuration (vast.ai)
GPU_HOST = os.environ.get('GPU_HOST', '82.141.118.40')
GPU_PORT = int(os.environ.get('GPU_PORT', '2674'))
GPU_USER = os.environ.get('GPU_USER', 'root')
MUSETALK_PATH = os.environ.get('MUSETALK_PATH', '/root/SadTalker')

# Storage paths (use local paths relative to this file)
BACKEND_DIR = Path(__file__).parent
AVATAR_OUTPUT_DIR = BACKEND_DIR / 'static' / 'avatar'
AVATAR_REFERENCE_DIR = BACKEND_DIR / 'static' / 'avatars'

# Ensure directories exist
AVATAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# Available voices (cached)
VOICE_CACHE = {}

# =============================================================================
# Helper Functions
# =============================================================================

def generate_request_id():
    """Generate unique request ID"""
    return f"{int(time.time())}_{os.urandom(4).hex()}"


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using ffprobe"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)
        ], capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception:
        # Fallback: estimate from file size (rough approximation for MP3)
        file_size = audio_path.stat().st_size
        return file_size / 16000  # ~128kbps MP3


def generate_tts_audio(text: str, voice_id: str, output_path: Path,
                       stability: float = 0.5, similarity_boost: float = 0.75) -> float:
    """
    Generate audio using ElevenLabs Flash v2.5

    Args:
        text: Text to synthesize
        voice_id: ElevenLabs voice ID
        output_path: Path to save MP3 file
        stability: Voice stability (0-1)
        similarity_boost: Similarity boost (0-1)

    Returns:
        Audio duration in seconds
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)

    if response.status_code != 200:
        error_detail = response.text[:500]
        raise Exception(f"ElevenLabs API error ({response.status_code}): {error_detail}")

    # Save audio file
    with open(output_path, 'wb') as f:
        f.write(response.content)

    # Get actual duration
    duration = get_audio_duration(output_path)

    return duration


def check_musetalk_available() -> bool:
    """Check if MuseTalk is available on GPU server"""
    try:
        result = subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=5',
            '-p', str(GPU_PORT), f'{GPU_USER}@{GPU_HOST}',
            f'test -d {MUSETALK_PATH} && echo OK'
        ], capture_output=True, text=True, timeout=15)
        return 'OK' in result.stdout
    except Exception:
        return False


def generate_lipsync_video(audio_path: Path, avatar_id: str, output_path: Path) -> bool:
    """
    Generate lip-sync video using MuseTalk on GPU server

    Args:
        audio_path: Path to audio file
        avatar_id: Avatar reference image ID
        output_path: Path for output video

    Returns:
        True if successful, False otherwise
    """
    # Get avatar reference image
    avatar_image = AVATAR_REFERENCE_DIR / f"{avatar_id}.png"
    if not avatar_image.exists():
        avatar_image = AVATAR_REFERENCE_DIR / "default.png"
        if not avatar_image.exists():
            return False

    try:
        # Transfer audio to GPU server
        gpu_audio_path = f"/tmp/avatar_audio_{audio_path.stem}.mp3"
        subprocess.run([
            'scp', '-o', 'StrictHostKeyChecking=no',
            '-P', str(GPU_PORT),
            str(audio_path),
            f'{GPU_USER}@{GPU_HOST}:{gpu_audio_path}'
        ], check=True, timeout=60)

        # Transfer avatar image to GPU server
        gpu_avatar_path = f"/tmp/avatar_image_{avatar_id}.png"
        subprocess.run([
            'scp', '-o', 'StrictHostKeyChecking=no',
            '-P', str(GPU_PORT),
            str(avatar_image),
            f'{GPU_USER}@{GPU_HOST}:{gpu_avatar_path}'
        ], check=True, timeout=60)

        # Run MuseTalk inference on GPU server
        gpu_output_path = f"/tmp/avatar_output_{output_path.stem}.mp4"
        musetalk_cmd = f'''
        cd {MUSETALK_PATH} && \
        python3 run_inference.py \
            --audio {gpu_audio_path} \
            --image {gpu_avatar_path} \
            --output {gpu_output_path} \
            --version v15
        '''

        subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            '-p', str(GPU_PORT),
            f'{GPU_USER}@{GPU_HOST}',
            musetalk_cmd
        ], check=True, timeout=300)  # 5 min timeout for processing

        # Transfer video back from GPU server
        subprocess.run([
            'scp', '-o', 'StrictHostKeyChecking=no',
            '-P', str(GPU_PORT),
            f'{GPU_USER}@{GPU_HOST}:{gpu_output_path}',
            str(output_path)
        ], check=True, timeout=120)

        # Cleanup GPU server temp files
        cleanup_cmd = f"rm -f {gpu_audio_path} {gpu_avatar_path} {gpu_output_path}"
        subprocess.run([
            'ssh', '-o', 'StrictHostKeyChecking=no',
            '-p', str(GPU_PORT),
            f'{GPU_USER}@{GPU_HOST}',
            cleanup_cmd
        ], timeout=30)

        return True

    except subprocess.TimeoutExpired:
        current_app.logger.error("MuseTalk processing timed out")
        return False
    except subprocess.CalledProcessError as e:
        current_app.logger.error(f"MuseTalk processing failed: {e}")
        return False
    except Exception as e:
        current_app.logger.error(f"Unexpected error in lip-sync generation: {e}")
        return False


# =============================================================================
# API Routes
# =============================================================================

@avatar_bp.route('/api/avatar/speak', methods=['POST'])
def avatar_speak():
    """
    Generate avatar speech with optional lip-sync

    Request Body:
        {
            "text": "Hello world!",
            "voice_id": "21m00Tcm4TlvDq8ikWAM",  // optional
            "avatar_id": "default",  // optional
            "generate_video": true,  // optional, default false
            "voice_settings": {  // optional
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

    Response:
        {
            "audio_url": "/static/avatar/123_abc.mp3",
            "video_url": "/static/avatar/123_abc.mp4",  // if video generated
            "duration": 3.5,
            "processing_time": 2.1,
            "request_id": "123_abc",
            "has_video": true
        }
    """
    start_time = time.time()
    data = request.json or {}

    # Extract parameters
    text = data.get('text', '').strip()
    voice_id = data.get('voice_id', ELEVENLABS_DEFAULT_VOICE)
    avatar_id = data.get('avatar_id', 'default')
    generate_video = data.get('generate_video', False)
    voice_settings = data.get('voice_settings', {})

    # Validation
    if not text:
        return jsonify({'error': 'Text is required'}), 400

    if len(text) > 1000:
        return jsonify({'error': 'Text too long (max 1000 characters)'}), 400

    try:
        request_id = generate_request_id()

        # Step 1: Generate TTS audio with ElevenLabs
        audio_path = AVATAR_OUTPUT_DIR / f"{request_id}.mp3"
        audio_duration = generate_tts_audio(
            text=text,
            voice_id=voice_id,
            output_path=audio_path,
            stability=voice_settings.get('stability', 0.5),
            similarity_boost=voice_settings.get('similarity_boost', 0.75)
        )

        response_data = {
            'audio_url': f'/static/avatar/{request_id}.mp3',
            'duration': round(audio_duration, 2),
            'request_id': request_id,
            'has_video': False
        }

        # Step 2: Generate lip-sync video (if requested and available)
        if generate_video:
            video_path = AVATAR_OUTPUT_DIR / f"{request_id}.mp4"
            if generate_lipsync_video(audio_path, avatar_id, video_path):
                response_data['video_url'] = f'/static/avatar/{request_id}.mp4'
                response_data['has_video'] = True

        response_data['processing_time'] = round(time.time() - start_time, 2)

        return jsonify(response_data)

    except Exception as e:
        current_app.logger.error(f"Avatar speak error: {e}")
        return jsonify({'error': str(e)}), 500


@avatar_bp.route('/api/avatar/voices', methods=['GET'])
def list_voices():
    """
    List available ElevenLabs voices

    Response:
        {
            "voices": [
                {"id": "...", "name": "Rachel", "preview_url": "..."},
                ...
            ]
        }
    """
    global VOICE_CACHE

    # Return cached voices if available (cache for 1 hour)
    cache_key = 'voices'
    if cache_key in VOICE_CACHE:
        cached_time, cached_data = VOICE_CACHE[cache_key]
        if time.time() - cached_time < 3600:  # 1 hour
            return jsonify(cached_data)

    try:
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": ELEVENLABS_API_KEY}

        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch voices'}), 500

        voices_data = response.json().get('voices', [])

        # Simplify and filter
        voices = [
            {
                'id': v['voice_id'],
                'name': v['name'],
                'category': v.get('category', 'generated'),
                'preview_url': v.get('preview_url'),
                'labels': v.get('labels', {})
            }
            for v in voices_data[:30]  # Limit to 30 voices
        ]

        result = {'voices': voices}

        # Cache result
        VOICE_CACHE[cache_key] = (time.time(), result)

        return jsonify(result)

    except Exception as e:
        current_app.logger.error(f"List voices error: {e}")
        return jsonify({'error': str(e)}), 500


@avatar_bp.route('/api/avatar/avatars', methods=['GET'])
def list_avatars():
    """
    List available avatar reference images

    Response:
        {
            "avatars": [
                {"id": "default", "name": "Default Avatar", "url": "/static/avatars/default.png"},
                ...
            ]
        }
    """
    avatars = []

    if AVATAR_REFERENCE_DIR.exists():
        for img_path in AVATAR_REFERENCE_DIR.glob('*.png'):
            avatars.append({
                'id': img_path.stem,
                'name': img_path.stem.replace('_', ' ').title(),
                'url': f'/static/avatars/{img_path.name}'
            })

    # Ensure default exists
    if not avatars:
        avatars.append({
            'id': 'default',
            'name': 'Default Avatar',
            'url': '/static/avatars/default.png'
        })

    return jsonify({'avatars': avatars})


@avatar_bp.route('/api/avatar/status', methods=['GET'])
def avatar_status():
    """
    Get avatar system status

    Response:
        {
            "elevenlabs": true,
            "musetalk": false,
            "gpu_server": "82.141.118.40:2674"
        }
    """
    # Check ElevenLabs
    elevenlabs_ok = False
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/user",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=5
        )
        elevenlabs_ok = response.status_code == 200
    except Exception:
        pass

    # Check MuseTalk availability
    musetalk_ok = check_musetalk_available()

    return jsonify({
        'elevenlabs': elevenlabs_ok,
        'musetalk': musetalk_ok,
        'gpu_server': f"{GPU_HOST}:{GPU_PORT}",
        'output_dir': str(AVATAR_OUTPUT_DIR),
        'api_key_configured': bool(ELEVENLABS_API_KEY)
    })


# Static file serving for avatar outputs
@avatar_bp.route('/static/avatar/<path:filename>')
def serve_avatar_output(filename):
    """Serve generated avatar files (audio/video)"""
    return send_from_directory(AVATAR_OUTPUT_DIR, filename)


@avatar_bp.route('/static/avatars/<path:filename>')
def serve_avatar_reference(filename):
    """Serve avatar reference images"""
    return send_from_directory(AVATAR_REFERENCE_DIR, filename)
