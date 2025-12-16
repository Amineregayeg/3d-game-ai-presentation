"use client";

import { DocPageLayout } from "@/components/docs/DocPageLayout";

const ttsIcon = (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
  </svg>
);

const phases = [
  {
    name: "Phase 1: Core TTS Integration",
    duration: "Week 1",
    tasks: [
      "Set up ElevenLabs API authentication",
      "Implement text-to-speech streaming",
      "Create voice selection interface",
      "Add audio buffer management",
      "Test latency with various text lengths",
    ],
    milestone: "ElevenLabs produces streamed audio output",
  },
  {
    name: "Phase 2: Lip-Sync Integration",
    duration: "Week 2",
    tasks: [
      "Set up Wav2Lip inference pipeline",
      "Implement audio-to-viseme mapping",
      "Create phoneme extraction from audio",
      "Build viseme blending system",
      "Test with avatar face mesh",
    ],
    milestone: "Avatar mouth syncs to audio",
  },
  {
    name: "Phase 3: Game Engine Bridge",
    duration: "Week 3",
    tasks: [
      "Implement Unity C# integration",
      "Implement Unreal Engine C++ integration",
      "Create WebSocket streaming protocol",
      "Build blend shape controller",
      "Test real-time performance",
    ],
    milestone: "Game engine avatars animate with speech",
  },
  {
    name: "Phase 4: Testing & Optimization",
    duration: "Week 4",
    tasks: [
      "Benchmark end-to-end latency",
      "Optimize audio streaming buffer",
      "Reduce lip-sync processing time",
      "Add fallback for network issues",
      "Test multi-language support",
    ],
    milestone: "< 200ms total latency achieved",
  },
  {
    name: "Phase 5: Production Hardening",
    duration: "Week 5",
    tasks: [
      "Implement error handling and retry logic",
      "Add logging and monitoring",
      "Create configuration management",
      "Write integration tests",
      "Document API and usage",
    ],
    milestone: "Production-ready deployment",
  },
];

const sections = [
  {
    title: "Overview",
    content: `The TTS + LipSync system enables realistic avatar speech animation by combining high-quality text-to-speech with precise lip synchronization. The pipeline flows from text input through audio generation to real-time blend shape animation.

Key features:
- ElevenLabs streaming TTS for natural voice synthesis
- Wav2Lip/SadTalker for neural lip-sync
- 22-viseme ARKit-compatible mapping
- Unity and Unreal Engine integration
- < 200ms end-to-end latency`,
  },
  {
    title: "ElevenLabs TTS",
    content: `ElevenLabs provides state-of-the-art text-to-speech with streaming support:

1. API Features:
   - WebSocket streaming for low latency
   - 30+ high-quality voices
   - Voice cloning capability
   - Emotion and style control

2. Streaming Pipeline:
   - Text sent via WebSocket
   - Audio chunks returned in real-time
   - MP3 or PCM format options
   - Configurable chunk size

3. Voice Selection:
   - Pre-built voices for different personalities
   - Custom voice cloning (requires samples)
   - Per-character voice assignment`,
    code: `# ElevenLabs Streaming
import websockets
import json

async def stream_tts(text, voice_id):
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "text": text,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }))
        async for message in ws:
            yield message  # Audio chunk`,
  },
  {
    title: "Lip-Sync Pipeline",
    content: `Two approaches for lip synchronization:

1. Wav2Lip (Preferred):
   - Neural network-based sync
   - Works with any audio input
   - Outputs video frames or blend shapes
   - ~30ms per frame inference

2. SadTalker (Alternative):
   - More expressive head movement
   - 3DMM face model output
   - Higher computational cost
   - Better for cinematic sequences

3. Phoneme-to-Viseme Mapping:
   - Extract phonemes from audio
   - Map to 22 ARKit visemes
   - Interpolate between keyframes`,
  },
  {
    title: "Viseme System",
    content: `22-viseme ARKit-compatible mapping for facial animation:

Standard Visemes:
- sil (silence)
- PP (p, b, m)
- FF (f, v)
- TH (th)
- DD (t, d)
- kk (k, g)
- CH (ch, j, sh)
- SS (s, z)
- nn (n, l)
- RR (r)
- aa (a)
- E (e)
- ih (i)
- oh (o)
- ou (u)

Blend Shape Weights:
- Each viseme maps to multiple blend shapes
- Smooth interpolation between visemes
- 60fps target for animation`,
    code: `// Unity Blend Shape Controller
void ApplyViseme(Viseme viseme, float weight) {
    var blendShapes = visemeToBlendShapes[viseme];
    foreach (var bs in blendShapes) {
        skinnedMeshRenderer.SetBlendShapeWeight(
            bs.index,
            bs.baseWeight * weight
        );
    }
}`,
  },
  {
    title: "Game Engine Integration",
    content: `Unity and Unreal Engine integration via WebSocket streaming:

Unity (C#):
- WebSocket client for audio streaming
- Audio Source for playback
- Skinned Mesh Renderer for blend shapes
- Coroutine-based animation loop

Unreal Engine (C++):
- WebSocket module for streaming
- Sound Wave for audio
- Morph Target for blend shapes
- Async task for processing

Streaming Protocol:
- Binary audio chunks (PCM/MP3)
- JSON viseme keyframes
- Timestamp synchronization`,
  },
  {
    title: "Latency Optimization",
    content: `Target: < 200ms end-to-end latency

Breakdown:
1. TTS API: ~100ms (streaming)
2. Audio Processing: ~30ms
3. Lip-Sync Inference: ~30ms
4. Network Transfer: ~20ms
5. Render: ~16ms (60fps)

Optimization Strategies:
- Pre-buffer audio chunks
- GPU acceleration for inference
- Predictive viseme generation
- Parallel processing pipeline`,
  },
];

const apiReference = [
  {
    endpoint: "/api/tts/generate",
    method: "POST",
    description: "Generate speech audio from text",
    parameters: [
      { name: "text", type: "string", description: "Text to convert to speech" },
      { name: "voice_id", type: "string", description: "ElevenLabs voice ID" },
      { name: "stream", type: "bool", description: "Enable streaming (default: true)" },
    ],
  },
  {
    endpoint: "/api/lipsync/process",
    method: "POST",
    description: "Generate lip-sync data from audio",
    parameters: [
      { name: "audio", type: "binary", description: "Audio file (WAV/MP3)" },
      { name: "method", type: "string", description: "wav2lip or sadtalker (default: wav2lip)" },
      { name: "fps", type: "int", description: "Output frame rate (default: 60)" },
    ],
  },
  {
    endpoint: "/api/avatar/speak",
    method: "POST",
    description: "Full pipeline: text to animated avatar",
    parameters: [
      { name: "text", type: "string", description: "Text for avatar to speak" },
      { name: "voice_id", type: "string", description: "Voice selection" },
      { name: "avatar_id", type: "string", description: "Target avatar mesh" },
    ],
  },
  {
    endpoint: "ws://api/stream",
    method: "WS",
    description: "WebSocket for real-time audio + viseme streaming",
    parameters: [
      { name: "format", type: "string", description: "Audio format: pcm or mp3" },
      { name: "viseme_format", type: "string", description: "arkit or custom" },
    ],
  },
];

export default function TTSLipsyncDocPage() {
  return (
    <DocPageLayout
      title="TTS + LipSync"
      subtitle="Text-to-Speech & Avatar Animation"
      description="Real-time avatar speech animation combining ElevenLabs streaming TTS with Wav2Lip neural lip-sync. Features 22-viseme ARKit mapping and native Unity/UE5 integration for game engines."
      gradient="from-rose-500 to-pink-600"
      accentColor="bg-rose-500"
      icon={ttsIcon}
      presentationLink="/avatar"
      technologies={["ElevenLabs", "Wav2Lip", "SadTalker", "WebSocket", "ARKit", "Unity", "Unreal Engine", "Blend Shapes"]}
      phases={phases}
      sections={sections}
      apiReference={apiReference}
    />
  );
}
