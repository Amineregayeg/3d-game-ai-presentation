# Full Demo Implementation Plan
## Complete End-to-End AI Assistant Pipeline

**Status:** Planning
**Date:** December 18, 2024
**Target:** Complete interactive demo with all components

---

## 1. User Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         FULL DEMO USER EXPERIENCE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STEP 0: Page Load                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Avatar auto-greets user with video explaining:                           │   │
│  │ "Welcome! I'm your 3D Game AI Assistant. You can ask me to..."          │   │
│  │ - Create 3D objects in Blender                                           │   │
│  │ - Generate game assets                                                    │   │
│  │ - Answer questions about Blender/3D modeling                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                             │
│  STEP 1: User Settings                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ - STT Engine: [Whisper (GPU)] [VoxFormer (Custom)]                       │   │
│  │ - Avatar: [Avatar 1] [Avatar 2] [Avatar 3]                               │   │
│  │ - Voice: [Rachel] [Josh] [Aria] ...                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                             │
│  STEP 2: User Input (Voice OR Text)                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 🎤 "Create a low-poly medieval sword in Blender"                         │   │
│  │ ───OR───                                                                  │   │
│  │ ⌨️ Type: "Create a low-poly medieval sword in Blender"                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                             │
│  STEP 3: STT Transcription (if voice)                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Whisper/VoxFormer processes audio → "Create a low-poly medieval sword"   │   │
│  │ Display: Confidence, Duration, Word-level timestamps                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                             │
│  STEP 4: RAG Pipeline (Visible 9-stage process)                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ ○ Orchestrator → ○ Query Analysis → ○ Dense → ○ Sparse → ○ RRF          │   │
│  │ → ○ Reranking → ○ Context → ○ Generation → ○ Validation                 │   │
│  │                                                                           │   │
│  │ Retrieved Docs: [Blender Mesh API] [Low-Poly Tutorial] [Sword Guide]     │   │
│  │ RAGAS Metrics: Faith 94% | Relevancy 91% | Complete 88%                  │   │
│  │                                                                           │   │
│  │ Response: "I'll create a low-poly sword for you. Here's the plan..."    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                             │
│  STEP 5: TTS + Avatar Response (Parallel)                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ ElevenLabs generates speech → SadTalker creates lip-sync video          │   │
│  │ [▶ Avatar Video Player]                                                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                             │
│  STEP 6: Blender MCP Execution (Visible steps)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ MCP Server connecting... → Blender Addon ready                           │   │
│  │ ► execute_blender_code: Creating blade mesh...                           │   │
│  │ ► execute_blender_code: Creating handle...                               │   │
│  │ ► execute_blender_code: Applying material...                             │   │
│  │ ► get_viewport_screenshot: Capturing result...                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                             │
│  STEP 7: Three.js 3D Preview                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    ╔═══════════════════════════╗                         │   │
│  │                    ║   3D Sword Model View    ║                         │   │
│  │                    ║       🗡️ (rotatable)     ║                         │   │
│  │                    ╚═══════════════════════════╝                         │   │
│  │ Controls: [Rotate] [Zoom] [Pan] [Export GLB] [Export FBX]               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Components to Implement

### 2.1 New Components (in `/src/components/full-demo/`)

| Component | Purpose | Priority |
|-----------|---------|----------|
| `GreetingPlayer.tsx` | Auto-play greeting video on load | HIGH |
| `SettingsPanel.tsx` | STT/Avatar/Voice selection | HIGH |
| `VoiceInput.tsx` | Microphone recording with waveform | HIGH |
| `TextInput.tsx` | Text input alternative | HIGH |
| `STTPanel.tsx` | STT results with confidence/timing | HIGH |
| `RAGPipeline.tsx` | 9-stage pipeline visualization (from rag_demo) | HIGH |
| `DocumentsPanel.tsx` | Retrieved docs with scores | MEDIUM |
| `ResponsePanel.tsx` | Generated answer with citations | HIGH |
| `RAGMetrics.tsx` | RAGAS gauges (Faith/Relevancy/Complete) | MEDIUM |
| `AvatarPlayer.tsx` | Video player for avatar response | HIGH |
| `BlenderMCPPanel.tsx` | MCP command execution steps | MEDIUM |
| `ThreeJSViewport.tsx` | 3D model viewer | MEDIUM |
| `MetricsBar.tsx` | Latency/processing time bar | LOW |

### 2.2 API Routes Needed

| Route | Purpose | Backend |
|-------|---------|---------|
| `/api/gpu/health` | GPU status | ✅ EXISTS |
| `/api/gpu/stt` | Whisper transcription | ✅ EXISTS |
| `/api/gpu/avatars` | List avatars | ✅ EXISTS |
| `/api/gpu/lipsync` | SadTalker generation | ✅ EXISTS |
| `/api/rag/query` | RAG pipeline | ✅ EXISTS (VPS) |
| `/api/avatar/speak` | TTS generation | ✅ EXISTS (VPS) |
| `/api/blender/connect` | MCP connection | ❌ NEW |
| `/api/blender/execute` | Execute MCP command | ❌ NEW |
| `/api/blender/scene` | Get scene info | ❌ NEW |
| `/api/blender/screenshot` | Get viewport | ❌ NEW |
| `/api/greeting/video` | Pre-generated greeting | ❌ NEW |

### 2.3 Dependencies to Add

```bash
# Three.js for 3D viewport
npm install three @types/three @react-three/fiber @react-three/drei

# WebSocket for Blender MCP
npm install socket.io-client
```

---

## 3. Implementation Phases

### Phase 1: Enhanced UI & Greeting (Priority: HIGH)

**Duration:** 1-2 days

**Tasks:**
1. Create `GreetingPlayer.tsx` component
   - Auto-play pre-generated greeting video on page load
   - Video content: "Welcome! I can help you create 3D assets..."
   - Skip button, mute toggle
   - State: `hasSeenGreeting` in localStorage

2. Create `SettingsPanel.tsx` component
   - STT engine selector (Whisper vs VoxFormer)
   - Avatar selector with thumbnails
   - Voice selector dropdown
   - Collapsible/expandable design

3. Improve main page layout
   - Two-column responsive design
   - Left: Input + Settings
   - Right: Output panels (stacked)

**Files:**
- `/src/components/full-demo/GreetingPlayer.tsx`
- `/src/components/full-demo/SettingsPanel.tsx`
- `/src/app/full_demo/page.tsx` (refactor)

---

### Phase 2: RAG Pipeline Visualization (Priority: HIGH)

**Duration:** 1-2 days

**Tasks:**
1. Port RAG components from `rag_demo`:
   - `PipelineStage` component (9 stages with tooltips)
   - `MetricGauge` component (circular progress)
   - `DocumentCard` component (with score breakdown)
   - `CitationPreview` component (HoverCard)

2. Create `RAGPipeline.tsx` wrapper
   - Real-time stage updates via SSE or polling
   - Animated transitions between stages
   - Error state handling

3. Create `ResponsePanel.tsx`
   - Markdown rendering for response
   - Inline citations with hover preview
   - Copy button

**Files:**
- `/src/components/full-demo/RAGPipeline.tsx`
- `/src/components/full-demo/DocumentsPanel.tsx`
- `/src/components/full-demo/ResponsePanel.tsx`
- `/src/components/full-demo/RAGMetrics.tsx`

---

### Phase 3: Avatar Video Response (Priority: HIGH)

**Duration:** 1 day

**Tasks:**
1. Create `AvatarPlayer.tsx` component
   - Video player with controls
   - Loading skeleton while generating
   - Progress indicator for SadTalker
   - Audio fallback if video fails

2. Update pipeline to handle TTS + Lipsync
   - Sequential: TTS → Lipsync
   - Show timing for each step

3. Pre-generate greeting video
   - Script the greeting text
   - Generate with ElevenLabs + SadTalker
   - Store as static asset or generate on-demand

**Files:**
- `/src/components/full-demo/AvatarPlayer.tsx`
- `/public/videos/greeting.mp4` (pre-generated)
- `/src/app/api/greeting/video/route.ts`

---

### Phase 4: Blender MCP Integration (Priority: MEDIUM)

**Duration:** 2-3 days

**Tasks:**
1. Create backend Blender MCP proxy
   - Socket connection to Blender addon
   - Command queue management
   - Error handling and timeouts

2. Create `BlenderMCPPanel.tsx`
   - Connection status indicator
   - Command execution log (scrollable)
   - Each command with: status, duration, output
   - Collapsible code blocks

3. Integrate with RAG response
   - Parse RAG response for Blender commands
   - Execute sequentially
   - Show progress for each command

**Files:**
- `/src/app/api/blender/connect/route.ts`
- `/src/app/api/blender/execute/route.ts`
- `/src/app/api/blender/scene/route.ts`
- `/src/components/full-demo/BlenderMCPPanel.tsx`
- `/backend/blender_mcp.py` (VPS)

---

### Phase 5: Three.js 3D Viewport (Priority: MEDIUM)

**Duration:** 2-3 days

**Tasks:**
1. Install Three.js dependencies
   ```bash
   npm install three @types/three @react-three/fiber @react-three/drei
   ```

2. Create `ThreeJSViewport.tsx`
   - GLB/GLTF model loader
   - OrbitControls for rotation/zoom
   - Grid helper and lighting
   - Screenshot capture
   - Export buttons (GLB, FBX)

3. Connect to Blender
   - Load model after Blender export
   - Real-time sync option
   - Preview updates

**Files:**
- `/src/components/full-demo/ThreeJSViewport.tsx`
- `/src/components/full-demo/ModelControls.tsx`

---

### Phase 6: Polish & Integration (Priority: LOW)

**Duration:** 1-2 days

**Tasks:**
1. Create `MetricsBar.tsx`
   - Total latency breakdown
   - Per-stage timing
   - Animated bar chart

2. Error handling improvements
   - Graceful degradation
   - Retry mechanisms
   - User-friendly messages

3. Mobile responsiveness
   - Stack layout on small screens
   - Touch-friendly controls

4. Accessibility
   - Keyboard navigation
   - Screen reader support
   - Focus management

**Files:**
- `/src/components/full-demo/MetricsBar.tsx`
- Update all components for a11y

---

## 4. Backend Requirements

### 4.1 VPS Backend Updates (`/home/developer/3d-game-ai/backend/`)

**New file: `blender_mcp.py`**
```python
"""
Blender MCP Proxy
Connects to Blender addon via socket and executes commands
"""

import socket
import json
from flask import Blueprint, request, jsonify

blender_bp = Blueprint('blender', __name__)

BLENDER_HOST = 'localhost'
BLENDER_PORT = 9876
blender_socket = None

@blender_bp.route('/api/blender/connect', methods=['POST'])
def connect():
    global blender_socket
    try:
        blender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blender_socket.settimeout(10)
        blender_socket.connect((BLENDER_HOST, BLENDER_PORT))
        return jsonify({"status": "connected"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@blender_bp.route('/api/blender/execute', methods=['POST'])
def execute():
    if not blender_socket:
        return jsonify({"error": "Not connected"}), 400

    data = request.json
    command = data.get('command')
    params = data.get('params', {})

    try:
        message = json.dumps({"type": command, "params": params})
        blender_socket.sendall(message.encode('utf-8'))

        response = blender_socket.recv(65536).decode('utf-8')
        return jsonify(json.loads(response))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@blender_bp.route('/api/blender/scene', methods=['GET'])
def get_scene():
    # Execute get_scene_info command
    return execute_command("get_scene_info", {})

@blender_bp.route('/api/blender/screenshot', methods=['GET'])
def get_screenshot():
    # Execute get_viewport_screenshot command
    return execute_command("get_viewport_screenshot", {"max_size": 800})
```

### 4.2 GPU Server Requirements

Already have in `full_demo_api.py`:
- ✅ Whisper STT
- ✅ SadTalker lip-sync
- ✅ ElevenLabs TTS proxy

Need to verify:
- ffmpeg audio conversion working
- Avatar images available

---

## 5. State Management

### 5.1 Main Page State

```typescript
interface FullDemoState {
  // Settings
  sttEngine: 'whisper' | 'voxformer';
  selectedAvatar: string;
  selectedVoice: string;

  // Pipeline stage
  currentStage: 'idle' | 'greeting' | 'recording' | 'stt' | 'rag' | 'tts' | 'lipsync' | 'blender' | 'complete' | 'error';

  // Results
  sttResult: STTResult | null;
  ragResult: RAGResult | null;
  ragStages: StageResult[];
  ragDocuments: Document[];
  ragMetrics: RAGMetrics | null;
  ttsResult: TTSResult | null;
  lipsyncResult: LipsyncResult | null;
  blenderCommands: BlenderCommand[];
  modelUrl: string | null;

  // UI state
  hasSeenGreeting: boolean;
  isSettingsOpen: boolean;
  error: string | null;
}
```

---

## 6. Pre-generated Assets Needed

### 6.1 Greeting Video

**Script:**
```
"Hello! Welcome to the 3D Game AI Assistant. I'm here to help you create
amazing 3D assets for your games. You can ask me to create objects in
Blender, like swords, shields, or even complex environments. Just click
the microphone and tell me what you'd like to create, or type your request
below. Let's build something amazing together!"
```

**Generation Steps:**
1. Generate audio with ElevenLabs (Rachel voice)
2. Create lip-sync video with SadTalker
3. Store as `/public/videos/greeting.mp4`

---

## 7. Testing Checklist

- [ ] Page loads and shows greeting video
- [ ] Settings panel works (STT/Avatar/Voice)
- [ ] Voice recording captures audio
- [ ] STT transcribes correctly (Whisper)
- [ ] STT transcribes correctly (VoxFormer)
- [ ] Text input works as alternative
- [ ] RAG pipeline shows all 9 stages
- [ ] Document cards display with scores
- [ ] RAGAS metrics show correctly
- [ ] TTS generates audio
- [ ] SadTalker creates video
- [ ] Avatar video plays correctly
- [ ] Blender MCP connects (when available)
- [ ] Blender commands execute
- [ ] Three.js viewport loads model
- [ ] Export buttons work
- [ ] Error states handled gracefully
- [ ] Mobile layout works
- [ ] Keyboard navigation works

---

## 8. Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: UI & Greeting | 1-2 days | None |
| Phase 2: RAG Pipeline | 1-2 days | Phase 1 |
| Phase 3: Avatar Response | 1 day | Phase 1 |
| Phase 4: Blender MCP | 2-3 days | Blender addon running |
| Phase 5: Three.js Viewport | 2-3 days | Phase 4 |
| Phase 6: Polish | 1-2 days | All phases |

**Total:** ~8-13 days

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Blender not available | Mock MCP responses, show demo video |
| GPU server down | Fallback to demo mode, cached responses |
| STT fails | Text input as alternative |
| TTS fails | Show text response only |
| Lipsync slow | Show audio immediately, video when ready |
| Three.js model fails | Show Blender screenshot instead |

---

## 10. Success Criteria

1. **User can complete full flow** from voice input to 3D output
2. **All pipeline stages visible** with real-time progress
3. **Avatar speaks responses** with lip-synced video
4. **Blender integration works** (or graceful fallback)
5. **3D model viewable** in Three.js viewport
6. **Metrics displayed** for all stages
7. **Error handling** is graceful and informative
8. **Mobile friendly** layout works

---

**Document Version:** 1.0
**Last Updated:** December 18, 2024
