# Full Demo Page - Technical Implementation Plan

**Created:** December 17, 2024
**Status:** APPROVED - READY FOR IMPLEMENTATION
**Route:** `/full_demo`

---

## 1. Overview

### Vision
A single, comprehensive demo page that showcases the **complete AI pipeline** for a 3D Game AI Assistant. The avatar is the central character - it greets the user, guides them, and responds intelligently while executing 3D commands in real-time.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE FLOW                           │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  USER    │    │   STT    │    │   RAG    │    │  AVATAR  │  │
│  │  SPEECH  │───>│ VoxFormer│───>│  Query   │───>│ Response │  │
│  │          │    │ /Whisper │    │ Pipeline │    │ + Video  │  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│                                                       │        │
│                                        ┌──────────────┘        │
│                                        ▼                       │
│                                   ┌──────────┐                 │
│                                   │ BLENDER  │                 │
│                                   │ 3D Exec  │                 │
│                                   └──────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

### User Experience Flow

#### Phase 1: Welcome (Automatic)
1. Page loads → Avatar appears with greeting animation
2. Avatar speaks: *"Hello! I'm your 3D Game AI Assistant. I can help you create 3D models, materials, and animations in Blender. Just click the microphone and tell me what you'd like to create!"*
3. 3D viewport shows empty Blender scene

#### Phase 2: User Interaction
1. User clicks microphone → Recording starts
2. User speaks: *"Create a low-poly tree with green leaves"*
3. User releases → Audio sent for transcription

#### Phase 3: Processing (Parallel)
1. **STT** transcribes speech → Shows confidence metrics
2. **RAG** processes query → Shows pipeline stages
3. Both complete → Avatar starts responding

#### Phase 4: Response + Execution (Simultaneous)
1. Avatar begins speaking explanation (video plays)
2. While avatar speaks, Blender commands execute in 3D viewport
3. User sees tree being built in real-time as avatar explains each step
4. Citations appear for referenced documentation

#### Phase 5: Complete
1. Avatar finishes: *"Your low-poly tree is ready! Would you like me to add textures or modify anything?"*
2. 3D viewport shows final result
3. User can continue conversation or download .blend file

---

## 2. Page Layout

### Responsive Grid Layout (Desktop)
```
┌─────────────────────────────────────────────────────────────────────┐
│                        HEADER / TITLE BAR                           │
│  "3D Game AI Assistant - Full Pipeline Demo"          [Settings ⚙️] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────┐  ┌────────────────────────────────────┐ │
│  │                       │  │                                    │ │
│  │   AVATAR DISPLAY      │  │       BLENDER VIEWPORT             │ │
│  │   (Video/Image)       │  │       (Live 3D Preview)            │ │
│  │                       │  │                                    │ │
│  │   [Speaking...]       │  │       [MCP Status: Connected]      │ │
│  └───────────────────────┘  └────────────────────────────────────┘ │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    INPUT SECTION                               │ │
│  │  ┌─────────────────────────────────────────────────────────┐  │ │
│  │  │  🎤 [Record] [Stop]  │  STT: [VoxFormer ▼] [Whisper]    │  │ │
│  │  │  Waveform Visualization                                  │  │ │
│  │  └─────────────────────────────────────────────────────────┘  │ │
│  │                                                               │ │
│  │  Transcription: "Create a low-poly tree with green leaves"   │ │
│  │  Confidence: 94.2% | RTF: 0.32 | Words: 8                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐ │
│  │    RAG PIPELINE         │  │      RESPONSE & EXECUTION        │ │
│  │  ┌─────────────────┐   │  │                                  │ │
│  │  │ 1. Orchestrator │✓  │  │  Avatar Response:                │ │
│  │  │ 2. Query Anal.  │✓  │  │  "I'll create a low-poly tree   │ │
│  │  │ 3. Dense Search │●  │  │   using a cylinder trunk and     │ │
│  │  │ 4. Sparse Search│○  │  │   ico-sphere leaves..."          │ │
│  │  │ 5. RRF Fusion   │○  │  │                                  │ │
│  │  │ 6. Reranking    │○  │  │  [Citations: [1] [2] [3]]        │ │
│  │  │ 7. Generation   │○  │  │                                  │ │
│  │  │ 8. Validation   │○  │  │  Blender Commands:               │ │
│  │  └─────────────────┘   │  │  ► bpy.ops.mesh.primitive_...    │ │
│  │                         │  │  ► bpy.context.object.scale...  │ │
│  │  Documents: 5 retrieved │  │  ► bpy.ops.object.modifier_...  │ │
│  │  Latency: 1.2s          │  │                                  │ │
│  └─────────────────────────┘  └──────────────────────────────────┘ │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        METRICS PANEL                                │
│  STT: 94.2% conf | RAG: 0.92 RAGAS | Latency: 3.2s | GPU: 78%      │
└─────────────────────────────────────────────────────────────────────┘
```

### Mobile Layout (Stacked)
```
┌─────────────────────┐
│      AVATAR         │
├─────────────────────┤
│   VOICE INPUT       │
│   + Transcription   │
├─────────────────────┤
│   RAG PIPELINE      │
├─────────────────────┤
│   RESPONSE          │
├─────────────────────┤
│   BLENDER (Mini)    │
├─────────────────────┤
│   METRICS           │
└─────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Main Page Component
```typescript
// /src/app/full_demo/page.tsx

interface FullDemoState {
  // Pipeline stage
  stage: 'idle' | 'recording' | 'transcribing' | 'processing' | 'responding' | 'executing';

  // STT
  sttEngine: 'voxformer' | 'whisper';
  audioBlob: Blob | null;
  transcription: TranscriptionResult | null;

  // RAG
  ragQuery: string;
  ragResponse: RAGResponse | null;
  ragStages: PipelineStage[];

  // Avatar
  avatarResponse: AvatarResponse | null;
  isAvatarSpeaking: boolean;

  // Blender
  blenderConnected: boolean;
  blenderCommands: BlenderCommand[];
  blenderPreview: string | null; // Base64 render

  // Settings
  settings: DemoSettings;
}

interface DemoSettings {
  sttEngine: 'voxformer' | 'whisper';
  voiceId: string;
  avatarId: string;
  enableBlender: boolean;
  autoExecute: boolean;
  showDebug: boolean;
}
```

### 3.2 Sub-Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `VoiceInput` | `/src/components/full-demo/VoiceInput.tsx` | Audio recording + waveform |
| `STTPanel` | `/src/components/full-demo/STTPanel.tsx` | Engine selection + metrics |
| `RAGPipeline` | `/src/components/full-demo/RAGPipeline.tsx` | Stage visualization |
| `AvatarDisplay` | `/src/components/full-demo/AvatarDisplay.tsx` | Video/image + speaking state |
| `BlenderViewport` | `/src/components/full-demo/BlenderViewport.tsx` | 3D preview + MCP status |
| `ResponsePanel` | `/src/components/full-demo/ResponsePanel.tsx` | Text + citations + commands |
| `MetricsBar` | `/src/components/full-demo/MetricsBar.tsx` | Combined metrics display |
| `SettingsDrawer` | `/src/components/full-demo/SettingsDrawer.tsx` | Configuration panel |

---

## 4. API Endpoints

### 4.1 New Endpoints Required

#### `/api/full-demo/pipeline` (POST)
Orchestrates the entire pipeline in one request.

```typescript
// Request
{
  audio: string;           // Base64 audio
  stt_engine: 'voxformer' | 'whisper';
  voice_id: string;
  avatar_id: string;
  enable_blender: boolean;
  auto_execute: boolean;
}

// Response (streamed via SSE)
{
  stage: string;
  data: {
    transcription?: TranscriptionResult;
    rag_response?: RAGResponse;
    avatar_response?: AvatarResponse;
    blender_commands?: BlenderCommand[];
    blender_result?: BlenderResult;
  };
  progress: number;        // 0-100
  timestamp: string;
}
```

#### `/api/stt/whisper` (POST)
Whisper transcription endpoint (alternative to VoxFormer).

```typescript
// Request
{
  audio: string;           // Base64 audio
  language?: string;       // Optional language hint
}

// Response
{
  text: string;
  segments: WhisperSegment[];
  language: string;
  duration: number;
  processing_time: number;
}
```

#### `/api/blender/execute` (POST)
Execute Blender commands via MCP.

```typescript
// Request
{
  commands: BlenderCommand[];
  render_preview: boolean;
}

// Response
{
  success: boolean;
  results: CommandResult[];
  preview_url?: string;    // Rendered preview
  error?: string;
}
```

#### `/api/blender/status` (GET)
Check Blender MCP connection status.

```typescript
// Response
{
  connected: boolean;
  blender_version?: string;
  scene_info?: {
    objects: number;
    selected: string[];
  };
}
```

### 4.2 Existing Endpoints to Reuse

| Endpoint | From | Use Case |
|----------|------|----------|
| `POST /api/transcribe` | `/demo` | VoxFormer STT |
| `POST /api/rag/query` | `/rag_demo` | RAG processing |
| `POST /api/avatar/speak` | `/avatar_demo` | TTS + lip-sync |
| `GET /api/avatar/voices` | `/avatar_demo` | Voice list |
| `GET /api/avatar/avatars` | `/avatar_demo` | Avatar list |

---

## 5. Backend Services

### 5.1 STT Service (GPU)

#### VoxFormer (Existing)
- **Location:** GPU server via SSH
- **Model:** Custom trained hybrid CTC-Attention
- **Metrics:** Confidence, RTF, word-level timing

#### Whisper (New)
- **Location:** Same GPU server
- **Model:** `openai/whisper-large-v3` or `faster-whisper`
- **Metrics:** Language detection, segment timestamps

```python
# /backend/stt_service.py

class STTService:
    def __init__(self):
        self.voxformer = VoxFormerInference()
        self.whisper = WhisperInference()

    def transcribe(self, audio_path: str, engine: str = 'voxformer'):
        if engine == 'voxformer':
            return self.voxformer.transcribe(audio_path)
        else:
            return self.whisper.transcribe(audio_path)
```

### 5.2 Blender MCP Service (New)

```python
# /backend/blender_mcp.py

class BlenderMCP:
    """Model Context Protocol client for Blender integration."""

    def __init__(self, blender_path: str = None):
        self.connected = False
        self.socket = None

    async def connect(self) -> bool:
        """Connect to Blender MCP server."""
        pass

    async def execute_command(self, command: str) -> dict:
        """Execute a single bpy command."""
        pass

    async def execute_script(self, script: str) -> dict:
        """Execute a full Python script in Blender."""
        pass

    async def get_render_preview(self, width: int = 512, height: int = 512) -> bytes:
        """Render current scene and return as PNG bytes."""
        pass

    async def get_scene_info(self) -> dict:
        """Get current scene information."""
        pass
```

### 5.3 Pipeline Orchestrator (New)

```python
# /backend/pipeline_orchestrator.py

class PipelineOrchestrator:
    """Orchestrates the full demo pipeline."""

    def __init__(self):
        self.stt_service = STTService()
        self.rag_service = RAGService()
        self.avatar_service = AvatarService()
        self.blender_mcp = BlenderMCP()

    async def run_pipeline(
        self,
        audio: bytes,
        stt_engine: str,
        voice_id: str,
        avatar_id: str,
        enable_blender: bool,
        auto_execute: bool
    ) -> AsyncGenerator[PipelineEvent, None]:
        """
        Run the full pipeline with streaming updates.

        Yields PipelineEvent at each stage:
        1. STT transcription
        2. RAG query processing
        3. Avatar response generation
        4. Blender command extraction
        5. Blender execution (if enabled)
        """

        # Stage 1: STT
        yield PipelineEvent('stt_start', progress=0)
        transcription = await self.stt_service.transcribe(audio, stt_engine)
        yield PipelineEvent('stt_complete', data=transcription, progress=20)

        # Stage 2: RAG
        yield PipelineEvent('rag_start', progress=20)
        rag_response = await self.rag_service.query(transcription.text)
        yield PipelineEvent('rag_complete', data=rag_response, progress=50)

        # Stage 3: Avatar
        yield PipelineEvent('avatar_start', progress=50)
        avatar_response = await self.avatar_service.speak(
            text=rag_response.answer,
            voice_id=voice_id,
            avatar_id=avatar_id
        )
        yield PipelineEvent('avatar_complete', data=avatar_response, progress=70)

        # Stage 4: Extract Blender commands
        blender_commands = self.extract_blender_commands(rag_response.answer)
        yield PipelineEvent('commands_extracted', data=blender_commands, progress=80)

        # Stage 5: Execute in Blender (if enabled)
        if enable_blender and auto_execute and blender_commands:
            yield PipelineEvent('blender_start', progress=80)
            blender_result = await self.blender_mcp.execute_commands(blender_commands)
            preview = await self.blender_mcp.get_render_preview()
            yield PipelineEvent('blender_complete', data={
                'result': blender_result,
                'preview': preview
            }, progress=100)
        else:
            yield PipelineEvent('pipeline_complete', progress=100)
```

---

## 6. Data Flow

### 6.1 Sequence Diagram

```
┌──────┐     ┌──────────┐     ┌─────┐     ┌─────┐     ┌────────┐     ┌─────────┐
│ User │     │ Frontend │     │ API │     │ STT │     │  RAG   │     │ Avatar  │
└──┬───┘     └────┬─────┘     └──┬──┘     └──┬──┘     └───┬────┘     └────┬────┘
   │              │              │           │            │               │
   │ Speak        │              │           │            │               │
   │─────────────>│              │           │            │               │
   │              │              │           │            │               │
   │              │ POST /pipeline           │            │               │
   │              │─────────────>│           │            │               │
   │              │              │           │            │               │
   │              │   SSE: stt_start         │            │               │
   │              │<─────────────│           │            │               │
   │              │              │ transcribe│            │               │
   │              │              │──────────>│            │               │
   │              │              │<──────────│            │               │
   │              │   SSE: stt_complete      │            │               │
   │              │<─────────────│           │            │               │
   │              │              │           │            │               │
   │              │   SSE: rag_start         │            │               │
   │              │<─────────────│           │            │               │
   │              │              │ query     │            │               │
   │              │              │───────────────────────>│               │
   │              │              │<───────────────────────│               │
   │              │   SSE: rag_complete      │            │               │
   │              │<─────────────│           │            │               │
   │              │              │           │            │               │
   │              │   SSE: avatar_start      │            │               │
   │              │<─────────────│           │            │               │
   │              │              │ speak     │            │               │
   │              │              │────────────────────────────────────────>│
   │              │              │<────────────────────────────────────────│
   │              │   SSE: avatar_complete   │            │               │
   │              │<─────────────│           │            │               │
   │              │              │           │            │               │
   │ Show Avatar  │              │           │            │               │
   │<─────────────│              │           │            │               │
   │              │              │           │            │               │
```

### 6.2 State Machine

```
                    ┌─────────┐
                    │  IDLE   │
                    └────┬────┘
                         │ User clicks Record
                         ▼
                    ┌─────────┐
                    │RECORDING│
                    └────┬────┘
                         │ User clicks Stop
                         ▼
               ┌─────────────────┐
               │  TRANSCRIBING   │
               └────────┬────────┘
                        │ STT complete
                        ▼
               ┌─────────────────┐
               │   PROCESSING    │──────┐
               │   (RAG Query)   │      │ Error
               └────────┬────────┘      │
                        │ RAG complete  ▼
                        ▼          ┌─────────┐
               ┌─────────────────┐ │  ERROR  │
               │   RESPONDING    │ └─────────┘
               │ (Avatar TTS)    │
               └────────┬────────┘
                        │ Avatar complete
                        ▼
               ┌─────────────────┐
               │   EXECUTING     │ (if Blender enabled)
               │ (Blender MCP)   │
               └────────┬────────┘
                        │ Complete
                        ▼
                    ┌─────────┐
                    │  DONE   │───> IDLE (reset)
                    └─────────┘
```

---

## 7. UI Components Detail

### 7.1 VoiceInput Component

```typescript
interface VoiceInputProps {
  onAudioCapture: (blob: Blob) => void;
  isRecording: boolean;
  isProcessing: boolean;
  sttEngine: 'voxformer' | 'whisper';
  onEngineChange: (engine: 'voxformer' | 'whisper') => void;
}

// Features:
// - Animated waveform during recording
// - Visual feedback for levels
// - Record/Stop button with state
// - Engine toggle (VoxFormer/Whisper)
// - Upload fallback option
```

### 7.2 STTPanel Component

```typescript
interface STTPanelProps {
  transcription: TranscriptionResult | null;
  isTranscribing: boolean;
  engine: 'voxformer' | 'whisper';
}

// Display:
// - Transcribed text with word highlighting
// - Confidence scores (color-coded)
// - Metrics: RTF, processing time, word count
// - Engine badge (VoxFormer/Whisper)
// - Word-level confidence breakdown (expandable)
```

### 7.3 RAGPipeline Component

```typescript
interface RAGPipelineProps {
  stages: PipelineStage[];
  activeStage: number;
  documents: Document[];
  metrics: RAGMetrics;
}

// Display:
// - 8 pipeline stages with status icons
// - Active stage highlighting
// - Retrieved documents (collapsed by default)
// - RAGAS scores visualization
// - Latency breakdown
```

### 7.4 AvatarDisplay Component

```typescript
interface AvatarDisplayProps {
  avatarId: string;
  videoUrl: string | null;
  audioUrl: string | null;
  isSpeaking: boolean;
  response: string;
}

// Features:
// - Video player with lip-sync
// - Fallback to static image + audio
// - Speaking indicator animation
// - Response text overlay (optional)
// - Avatar selector dropdown
```

### 7.5 BlenderViewport Component

```typescript
interface BlenderViewportProps {
  connected: boolean;
  previewUrl: string | null;
  commands: BlenderCommand[];
  isExecuting: boolean;
  onExecute: (commands: BlenderCommand[]) => void;
}

// Features:
// - Live render preview image
// - Connection status badge
// - Command list with syntax highlighting
// - Execute button (manual trigger)
// - Error display if failed
```

### 7.6 MetricsBar Component

```typescript
interface MetricsBarProps {
  sttMetrics: {
    confidence: number;
    rtf: number;
    engine: string;
  };
  ragMetrics: {
    ragas: number;
    latency: number;
    documentsRetrieved: number;
  };
  totalLatency: number;
  gpuUtilization: number;
}

// Display:
// - Compact horizontal bar
// - Key metrics with icons
// - Color-coded status indicators
// - Expandable for details
```

---

## 8. Styling & Theme

### Color Scheme
```css
/* Primary gradient - Cyan to Purple (matches /technical) */
--gradient-primary: linear-gradient(135deg, #06b6d4, #8b5cf6);

/* Background */
--bg-dark: #0f172a;
--bg-card: rgba(30, 41, 59, 0.8);

/* Status colors */
--status-success: #22c55e;
--status-warning: #f59e0b;
--status-error: #ef4444;
--status-processing: #3b82f6;

/* Accents */
--accent-cyan: #06b6d4;
--accent-purple: #8b5cf6;
--accent-emerald: #10b981;
```

### Animation Library
- **Framer Motion** for component transitions
- **CSS animations** for loading states
- **Canvas** for waveform visualization

---

## 9. Implementation Phases

### Phase 1: Core Layout & STT (Day 1)
- [ ] Create `/full_demo` page with layout grid
- [ ] Implement `VoiceInput` component (reuse from `/demo`)
- [ ] Add STT engine toggle (VoxFormer/Whisper)
- [ ] Create `/api/stt/whisper` endpoint
- [ ] Implement `STTPanel` with metrics

### Phase 2: RAG Integration (Day 1-2)
- [ ] Implement `RAGPipeline` component (adapt from `/rag_demo`)
- [ ] Connect to existing `/api/rag/query`
- [ ] Add document display with citations
- [ ] Implement RAGAS visualization

### Phase 3: Avatar Integration (Day 2)
- [ ] Implement `AvatarDisplay` component (adapt from `/avatar_demo`)
- [ ] Connect to existing `/api/avatar/speak`
- [ ] Add voice/avatar selection
- [ ] Handle video playback + fallback

### Phase 4: Blender MCP (Day 2-3)
- [ ] Implement Blender MCP backend service
- [ ] Create `/api/blender/execute` endpoint
- [ ] Implement `BlenderViewport` component
- [ ] Add command extraction from RAG response
- [ ] Handle render preview

### Phase 5: Pipeline Orchestration (Day 3)
- [ ] Create `PipelineOrchestrator` backend
- [ ] Implement SSE streaming for `/api/full-demo/pipeline`
- [ ] Add state machine in frontend
- [ ] Connect all components

### Phase 6: Polish & Testing (Day 3-4)
- [ ] Add error handling throughout
- [ ] Implement loading states
- [ ] Add settings drawer
- [ ] Mobile responsive layout
- [ ] Performance optimization
- [ ] End-to-end testing

---

## 10. Dependencies

### Frontend
```json
{
  "dependencies": {
    "framer-motion": "^10.x",  // Already installed
    "lucide-react": "^0.x",    // Already installed
    "@radix-ui/*": "^1.x"      // Already installed
  }
}
```

### Backend (New)
```python
# requirements.txt additions
faster-whisper>=0.10.0    # For Whisper STT
blender-mcp>=0.1.0        # For Blender integration (or custom)
```

### External Services
- **VoxFormer**: GPU server (existing)
- **Whisper**: Same GPU server (new deployment)
- **RAG**: Existing backend
- **ElevenLabs**: Existing integration
- **MuseTalk**: Existing integration
- **Blender MCP**: New setup required

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU server unavailable | High | Fallback to demo mode with mock data |
| Blender MCP connection fails | Medium | Make Blender optional, show commands only |
| High latency (>10s) | Medium | Show progress at each stage, streaming updates |
| Audio recording fails | Medium | Provide file upload alternative |
| ElevenLabs rate limit | Low | Queue requests, show fallback avatar |

---

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| End-to-end latency | < 8 seconds |
| STT accuracy (WER) | < 15% (VoxFormer), < 10% (Whisper) |
| RAG relevance (RAGAS) | > 0.85 |
| UI responsiveness | < 100ms interactions |
| Mobile usability | Fully functional on tablet+ |

---

## 13. File Structure

```
/src/app/full_demo/
├── page.tsx                    # Main page component
├── layout.tsx                  # Layout with metadata
└── loading.tsx                 # Loading skeleton

/src/components/full-demo/
├── VoiceInput.tsx              # Audio recording
├── STTPanel.tsx                # Transcription display
├── RAGPipeline.tsx             # Pipeline visualization
├── AvatarDisplay.tsx           # Avatar video/audio
├── BlenderViewport.tsx         # 3D preview
├── ResponsePanel.tsx           # Answer + commands
├── MetricsBar.tsx              # Combined metrics
├── SettingsDrawer.tsx          # Configuration
└── index.ts                    # Barrel export

/src/app/api/
├── full-demo/
│   └── pipeline/
│       └── route.ts            # SSE pipeline endpoint
├── stt/
│   └── whisper/
│       └── route.ts            # Whisper endpoint
└── blender/
    ├── execute/
    │   └── route.ts            # Execute commands
    └── status/
        └── route.ts            # Connection status

/backend/
├── stt_service.py              # STT orchestration
├── blender_mcp.py              # Blender MCP client
└── pipeline_orchestrator.py    # Full pipeline
```

---

## 14. Design Decisions (RESOLVED)

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Blender Setup** | Hybrid: Three.js (browser) + Blender (GPU server) | Three.js for real-time preview in browser; Blender headless on GPU for actual execution & final renders |
| **Whisper Model** | `openai-whisper` (official) | Accuracy is priority over speed |
| **Avatar Behavior** | Start automatically with greeting | Creates engaging experience; avatar explains capabilities before user interaction |
| **Command Execution** | Auto-execute | User confirmed; no manual confirmation needed |
| **Response + Execution** | Simultaneous | Avatar speaks while Blender executes in parallel |

---

## 15. Blender Integration Architecture

### Hybrid Approach: Three.js + Blender

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       BROWSER (Three.js)                                │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    BlenderViewport Component                     │  │
│   │   ┌───────────────────────────────────────────────────────┐     │  │
│   │   │                   Three.js Scene                       │     │  │
│   │   │   - Real-time 3D preview (wireframe/solid)             │     │  │
│   │   │   - Orbit controls for user navigation                 │     │  │
│   │   │   - Progressive updates as commands execute            │     │  │
│   │   │   - Lightweight approximation of Blender output        │     │  │
│   │   └───────────────────────────────────────────────────────┘     │  │
│   │                                                                   │  │
│   │   [Download .blend] [View Final Render]                          │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    │ WebSocket / SSE                    │
│                                    ▼                                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
                                     │
┌────────────────────────────────────┼────────────────────────────────────┐
│                       GPU SERVER (Blender Headless)                     │
│                                    ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    Blender MCP Server                            │  │
│   │   - Headless Blender instance                                    │  │
│   │   - Executes bpy commands                                        │  │
│   │   - Generates actual .blend files                                │  │
│   │   - Produces high-quality renders                                │  │
│   │   - Streams scene state updates to frontend                      │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   Benefits:                                                             │
│   - No Blender installation required on client                          │
│   - GPU-accelerated rendering                                           │
│   - Consistent results across all users                                 │
│   - Three.js provides immediate visual feedback                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Scene Sync Protocol

```typescript
// Three.js receives simplified scene updates from Blender
interface SceneUpdate {
  type: 'add' | 'modify' | 'delete';
  object: {
    name: string;
    type: 'mesh' | 'light' | 'camera';
    geometry?: {
      vertices: number[];
      faces: number[];
    };
    position: [number, number, number];
    rotation: [number, number, number];
    scale: [number, number, number];
    material?: {
      color: string;
      metalness?: number;
      roughness?: number;
    };
  };
}
```

---

## 16. Expected Output Examples

### Example 1: "Create a low-poly tree with green leaves"

#### STT Output
```json
{
  "text": "Create a low-poly tree with green leaves",
  "confidence": 0.942,
  "words": [
    {"word": "Create", "confidence": 0.98, "start": 0.0, "end": 0.4},
    {"word": "a", "confidence": 0.95, "start": 0.4, "end": 0.5},
    {"word": "low-poly", "confidence": 0.89, "start": 0.5, "end": 1.0},
    {"word": "tree", "confidence": 0.97, "start": 1.0, "end": 1.3},
    {"word": "with", "confidence": 0.94, "start": 1.3, "end": 1.5},
    {"word": "green", "confidence": 0.96, "start": 1.5, "end": 1.8},
    {"word": "leaves", "confidence": 0.93, "start": 1.8, "end": 2.2}
  ],
  "rtf": 0.32,
  "processing_time_ms": 704,
  "engine": "whisper"
}
```

#### RAG Pipeline Output
```json
{
  "stages": [
    {"name": "Orchestrator", "status": "complete", "duration_ms": 12},
    {"name": "Query Analysis", "status": "complete", "duration_ms": 89},
    {"name": "Dense Search", "status": "complete", "duration_ms": 156, "results": 12},
    {"name": "Sparse Search", "status": "complete", "duration_ms": 43, "results": 8},
    {"name": "RRF Fusion", "status": "complete", "duration_ms": 5, "merged": 15},
    {"name": "Reranking", "status": "complete", "duration_ms": 234, "top_k": 5},
    {"name": "Generation", "status": "complete", "duration_ms": 1240},
    {"name": "Validation", "status": "complete", "duration_ms": 89, "score": 0.92}
  ],
  "documents_retrieved": 5,
  "ragas_score": 0.92,
  "total_latency_ms": 1868,
  "citations": [
    {"id": 1, "source": "blender_api/mesh_primitives.md", "relevance": 0.94},
    {"id": 2, "source": "tutorials/low_poly_modeling.md", "relevance": 0.89},
    {"id": 3, "source": "materials/vertex_colors.md", "relevance": 0.85}
  ]
}
```

#### Avatar Response (Text)
```
"I'll create a low-poly tree for you! I'm using a cylinder for the trunk with a
brown material, and stacking three ico-spheres with increasing subdivision for the
foliage. Each sphere gets a slightly different shade of green for visual depth.
Let me execute that in Blender now... Perfect! Your tree is ready. Would you like
me to add more trees, change the colors, or add some ground beneath it?"
```

#### Blender Commands Generated
```python
# Tree trunk
bpy.ops.mesh.primitive_cylinder_add(radius=0.3, depth=2.0, location=(0, 0, 1))
trunk = bpy.context.active_object
trunk.name = "TreeTrunk"
mat_trunk = bpy.data.materials.new("BrownBark")
mat_trunk.diffuse_color = (0.35, 0.22, 0.1, 1.0)
trunk.data.materials.append(mat_trunk)

# Foliage layers
for i, (size, height, shade) in enumerate([
    (1.2, 2.5, 0.4),
    (0.9, 3.2, 0.5),
    (0.6, 3.8, 0.6)
]):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=size, location=(0, 0, height))
    foliage = bpy.context.active_object
    foliage.name = f"Foliage_{i+1}"
    mat = bpy.data.materials.new(f"Green_{i+1}")
    mat.diffuse_color = (0.1, shade, 0.15, 1.0)
    foliage.data.materials.append(mat)
```

#### Three.js Preview (Browser)
```
┌─────────────────────────────────────────┐
│                                         │
│              🌲 (rendered)              │
│         ╱ ▼ ╲   ← Green ico-sphere      │
│        ╱     ╲                          │
│       ╱   ▼   ╲  ← Larger green sphere  │
│      ╱         ╲                        │
│     ╱     ▼     ╲ ← Largest green       │
│           █                             │
│           █     ← Brown cylinder trunk  │
│           █                             │
│    ───────────────                      │
│    Ground plane (optional)              │
│                                         │
│   [🔄 Orbit] [📷 Render] [💾 Download]  │
└─────────────────────────────────────────┘
```

### Example 2: "Add a metallic sphere with reflections"

#### STT Output
```json
{
  "text": "Add a metallic sphere with reflections",
  "confidence": 0.967,
  "engine": "whisper",
  "rtf": 0.28
}
```

#### Avatar Response
```
"Adding a metallic sphere with a highly reflective material! I'm setting the
metalness to 1.0 and roughness to 0.1 for that mirror-like finish. The sphere
will pick up reflections from the environment."
```

#### Blender Commands
```python
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 1))
sphere = bpy.context.active_object
sphere.name = "MetallicSphere"
mat = bpy.data.materials.new("ChromeMetal")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Metallic"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = 0.1
bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.85, 1.0)
sphere.data.materials.append(mat)
```

---

## 17. Complete Interface Mockup

### Desktop Layout (1920x1080)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ◀ Back   🎮 3D GAME AI ASSISTANT - Full Pipeline Demo            ⚙️ Settings   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────┐    ┌─────────────────────────────────────────────┐│
│  │                          │    │                                             ││
│  │       AVATAR             │    │              THREE.JS VIEWPORT              ││
│  │    ┌─────────────┐       │    │                                             ││
│  │    │             │       │    │                   🌲                        ││
│  │    │  [Avatar    │       │    │                  ╱▼╲                        ││
│  │    │   Video]    │       │    │                 ╱   ╲                       ││
│  │    │             │       │    │                    █                        ││
│  │    │  Speaking...|       │    │                    █                        ││
│  │    └─────────────┘       │    │                                             ││
│  │                          │    │    ┌──────────────────────────────────────┐ ││
│  │    🔊 ▓▓▓▓▓▓▓▓▒▒▒▒░░░░  │    │    │ 🟢 Blender Connected | Objects: 4   │ ││
│  │    Voice: Rachel         │    │    └──────────────────────────────────────┘ ││
│  └──────────────────────────┘    └─────────────────────────────────────────────┘│
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │  🎤  VOICE INPUT                                    STT: [Whisper ▼]        ││
│  │  ┌────────────────────────────────────────────────────────────────────────┐ ││
│  │  │  ~~~▁▂▃▄▅▆▇█▇▆▅▄▃▂▁~~~▁▂▃▄▅▆▇█▇▆▅▄▃▂▁~~~  Waveform               │ ││
│  │  └────────────────────────────────────────────────────────────────────────┘ ││
│  │  [ 🔴 Recording... ]  [ ⬛ Stop ]                                            ││
│  │                                                                              ││
│  │  📝 Transcription: "Create a low-poly tree with green leaves"                ││
│  │  📊 Confidence: 94.2% │ ⏱️ RTF: 0.32 │ 🔤 Words: 8 │ 🎯 Engine: Whisper      ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐│
│  │  📊 RAG PIPELINE                │  │  💬 RESPONSE                            ││
│  │                                 │  │                                         ││
│  │  ✅ 1. Orchestrator      12ms   │  │  "I'll create a low-poly tree for you! ││
│  │  ✅ 2. Query Analysis    89ms   │  │   I'm using a cylinder for the trunk   ││
│  │  ✅ 3. Dense Search     156ms   │  │   with a brown material, and stacking  ││
│  │  ✅ 4. Sparse Search     43ms   │  │   three ico-spheres..."                 ││
│  │  ✅ 5. RRF Fusion         5ms   │  │                                         ││
│  │  ✅ 6. Reranking        234ms   │  │  📚 Citations: [1] mesh_primitives.md   ││
│  │  🔄 7. Generation      1240ms   │  │               [2] low_poly_modeling.md  ││
│  │  ⏳ 8. Validation         -     │  │                                         ││
│  │                                 │  │  ▶ BLENDER COMMANDS                     ││
│  │  📄 Documents: 5 retrieved      │  │  ┌─────────────────────────────────────┐││
│  │  🎯 RAGAS Score: 0.92           │  │  │ bpy.ops.mesh.primitive_cylinder... │││
│  │  ⏱️ Latency: 1.87s              │  │  │ trunk.name = "TreeTrunk"           │││
│  │                                 │  │  │ bpy.ops.mesh.primitive_ico_sphere..│││
│  └─────────────────────────────────┘  │  └─────────────────────────────────────┘││
│                                       │  [ ▶️ Execute ] [ 📋 Copy ] [ 💾 .blend ]││
│                                       └─────────────────────────────────────────┘│
│                                                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  📈 STT: 94.2% │ RAG: 0.92 RAGAS │ ⏱️ Total: 3.2s │ 🖥️ GPU: 78% │ 📦 Blender: ✅ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Settings Panel

```
┌─────────────────────────────────────────┐
│  ⚙️ SETTINGS                       [X]  │
├─────────────────────────────────────────┤
│                                         │
│  🎤 Speech-to-Text                      │
│  ┌─────────────────────────────────┐    │
│  │ ○ VoxFormer (Custom trained)    │    │
│  │ ● Whisper (OpenAI - Recommended)│    │
│  └─────────────────────────────────┘    │
│                                         │
│  🗣️ Voice                               │
│  ┌─────────────────────────────────┐    │
│  │ Rachel (Female)              ▼  │    │
│  └─────────────────────────────────┘    │
│                                         │
│  👤 Avatar                              │
│  ┌─────────────────────────────────┐    │
│  │ Default Assistant            ▼  │    │
│  └─────────────────────────────────┘    │
│                                         │
│  🎨 Blender                             │
│  ┌─────────────────────────────────┐    │
│  │ ☑️ Enable 3D Execution          │    │
│  │ ☑️ Auto-execute commands        │    │
│  │ ☑️ Show live preview            │    │
│  └─────────────────────────────────┘    │
│                                         │
│  🐛 Debug                               │
│  ┌─────────────────────────────────┐    │
│  │ ☐ Show detailed metrics         │    │
│  │ ☐ Log API responses             │    │
│  └─────────────────────────────────┘    │
│                                         │
│  [ Apply ]              [ Reset ]       │
└─────────────────────────────────────────┘
```

---

## 18. Ready for Implementation ✅

### Prerequisites Completed
- ✅ Frontend demos analyzed (`/demo`, `/avatar_demo`, `/rag_demo`)
- ✅ Backend APIs documented (all endpoints mapped)
- ✅ Technical architecture designed
- ✅ Component structure planned
- ✅ Data flow defined
- ✅ Implementation phases outlined
- ✅ Design decisions resolved
- ✅ Blender integration architecture (Three.js + GPU)
- ✅ Expected outputs documented with examples
- ✅ Complete interface mockup

### Implementation Summary

| Component | Source | New Work |
|-----------|--------|----------|
| Voice Input | `/demo` | Minor adaptation |
| STT Engine Toggle | New | Whisper endpoint + UI |
| RAG Pipeline | `/rag_demo` | Adapt visualization |
| Avatar Display | `/avatar_demo` | Add auto-greeting |
| Blender Viewport | New | Three.js scene + MCP |
| Pipeline Orchestrator | New | SSE streaming backend |
| Metrics Bar | New | Aggregate from components |

### Total Estimated Components
- **Frontend**: 8 new/adapted components
- **Backend**: 4 new endpoints + 2 services
- **External**: Blender MCP setup on GPU server

---

**STATUS: APPROVED - READY FOR IMPLEMENTATION**

---

*Document created: December 17, 2024*
*Last updated: December 18, 2024*
