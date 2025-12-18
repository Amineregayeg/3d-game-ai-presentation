# 3D Game AI Assistant - Architecture Video Plan

## Overview

**Video Title:** 3D Game AI Assistant - Complete Architecture Explainer
**Duration:** ~11 minutes (660 seconds)
**Frame Rate:** 30 FPS
**Resolution:** 1920x1080 (1080p)
**Total Frames:** ~19,800

## Project Components

| Component | Slides | Architecture Elements | Color Theme |
|-----------|--------|----------------------|-------------|
| **VoxFormer STT** | 14 slides | Audio→WavLM→Conformer→Decoder→CTC | Cyan/Purple (#06b6d4/#a855f7) |
| **RAG System** | 8 slides | Query→Retrieval→Rerank→Generate→Validate | Emerald/Cyan (#10b981/#06b6d4) |
| **MCP/Blender** | 9 slides | Claude→MCP→Blender→Assets→Export | Orange/Amber (#f97316/#fbbf24) |
| **Avatar TTS+Lipsync** | 9 slides | Text→TTS→LipSync→Visemes→Engine | Rose/Pink (#f43f5e/#ec4899) |

## Assets

- Logo: `/public/logo.svg`, `/public/logo.png`
- All architecture diagrams extracted from slide components
- Color schemes defined per component

---

## Scene Breakdown

### Scene 0: Opening Logo Animation
**Duration:** 0:00 - 0:08 (8 seconds, 240 frames)

```
[Black screen]
T=0.0s (F0):    Logo fades in from center (scale 0.8→1.0, ease-out, 800ms)
T=0.5s (F15):   Logo particles/glow effect pulse (cyan→purple gradient)
T=1.5s (F45):   Project name appears below: "3D Game AI Assistant"
T=2.5s (F75):   Tagline fades in: "Intelligent NPC Pipeline"
T=4.0s (F120):  Logo shrinks and moves to corner
T=5.0s (F150):  Transition to main interface
T=8.0s (F240):  Scene complete
```

**Components:**
- AnimatedLogo.tsx
- GlowEffect.tsx
- TextReveal.tsx

---

### Scene 1: Project Overview Dropdown
**Duration:** 0:08 - 0:25 (17 seconds, 510 frames)

```
[Modern dark UI with gradient background]
T=0.0s (F0):    Central component card appears (glass morphism effect)
T=0.3s (F9):    "3D Game AI Assistant" title with gradient text
T=0.8s (F24):   Click animation on card
T=1.2s (F36):   Dropdown expands with 4 components (staggered 200ms each):

  ┌─────────────────────────────────────┐
  │  🎙️ VoxFormer STT          ○       │  ← Cyan accent
  │  🔍 Advanced RAG System    ○       │  ← Emerald accent
  │  🗣️ Avatar TTS + LipSync   ○       │  ← Rose accent
  │  🎨 Blender MCP Bridge     ○       │  ← Orange accent
  └─────────────────────────────────────┘

T=3.0s (F90):   Brief description appears for each on hover simulation
T=5.0s (F150):  Component 1 (VoxFormer) highlights with glow
T=17.0s (F510): Scene complete
```

**Components:**
- DropdownCard.tsx
- ComponentItem.tsx
- HoverDescription.tsx

---

### Scene 2: VoxFormer Deep Dive
**Duration:** 0:25 - 3:00 (155 seconds, 4,650 frames)

#### 2.1 Title Card (0:25 - 0:35, 10s, 300 frames)
```
T=0.0s: VoxFormer logo/badge scales in (ease-out, 600ms)
T=0.5s: "Speech-to-Text Transformer" subtitle slides in
T=1.0s: Key metrics fly in: "142M params | <200ms latency | <3.5% WER"
T=3.0s: Metrics pulse with glow
T=5.0s: Transition to architecture
```

#### 2.2 High-Level Architecture (0:35 - 1:00, 25s, 750 frames)
```
[Full pipeline diagram animates left-to-right]
T=0.0s:  "Raw Audio" box appears with waveform
T=2.0s:  Arrow draws → "DSP Pipeline" fades in (6 stages icon)
T=4.0s:  Arrow draws → "WavLM" fades in (95M params badge)
T=6.0s:  Arrow draws → "Adapter" fades in (768→512 label)
T=8.0s:  Arrow draws → "Zipformer Encoder" fades in (6 blocks)
T=10.0s: Arrow draws → "Transformer Decoder" fades in (4 layers)
T=12.0s: Arrow draws → "Text Output" with checkmark
T=15.0s: Full diagram pulses, labels appear
```

#### 2.3 DSP Pipeline Detail (1:00 - 1:25, 25s, 750 frames)
```
[6-stage vertical flow with waveform visualizations]
T=0.0s:  Title "Audio Frontend" appears
T=2.0s:  Stage 1: Signal Conditioning (DC removal, pre-emphasis)
T=5.0s:  Stage 2: Voice Activity Detection (energy bars)
T=8.0s:  Stage 3: Noise Estimation (MCRA visualization)
T=11.0s: Stage 4: Noise Reduction (spectral subtraction)
T=14.0s: Stage 5: Echo Cancellation (adaptive filter)
T=17.0s: Stage 6: Voice Isolation (deep attractor)
T=20.0s: Clean waveform output animation
```

#### 2.4 WavLM Integration (1:25 - 1:45, 20s, 600 frames)
```
[Frozen backbone visualization]
T=0.0s:  Title "WavLM-Base (95M params)"
T=2.0s:  12 transformer layers shown as stacked blocks
T=5.0s:  "FROZEN" badge appears with lock icon
T=8.0s:  Weighted sum combination animated (layer weights)
T=12.0s: 768→512 adapter projection highlighted
T=15.0s: Output: 512-dim features @ 50fps
```

#### 2.5 Conformer Block (1:45 - 2:10, 25s, 750 frames)
```
[Detailed block architecture]
T=0.0s:  Title "Zipformer Encoder Block"
T=3.0s:  FFN(½) block appears
T=6.0s:  → Multi-Head Attention (8 heads) with RoPE
T=9.0s:  → Depthwise Convolution (kernel=31)
T=12.0s: → FFN(½) block
T=15.0s: RoPE rotation visualization on unit circle
T=18.0s: Flash Attention memory comparison: O(N²) vs O(N)
T=22.0s: Full block diagram complete
```

#### 2.6 Hybrid Loss (2:10 - 2:30, 20s, 600 frames)
```
[CTC + Cross-Entropy visualization]
T=0.0s:  Title "Hybrid Loss Function"
T=3.0s:  CTC alignment grid animation (time × labels)
T=8.0s:  Paths through lattice animated (monotonic)
T=12.0s: Loss weight slider: 0.3 CTC | 0.7 CE
T=15.0s: Label smoothing distribution curve
T=18.0s: Combined loss formula appears
```

#### 2.7 Streaming & Deployment (2:30 - 2:50, 20s, 600 frames)
```
[Real-time pipeline]
T=0.0s:  Title "Streaming Inference"
T=3.0s:  Chunked audio processing (160-200ms chunks)
T=6.0s:  KV-cache visualization (reused keys/values)
T=10.0s: Latency breakdown bar chart:
         - Audio: 20ms
         - WavLM: 40ms
         - Encoder: 60ms
         - Decoder: 40ms
         - Total: 160ms
T=14.0s: Model compression: FP32→FP16→INT8
T=17.0s: Size reduction: 285MB → 145MB → 75MB
```

#### 2.8 Return to Dropdown (2:50 - 3:00, 10s, 300 frames)
```
T=0.0s:  VoxFormer section zooms out
T=2.0s:  Collapse animation
T=4.0s:  Dropdown reappears
T=6.0s:  VoxFormer row gets GREEN CHECKMARK ✓ (scale + glow)
T=8.0s:  RAG System row highlights with emerald glow
```

---

### Scene 3: RAG System Deep Dive
**Duration:** 3:00 - 5:15 (135 seconds, 4,050 frames)

#### 3.1 Title Card (3:00 - 3:10, 10s, 300 frames)
```
T=0.0s: RAG badge scales in with emerald glow
T=0.5s: "Advanced RAG System" title
T=1.0s: "Retrieval-Augmented Generation for Game Development"
T=2.0s: Metrics: "3,885 docs | 0.82 RAGAS | <5% hallucination"
```

#### 3.2 Full Architecture (3:10 - 3:40, 30s, 900 frames)
```
[Pipeline animates with data particles flowing]
T=0.0s:  User Query box appears
T=3.0s:  → Query Transformer
T=6.0s:  Split into dual paths:
         ├─ Dense Search (MiniLM, cyan)
         └─ Sparse Search (BM25, blue)
T=12.0s: Both paths merge → RRF Fusion (purple)
T=15.0s: → Cross-Encoder Rerank (amber)
T=18.0s: → GPT-5.1 Generation
T=21.0s: → Validation Loop (circular arrow)
T=24.0s: → Final Answer
T=27.0s: Full diagram with data flow animation
```

#### 3.3 Hybrid Retrieval (3:40 - 4:10, 30s, 900 frames)
```
[Dual path visualization]
T=0.0s:  Title "Hybrid Retrieval"
T=3.0s:  Dense Search: 3D vector space visualization
         - Documents as points in space
         - Query vector searches nearest neighbors
T=10.0s: Sparse Search: Document with keyword highlighting
         - TF-IDF scoring animation
         - BM25 formula appears
T=17.0s: RRF Fusion:
         - Formula: score(d) = Σ 1/(k + rank)
         - Animated calculation with k=60
T=24.0s: Top-50 candidates highlighted
```

#### 3.4 HNSW Index (4:10 - 4:30, 20s, 600 frames)
```
[3-layer hierarchical graph]
T=0.0s:  Title "HNSW Vector Index"
T=3.0s:  Layer 2 (sparse) - few nodes
T=6.0s:  Layer 1 (medium) - more nodes
T=9.0s:  Layer 0 (dense) - all nodes
T=12.0s: Search path animated:
         - Start at top layer
         - Descend through layers
         - Find nearest neighbors
T=17.0s: O(log N) complexity badge
```

#### 3.5 Cross-Encoder Reranking (4:30 - 4:50, 20s, 600 frames)
```
[Comparison diagram]
T=0.0s:  Title "Cross-Encoder Reranking"
T=3.0s:  Bi-Encoder (fast, separate encoding)
T=7.0s:  Cross-Encoder (accurate, joint encoding)
T=11.0s: Query-Doc pairs feed through transformer
T=14.0s: Relevance scores 0.0-1.0 appear
T=17.0s: Top-10 selection highlighted with checkmarks
```

#### 3.6 Validation Loop (4:50 - 5:05, 15s, 450 frames)
```
[Circular flow diagram]
T=0.0s:  Title "Agentic Validation"
T=2.0s:  Generate box
T=4.0s:  → Validate diamond (4 checks):
         1. Syntax Check
         2. API Verify
         3. Version Match
         4. Hallucination Detection
T=8.0s:  Decision paths:
         ├─ Pass → Output (green)
         └─ Fail → Retry (red, max 3)
T=12.0s: Retry loop animation
```

#### 3.7 Return to Dropdown (5:05 - 5:15, 10s, 300 frames)
```
T=0.0s:  RAG section zooms out
T=3.0s:  Dropdown appears
T=5.0s:  RAG row gets GREEN CHECKMARK ✓
T=7.0s:  Avatar row highlights with rose glow
```

---

### Scene 4: Avatar TTS + LipSync Deep Dive
**Duration:** 5:15 - 7:30 (135 seconds, 4,050 frames)

#### 4.1 Title Card (5:15 - 5:25, 10s, 300 frames)
```
T=0.0s: Avatar badge scales in with rose glow
T=0.5s: "Avatar TTS + Lip Synchronization"
T=1.0s: "Bringing NPCs to Life"
T=2.0s: Metrics: "75ms TTFB | 4.14 MOS | 95% sync accuracy"
```

#### 4.2 Full Pipeline (5:25 - 5:55, 30s, 900 frames)
```
[Pipeline with audio waveform flowing through]
T=0.0s:  RAG Output box
T=3.0s:  → Text Processing (SSML tags visible)
T=6.0s:  → ElevenLabs TTS (cloud icon)
T=9.0s:  Audio stream splits:
         ├─ Audio Playback (speaker icon)
         └─ Lip-Sync Engine (mouth icon)
T=15.0s: Both paths merge → Blend Shapes
T=18.0s: → Avatar face animation
T=22.0s: Waveform and mouth sync animation
```

#### 4.3 ElevenLabs Integration (5:55 - 6:20, 25s, 750 frames)
```
[WebSocket streaming visualization]
T=0.0s:  Title "ElevenLabs TTS"
T=3.0s:  WebSocket connection diagram
T=6.0s:  Voice waveform generation animation
T=10.0s: SSML emotion tags highlighted:
         <emotion name="friendly">
         <prosody rate="slow">
         <break time="500ms"/>
T=15.0s: Voice library cards:
         - Rachel (Warm)
         - Antoni (Deep)
         - Bella (Cheerful)
T=20.0s: Performance metrics: 75ms TTFB, 4.14 MOS
```

#### 4.4 Lip-Sync Engines (6:20 - 6:50, 30s, 900 frames)
```
[Hybrid approach comparison]
T=0.0s:  Title "Hybrid Lip-Sync Strategy"
T=3.0s:  Wav2Lip card (80% usage):
         - Real-time
         - Speaker-agnostic
         - 100-300ms latency
T=10.0s: SadTalker card (20% usage):
         - Emotional expressions
         - Head poses
         - 200-500ms latency
T=17.0s: Side-by-side comparison animation:
         - Standard dialogue → Wav2Lip
         - Cinematic moment → SadTalker
T=25.0s: Use case decision tree
```

#### 4.5 Viseme Mapping (6:50 - 7:10, 20s, 600 frames)
```
[Phoneme to viseme visualization]
T=0.0s:  Title "Viseme → Blend Shape Mapping"
T=3.0s:  Audio waveform with phoneme markers
T=6.0s:  Phonemes → 22 Visemes table
T=10.0s: Core 6 visemes animated:
         A (wide open) → B/M/P (closed) → E (smile) →
         F/V (lip-teeth) → O (rounded) → U (tense)
T=15.0s: Blend shape sliders animation (0-1 values)
T=18.0s: Avatar face responding to blend shapes
```

#### 4.6 Game Engine Integration (7:10 - 7:20, 10s, 300 frames)
```
[Split screen: Unity and UE5]
T=0.0s:  Title "Game Engine Integration"
T=2.0s:  Unity side: C# code snippet
T=5.0s:  UE5 side: C++ code snippet
T=7.0s:  MetaHuman audio-driven lip-sync diagram
```

#### 4.7 Return to Dropdown (7:20 - 7:30, 10s, 300 frames)
```
T=0.0s:  Avatar section zooms out
T=3.0s:  Dropdown appears
T=5.0s:  Avatar row gets GREEN CHECKMARK ✓
T=7.0s:  MCP row highlights with orange glow
```

---

### Scene 5: Blender MCP Bridge Deep Dive
**Duration:** 7:30 - 9:45 (135 seconds, 4,050 frames)

#### 5.1 Title Card (7:30 - 7:40, 10s, 300 frames)
```
T=0.0s: MCP badge scales in with orange glow
T=0.5s: "Blender MCP Bridge"
T=1.0s: "AI-Powered 3D Asset Generation"
T=2.0s: Metrics: "24 tools | 5 asset sources | <100ms socket"
```

#### 5.2 MCP Protocol (7:40 - 8:05, 25s, 750 frames)
```
[Protocol primitives visualization]
T=0.0s:  Title "Model Context Protocol"
T=3.0s:  5 primitives appear as icons:
         Prompts | Resources | Tools | Roots | Sampling
T=8.0s:  JSON-RPC 2.0 message animation:
         {
           "jsonrpc": "2.0",
           "method": "tools/call",
           "params": { "name": "create_mesh" }
         }
T=15.0s: Request → Response flow animation
T=20.0s: Anthropic logo + "Open Standard" badge
```

#### 5.3 System Architecture (8:05 - 8:35, 30s, 900 frames)
```
[Full system diagram]
T=0.0s:  User Request (speech bubble)
T=3.0s:  → Claude AI (brain icon)
T=6.0s:  → MCP Server (Python SDK)
T=9.0s:  → TCP Socket (localhost:9876)
T=12.0s: → Blender Addon (Blender logo)
T=15.0s: → 3D Asset output
T=18.0s: → Game Engines (Unity + UE5 logos)
T=22.0s: Bidirectional arrows for communication
T=26.0s: Latency labels appear (<100ms, <50ms)
```

#### 5.4 24 Tools (8:35 - 9:00, 25s, 750 frames)
```
[5 categories with tool icons]
T=0.0s:  Title "24 MCP Tools"
T=3.0s:  Scene Operations (3 tools) - eye icon
         get_scene_info, get_object_info, get_screenshot
T=7.0s:  Code Execution (1 tool) - code icon
         execute_blender_code
T=10.0s: Asset Search (3 tools) - search icon
         search_sketchfab, search_polyhaven, get_categories
T=14.0s: Asset Download (2 tools) - download icon
         download_sketchfab, download_polyhaven
T=18.0s: AI Generation (6 tools) - sparkle icon
         generate_hyper3d, generate_hunyuan, poll_status...
T=22.0s: All tools pulse with glow
```

#### 5.5 Asset Sources (9:00 - 9:20, 20s, 600 frames)
```
[Priority fallback chain visualization]
T=0.0s:  Title "Multi-Source Asset Acquisition"
T=3.0s:  Priority chain appears:
         1. Sketchfab (Professional, 10-30s)
         2. Poly Haven (Free, 5-20s)
         3. Hyper3D Rodin (AI, 30-120s)
         4. Hunyuan3D (AI, 30-120s)
         5. Python Scripts (Procedural, <5s)
T=10.0s: Fallback arrows animate between sources
T=14.0s: "Asset not found → Try next source" animation
T=17.0s: Success checkmark at end
```

#### 5.6 Game Engine Export (9:20 - 9:35, 15s, 450 frames)
```
[Export configuration]
T=0.0s:  Title "Game Engine Export"
T=2.0s:  Format icons: FBX, GLTF, OBJ, USD
T=5.0s:  Material mapping table:
         Blender → Unity → UE5
         Base Color → Albedo → Base Color
         Roughness → 1-Smoothness → Roughness
T=10.0s: Export settings code snippet
T=13.0s: Unity and UE5 import preview
```

#### 5.7 Return to Dropdown (9:35 - 9:45, 10s, 300 frames)
```
T=0.0s:  MCP section zooms out
T=3.0s:  Dropdown appears
T=5.0s:  MCP row gets GREEN CHECKMARK ✓
T=7.0s:  ALL 4 COMPONENTS NOW CHECKED
T=9.0s:  Celebration particles effect
```

---

### Scene 6: System Integration & Use Cases
**Duration:** 9:45 - 10:45 (60 seconds, 1,800 frames)

#### 6.1 Full System Diagram (9:45 - 10:15, 30s, 900 frames)
```
[All 4 components connected in one diagram]
T=0.0s:  Title "Complete System Integration"
T=3.0s:  Player Voice input (microphone icon)
T=5.0s:  → VoxFormer STT (cyan box)
T=8.0s:  → RAG System (emerald box)
T=11.0s: Response splits:
         ├─ Avatar Pipeline (rose box)
         │   └─ Voice + Animation output
         └─ MCP Pipeline (orange box)
             └─ 3D Asset output
T=18.0s: Data flow particles animate through system
T=22.0s: Color-coded connections pulse
T=26.0s: End-to-end latency label: "<500ms"
```

#### 6.2 Use Cases (10:15 - 10:35, 20s, 600 frames)
```
[3 scenario cards animate in]
T=0.0s:  Title "Real-World Use Cases"
T=3.0s:  Card 1: "Voice-Controlled NPC Dialogue"
         Components: VoxFormer + RAG + Avatar
         Demo: Player speaks → NPC responds with animation
T=9.0s:  Card 2: "Dynamic Asset Creation"
         Components: RAG + MCP
         Demo: "Create a magical sword" → 3D model appears
T=15.0s: Card 3: "Full Immersive Experience"
         Components: All 4
         Demo: Voice command → Intelligent response + visuals
```

#### 6.3 Key Metrics Summary (10:35 - 10:45, 10s, 300 frames)
```
[Dashboard with animated counters]
T=0.0s:  Stats dashboard appears
T=2.0s:  End-to-end latency: <500ms (counter animation)
T=3.0s:  Voice recognition: <3.5% WER
T=4.0s:  Retrieval accuracy: 0.82 RAGAS
T=5.0s:  Lip-sync accuracy: 95%+
T=6.0s:  Asset generation: 30-120s
T=8.0s:  All metrics pulse with success glow
```

---

### Scene 7: Closing
**Duration:** 10:45 - 11:00 (15 seconds, 450 frames)

```
T=0.0s (F0):    All components collapse into center point
T=2.0s (F60):   Logo scales up from center (ease-out, 800ms)
T=4.0s (F120):  Logo glow effect (cyan→purple→emerald→orange cycle)
T=6.0s (F180):  Tagline appears: "The Future of Intelligent NPCs"
T=8.0s (F240):  Subtitle: "AI-Powered Game Development"
T=10.0s (F300): Links/contact fade in (optional)
T=12.0s (F360): Logo pulse animation
T=14.0s (F420): Fade to black
T=15.0s (F450): Scene complete
```

---

## Technical Implementation

### Remotion Project Structure
```
/home/developer/remotion/
├── src/
│   ├── compositions/
│   │   └── ArchitectureVideo/
│   │       ├── index.tsx                 # Main composition (19,800 frames)
│   │       ├── scenes/
│   │       │   ├── S0_LogoIntro.tsx      # 240 frames
│   │       │   ├── S1_ProjectDropdown.tsx # 510 frames
│   │       │   ├── S2_VoxFormer.tsx      # 4,650 frames
│   │       │   ├── S3_RAGSystem.tsx      # 4,050 frames
│   │       │   ├── S4_Avatar.tsx         # 4,050 frames
│   │       │   ├── S5_MCP.tsx            # 4,050 frames
│   │       │   ├── S6_Integration.tsx    # 1,800 frames
│   │       │   └── S7_Closing.tsx        # 450 frames
│   │       └── components/
│   │           ├── AnimatedLogo.tsx
│   │           ├── DropdownMenu.tsx
│   │           ├── ComponentCard.tsx
│   │           ├── ArchitectureDiagram.tsx
│   │           ├── FlowArrow.tsx
│   │           ├── DataParticle.tsx
│   │           ├── Checkmark.tsx
│   │           ├── MetricCounter.tsx
│   │           ├── CodeBlock.tsx
│   │           └── GlowEffect.tsx
│   ├── utils/
│   │   ├── animations.ts                 # Easing functions, springs
│   │   ├── colors.ts                     # Theme color constants
│   │   ├── timings.ts                    # Frame/second conversions
│   │   └── fonts.ts                      # Font loading
│   ├── assets/
│   │   └── logo.svg
│   ├── Root.tsx                          # Composition registry
│   └── index.tsx                         # Entry point
├── public/
│   └── logo.svg
├── remotion.config.ts
├── package.json
└── tsconfig.json
```

### Color Constants
```typescript
export const COLORS = {
  // Background
  slate950: '#020617',
  slate900: '#0f172a',
  slate800: '#1e293b',

  // VoxFormer (Cyan/Purple)
  cyan400: '#22d3ee',
  cyan500: '#06b6d4',
  purple400: '#c084fc',
  purple500: '#a855f7',

  // RAG (Emerald/Cyan)
  emerald400: '#34d399',
  emerald500: '#10b981',

  // Avatar (Rose/Pink)
  rose400: '#fb7185',
  rose500: '#f43f5e',
  pink400: '#f472b6',
  pink500: '#ec4899',

  // MCP (Orange/Amber)
  orange400: '#fb923c',
  orange500: '#f97316',
  amber400: '#fbbf24',
  amber500: '#f59e0b',

  // Accents
  white: '#ffffff',
  green500: '#22c55e',  // Checkmarks
};
```

### Animation Easing
```typescript
import { Easing } from 'remotion';

export const EASING = {
  // Standard transitions
  entrance: Easing.out(Easing.cubic),
  exit: Easing.in(Easing.cubic),

  // Emphasis
  bounce: Easing.elastic(1.2),
  spring: Easing.out(Easing.back(1.7)),

  // Data flow
  smooth: Easing.inOut(Easing.quad),

  // UI elements
  snappy: Easing.out(Easing.quad),
};
```

### Frame Timing Reference
```typescript
export const FPS = 30;

export const SCENES = {
  logo: { start: 0, duration: 240 },           // 8s
  dropdown: { start: 240, duration: 510 },     // 17s
  voxformer: { start: 750, duration: 4650 },   // 155s
  rag: { start: 5400, duration: 4050 },        // 135s
  avatar: { start: 9450, duration: 4050 },     // 135s
  mcp: { start: 13500, duration: 4050 },       // 135s
  integration: { start: 17550, duration: 1800 }, // 60s
  closing: { start: 19350, duration: 450 },    // 15s
};

export const TOTAL_FRAMES = 19800; // ~11 minutes
```

---

## Render Configuration

### remotion.config.ts
```typescript
import { Config } from 'remotion';

Config.setFrameRate(30);
Config.setDimensions(1920, 1080);
Config.setConcurrency(4);
Config.setChromiumHeadlessMode(true);
Config.setChromiumMultiProcessOnLinux(true);
Config.setTimeoutInMilliseconds(300000); // 5 min for long scenes
Config.setCodec('h264');
```

### Render Command
```bash
cd /home/developer/remotion
npx remotion render src/index.tsx ArchitectureVideo output/architecture-video.mp4 \
  --codec h264 \
  --crf 18 \
  --concurrency 4 \
  --log verbose
```

### Estimated Render Time
- **VPS Specs:** 10 CPU cores, 17GB RAM
- **Estimated Time:** 30-60 minutes for full 11-minute video
- **Output Size:** ~200-400 MB (H.264, CRF 18)

---

## Development Phases

### Phase 1: Setup & Shared Components (1-2 hours)
- [ ] Initialize Remotion project structure
- [ ] Create color constants and utilities
- [ ] Build AnimatedLogo component
- [ ] Build DropdownMenu component
- [ ] Build FlowArrow component
- [ ] Build Checkmark component

### Phase 2: Scene 0-1 (1-2 hours)
- [ ] Logo intro animation
- [ ] Project dropdown animation
- [ ] Component highlighting

### Phase 3: VoxFormer Section (2-3 hours)
- [ ] Title card
- [ ] Architecture diagram
- [ ] DSP pipeline visualization
- [ ] WavLM integration
- [ ] Conformer block
- [ ] Hybrid loss
- [ ] Streaming inference

### Phase 4: RAG Section (2-3 hours)
- [ ] Title card
- [ ] Full architecture
- [ ] Hybrid retrieval
- [ ] HNSW visualization
- [ ] Cross-encoder reranking
- [ ] Validation loop

### Phase 5: Avatar Section (2-3 hours)
- [ ] Title card
- [ ] Full pipeline
- [ ] ElevenLabs integration
- [ ] Lip-sync engines
- [ ] Viseme mapping
- [ ] Game engine integration

### Phase 6: MCP Section (2-3 hours)
- [ ] Title card
- [ ] MCP protocol
- [ ] System architecture
- [ ] 24 tools visualization
- [ ] Asset sources
- [ ] Game engine export

### Phase 7: Integration & Closing (1-2 hours)
- [ ] Full system diagram
- [ ] Use cases
- [ ] Metrics dashboard
- [ ] Closing animation

### Phase 8: Polish & Render (1-2 hours)
- [ ] Timing adjustments
- [ ] Transition smoothing
- [ ] Test render
- [ ] Final render

**Total Estimated Time:** 12-20 hours

---

## Notes

- All timings are in seconds (T=) and frames (F=) at 30fps
- Color themes match existing slide presentations
- Architecture diagrams should mirror slide content
- Animations follow research best practices (ease-out entrances, 600-1200ms durations)
- Stagger related elements by 200-300ms for rhythm
