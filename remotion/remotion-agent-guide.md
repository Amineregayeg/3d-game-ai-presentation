# Remotion + AI Agents Integration Guide
## Reliable Video Generation Automation for Claude Code & Node.js Environments

**Last Updated:** December 2025  
**Remotion Version:** v4.0+  
**Target Use Case:** Automated video generation via AI agents (Claude Code, agentic systems)

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation & Setup](#installation--setup)
3. [Project Structure](#project-structure)
4. [API Fundamentals](#api-fundamentals)
5. [Headless Rendering](#headless-rendering)
6. [Docker Deployment](#docker-deployment)
7. [Known Failure Modes & Prevention](#known-failure-modes--prevention)
8. [Best Practices for AI Agents](#best-practices-for-ai-agents)
9. [Reference Implementation](#reference-implementation)

---

## System Requirements

### Minimum System Specs for Local Rendering

- **Node.js:** 18.0.0 or higher (LTS recommended: 20+)
- **RAM:** 4GB minimum (8GB+ recommended for concurrent rendering)
- **CPU:** Multi-core processor (concurrency defaults to 50% of CPU threads)
- **Disk:** 2GB free for node_modules + temporary frame buffers
- **OS:** macOS 10.15+, Ubuntu 20.04+, Debian 10+, Windows (WSL2 recommended)

### Bundled vs External Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| **FFmpeg** | Bundled (v4.0+) | Auto-installs from internet on first render; no manual setup needed |
| **Chromium/Chrome** | Auto-downloaded | Remotion manages browser lifecycle; ~200MB download |
| **Node.js** | External | Must be pre-installed |

### Architecture Support

**FFmpeg auto-install supported on:**
- Linux x86_64
- macOS Intel
- macOS Apple Silicon (M1/M2/M3)
- Windows x86_64

**For other architectures:** Supply custom FFmpeg binaries via `binariesDirectory` option.

---

## Installation & Setup

### Step 1: Initialize Node.js Project

```bash
mkdir my-video-generator
cd my-video-generator
npm init -y
```

### Step 2: Install Remotion Core Packages

```bash
# Minimal installation
npm install remotion @remotion/cli @remotion/renderer

# OR full-featured installation (recommended for agents)
npm install remotion \
  @remotion/cli \
  @remotion/renderer \
  @remotion/bundler \
  react \
  react-dom
```

**Why this stack for agents?**
- `remotion`: Core library for video composition (React components)
- `@remotion/cli`: CLI for preview/testing/one-off renders
- `@remotion/renderer`: Programmatic API (used by agents)
- `@remotion/bundler`: Webpack-based bundler for server-side rendering
- `react` / `react-dom`: Required for composition definitions

### Step 3: TypeScript Configuration (Optional but Recommended)

```bash
npm install -D typescript @types/react @types/node
npx tsc --init
```

**Minimal tsconfig.json:**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["ES2020", "DOM"],
    "jsx": "react",
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "allowSyntheticDefaultImports": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "out"]
}
```

### Step 4: Pre-install FFmpeg Binaries (Server Environments)

On servers where you want to avoid delays on first render:

```bash
npx remotion install ffmpeg
npx remotion install ffprobe
```

This triggers auto-install without executing a render. Binaries land in `node_modules/@remotion/compositor-*/ffmpeg`.

### Step 5: Verify Installation

```bash
npx remotion --version
# Output: remotion version 4.0.278 (example)

node -e "const {renderMedia} = require('@remotion/renderer'); console.log('API available')"
# Output: API available
```

---

## Project Structure

### Recommended Folder Layout for AI Agent Usage

```
my-video-generator/
├── src/
│   ├── compositions/           # Video composition definitions (React)
│   │   ├── HelloWorld.tsx       # Example composition
│   │   ├── animations/          # Reusable animation logic
│   │   └── index.tsx            # Composition registry
│   ├── utils/
│   │   ├── render.ts            # renderMedia() wrapper with error handling
│   │   ├── schemas.ts           # Input validation (Zod/Joi)
│   │   └── logger.ts            # Structured logging
│   ├── templates/               # Pre-built templates for common use cases
│   │   ├── intro-video.tsx
│   │   ├── slide-show.tsx
│   │   └── title-card.tsx
│   └── index.ts                 # Main export for agent SDK
├── remotion.config.ts           # Remotion configuration
├── package.json
├── tsconfig.json
└── .env.example                 # Environment variables
```

### Example: remotion.config.ts

```typescript
import { Config } from 'remotion';

Config.setFrameRate(30);                    // 24, 25, 30, 60 are common
Config.setDimensions(1920, 1080);           // 1080p default
Config.setDurationInFrames(300);            // 10 seconds @ 30fps
Config.setCodec('h264');                    // H.264 for broad compatibility
Config.setConcurrency(4);                   // Render 4 frames in parallel
Config.setChromiumHeadlessMode(true);       // Always headless for agents
Config.setChromiumMultiProcessOnLinux(true);// Better Linux stability (v4.0.42+)

// For AI agent environments: increase timeouts
Config.setTimeoutInMilliseconds(60000);     // 60s instead of default 30s

// For GPU acceleration (if available)
// Config.setChromeMode('chrome-for-testing'); // Use native Chrome on Linux
// Config.setChromiumOpenGlRenderer('angle-egl'); // Use Angle EGL renderer
```

### Example: src/compositions/HelloWorld.tsx

```typescript
import React from 'react';
import { AbsoluteFill, useVideoConfig } from 'remotion';

export interface HelloWorldProps {
  text: string;
  durationInSeconds: number;
}

export const HelloWorld: React.FC<HelloWorldProps> = ({
  text,
  durationInSeconds,
}) => {
  const { fps } = useVideoConfig();
  
  return (
    <AbsoluteFill style={{ backgroundColor: 'white' }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          fontSize: '72px',
          fontWeight: 'bold',
          color: '#000',
        }}
      >
        {text}
      </div>
    </AbsoluteFill>
  );
};

export const compositions = [
  {
    id: 'HelloWorld',
    component: HelloWorld,
    durationInFrames: 300,
    fps: 30,
    width: 1920,
    height: 1080,
    defaultProps: {
      text: 'Hello, Remotion!',
      durationInSeconds: 10,
    },
  },
];
```

---

## API Fundamentals

### Two Approaches: CLI vs Programmatic

| Feature | CLI (npx remotion) | Programmatic API |
|---------|------------------|------------------|
| **Use Case** | Manual renders, testing | Automated systems, agents |
| **Invocation** | Shell command | Node.js function call |
| **Error Handling** | Exit codes | Promise rejection |
| **Concurrency** | Single render at a time | Multiple renders in parallel |
| **Suitable for Agents?** | ❌ Spawning shell = overhead | ✅ Direct function calls |

### CLI Usage (Reference Only)

```bash
# Preview in browser (for development/testing)
npx remotion preview src/index.tsx

# Render to file
npx remotion render src/index.tsx HelloWorld output.mp4

# With options
npx remotion render src/index.tsx HelloWorld output.mp4 \
  --codec h264 \
  --concurrency 4 \
  --log verbose
```

### Programmatic API (Recommended for Agents)

The `renderMedia()` function from `@remotion/renderer` is the backbone for automation.

#### Basic Flow

```typescript
import { renderMedia } from '@remotion/renderer';
import { bundle } from '@remotion/bundler';
import { selectComposition } from '@remotion/bundler';

async function renderVideo() {
  // Step 1: Bundle the Remotion project (create webpack bundle)
  const bundleLocation = await bundle({
    entryPoint: require.resolve('./src/index.tsx'),
    webpackOverride: (config) => config,
  });

  // Step 2: Select which composition to render
  const compositions = await selectComposition({
    serveUrl: bundleLocation,
    id: 'HelloWorld',
  });

  // Step 3: Render to file
  const result = await renderMedia({
    composition: compositions,
    serveUrl: bundleLocation,
    codec: 'h264',
    outputLocation: './output.mp4',
    inputProps: {
      text: 'Hello from AI!',
      durationInSeconds: 10,
    },
  });

  console.log('Rendered:', result);
}

renderVideo().catch(console.error);
```

#### Complete renderMedia() Signature (Key Options)

```typescript
interface RenderMediaOptions {
  // Identifiers
  composition: VideoConfig;          // From selectComposition()
  serveUrl: string;                  // Webpack bundle path or URL
  
  // Output
  outputLocation?: string;           // File path (optional = returns Buffer)
  codec: 'h264' | 'h265' | 'vp8' | 'vp9' | 'gif' | 'prores' | ...;
  
  // Input
  inputProps?: Record<string, any>;  // Data passed to composition
  
  // Performance
  concurrency?: number | string;     // Number of parallel frames (default: 50% CPU)
  frameRange?: number | [number, number]; // Render subset of frames
  
  // Encoding
  crf?: number;                      // Quality (0-51, lower = better, default 23)
  videoBitrate?: string;             // e.g., '10M', '5000k'
  audioBitrate?: string;             // e.g., '320k', '128k'
  audioCodec?: 'aac' | 'mp3' | 'opus' | 'pcm-16';
  imageFormat?: 'jpeg' | 'png';      // Frame format before encoding
  
  // Browser/Chromium
  browserExecutable?: string;        // Path to Chrome binary
  chromiumOptions?: ChromiumOptions; // Advanced browser flags
  disableWebSecurity?: boolean;
  gl?: 'angle' | 'egl' | 'swiftshader' | 'vulkan';
  
  // Timeouts & Limits
  timeoutInMilliseconds?: number;    // Default 30000ms (INCREASE for AI agents)
  
  // Callbacks
  onProgress?: (progress: RenderMediaOnProgress) => void;
  onStart?: (data: OnStartData) => void;
  onBrowserLog?: (log: BrowserLog) => void;
  onDownload?: (src: string) => DownloadProgress;
  
  // Hardware
  hardwareAcceleration?: 'disable' | 'if-possible' | 'required';
  offthreadVideoThreads?: number;    // Threads for parallel video extraction
}
```

---

## Headless Rendering

### What "Headless" Means for Remotion

Remotion renders videos via a **headless Chromium browser**—no GUI, no window, pure frame-by-frame output. This is essential for automation.

### Headless Browser Modes (v4.0+)

Remotion supports two Chromium modes:

#### 1. **Headless Shell** (Default, Recommended)

- **Pros:** Fast, stable, minimal deps, no GPU needed
- **Cons:** Can't leverage GPU on Linux
- Configuration: `Config.setChromeMode('headless-shell')` (v4.0.248+)
- **Best for:** General-purpose video rendering, CI/CD, containers

#### 2. **Chrome for Testing** (GPU-Enabled)

- **Pros:** Supports GPU acceleration on Linux (WebGL, gradients, filters, transforms)
- **Cons:** Larger binary, requires GPU drivers installed
- Configuration: `Config.setChromeMode('chrome-for-testing')`
- **Best for:** Heavy WebGL content (Three.js, Mapbox), complex CSS filters

### Critical Headless Configuration for Agents

```typescript
// remotion.config.ts
import { Config } from 'remotion';

// MUST be true for agents (no GUI expected)
Config.setChromiumHeadlessMode(true);

// On Linux, use multi-process rendering (v4.0.42+)
Config.setChromiumMultiProcessOnLinux(true);

// For agent rendering, extend timeout significantly
Config.setTimeoutInMilliseconds(120000); // 2 minutes for safety
```

### Browser Lifecycle Management

When using `renderMedia()` without a pre-opened browser:

1. **Automatic browser start** → Remotion opens a fresh Chromium instance
2. **Render phase** → Frames rendered in parallel
3. **Encoding phase** → FFmpeg encodes PNG/JPEG frames to video
4. **Automatic cleanup** → Browser closed automatically

**For batch operations (multiple renders):**

```typescript
import { openBrowser, closeBrowser } from '@remotion/renderer';

async function batchRender(jobs: RenderJob[]) {
  const browser = await openBrowser();
  
  try {
    for (const job of jobs) {
      await renderMedia({
        ...job,
        puppeteerInstance: browser, // Reuse browser instance
      });
    }
  } finally {
    await closeBrowser(browser);
  }
}
```

**Benefit:** Opens browser once, saves ~3-5 seconds per render.

---

## Docker Deployment

### Production-Ready Dockerfile

```dockerfile
FROM node:20-bullseye

WORKDIR /app

# Install system dependencies for Chromium
RUN apt-get update && apt-get install -y \
  libglib2.0-0 \
  libx11-6 \
  libxext6 \
  libxss1 \
  libxtst6 \
  libxdamage1 \
  libxrandr2 \
  libgconf-2-4 \
  libxkbcommon0 \
  libnss3 \
  libcairo2 \
  libpango-1.0-0 \
  libpangocairo-1.0-0 \
  libgbm1 \
  libasound2 \
  xdg-utils \
  fonts-liberation \
  fonts-dejavu \
  fonts-noto \
  --no-install-recommends && \
  rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package*.json ./

# Install Node dependencies
RUN npm ci --omit=dev

# Copy source code
COPY src ./src
COPY remotion.config.ts .
COPY tsconfig.json .

# Pre-install FFmpeg and Chromium to avoid first-render delays
RUN npm run build && \
  npx remotion install ffmpeg && \
  npx remotion install ffprobe

# Create output directory
RUN mkdir -p /app/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD node -e "require('@remotion/renderer')" || exit 1

EXPOSE 3000

CMD ["node", "dist/server.js"]
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  remotion-renderer:
    build: .
    container_name: remotion-renderer
    environment:
      NODE_ENV: production
      LOG_LEVEL: info
      # Prevent OOM kills during rendering
      NODE_OPTIONS: "--max-old-space-size=4096"
    ports:
      - "3000:3000"
    volumes:
      # Persistent storage for outputs
      - ./output:/app/output
      # Share render cache across container restarts
      - remotion-cache:/app/node_modules/@remotion/compositor-linux-x64/bin
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: on-failure
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

volumes:
  remotion-cache:
```

### Docker Best Practices for Remotion

1. **Use Node.js 20+ Bullseye/Bookworm:** Latest stable with all system libraries
2. **Pre-install FFmpeg/Chromium in image:** Avoid first-render delays in production
3. **Bind mount `/output`:** Let container write to host filesystem
4. **Set memory limits:** Use `NODE_OPTIONS="--max-old-space-size=4096"`
5. **Single-process rendering on Linux:** Already enabled in `config`, but `--multi-process` if GPU available
6. **Health checks:** Include simple validation in startup

### Serverless (AWS Lambda) Deployment

For scalable parallel rendering:

```bash
# Install Lambda-specific package
npm install @remotion/lambda

# Deploy Remotion Lambda function
npx remotion lambda functions deploy --memory=2048

# Render via Lambda
npx remotion lambda render --serve-url=... --composition-id=HelloWorld

# Track progress
npx remotion lambda progress --renderId=<id>
```

Benefits:
- Render multiple videos in parallel
- Auto-scaling based on demand
- S3-based artifact storage
- Cost-effective for batch operations

---

## Known Failure Modes & Prevention

### 1. Timeout Errors

#### Problem
```
Error: A delayRender() was called but not cleared after 30000ms.
```

#### Root Causes
- Google Fonts loading all weights/subsets (v4.0+)
- External HTTP requests hanging
- Missing images/assets
- Heavy CPU-bound composition

#### Prevention & Solutions

```typescript
// ❌ WRONG: Loads all font variants
import { loadFont } from '@remotion/google-fonts/Inter';
loadFont();

// ✅ CORRECT: Load only needed weights
import { loadFont } from '@remotion/google-fonts/Inter';
loadFont('normal', { 
  weights: ['400', '700'],
  subsets: ['latin'],
});

// Also increase timeout in config
Config.setTimeoutInMilliseconds(120000); // 2 minutes for agents
```

### 2. Memory Exhaustion

#### Problem
```
FATAL ERROR: CALL_AND_RETRY_LAST Allocation failed - JavaScript heap out of memory
```

#### Root Causes
- Rendering with `concurrency` too high for available RAM
- Storing frames in memory instead of disk
- `<OffthreadVideo>` with large cache

#### Prevention & Solutions

```typescript
// renderMedia() with controlled concurrency
await renderMedia({
  // ...
  concurrency: 2, // Lower than CPU count if RAM is limited
  offthreadVideoThreads: 1, // Reduce parallel video extraction
  offthreadVideoCacheSizeInBytes: 1024 * 1024 * 512, // 512MB instead of 50% RAM
});

// In Docker: set Node memory limit
NODE_OPTIONS="--max-old-space-size=4096"
```

**Memory Formula:**
```
Estimated RAM = (1920 × 1080 × 4 bytes) × frames_in_parallel × 1.5
             = (8.3 MB × concurrency × 1.5)
```

For 4K@30fps with concurrency=4: ~150MB minimum.

### 3. Chromium Download Failures

#### Problem
```
Error: Chrome download failed. Chromium revision not available.
```

#### Root Causes
- Network interruption during auto-install
- Unsupported architecture (ARM, etc.)
- Disk full during binary download

#### Prevention & Solutions

```typescript
// Pre-install Chromium in CI/Docker
npx remotion install-ffmpeg && npx remotion install-ffprobe

// Provide custom Chromium path if needed
await renderMedia({
  // ...
  browserExecutable: '/usr/bin/chromium-browser',
});

// Or pass pre-opened browser instance
const browser = await openBrowser();
await renderMedia({
  // ...
  puppeteerInstance: browser,
});
```

### 4. Font Loading Errors

#### Problem
```
Error: Failed to load font from http://fonts.googleapis.com/...
Timeout waiting for font: Inter
```

#### Root Causes
- Network timeout fetching Google Fonts
- Font subset not supported in browser
- Font file corrupted during download

#### Prevention & Solutions

```typescript
// Use @remotion/google-fonts with specific subsets
import { loadFont } from '@remotion/google-fonts/Inter';

export const MyVideo = () => {
  loadFont('normal', {
    weights: ['400', '700'],
    subsets: ['latin'], // Only Latin, not Cyrillic
  });
  // ...
};

// OR self-host fonts
import local from './fonts/Inter-Regular.ttf';

export const MyVideo = () => {
  const { fontFamily } = useFont('Inter');
  // ...
};
```

### 5. GPU Not Available in Headless Mode

#### Problem
```
WebGL context not available
Gradients/filters extremely slow in headless rendering
```

#### Root Causes
- Default `headless-shell` mode disables GPU
- GL renderer set to incompatible option
- GPU drivers missing in Docker

#### Prevention & Solutions

```typescript
// Enable GPU-capable rendering on Linux
Config.setChromeMode('chrome-for-testing'); // (v4.0.248+)

// Specify GL renderer
await renderMedia({
  // ...
  gl: 'angle-egl', // or 'vulkan', 'swangle'
});

// In Docker, ensure GPU drivers are available
FROM nvidia/cuda:12.0-runtime-ubuntu22.04
# OR use glvnd libraries
RUN apt-get install -y libglvnd0

// Monitor GPU usage
// Use lighter alternatives to WebGL for headless (CSS transforms preferred)
```

### 6. FFmpeg Codec Not Available

#### Problem
```
Error: Output #0, hvc1, to 'output.mp4': Unknown encoder 'libx265'
Unknown encoder for codec id H265 (18)
```

#### Root Causes
- H.265/VP9 codec not compiled into bundled FFmpeg
- Attempting hardware acceleration on unsupported system

#### Prevention & Solutions

```typescript
// Stick to widely-supported codecs
const SAFE_CODECS = {
  h264: 'Broad compatibility, moderate compression',
  vp8: 'Open source, smaller file size',
  gif: 'No codec needed, animated GIFs',
  prores: 'Professional workflows (large files)',
};

// If H.265 needed, supply custom FFmpeg
await renderMedia({
  // ...
  codec: 'h264', // Default to H.264
  // ffmpegOverride can extend args but can't add encoders
});
```

### 7. Concurrent Rendering Deadlock

#### Problem
```
Render hangs indefinitely
No error, just freezes at 40% completion
```

#### Root Causes
- Browser deadlock under high concurrency
- Event loop starvation
- Memory pressure causing GC pauses

#### Prevention & Solutions

```typescript
// Set concurrency conservatively
await renderMedia({
  // ...
  concurrency: Math.floor(os.cpus().length / 2), // Half CPU threads
  disallowParallelEncoding: true, // Encoding sequential, rendering parallel
});

// Add timeout signal for safety
import { makeCancelSignal } from '@remotion/renderer';
const signal = makeCancelSignal();

setTimeout(() => signal.cancel(), 600000); // 10-minute hard timeout

await renderMedia({
  // ...
  cancelSignal: signal,
});
```

### 8. File System Write Failures

#### Problem
```
Error: EACCES: permission denied, open '/output/video.mp4'
Error: ENOSPC: no space left on device
```

#### Prevention & Solutions

```typescript
import * as fs from 'fs';
import * as path from 'path';

// Pre-flight checks before rendering
function validateOutputPath(outputPath: string) {
  const dir = path.dirname(outputPath);
  
  // Ensure directory exists
  fs.mkdirSync(dir, { recursive: true });
  
  // Check writability
  try {
    fs.accessSync(dir, fs.constants.W_OK);
  } catch {
    throw new Error(`No write permission for ${dir}`);
  }
  
  // Check free disk space (rough estimate)
  const stats = fs.statSync(dir);
  // You'd need disk-usage lib for precise check
  console.log(`Writing to: ${dir}`);
}
```

---

## Best Practices for AI Agents

### 1. Error Handling & Graceful Degradation

```typescript
import { renderMedia, openBrowser, closeBrowser } from '@remotion/renderer';

interface RenderOptions {
  composition: string;
  inputProps: Record<string, any>;
  outputPath: string;
  timeoutMs?: number;
  maxRetries?: number;
}

async function robustRender(opts: RenderOptions, retries = 0): Promise<string> {
  const { composition, inputProps, outputPath, maxRetries = 3 } = opts;
  
  try {
    // Pre-flight validation
    if (!outputPath) throw new Error('outputPath required');
    
    // Get bundle
    const bundleLocation = await bundle({
      entryPoint: require.resolve('./src/index.tsx'),
    });
    
    // Select composition
    const comp = await selectComposition({
      serveUrl: bundleLocation,
      id: composition,
      inputProps,
    });
    
    if (!comp) throw new Error(`Composition "${composition}" not found`);
    
    // Render with timeout
    const result = await Promise.race([
      renderMedia({
        composition: comp,
        serveUrl: bundleLocation,
        codec: 'h264',
        outputLocation: outputPath,
        inputProps,
        onProgress: ({ progress }) => {
          console.log(`Progress: ${Math.round(progress * 100)}%`);
        },
      }),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Render timeout')), opts.timeoutMs || 600000)
      ),
    ]);
    
    return outputPath;
  } catch (error) {
    if (retries < (maxRetries || 3)) {
      console.log(`Retry ${retries + 1}/${maxRetries}:`, error);
      await new Promise(r => setTimeout(r, 2000 * (retries + 1))); // Exponential backoff
      return robustRender(opts, retries + 1);
    }
    throw error;
  }
}
```

### 2. Input Validation Before Rendering

```typescript
import { z } from 'zod';

const VideoConfigSchema = z.object({
  title: z.string().min(1).max(200),
  duration: z.number().min(1).max(600),
  bgColor: z.string().regex(/^#[0-9A-F]{6}$/i),
  fontSize: z.number().min(12).max(200),
});

type VideoConfig = z.infer<typeof VideoConfigSchema>;

async function validateAndRender(props: unknown, composition: string) {
  // Validate input before touching browser
  const validated = VideoConfigSchema.parse(props);
  
  return robustRender({
    composition,
    inputProps: validated,
    outputPath: `/output/${Date.now()}.mp4`,
  });
}
```

### 3. Structured Logging for Debugging

```typescript
interface RenderLog {
  timestamp: string;
  level: 'info' | 'warn' | 'error';
  stage: 'bundle' | 'render' | 'encode';
  message: string;
  duration?: number;
  error?: string;
}

const logs: RenderLog[] = [];

async function loggedRender(opts: RenderOptions) {
  const startTime = Date.now();
  
  try {
    logs.push({
      timestamp: new Date().toISOString(),
      level: 'info',
      stage: 'bundle',
      message: `Starting render: ${opts.composition}`,
    });
    
    // ... render logic
    
    logs.push({
      timestamp: new Date().toISOString(),
      level: 'info',
      stage: 'render',
      message: 'Render complete',
      duration: Date.now() - startTime,
    });
    
    return logs;
  } catch (error) {
    logs.push({
      timestamp: new Date().toISOString(),
      level: 'error',
      stage: 'render',
      message: 'Render failed',
      error: String(error),
      duration: Date.now() - startTime,
    });
    throw error;
  }
}
```

### 4. Resource Pooling & Cleanup

```typescript
import pLimit from 'p-limit';

class RenderPool {
  private browser: Browser | null = null;
  private limit: ReturnType<typeof pLimit>;
  
  constructor(maxConcurrentRenders = 2) {
    this.limit = pLimit(maxConcurrentRenders);
  }
  
  async initialize() {
    this.browser = await openBrowser();
  }
  
  async render(opts: RenderOptions): Promise<string> {
    return this.limit(async () => {
      if (!this.browser) throw new Error('Pool not initialized');
      
      return renderMedia({
        ...opts,
        puppeteerInstance: this.browser,
      });
    });
  }
  
  async cleanup() {
    if (this.browser) {
      await closeBrowser(this.browser);
    }
  }
}

// Usage
const pool = new RenderPool(2);
await pool.initialize();

const results = await Promise.all([
  pool.render({ /* ... */ }),
  pool.render({ /* ... */ }),
  pool.render({ /* ... */ }),
]);

await pool.cleanup();
```

### 5. Composition Registry for Discovery

```typescript
// src/compositions/index.ts
export interface CompositionMetadata {
  id: string;
  name: string;
  description: string;
  defaultProps: Record<string, any>;
  inputSchema: z.ZodSchema;
}

export const COMPOSITIONS: CompositionMetadata[] = [
  {
    id: 'HelloWorld',
    name: 'Hello World',
    description: 'Simple text rendering',
    defaultProps: { text: 'Hello' },
    inputSchema: z.object({ text: z.string() }),
  },
  {
    id: 'Slideshow',
    name: 'Image Slideshow',
    description: 'Animate through image sequence',
    defaultProps: { images: [], duration: 3 },
    inputSchema: z.object({
      images: z.array(z.string().url()),
      duration: z.number().min(1),
    }),
  },
];

// Agents can query available compositions
export function getComposition(id: string) {
  return COMPOSITIONS.find(c => c.id === id);
}
```

---

## Reference Implementation

### Complete Minimal Server for AI Agents

```typescript
// src/index.ts
import express, { Express, Request, Response } from 'express';
import { renderMedia, bundle, selectComposition } from '@remotion/renderer';
import { z } from 'zod';
import * as fs from 'fs/promises';
import * as path from 'path';

const app: Express = express();
app.use(express.json());

const OUTPUT_DIR = path.join(__dirname, '..', 'output');

// Ensure output directory exists
fs.mkdir(OUTPUT_DIR, { recursive: true }).catch(console.error);

// Validation schema
const RenderRequestSchema = z.object({
  composition: z.string(),
  inputProps: z.record(z.any()),
  fileName: z.string().regex(/^[a-zA-Z0-9_-]+$/),
});

// Health check
app.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// List available compositions
app.get('/compositions', async (req: Request, res: Response) => {
  try {
    const bundleLocation = await bundle({
      entryPoint: require.resolve('./compositions/index.tsx'),
    });
    
    const compositions = await selectComposition({
      serveUrl: bundleLocation,
    });
    
    res.json({ compositions: compositions || [] });
  } catch (error) {
    res.status(500).json({ error: String(error) });
  }
});

// Render video
app.post('/render', async (req: Request, res: Response) => {
  try {
    const { composition, inputProps, fileName } = RenderRequestSchema.parse(req.body);
    const outputPath = path.join(OUTPUT_DIR, `${fileName}.mp4`);
    
    // Bundle
    const bundleLocation = await bundle({
      entryPoint: require.resolve('./compositions/index.tsx'),
    });
    
    // Select composition
    const comp = await selectComposition({
      serveUrl: bundleLocation,
      id: composition,
      inputProps,
    });
    
    if (!comp) {
      return res.status(404).json({ error: `Composition "${composition}" not found` });
    }
    
    // Render
    await renderMedia({
      composition: comp,
      serveUrl: bundleLocation,
      codec: 'h264',
      outputLocation: outputPath,
      inputProps,
      onProgress: ({ progress }) => {
        // Could emit progress via WebSocket here
        console.log(`Rendering: ${Math.round(progress * 100)}%`);
      },
    });
    
    res.json({
      success: true,
      outputPath,
      url: `/output/${fileName}.mp4`,
    });
  } catch (error) {
    res.status(400).json({ error: String(error) });
  }
});

// Download video
app.get('/output/:fileName', (req: Request, res: Response) => {
  const filePath = path.join(OUTPUT_DIR, req.params.fileName);
  res.download(filePath);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Render server listening on port ${PORT}`);
});
```

### package.json

```json
{
  "name": "remotion-ai-renderer",
  "version": "1.0.0",
  "description": "AI-agent-ready video generation with Remotion",
  "main": "dist/index.js",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "ts-node src/index.ts",
    "test": "jest",
    "preview": "remotion preview src/compositions/index.tsx"
  },
  "dependencies": {
    "@remotion/bundler": "^4.0.278",
    "@remotion/cli": "^4.0.278",
    "@remotion/renderer": "^4.0.278",
    "express": "^4.18.2",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "zod": "^3.22.4"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/node": "^20.10.6",
    "@types/react": "^18.2.37",
    "typescript": "^5.3.3"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
```

---

## Troubleshooting Checklist

| Issue | Command | Solution |
|-------|---------|----------|
| "Cannot find module 'remotion'" | `npm list remotion` | Run `npm install remotion` |
| FFmpeg download fails | `npx remotion install ffmpeg` | Pre-install in CI/Docker |
| Render timeout | Check `timeoutInMilliseconds` | Increase to 120000+ |
| Out of memory | Check `concurrency` | Lower to 2-3 |
| Chromium not found | `echo $PATH` | Install Node properly via nvm |
| Docker build fails | `docker build --no-cache` | Update base image |

---

## Official References

- **Main Docs:** https://www.remotion.dev/docs
- **renderMedia() API:** https://www.remotion.dev/docs/renderer/render-media
- **Bundle & Select:** https://www.remotion.dev/docs/bundler
- **Chromium Configuration:** https://www.remotion.dev/docs/chromium-flags
- **GPU Rendering:** https://www.remotion.dev/docs/gpu
- **Font Loading:** https://www.remotion.dev/docs/troubleshooting/font-loading-errors
- **Serverless Lambda:** https://www.remotion.dev/docs/lambda
- **GitHub Examples:** https://github.com/remotion-dev/examples

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Maintainer:** Remotion Community  
**For AI Agents:** Claude Code, agentic systems, automation frameworks
