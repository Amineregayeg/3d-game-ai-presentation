# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Two Projects - Don't Confuse Them!

### FreelanceHub (Task Tracker)
- **Location**: `/mnt/d/freelance-hub/`
- **VPS**: `5.249.161.66:3002` (`/var/www/freelance-hub/`)
- **Purpose**: Amine's personal project management tool
- **Use For**: Updating task status in backlog database ONLY
- **DO NOT**: Implement product features here

### Salesforce Avatar AI (Hatem's Product)
- **Location**: `/mnt/d/3d-game-ai-presentation/`
- **VPS**: `5.249.161.66:3000` (`/home/developer/3d-game-ai/`)
- **Purpose**: Consulting Delivery OS - the actual product
- **Use For**: Implementing ALL sprint features from the backlog
- **Routes**: `/salesforce_demo`, `/dashboard`, `/login`

### Workflow for Any Sprint Task
1. **Implement** in Salesforce Avatar AI (`/mnt/d/3d-game-ai-presentation/`)
2. **Deploy** to VPS (`5.249.161.66:3000`)
3. **Update status** in FreelanceHub database:
   ```bash
   sshpass -p 'AiDev123123123.' ssh root@5.249.161.66 \
     "sudo -u postgres psql -d freelancehub -c \"UPDATE tasks SET status='done' WHERE title ILIKE '%task name%';\""
   ```

---

## Project Overview

This is a Next.js presentation application showcasing a 3D Game Generation AI Assistant system. The project features three main presentation routes:
- **Home (`/`)**: High-level business presentation about the AI system
- **Technical (`/technical`)**: Deep-dive technical presentation about VoxFormer (custom Speech-to-Text Transformer architecture)
- **RAG (`/rag`)**: Advanced RAG (Retrieval-Augmented Generation) system architecture presentation

The application is a full-screen, keyboard-navigable slide presentation with responsive design using Tailwind CSS and shadcn/ui components.

### Documentation-to-Presentation Pattern

The codebase follows a unique architecture pattern where **detailed technical documentation is transformed into beautiful visual presentations**:

**VoxFormer STT:**
- Source: `/docs/technical/STT_ARCHITECTURE_PLAN.md` (1,400+ line comprehensive technical specification)
- Visualization: `/src/components/tech-slides/` with cyan/purple color scheme

**Advanced RAG:**
- Source: `/docs/technical/RAG_ARCHITECTURE_PLAN.md` (1,500+ line RAG architecture specification)
- Visualization: `/src/components/rag-slides/` with emerald/cyan color scheme

This allows maintaining production-grade technical specifications alongside engaging visual presentations

## Development Commands

- `npm run dev` - Start development server (runs on http://localhost:3000)
- `npm run build` - Build production bundle
- `npm start` - Start production server
- `npm run lint` - Run ESLint to check code quality

## Project Structure

### Core Directories

- **`src/app/`** - Next.js App Router routes
  - `page.tsx` - Main presentation (business overview)
  - `technical/page.tsx` - Technical deep-dive presentation (VoxFormer STT)
  - `rag/page.tsx` - RAG system architecture presentation
  - `layout.tsx` - Root layout with metadata and font configuration
  - `globals.css` - Global Tailwind styles

- **`src/components/`** - React components
  - `slides/` - Business presentation slide components
  - `tech-slides/` - VoxFormer technical presentation slide components
  - `rag-slides/` - RAG system presentation slide components
  - `ui/` - Reusable shadcn/ui components (accordion, badge, card, progress, tabs, table, separator)

- **`src/lib/`** - Utility functions
  - `utils.ts` - Contains `cn()` utility for merging Tailwind classes (clsx + tailwind-merge)

### Configuration Files

- `tsconfig.json` - TypeScript configuration with path alias `@/*` pointing to `src/`
- `next.config.ts` - Next.js configuration (currently minimal)
- `postcss.config.mjs` - PostCSS config using Tailwind v4 (`@tailwindcss/postcss`)
- `eslint.config.mjs` - ESLint configuration using Next.js presets (core-web-vitals and typescript)
- `components.json` - shadcn/ui configuration with Lucide icons and New York style

## Architecture Patterns

### Slide Components

Both presentation routes follow a similar pattern:
1. Main page component manages slide state (`currentSlide`) with React hooks
2. Keyboard navigation (arrow keys, space, enter, home, end, backspace)
3. Navigation UI (prev/next buttons, slide dots, keyboard hints)
4. Renders current slide component dynamically from an array

### Slide Wrapper Components

- **`SlideWrapper`** (business slides) - Creates a centered slide with:
  - Gradient background (slate colors with cyan/purple orbs)
  - Grid pattern background
  - Slide number display (bottom-right)
  - Relative z-index layering for overlays

- **`TechSlideWrapper`** (technical slides) - Creates a technical-style slide with:
  - Similar gradient background with additional emerald orb
  - Circuit pattern SVG overlay
  - Title bar with colored underline
  - Slide counter with dots and padding
  - More layered background effects

### Styling Approach

- **Tailwind CSS v4** with PostCSS support
- **Dark mode** enabled globally (dark class on root elements)
- **Color scheme**: Slate base with cyan/purple accents and gradients
- **Typography**: Geist font family (default), Geist Mono for code/monospace
- **Responsive**: Full viewport slides with backdrop-blur effects
- **shadcn/ui components** used for structured UI elements (cards, tables, accordions, progress bars)

## Adding New Slides

1. Create component in `src/components/slides/` or `src/components/tech-slides/`
2. Export default function component accepting `{ slideNumber, totalSlides }` props
3. Wrap content with `SlideWrapper` or `TechSlideWrapper` component
4. Add to exports in respective `index.ts` file
5. Add component to slides array in page.tsx or technical/page.tsx
6. Keyboard navigation and UI controls automatically work

## Key Dependencies

- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **Tailwind CSS 4** - Utility-first styling
- **shadcn/ui** - High-quality UI components
- **Radix UI** - Headless component primitives (accordion, progress, separator, tabs, slot)
- **Lucide React** - Icon library
- **TypeScript** - Type safety
- **ESLint** - Code quality

## Type Safety

All slide components are typed with explicit props interfaces. The project uses strict TypeScript with proper type checking enabled.

## Technical Documentation

The `/docs` directory contains comprehensive technical specifications:

- **`/docs/technical/STT_ARCHITECTURE_PLAN.md`** - Complete VoxFormer (Speech-to-Text Transformer) specification with:
  - Audio frontend (STFT, Mel filter banks)
  - Rotary position embeddings (RoPE)
  - Multi-head self-attention with detailed implementations
  - Conformer convolution modules
  - SwiGLU feed-forward networks
  - CTC loss with forward-backward algorithm
  - Full model architecture and training strategy
  - Implementation roadmap with phases

- **`/docs/technical/RAG_ARCHITECTURE_PLAN.md`** - Complete Advanced RAG system specification with:
  - Hybrid retrieval (dense vector + sparse BM25)
  - BGE-M3 embeddings (4,096 dimensions) with HNSW indexing
  - RRF (Reciprocal Rank Fusion) for result merging
  - Cross-encoder reranking (MiniLM)
  - Agentic query transformation and validation loops
  - RAGAS evaluation framework
  - PostgreSQL + pgvector implementation
  - 16-week implementation roadmap

- **`/docs/technical/DSP_VOICE_ISOLATION_PLAN.md`** - Voice isolation pipeline specifications

- **`/docs/RAG-research.md`** - Advanced RAG system research notes (source for RAG_ARCHITECTURE_PLAN.md)

These documentation files serve as the source of truth for the technical presentation slides.

## Consulting Delivery OS - Product Backlog

**CRITICAL: Update the backlog after EVERY implementation task.**

After completing any task:
1. Update `docs/PRODUCT_BACKLOG_6_MONTHS.md` - change status ⬜→🔄→✅
2. Update VPS PostgreSQL database: `sudo -u postgres psql -d freelancehub -c "UPDATE tasks SET status='done' WHERE title ILIKE '%task name%';"`
3. Sync to external backup: `cp docs/PRODUCT_BACKLOG_6_MONTHS.* /mnt/d/freelance-hub/scripts/hatem-backlog/`
4. Ensure `scheduled_date` matches `completed_at` for done tasks (no future dates for completed work)

The Consulting Delivery OS is a multi-agent platform for Salesforce consulting. The backlog tracks all work items.

### Backlog Files

| File | Purpose | Update Frequency |
|------|---------|------------------|
| `docs/PRODUCT_BACKLOG_6_MONTHS.md` | Master backlog with status | After each task |
| `docs/PRODUCT_BACKLOG_6_MONTHS.csv` | Jira-importable CSV | Weekly sync |
| `docs/SPRINT_PLAN_6_MONTHS.md` | Sprint-by-sprint plan | Sprint planning |

**External Copy:** `/mnt/d/freelance-hub/scripts/hatem-backlog/` (keep in sync)

### Related V2 Documents (Full Scope)

| Document | Scope | Sprints |
|----------|-------|---------|
| `docs/CONSULTING_DELIVERY_OS_ROADMAP_V2.md` | Full vision (7 agents, all integrations) | 30 weeks |
| `docs/BACKLOG_JIRA_IMPORT_V2.csv` | Complete backlog (367 points) | 18 sprints |
| `docs/SPRINT_PLANNING_SUMMARY_V2.md` | Full sprint plan | 36 weeks |

### 6-Month MVP vs Full Scope

The 6-month backlog is a **reduced scope** MVP:

| Aspect | 6-Month MVP | Full Scope (V2) |
|--------|-------------|-----------------|
| Duration | 26 weeks | 36 weeks |
| Agents | 4 (Discovery, Scoping, Designer, Delivery) | 7 (+ Challenger, QA, RunOps) |
| Integrations | Jira, Slack | Jira, GitHub, Teams, ServiceNow, CI/CD |
| Platforms | Salesforce only | Salesforce, Microsoft, Adobe |

### Backlog Update Workflow

After completing any task:

1. **Update status** in `docs/PRODUCT_BACKLOG_6_MONTHS.md`:
   - Change `⬜ TODO` → `🔄 IN_PROGRESS` → `✅ DONE`
2. **Update CSV** in `docs/PRODUCT_BACKLOG_6_MONTHS.csv`:
   - Change Status column: `To Do` → `In Progress` → `Done`
3. **Sync external copy**:
   ```bash
   cp docs/PRODUCT_BACKLOG_6_MONTHS.* /mnt/d/freelance-hub/scripts/hatem-backlog/
   ```
4. **Update sprint progress** in `docs/SPRINT_PLAN_6_MONTHS.md` if sprint changes

### Current Blockers (Fix First)

1. **Flask blueprints not registered** - Add 2 lines to `backend/app.py`:
   ```python
   from salesforce_rag_api import salesforce_rag_bp
   from salesforce_mcp_api import salesforce_mcp_bp
   app.register_blueprint(salesforce_rag_bp)
   app.register_blueprint(salesforce_mcp_bp)
   ```
2. **Data not indexed** - Run ingestion scripts
3. **Auth UI not connected** - Wire login to backend

- **`/MANIM_VIDEO_PLAN.md`** - Comprehensive plan for creating Manim-based animated videos explaining the VoxFormer STT architecture

- **`/manim-videos/`** - Complete Manim video implementation (based on 3b1b's manim library):
  - `scenes/01_audio_pipeline.py` - Audio Frontend (6 scenes)
  - `scenes/02_transformer_foundations.py` - Attention & RoPE (8 scenes)
  - `scenes/03_conformer_block.py` - Conformer architecture (6 scenes)
  - `scenes/04_ctc_loss.py` - CTC training (7 scenes)
  - `scenes/05_full_pipeline.py` - Integration (6 scenes)
  - `custom/` - Reusable Manim components (colors, DSP, transformer, CTC)
  - `render_all.py` - Main rendering script with quality presets
  - **Total: 33 scenes across 5 videos (~45-60 min content)**

  Usage:
  ```bash
  cd manim-videos
  python render_all.py --list           # List all scenes
  python render_all.py --video 1        # Render Video 1
  python render_all.py --all --quality 4k  # Render all in 4K
  ```

## VPS Deployment

> **Full security details**: See `/docs/VPS_ACCESS.md`

### Server Access (Fresh Install - January 2026)
- **Host**: `5.249.161.66`
- **SSH Port**: `22`
- **Hostname**: `gold-raccoon-61739`

**SSH Key Access (Preferred):**
```bash
ssh vps-zap
# or
ssh -i ~/.ssh/vps_5.249.161.66 root@5.249.161.66
```

**Password Access (Fallback):**
```bash
sshpass -p 'AiDev123123123.' ssh root@5.249.161.66
```

### Live URLs
- **Frontend**: http://5.249.161.66:3000
- **Backend API**: http://5.249.161.66:5000

### Server Environment
- **OS**: Ubuntu 20.04.6 LTS (Focal Fossa)
- **RAM**: 31 GB
- **Disk**: 492 GB
- **Process Manager**: pm2 (to be installed)

### Security Configuration
- **Firewall**: UFW enabled (ports 22, 80, 443, 3000, 5000, 5001, 5002, 9876)
- **Malware Detection**: rkhunter, chkrootkit, ClamAV installed
- **SSH**: Key + password auth enabled (non-aggressive limits)

### Project Structure on VPS
```
/home/developer/
├── 3d-game-ai/
│   ├── frontend/          # Next.js app
│   │   ├── src/
│   │   ├── public/
│   │   ├── .env.local     # NEXT_PUBLIC_API_URL=http://5.249.161.66:5000
│   │   └── ...
│   ├── backend/           # Flask API
│   │   ├── app.py
│   │   ├── venv/          # Python virtual environment
│   │   └── ...
│   ├── voxformer/         # VoxFormer STT project
│   └── ecosystem.config.js # pm2 configuration
├── voxformer_checkpoints/ # Trained model checkpoints (~12GB)
│   ├── best_final_stage1.pt  # Primary Stage 1 checkpoint
│   └── ...
├── voxformer_backup/      # VoxFormer source code backup
├── best_final.pt          # Stage 1 final checkpoint (1.7GB)
└── malek/                 # Other projects
```

### Deployment Workflow

**IMPORTANT: Always work locally first, then deploy to VPS.**

1. **Make changes locally** in `/mnt/d/3d-game-ai-presentation/`
2. **Test locally** with `npm run dev`
3. **Upload changed files to VPS** using scp:
   ```bash
   # Single file
   sshpass -p '<VPS_PASSWORD>' scp -o StrictHostKeyChecking=no /mnt/d/3d-game-ai-presentation/src/path/to/file.tsx root@5.249.161.66:/home/developer/3d-game-ai/frontend/src/path/to/

   # Multiple files or directory
   sshpass -p '<VPS_PASSWORD>' scp -r -o StrictHostKeyChecking=no /mnt/d/3d-game-ai-presentation/src/components/ root@5.249.161.66:/home/developer/3d-game-ai/frontend/src/
   ```
4. **Rebuild on VPS**:
   ```bash
   sshpass -p '<VPS_PASSWORD>' ssh -o StrictHostKeyChecking=no root@5.249.161.66 'cd /home/developer/3d-game-ai/frontend && npm run build'
   ```
5. **Restart services**:
   ```bash
   sshpass -p '<VPS_PASSWORD>' ssh -o StrictHostKeyChecking=no root@5.249.161.66 'pm2 restart frontend'
   ```

### Useful pm2 Commands
```bash
pm2 status              # Check service status
pm2 restart frontend    # Restart frontend
pm2 restart backend     # Restart backend
pm2 logs frontend       # View frontend logs
pm2 logs backend        # View backend logs
pm2 save                # Save current process list
```

### Backend Notes
- Flask app with SQLAlchemy (SQLite database)
- CORS configured for localhost:3000 and VPS IP
- Vault password: `admin123`
- Virtual environment at `backend/venv/`
- Dependencies: flask, flask-cors, flask-sqlalchemy, gunicorn, bcrypt, pyjwt

### Old VPS (DEPRECATED - Expired Dec 2024)
- Host: `134.255.234.188` - NO LONGER AVAILABLE

## GPU Server (Vast.ai - Full Demo Pipeline)

### Current Instance (December 2024)
- **Host**: `80.188.223.202`
- **SSH Port**: `17757`
- **Proxy**: `ssh -p 31523 root@ssh2.vast.ai`
- **GPU**: H100 or RTX 4090 (varies)
- **Provider**: Vast.ai

### Connection
```bash
# Direct SSH
ssh -p 17757 root@80.188.223.202

# Via proxy
ssh -p 31523 root@ssh2.vast.ai

# Via VPS
sshpass -p '<VPS_PASSWORD>' ssh root@5.249.161.66 \
  'ssh -p 17757 root@80.188.223.202 "command"'
```

### Required SSH Tunnels (IMPORTANT)

**Before testing /demo or /full_demo, these tunnels MUST be active on VPS:**

| Port | Service | Purpose |
|------|---------|---------|
| 5001 | Whisper + SadTalker | STT and lip-sync for /full_demo |
| 5002 | VoxFormer | Custom STT for /demo |

**Start tunnels on VPS:**
```bash
# SSH into VPS first, then run:
ssh -p 17757 -f -N -L 5001:localhost:5001 root@80.188.223.202
ssh -p 17757 -f -N -L 5002:localhost:5002 root@80.188.223.202
```

**Check if tunnels are active:**
```bash
ps aux | grep "ssh.*500" | grep -v grep
# Should show both 5001 and 5002 tunnels
```

**Test tunnels:**
```bash
curl -s http://localhost:5001/health  # Whisper/SadTalker
curl -s http://localhost:5002/health  # VoxFormer
```

**Note:** Tunnels do NOT persist after VPS reboot. Re-run the commands above if services return "connection failed".

### GPU Server Structure (Stage 4)
```
/root/voxformer/
├── configs/
│   └── stage4.yaml              # CTC-only training config
├── checkpoints/
│   ├── stage2/best.pt           # Starting checkpoint (1.8GB)
│   └── stage4/                  # Stage 4 outputs
├── data/LibriSpeech/
│   ├── train-clean-100/         # 28,539 samples (6.6GB)
│   └── dev-clean/               # Evaluation set
├── src/                         # Model code
├── scripts/train.py             # Training script
├── tokenizer/                   # BPE tokenizer (vocab=2000)
├── train_stage4.sh              # Main launcher
├── auto_backup.sh               # Backup to VPS every 5 min
├── metrics_updater.sh           # Updates metrics.json for dashboard
└── metrics.json                 # Real-time metrics for dashboard
```

### Stage 4 Training Commands
```bash
# Start Stage 4 training (in tmux)
ssh -p 17757 root@145.236.166.111
cd /root/voxformer
tmux new -s training
./train_stage4.sh
# Detach: Ctrl+B, D

# Monitor training
tmux attach -t training
# or
tmux capture-pane -t training -p -S -20

# Check GPU
nvidia-smi
```

### Stage 4 Configuration
- **Epochs**: 5
- **Batch size**: 8 (effective 32 with gradient accumulation)
- **Learning rate**: 5e-6
- **Loss**: CTC-only (ce_weight=0)
- **Checkpoint interval**: 500 steps (~7 min)
- **Backup interval**: 5 minutes to VPS
- **Target WER**: < 15%

### Training Dashboard
- **URL**: http://5.249.161.66:3000/training
- **Features**:
  - Real-time epoch/step progress
  - Loss tracking (CTC)
  - WER display (when available)
  - GPU utilization & memory
  - Backup status with timestamp
  - ETA calculation

### VPS Backup Location
```
/home/developer/voxformer_checkpoints/
├── stage2/best.pt               # Stage 2 checkpoint (source)
├── stage4/                      # Stage 4 backups
│   ├── metrics.json             # Dashboard reads this
│   ├── best.pt                  # Best checkpoint
│   ├── latest.pt                # Most recent checkpoint
│   ├── epoch_history.log        # Completed epochs
│   └── training.log             # Full training log
```

### Recovery (if GPU instance crashes)
```bash
# On new GPU instance, download from VPS:
sshpass -p '    ' scp developer@5.249.161.66:/home/developer/voxformer_checkpoints/stage4/latest.pt \
  /root/voxformer/checkpoints/stage4/

# Update config to resume:
sed -i 's|resume_from:.*|resume_from: checkpoints/stage4/latest.pt|' configs/stage4.yaml

# Continue training:
./train_stage4.sh
```

### Training Documentation
- **Plan**: `/docs/technical/VOXFORMER_STAGE4_PLAN.md`
- **Architecture**: `/docs/technical/STT_ARCHITECTURE_PLAN.md`

### Avatar Demo
- **URL**: http://5.249.161.66:3000/avatar_demo
- **API**: POST http://5.249.161.66:5000/api/avatar/speak
- Uses: ElevenLabs TTS + SadTalker lip-sync

## Full Demo Pipeline (CRITICAL - Keep Intact)

The Full Demo at `/full_demo` integrates multiple services. **These must be preserved when making changes:**

### Critical Services Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FULL DEMO PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │   Blender   │    │  SadTalker  │    │  VoxFormer  │             │
│  │     MCP     │    │   Lipsync   │    │     STT     │             │
│  │  Port 9876  │    │  GPU :5001  │    │  Flask API  │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                  │                  │                      │
│        ▼                  ▼                  ▼                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Flask Backend (Port 5000)                       │   │
│  │  /api/blender/generate  /api/gpu/lipsync  /api/voxformer/*  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Next.js Frontend (Port 3000)                    │   │
│  │                    /full_demo page                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. Blender MCP Server (DO NOT BREAK)

**Purpose**: Real-time 3D model generation via Claude + Blender

**Location**: VPS (5.249.161.66)
- **Port**: 9876
- **Process**: Blender running in GUI mode with Xvfb
- **Script**: `/root/blender_server_gui.py`
- **Startup**: `/root/start_blender_mcp.sh`
- **Systemd**: `blender-mcp.service` (auto-start on boot)

**Critical Fix Applied**: Must run Blender in **GUI mode** (not `--background`) with Xvfb for proper context. The glTF exporter requires `bpy.context.active_object` which only exists in GUI mode.

**Check Status**:
```bash
ss -tlnp | grep 9876  # Should show blender listening
curl -s http://localhost:5000/api/blender/health  # Should show blender_connected: true
```

**If Broken - Recovery**:
```bash
/root/start_blender_mcp.sh
pm2 restart backend
```

### 2. SadTalker Lipsync (DO NOT BREAK)

**Purpose**: Generate talking avatar videos from audio

**Location**: GPU Server (Vast.ai)
- **Host**: `80.188.223.202` (or current Vast.ai instance)
- **Port**: `17757` (SSH), `5001` (API via tunnel)
- **Script**: `/root/full_demo_api.py`

**VPS Proxy**: Flask backend proxies to GPU via SSH tunnel
- **Endpoint**: `POST /api/gpu/lipsync`
- **Timeout**: 600 seconds (for long videos)
- **Proxy Code**: `/home/developer/3d-game-ai/backend/gpu_proxy.py`

**Check Status**:
```bash
# From VPS
ssh -p 17757 root@80.188.223.202 "curl -s http://localhost:5001/health"
```

### 3. Whisper STT (Current - Works)

**Purpose**: Speech-to-text transcription

**Location**: GPU Server
- **Endpoint**: `POST /api/gpu/stt`
- Uses OpenAI Whisper model on GPU

### 4. VoxFormer STT (To Be Implemented)

**Purpose**: Custom speech-to-text using trained VoxFormer model

**Recommended Checkpoint**: `/home/developer/voxformer_checkpoints/stage2/best.pt`
- Size: 1.8GB
- Tokenizer: `/home/developer/voxformer_backup/tokenizer/`
- Source: `/home/developer/voxformer_backup/src/`

**Target Endpoint**: `POST /api/voxformer/transcribe`

**Documentation**: `/docs/technical/VOXFORMER_DEPLOYMENT.md`

### Service Dependencies

| Service | Port | Depends On | Safe to Restart |
|---------|------|------------|-----------------|
| Frontend | 3000 | Backend | Yes |
| Backend | 5000 | Blender MCP, GPU tunnel | Yes (with care) |
| Blender MCP | 9876 | Xvfb | Yes (use startup script) |
| GPU API | 5001 | Vast.ai instance | Check GPU first |

### Before Making Changes

1. **Check all services are running**:
   ```bash
   pm2 status
   ss -tlnp | grep -E '3000|5000|9876'
   ```

2. **Test critical endpoints**:
   ```bash
   curl http://localhost:5000/api/blender/health
   curl http://localhost:5000/api/gpu/health
   ```

3. **After changes, verify**:
   - Blender generation still works
   - Lipsync still works
   - No 500 errors in pm2 logs

## GPU Server (Vast.ai - Updated December 2024)

### Current Instance
- **Host**: `80.188.223.202`
- **SSH Port**: `17757`
- **Alternative**: `ssh -p 31523 root@ssh2.vast.ai`
- **GPU**: H100 or RTX 4090 (varies by instance)

### Services Running on GPU
1. **SadTalker API** (`/root/full_demo_api.py`) - Port 5001
2. **Whisper STT** - Integrated in full_demo_api.py

### GPU Backup Location (VPS)
```
/home/developer/gpu_backup_20241218/
├── full_demo_api.py      # SadTalker + Whisper API
└── fix_stt.py            # STT fixes
```
