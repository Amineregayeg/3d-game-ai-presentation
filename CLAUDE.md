# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### Server Access (NEW VPS - Valid for 6 months from Dec 2024)
- **Host**: `5.249.161.66`
- **SSH Port**: `22`
- **Root Access**: `ssh root@5.249.161.66` (password: `<VPS_PASSWORD>`)
- **Developer User**: `ssh developer@5.249.161.66` (password: <DEV_PASSWORD>)

### Live URLs
- **Frontend**: http://5.249.161.66:3000
- **Backend API**: http://5.249.161.66:5000

### Server Environment
- **OS**: Debian 13 (Trixie)
- **Node.js**: v20.19.6
- **Python**: 3.13.5
- **Process Manager**: pm2 (installed globally)

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

## GPU Server (Vast.ai - VoxFormer Training)

### Current Instance (Stage 4 - December 2024)
- **Host**: `145.236.166.111`
- **SSH Port**: `17757`
- **Instance ID**: 28907731
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: AMD EPYC 7702 (16 cores)
- **RAM**: 64.4 GB
- **Provider**: Vast.ai (~$0.40/hr)

### Connection
```bash
# Direct SSH
ssh -p 17757 root@145.236.166.111

# Via VPS (passwordless - SSH key configured)
sshpass -p '<VPS_PASSWORD>' ssh root@5.249.161.66 \
  'ssh -p 17757 root@145.236.166.111 "command"'
```

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
