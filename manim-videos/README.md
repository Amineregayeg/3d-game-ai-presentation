# VoxFormer STT Manim Videos

Animated video explanations of the VoxFormer Speech-to-Text Transformer architecture using [Manim](https://github.com/3b1b/manim) (3Blue1Brown's animation library).

## Overview

This project creates a 5-part video series (~45-60 minutes total) explaining the VoxFormer architecture:

| Video | Topic | Duration | Scenes |
|-------|-------|----------|--------|
| 1 | Audio Pipeline | 8-10 min | 6 |
| 2 | Transformer Foundations | 10-12 min | 8 |
| 3 | Conformer Block | 10-12 min | 6 |
| 4 | CTC Loss | 8-10 min | 7 |
| 5 | Full Pipeline | 8-10 min | 6 |

**Total: 33 scenes**

## Prerequisites

### Install Manim (3b1b version)

```bash
# Install manimgl (NOT manim community edition)
pip install manimgl

# Required dependencies
# FFmpeg, OpenGL, LaTeX (optional)
```

### System Requirements

- Python 3.8+
- FFmpeg
- OpenGL support
- 8GB+ RAM recommended
- GPU recommended for 4K rendering

## Project Structure

```
manim-videos/
├── scenes/                    # Video scene files
│   ├── 01_audio_pipeline.py   # Video 1: Audio processing
│   ├── 02_transformer_foundations.py  # Video 2: Attention, RoPE
│   ├── 03_conformer_block.py  # Video 3: Conformer architecture
│   ├── 04_ctc_loss.py         # Video 4: CTC training
│   └── 05_full_pipeline.py    # Video 5: Integration
├── custom/                    # Custom Manim components
│   ├── colors.py              # Color palette (matches project theme)
│   ├── dsp_components.py      # Audio/DSP visualizations
│   ├── transformer_components.py  # Transformer visualizations
│   └── ctc_components.py      # CTC algorithm visualizations
├── assets/                    # Media assets
│   ├── audio_samples/
│   ├── spectrograms/
│   └── reference_images/
├── output/                    # Rendered videos
├── render_all.py              # Main render script
├── manim_imports_ext.py       # Universal imports
└── custom_config.yml          # Manim configuration
```

## Usage

### List All Scenes

```bash
python render_all.py --list
```

### Render Specific Video

```bash
# Render Video 1 (Audio Pipeline)
python render_all.py --video 1

# Preview mode (fast, low quality)
python render_all.py --video 1 --preview

# High quality
python render_all.py --video 1 --quality high
```

### Render Specific Scene

```bash
python render_all.py --scene AudioWaveformScene
python render_all.py --scene RoPEScene --preview
```

### Render All Videos

```bash
# Medium quality (default)
python render_all.py --all

# Production quality (4K)
python render_all.py --all --quality 4k
```

### Direct Manim Commands

```bash
# Preview a scene interactively
manimgl scenes/01_audio_pipeline.py AudioWaveformScene

# Render to file
manimgl -qh scenes/01_audio_pipeline.py AudioWaveformScene
```

## Quality Presets

| Preset | Resolution | FPS | Use Case |
|--------|------------|-----|----------|
| preview | 480p | 15 | Quick iteration |
| low | 480p | 15 | Testing |
| medium | 720p | 30 | Development |
| high | 1080p | 60 | Good quality |
| production | 1440p | 60 | High quality |
| 4k | 2160p | 60 | Final output |

## Custom Components

### DSP Components (`custom/dsp_components.py`)

- `AudioWaveform` - Animated waveform display
- `SpeechWaveform` - Speech-like complex waveform
- `STFTWindow` - Sliding window with Hann function
- `FrequencySpectrum` - Bar chart spectrum display
- `MelFilterBank` - Triangular filter visualization
- `SpectrogramDisplay` - Heatmap spectrogram
- `ProcessingBlock` - Generic processing block
- `DataFlowArrow` - Animated data flow arrows

### Transformer Components (`custom/transformer_components.py`)

- `AttentionHead` - Single attention matrix
- `MultiHeadAttention` - Multiple attention heads
- `RotaryEmbeddingVisualization` - RoPE 2D rotation
- `ConformerBlock` - Full conformer structure
- `TransformerStack` - Stacked layers
- `FeedForwardNetwork` - FFN with SwiGLU
- `ConvolutionModule` - Depthwise convolution
- `QueryKeyValueProjection` - Q,K,V projection
- `ScaledDotProductAttention` - Attention formula

### CTC Components (`custom/ctc_components.py`)

- `CTCTrellis` - Dynamic programming trellis
- `CTCForwardVisualization` - Forward algorithm
- `CTCBackwardVisualization` - Backward algorithm
- `CTCLossVisualization` - Loss computation
- `CTCDecoder` - Decoding process
- `AlignmentVisualization` - Multiple alignments
- `BeamSearchVisualization` - Beam search tree

## Color Palette

Matches the project's Tailwind theme:

```python
# Primary colors
CYAN_500 = "#06b6d4"    # Primary accent
PURPLE_500 = "#a855f7"  # Secondary accent
EMERALD_500 = "#10b981" # Success/positive
AMBER_500 = "#f59e0b"   # Warning/highlight
ROSE_500 = "#f43f5e"    # Error/important

# Background
SLATE_900 = "#0f172a"   # Main background
SLATE_800 = "#1e293b"   # Surface
```

## Estimated Render Times

| Quality | Per Scene | Full Series |
|---------|-----------|-------------|
| preview | ~10 sec | ~5 min |
| medium | ~1 min | ~30 min |
| high | ~3 min | ~1.5 hours |
| 4k | ~10 min | ~5 hours |

*Times vary based on scene complexity and hardware*

## Development

### Adding New Scenes

1. Create scene class in appropriate video file
2. Add to scene list in `render_all.py`
3. Test with preview: `manimgl -pql file.py SceneName`

### Creating Custom Components

1. Add to appropriate file in `custom/`
2. Import in `custom/__init__.py`
3. Use in scene files

### Testing

```bash
# Interactive preview
manimgl scenes/01_audio_pipeline.py -se 10

# This drops into IPython at line 10 for debugging
```

## Integration

These videos complement the web-based presentations at:
- `/technical` - VoxFormer STT slides (React/Next.js)
- `/rag` - RAG system slides (React/Next.js)

Consider embedding completed videos in the web presentations or linking to a YouTube playlist.

## Credits

- Animation library: [Manim (3b1b version)](https://github.com/3b1b/manim)
- Inspired by: [3Blue1Brown](https://www.3blue1brown.com/)
- Architecture: VoxFormer STT from `/docs/technical/STT_ARCHITECTURE_PLAN.md`

## License

Same as parent project.
