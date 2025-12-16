"""
VoxFormer STT Manim Video Scenes
================================

5-part video series explaining the VoxFormer architecture:

1. Audio Pipeline (01_audio_pipeline.py)
   - Waveforms, STFT, Mel filterbanks, normalization

2. Transformer Foundations (02_transformer_foundations.py)
   - Conv subsampling, RoPE, attention mechanism

3. Conformer Block (03_conformer_block.py)
   - FFN, MHSA, convolution module, stacking

4. CTC Loss (04_ctc_loss.py)
   - Alignment problem, forward-backward, decoding

5. Full Pipeline (05_full_pipeline.py)
   - End-to-end architecture, training, inference
"""

from .scenes_01_audio_pipeline import *
from .scenes_02_transformer_foundations import *
from .scenes_03_conformer_block import *
from .scenes_04_ctc_loss import *
from .scenes_05_full_pipeline import *
