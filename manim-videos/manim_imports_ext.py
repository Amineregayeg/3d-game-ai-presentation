"""
VoxFormer STT Manim Video Project
Universal imports file

Usage:
    from manim_imports_ext import *
"""

# Standard Manim imports
from manim import *
import numpy as np
import math

# Custom components
from custom.colors import *
from custom.dsp_components import *
from custom.transformer_components import *
from custom.ctc_components import *

# Common configuration
config.background_color = SLATE_900
config.frame_rate = 30

# Default text settings
Text.set_default(font="sans-serif")
