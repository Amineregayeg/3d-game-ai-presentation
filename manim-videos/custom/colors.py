"""
Color palette for VoxFormer STT visualization
Matching the project's Tailwind theme
"""

from manim import *

# Primary colors (matching Tailwind slate theme)
SLATE_50 = "#f8fafc"
SLATE_100 = "#f1f5f9"
SLATE_200 = "#e2e8f0"
SLATE_300 = "#cbd5e1"
SLATE_400 = "#94a3b8"
SLATE_500 = "#64748b"
SLATE_600 = "#475569"
SLATE_700 = "#334155"
SLATE_800 = "#1e293b"
SLATE_900 = "#0f172a"
SLATE_950 = "#020617"

# Accent colors
CYAN_400 = "#22d3ee"
CYAN_500 = "#06b6d4"
CYAN_600 = "#0891b2"

PURPLE_400 = "#c084fc"
PURPLE_500 = "#a855f7"
PURPLE_600 = "#9333ea"

EMERALD_400 = "#34d399"
EMERALD_500 = "#10b981"
EMERALD_600 = "#059669"

AMBER_400 = "#fbbf24"
AMBER_500 = "#f59e0b"
AMBER_600 = "#d97706"

ROSE_400 = "#fb7185"
ROSE_500 = "#f43f5e"
ROSE_600 = "#e11d48"

BLUE_400 = "#60a5fa"
BLUE_500 = "#3b82f6"
BLUE_600 = "#2563eb"

# Semantic colors
PRIMARY = CYAN_500
SECONDARY = PURPLE_500
ACCENT = EMERALD_500
BACKGROUND = SLATE_900
SURFACE = SLATE_800
TEXT_PRIMARY = SLATE_50
TEXT_SECONDARY = SLATE_400
TEXT_MUTED = SLATE_500

# Audio pipeline colors
WAVEFORM_COLOR = CYAN_400
SPECTROGRAM_LOW = SLATE_900
SPECTROGRAM_HIGH = CYAN_400
MEL_FILTER_COLOR = EMERALD_400
STFT_WINDOW_COLOR = AMBER_400

# Transformer colors
QUERY_COLOR = CYAN_400
KEY_COLOR = PURPLE_400
VALUE_COLOR = EMERALD_400
ATTENTION_COLOR = AMBER_400
FFN_COLOR = ROSE_400
CONV_COLOR = BLUE_400

# Training colors
LOSS_COLOR = ROSE_400
GRADIENT_COLOR = PURPLE_400
WEIGHT_COLOR = EMERALD_400

# Gradient definitions for heatmaps
def get_spectrogram_gradient():
    """Returns color gradient for spectrogram visualization"""
    return [SLATE_900, SLATE_700, CYAN_600, CYAN_400, SLATE_50]

def get_attention_gradient():
    """Returns color gradient for attention weights"""
    return [SLATE_900, PURPLE_600, PURPLE_400, AMBER_400, SLATE_50]

def get_loss_gradient():
    """Returns color gradient for loss visualization"""
    return [EMERALD_500, AMBER_500, ROSE_500]

# Helper function to interpolate colors
def interpolate_color(color1, color2, t):
    """Interpolate between two hex colors"""
    c1 = color_to_rgb(color1)
    c2 = color_to_rgb(color2)
    result = [c1[i] + t * (c2[i] - c1[i]) for i in range(3)]
    return rgb_to_color(result)
