"""
Transformer architecture components for visualization
Includes: Attention, RoPE, Conformer blocks, FFN
"""

from manim import *
import numpy as np
from .colors import *


class AttentionHead(VGroup):
    """
    Single attention head visualization
    """

    def __init__(
        self,
        seq_len=5,
        head_dim=64,
        width=3,
        height=3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.seq_len = seq_len
        self.head_dim = head_dim

        # Create attention matrix visualization
        self.matrix = self._create_attention_matrix(width, height)
        self.add(self.matrix)

    def _create_attention_matrix(self, width, height):
        """Create visual attention weight matrix"""
        matrix = VGroup()

        cell_size = min(width, height) / self.seq_len

        # Generate random attention weights (softmaxed)
        weights = np.random.rand(self.seq_len, self.seq_len)
        weights = np.exp(weights) / np.exp(weights).sum(axis=1, keepdims=True)

        for i in range(self.seq_len):
            for j in range(self.seq_len):
                x = (j - self.seq_len / 2 + 0.5) * cell_size
                y = (self.seq_len / 2 - i - 0.5) * cell_size

                cell = Square(
                    side_length=cell_size * 0.9,
                    fill_color=self._weight_to_color(weights[i, j]),
                    fill_opacity=1,
                    stroke_width=0
                )
                cell.move_to([x, y, 0])
                matrix.add(cell)

        return matrix

    def _weight_to_color(self, weight):
        """Convert attention weight to color"""
        return interpolate_color(SLATE_800, ATTENTION_COLOR, weight)


class MultiHeadAttention(VGroup):
    """
    Multi-head attention visualization showing multiple heads
    """

    def __init__(
        self,
        n_heads=8,
        seq_len=5,
        spacing=0.3,
        head_size=1.5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_heads = n_heads
        self.heads = []

        # Arrange heads in a row
        total_width = n_heads * head_size + (n_heads - 1) * spacing

        for i in range(n_heads):
            head = AttentionHead(
                seq_len=seq_len,
                width=head_size,
                height=head_size
            )
            x_pos = (i - n_heads / 2 + 0.5) * (head_size + spacing)
            head.move_to([x_pos, 0, 0])

            # Add head number label
            label = Text(f"H{i+1}", font_size=14, color=SLATE_400)
            label.next_to(head, DOWN, buff=0.1)

            self.heads.append(head)
            self.add(head, label)

        # Add title
        title = Text("Multi-Head Attention", font_size=20, color=SLATE_50)
        title.next_to(self, UP, buff=0.3)
        self.add(title)


class RotaryEmbeddingVisualization(VGroup):
    """
    Visualization of Rotary Position Embeddings (RoPE)
    Shows 2D rotation of vector pairs
    """

    def __init__(
        self,
        dim=4,
        max_pos=8,
        radius=1.5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.max_pos = max_pos
        self.radius = radius

        # Create multiple 2D subspaces
        self.subspaces = self._create_subspaces()
        self.add(self.subspaces)

    def _create_subspaces(self):
        """Create 2D rotation visualizations for each dimension pair"""
        subspaces = VGroup()

        n_pairs = self.dim // 2
        spacing = 3.5

        for pair_idx in range(n_pairs):
            # Create circle (unit circle for rotation)
            circle = Circle(
                radius=self.radius,
                color=SLATE_600,
                stroke_width=1
            )

            # Position
            x_pos = (pair_idx - n_pairs / 2 + 0.5) * spacing
            circle.move_to([x_pos, 0, 0])

            # Add dimension label
            dim_label = Text(
                f"dims {2*pair_idx}, {2*pair_idx+1}",
                font_size=14,
                color=SLATE_400
            )
            dim_label.next_to(circle, DOWN, buff=0.2)

            # Create vectors at different positions
            theta_base = 10000 ** (-2 * pair_idx / self.dim)
            vectors = VGroup()

            for pos in range(min(4, self.max_pos)):
                angle = pos * theta_base
                end_point = [
                    x_pos + self.radius * 0.8 * np.cos(angle),
                    self.radius * 0.8 * np.sin(angle),
                    0
                ]

                vec = Arrow(
                    start=[x_pos, 0, 0],
                    end=end_point,
                    color=interpolate_color(CYAN_400, PURPLE_400, pos / 4),
                    stroke_width=2,
                    buff=0
                )
                vectors.add(vec)

                # Position label
                pos_label = Text(f"p={pos}", font_size=10, color=SLATE_500)
                pos_label.move_to(end_point)
                pos_label.shift(
                    0.3 * np.array([np.cos(angle), np.sin(angle), 0])
                )
                vectors.add(pos_label)

            subspaces.add(circle, dim_label, vectors)

        return subspaces


class ConformerBlock(VGroup):
    """
    Visual representation of a Conformer block
    """

    def __init__(
        self,
        width=3,
        height=6,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Components
        components = [
            ("FFN (½)", FFN_COLOR, "Feed-Forward"),
            ("MHSA", ATTENTION_COLOR, "Multi-Head Self-Attention"),
            ("Conv", CONV_COLOR, "Convolution Module"),
            ("FFN (½)", FFN_COLOR, "Feed-Forward"),
            ("LayerNorm", SLATE_500, "Layer Normalization"),
        ]

        block_height = height / len(components)
        y_offset = height / 2 - block_height / 2

        self.blocks = []

        for i, (name, color, desc) in enumerate(components):
            block = RoundedRectangle(
                width=width,
                height=block_height * 0.8,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color,
                stroke_width=2
            )
            y_pos = y_offset - i * block_height
            block.move_to([0, y_pos, 0])

            label = Text(name, font_size=16, color=SLATE_50)
            label.move_to(block.get_center())

            self.blocks.append(block)
            self.add(block, label)

            # Add arrows between blocks
            if i < len(components) - 1:
                arrow = Arrow(
                    start=[0, y_pos - block_height * 0.4, 0],
                    end=[0, y_pos - block_height * 0.6, 0],
                    color=SLATE_600,
                    stroke_width=2,
                    buff=0
                )
                self.add(arrow)

        # Residual connection (visual)
        residual = CurvedArrow(
            start_point=[-width/2 - 0.3, y_offset + block_height * 0.3, 0],
            end_point=[-width/2 - 0.3, -y_offset - block_height * 0.3, 0],
            color=EMERALD_400,
            stroke_width=2
        )
        res_label = Text("+", font_size=24, color=EMERALD_400)
        res_label.next_to(residual, LEFT, buff=0.1)
        self.add(residual, res_label)

        # Title
        title = Text("Conformer Block", font_size=20, color=SLATE_50)
        title.next_to(self, UP, buff=0.3)
        self.add(title)


class TransformerStack(VGroup):
    """
    Stack of transformer/conformer blocks
    """

    def __init__(
        self,
        n_layers=6,
        block_width=2,
        block_height=0.8,
        spacing=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_layers = n_layers

        total_height = n_layers * (block_height + spacing)

        for i in range(n_layers):
            block = RoundedRectangle(
                width=block_width,
                height=block_height,
                corner_radius=0.1,
                fill_color=CYAN_500,
                fill_opacity=0.2,
                stroke_color=CYAN_500,
                stroke_width=2
            )

            y_pos = total_height / 2 - (i + 0.5) * (block_height + spacing)
            block.move_to([0, y_pos, 0])

            label = Text(f"Layer {i+1}", font_size=14, color=SLATE_50)
            label.move_to(block.get_center())

            self.add(block, label)


class FeedForwardNetwork(VGroup):
    """
    Visualization of Feed-Forward Network with SwiGLU
    """

    def __init__(
        self,
        d_model=512,
        d_ff=2048,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Input layer
        input_layer = self._create_layer(d_model, "Input", SLATE_500, 0)

        # Expand layer
        expand_layer = self._create_layer(d_ff, "Expand", FFN_COLOR, 2)

        # Gate layer (for SwiGLU)
        gate_layer = self._create_layer(d_ff, "Gate", AMBER_400, 2)
        gate_layer.shift(RIGHT * 2)

        # SwiGLU activation
        swiglu = Text("SwiGLU", font_size=16, color=AMBER_400)
        swiglu.move_to([2, 1, 0])

        # Output layer
        output_layer = self._create_layer(d_model, "Output", SLATE_500, 4)

        # Arrows
        arrow1 = Arrow(start=[0, 0, 0], end=[0, 2, 0], color=SLATE_600, buff=0.2)
        arrow2 = Arrow(start=[0, 2, 0], end=[0, 4, 0], color=SLATE_600, buff=0.2)

        self.add(input_layer, expand_layer, output_layer, arrow1, arrow2, swiglu)

    def _create_layer(self, size, name, color, y_pos):
        """Create a layer visualization"""
        layer = VGroup()

        rect = RoundedRectangle(
            width=2,
            height=0.6,
            corner_radius=0.1,
            fill_color=color,
            fill_opacity=0.2,
            stroke_color=color,
            stroke_width=2
        )
        rect.move_to([0, y_pos, 0])

        label = Text(f"{name} ({size})", font_size=12, color=SLATE_50)
        label.move_to(rect.get_center())

        layer.add(rect, label)
        return layer


class ConvolutionModule(VGroup):
    """
    Visualization of the convolution module in Conformer
    """

    def __init__(
        self,
        kernel_size=31,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Sequence visualization
        seq_len = 15
        cell_width = 0.4

        sequence = VGroup()
        for i in range(seq_len):
            cell = Square(
                side_length=cell_width * 0.9,
                fill_color=SLATE_700,
                fill_opacity=0.8,
                stroke_color=SLATE_600,
                stroke_width=1
            )
            cell.move_to([(i - seq_len / 2) * cell_width, 0, 0])
            sequence.add(cell)

        self.add(sequence)

        # Kernel highlight
        kernel_display_size = 5  # Show 5 cells as kernel
        kernel_rect = Rectangle(
            width=kernel_display_size * cell_width,
            height=cell_width * 1.2,
            fill_color=CONV_COLOR,
            fill_opacity=0.3,
            stroke_color=CONV_COLOR,
            stroke_width=2
        )
        kernel_rect.move_to([0, 0, 0])
        self.add(kernel_rect)

        # Label
        label = Text(f"Kernel Size: {kernel_size}", font_size=16, color=SLATE_400)
        label.next_to(sequence, DOWN, buff=0.3)
        self.add(label)

        # Depthwise conv explanation
        dw_label = Text("Depthwise Separable Convolution", font_size=14, color=CONV_COLOR)
        dw_label.next_to(label, DOWN, buff=0.2)
        self.add(dw_label)


class QueryKeyValueProjection(VGroup):
    """
    Visualization of Q, K, V projections
    """

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        **kwargs
    ):
        super().__init__(**kwargs)

        head_dim = d_model // n_heads

        # Input
        input_rect = RoundedRectangle(
            width=3,
            height=0.8,
            corner_radius=0.1,
            fill_color=SLATE_600,
            fill_opacity=0.3,
            stroke_color=SLATE_500,
            stroke_width=2
        )
        input_label = Text(f"X ({d_model})", font_size=14, color=SLATE_50)
        input_label.move_to(input_rect.get_center())
        input_rect.move_to([0, 2, 0])
        input_label.move_to([0, 2, 0])

        # Q, K, V projections
        projections = [
            ("Q", QUERY_COLOR, -2.5),
            ("K", KEY_COLOR, 0),
            ("V", VALUE_COLOR, 2.5),
        ]

        self.add(input_rect, input_label)

        for name, color, x_pos in projections:
            rect = RoundedRectangle(
                width=2,
                height=0.8,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color,
                stroke_width=2
            )
            rect.move_to([x_pos, 0, 0])

            label = Text(f"{name} ({d_model})", font_size=12, color=SLATE_50)
            label.move_to(rect.get_center())

            # Arrow from input
            arrow = Arrow(
                start=[0, 1.5, 0],
                end=[x_pos, 0.5, 0],
                color=SLATE_600,
                stroke_width=2,
                buff=0.1
            )

            # Projection matrix label
            matrix_label = Text(f"W_{name}", font_size=10, color=color)
            matrix_label.move_to(arrow.get_center())
            matrix_label.shift(UP * 0.2 + RIGHT * (x_pos / 4))

            self.add(rect, label, arrow, matrix_label)

        # Reshape notation
        reshape = Text(
            f"Reshape to ({n_heads} heads × {head_dim} dim)",
            font_size=12,
            color=SLATE_400
        )
        reshape.move_to([0, -1, 0])
        self.add(reshape)


class ScaledDotProductAttention(VGroup):
    """
    Visual representation of scaled dot-product attention formula
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Formula components
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=32
        )
        formula.set_color(SLATE_50)

        # Color different parts
        # This would need index-based coloring in actual implementation

        self.add(formula)

        # Step breakdown
        steps = VGroup()
        step_texts = [
            ("1. Compute scores:", r"S = QK^T", QUERY_COLOR),
            ("2. Scale:", r"S' = S / \sqrt{d_k}", SLATE_400),
            ("3. Softmax:", r"W = \text{softmax}(S')", ATTENTION_COLOR),
            ("4. Weight values:", r"O = W \cdot V", VALUE_COLOR),
        ]

        for i, (desc, math, color) in enumerate(step_texts):
            step = VGroup()
            desc_text = Text(desc, font_size=14, color=SLATE_400)
            math_text = MathTex(math, font_size=18)
            math_text.set_color(color)
            math_text.next_to(desc_text, RIGHT, buff=0.2)
            step.add(desc_text, math_text)
            step.move_to([0, -1.5 - i * 0.6, 0])
            steps.add(step)

        self.add(steps)
