"""
Video 2: Transformer Foundations
Duration: ~10-12 minutes

Scenes:
1. ConvSubsamplingScene - Temporal reduction
2. PositionEncodingIntro - Why position matters
3. RoPEScene - Rotary Position Embeddings
4. AttentionIntroScene - Basic attention mechanism
5. ScaledDotProductScene - Full attention formula
"""

from manim import *
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from custom.colors import *
from custom.dsp_components import *
from custom.transformer_components import *


class TransformerIntroScene(Scene):
    """
    Opening scene for transformer foundations
    """

    def construct(self):
        # Title
        title = Text("Part 2: Transformer Foundations", font_size=48, color=SLATE_50)
        subtitle = Text("Attention Mechanisms & Position Encoding", font_size=24, color=PURPLE_400)
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1)
        self.wait(2)

        # Key concepts
        concepts = VGroup()
        concept_list = [
            "• Convolutional Subsampling",
            "• Rotary Position Embeddings (RoPE)",
            "• Scaled Dot-Product Attention",
            "• Multi-Head Attention"
        ]

        for i, text in enumerate(concept_list):
            t = Text(text, font_size=18, color=SLATE_400)
            t.move_to([0, -0.5 - i * 0.5, 0])
            concepts.add(t)

        self.play(FadeIn(concepts, lag_ratio=0.2), run_time=2)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class ConvSubsamplingScene(Scene):
    """
    Scene 1: Convolutional Subsampling
    Shows how we reduce temporal resolution
    """

    def construct(self):
        # Title
        title = Text("Convolutional Subsampling", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Problem statement
        problem = Text(
            "Problem: Sequence too long for efficient attention (O(T²) complexity)",
            font_size=16,
            color=ROSE_400
        )
        problem.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(problem), run_time=1)

        # Input representation
        input_seq = VGroup()
        n_frames = 20
        frame_width = 0.5

        for i in range(n_frames):
            frame = Rectangle(
                width=frame_width * 0.9,
                height=1.5,
                fill_color=CYAN_500,
                fill_opacity=0.3,
                stroke_color=CYAN_400,
                stroke_width=1
            )
            frame.move_to([(i - n_frames / 2) * frame_width, 1, 0])
            input_seq.add(frame)

        input_label = Text("Input: T frames (mel-spectrogram)", font_size=14, color=CYAN_400)
        input_label.next_to(input_seq, LEFT, buff=0.3)

        self.play(FadeIn(input_seq, input_label), run_time=1.5)

        # Conv Layer 1
        conv1_label = Text("Conv2D (stride=2)", font_size=14, color=AMBER_400)
        conv1_label.move_to([0, 0, 0])

        arrow1 = Arrow(
            start=[0, 0.2, 0],
            end=[0, -0.2, 0],
            color=SLATE_600,
            stroke_width=2
        )

        self.play(FadeIn(conv1_label, arrow1), run_time=0.5)

        # After conv1: T/2 frames
        mid_seq = VGroup()
        n_mid = n_frames // 2

        for i in range(n_mid):
            frame = Rectangle(
                width=frame_width * 0.9,
                height=1.5,
                fill_color=AMBER_500,
                fill_opacity=0.3,
                stroke_color=AMBER_400,
                stroke_width=1
            )
            frame.move_to([(i - n_mid / 2) * frame_width, -1, 0])
            mid_seq.add(frame)

        mid_label = Text("T/2 frames", font_size=14, color=AMBER_400)
        mid_label.next_to(mid_seq, LEFT, buff=0.3)

        self.play(FadeIn(mid_seq, mid_label), run_time=1)

        # Conv Layer 2
        conv2_label = Text("Conv2D (stride=2)", font_size=14, color=EMERALD_400)
        conv2_label.move_to([0, -2, 0])

        arrow2 = Arrow(
            start=[0, -1.8, 0],
            end=[0, -2.2, 0],
            color=SLATE_600,
            stroke_width=2
        )

        self.play(FadeIn(conv2_label, arrow2), run_time=0.5)

        # After conv2: T/4 frames
        output_seq = VGroup()
        n_out = n_frames // 4

        for i in range(n_out):
            frame = Rectangle(
                width=frame_width * 0.9,
                height=1.5,
                fill_color=EMERALD_500,
                fill_opacity=0.3,
                stroke_color=EMERALD_400,
                stroke_width=1
            )
            frame.move_to([(i - n_out / 2) * frame_width, -3, 0])
            output_seq.add(frame)

        output_label = Text("T/4 frames → Transformer", font_size=14, color=EMERALD_400)
        output_label.next_to(output_seq, LEFT, buff=0.3)

        self.play(FadeIn(output_seq, output_label), run_time=1)
        self.wait(1)

        # Efficiency stats
        stats = VGroup()
        stat_texts = [
            "Original: T = 1000 frames",
            "After subsampling: T' = 250 frames",
            "Attention complexity: O(1M) → O(62.5K)",
            "16× reduction in computation!"
        ]

        for i, text in enumerate(stat_texts):
            color = EMERALD_400 if i == 3 else SLATE_400
            t = Text(text, font_size=14, color=color)
            t.to_corner(DR)
            t.shift(UP * (2 - i * 0.4))
            stats.add(t)

        self.play(FadeIn(stats, lag_ratio=0.3), run_time=2)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class PositionEncodingIntroScene(Scene):
    """
    Scene 2: Why position encoding matters
    """

    def construct(self):
        # Title
        title = Text("Why Position Matters", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Problem: Attention is permutation-invariant
        problem = Text(
            "Self-attention is permutation-invariant",
            font_size=20,
            color=ROSE_400
        )
        problem.next_to(title, DOWN, buff=0.3)

        self.play(FadeIn(problem), run_time=1)

        # Show example: same attention regardless of order
        seq1 = VGroup()
        tokens1 = ["The", "cat", "sat"]
        for i, token in enumerate(tokens1):
            box = RoundedRectangle(
                width=1.2,
                height=0.6,
                corner_radius=0.1,
                fill_color=CYAN_500,
                fill_opacity=0.3,
                stroke_color=CYAN_400
            )
            box.move_to([-2 + i * 1.5, 0.5, 0])
            label = Text(token, font_size=14, color=SLATE_50)
            label.move_to(box.get_center())
            seq1.add(box, label)

        seq2 = VGroup()
        tokens2 = ["sat", "The", "cat"]
        for i, token in enumerate(tokens2):
            box = RoundedRectangle(
                width=1.2,
                height=0.6,
                corner_radius=0.1,
                fill_color=CYAN_500,
                fill_opacity=0.3,
                stroke_color=CYAN_400
            )
            box.move_to([-2 + i * 1.5, -0.5, 0])
            label = Text(token, font_size=14, color=SLATE_50)
            label.move_to(box.get_center())
            seq2.add(box, label)

        same_label = Text("Same attention scores!", font_size=14, color=ROSE_400)
        same_label.move_to([4, 0, 0])

        self.play(FadeIn(seq1, seq2), run_time=1)
        self.play(FadeIn(same_label), run_time=0.5)
        self.wait(1)

        # Solution
        solution = Text(
            "Solution: Encode position information into the representation",
            font_size=18,
            color=EMERALD_400
        )
        solution.move_to([0, -1.5, 0])

        self.play(FadeIn(solution), run_time=1)

        # Types of position encoding
        types = VGroup()
        type_data = [
            ("Sinusoidal", "Fixed pattern based on sine/cosine", SLATE_400),
            ("Learned", "Trainable position embeddings", SLATE_400),
            ("RoPE", "Rotary Position Embedding (our choice)", PURPLE_400),
        ]

        for i, (name, desc, color) in enumerate(type_data):
            name_text = Text(name + ":", font_size=16, color=color)
            desc_text = Text(desc, font_size=14, color=SLATE_500)
            desc_text.next_to(name_text, RIGHT, buff=0.2)

            item = VGroup(name_text, desc_text)
            item.move_to([0, -2.2 - i * 0.5, 0])
            types.add(item)

        self.play(FadeIn(types, lag_ratio=0.3), run_time=1.5)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class RoPEScene(Scene):
    """
    Scene 3: Rotary Position Embeddings
    Shows how RoPE works
    """

    def construct(self):
        # Title
        title = Text("Rotary Position Embeddings (RoPE)", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Key insight
        insight = Text(
            "Key Idea: Encode position by rotating vector pairs",
            font_size=18,
            color=PURPLE_400
        )
        insight.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(insight), run_time=1)

        # Create 2D rotation visualization
        circle = Circle(
            radius=2,
            color=SLATE_600,
            stroke_width=1
        )
        circle.move_to([0, -0.5, 0])

        # Unit circle labels
        angle_labels = VGroup()
        for angle, label in [(0, "0°"), (PI/2, "90°"), (PI, "180°"), (3*PI/2, "270°")]:
            pos = circle.get_center() + 2.3 * np.array([np.cos(angle), np.sin(angle), 0])
            t = Text(label, font_size=12, color=SLATE_500)
            t.move_to(pos)
            angle_labels.add(t)

        self.play(Create(circle), FadeIn(angle_labels), run_time=1)

        # Show vectors at different positions
        vectors = VGroup()
        colors = [CYAN_400, PURPLE_400, EMERALD_400, AMBER_400]
        positions = [0, 1, 2, 3]

        theta_base = PI / 6  # Base rotation angle

        for pos, color in zip(positions, colors):
            angle = pos * theta_base
            end_point = circle.get_center() + 1.8 * np.array([np.cos(angle), np.sin(angle), 0])

            vec = Arrow(
                start=circle.get_center(),
                end=end_point,
                color=color,
                stroke_width=3,
                buff=0
            )

            label = Text(f"pos={pos}", font_size=10, color=color)
            label.move_to(end_point + 0.3 * np.array([np.cos(angle), np.sin(angle), 0]))

            vectors.add(vec, label)

        self.play(FadeIn(vectors, lag_ratio=0.2), run_time=2)
        self.wait(1)

        # Formula
        formula = MathTex(
            r"R_{\theta,m} = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}",
            font_size=24
        )
        formula.set_color(SLATE_300)
        formula.to_corner(DL)

        self.play(Write(formula), run_time=2)
        self.wait(1)

        # Key property
        property_text = VGroup()
        prop1 = Text("Key Property:", font_size=16, color=EMERALD_400)
        prop2 = MathTex(r"\langle q_m', k_n' \rangle \text{ depends only on } (m - n)", font_size=20)
        prop2.set_color(SLATE_300)
        prop3 = Text("→ Natural relative position encoding!", font_size=14, color=SLATE_400)

        prop1.to_corner(DR).shift(UP * 1.5)
        prop2.next_to(prop1, DOWN, buff=0.2)
        prop3.next_to(prop2, DOWN, buff=0.2)
        property_text.add(prop1, prop2, prop3)

        self.play(FadeIn(property_text, lag_ratio=0.3), run_time=1.5)
        self.wait(2)

        # Implementation code
        code_box = RoundedRectangle(
            width=6,
            height=2,
            corner_radius=0.1,
            fill_color=SLATE_900,
            fill_opacity=0.9,
            stroke_color=SLATE_700
        )
        code_box.to_corner(DR).shift(DOWN * 0.5)

        code_text = VGroup()
        lines = [
            "# Apply RoPE",
            "q_rot = q * cos + rotate_half(q) * sin",
            "k_rot = k * cos + rotate_half(k) * sin",
        ]
        for i, line in enumerate(lines):
            color = SLATE_500 if line.startswith("#") else CYAN_400
            t = Text(line, font_size=10, color=color, font="monospace")
            t.move_to(code_box.get_center() + UP * (0.5 - i * 0.4) + LEFT * 0.5)
            t.align_to(code_box.get_left() + RIGHT * 0.3, LEFT)
            code_text.add(t)

        self.play(FadeIn(code_box, code_text), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class AttentionIntroScene(Scene):
    """
    Scene 4: Introduction to Attention
    Basic concept explanation
    """

    def construct(self):
        # Title
        title = Text("Attention Mechanism", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Intuition
        intuition = Text(
            '"Which parts of the input should I focus on?"',
            font_size=18,
            color=AMBER_400
        )
        intuition.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(intuition), run_time=1)

        # Input sequence
        seq = VGroup()
        tokens = ["Audio", "is", "complex", "signal"]
        n_tokens = len(tokens)

        for i, token in enumerate(tokens):
            box = RoundedRectangle(
                width=1.5,
                height=0.8,
                corner_radius=0.1,
                fill_color=CYAN_500,
                fill_opacity=0.3,
                stroke_color=CYAN_400
            )
            box.move_to([(i - n_tokens / 2 + 0.5) * 2, 0.5, 0])
            label = Text(token, font_size=14, color=SLATE_50)
            label.move_to(box.get_center())
            seq.add(VGroup(box, label))

        self.play(FadeIn(seq), run_time=1)

        # Highlight one token and show attention
        query_idx = 2  # "complex"
        query_box = seq[query_idx][0]

        self.play(
            query_box.animate.set_stroke(color=AMBER_400, width=3),
            run_time=0.5
        )

        query_label = Text("Query: 'complex'", font_size=14, color=AMBER_400)
        query_label.next_to(query_box, UP, buff=0.3)
        self.play(FadeIn(query_label), run_time=0.5)

        # Show attention weights to all tokens
        weights = [0.1, 0.15, 0.5, 0.25]  # Attention weights

        weight_arrows = VGroup()
        weight_labels = VGroup()

        for i, (token_group, weight) in enumerate(zip(seq, weights)):
            box = token_group[0]

            # Arrow from query to this token
            if i != query_idx:
                arrow = Arrow(
                    start=query_box.get_bottom(),
                    end=box.get_top(),
                    color=interpolate_color(SLATE_600, ATTENTION_COLOR, weight),
                    stroke_width=1 + weight * 4,
                    buff=0.1
                )
                weight_arrows.add(arrow)

            # Weight label
            w_label = Text(f"{weight:.2f}", font_size=12, color=SLATE_400)
            w_label.next_to(box, DOWN, buff=0.3)
            weight_labels.add(w_label)

        self.play(FadeIn(weight_arrows), run_time=1)
        self.play(FadeIn(weight_labels), run_time=0.5)
        self.wait(1)

        # Attention matrix visualization
        attention_title = Text("Attention Matrix", font_size=16, color=SLATE_50)
        attention_title.move_to([0, -1.5, 0])

        # Create small attention matrix
        matrix_size = 1.5
        cell_size = matrix_size / n_tokens

        matrix = VGroup()
        # Random attention weights for visualization
        attn_weights = np.random.rand(n_tokens, n_tokens)
        attn_weights = np.exp(attn_weights) / np.exp(attn_weights).sum(axis=1, keepdims=True)

        for i in range(n_tokens):
            for j in range(n_tokens):
                cell = Square(
                    side_length=cell_size * 0.95,
                    fill_color=interpolate_color(SLATE_800, ATTENTION_COLOR, attn_weights[i, j]),
                    fill_opacity=1,
                    stroke_width=0
                )
                cell.move_to([
                    (j - n_tokens / 2 + 0.5) * cell_size,
                    -2.5 - (i - n_tokens / 2 + 0.5) * cell_size,
                    0
                ])
                matrix.add(cell)

        self.play(FadeIn(attention_title, matrix), run_time=1.5)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class ScaledDotProductScene(Scene):
    """
    Scene 5: Scaled Dot-Product Attention formula
    """

    def construct(self):
        # Title
        title = Text("Scaled Dot-Product Attention", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Main formula
        formula = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=36
        )
        formula.set_color(SLATE_50)
        formula.move_to([0, 1.5, 0])

        self.play(Write(formula), run_time=2)
        self.wait(1)

        # Step-by-step breakdown
        steps = VGroup()
        step_data = [
            ("1. Compute similarity:", r"S = QK^T", "Query-Key dot product", QUERY_COLOR),
            ("2. Scale:", r"S' = \frac{S}{\sqrt{d_k}}", "Prevent gradient vanishing", SLATE_400),
            ("3. Normalize:", r"W = \text{softmax}(S')", "Convert to probabilities", ATTENTION_COLOR),
            ("4. Aggregate:", r"O = WV", "Weighted sum of values", VALUE_COLOR),
        ]

        for i, (step_name, math_str, desc, color) in enumerate(step_data):
            step = VGroup()

            name = Text(step_name, font_size=14, color=SLATE_400)
            math = MathTex(math_str, font_size=20)
            math.set_color(color)
            description = Text(desc, font_size=12, color=SLATE_500)

            name.move_to([-4, -0.3 - i * 0.8, 0])
            math.next_to(name, RIGHT, buff=0.3)
            description.next_to(math, RIGHT, buff=0.3)

            step.add(name, math, description)
            steps.add(step)

        self.play(FadeIn(steps, lag_ratio=0.3), run_time=3)
        self.wait(2)

        # Why scale?
        scale_box = RoundedRectangle(
            width=8,
            height=1.5,
            corner_radius=0.1,
            fill_color=SLATE_800,
            fill_opacity=0.5,
            stroke_color=AMBER_400
        )
        scale_box.move_to([0, -3, 0])

        scale_title = Text("Why √d_k?", font_size=16, color=AMBER_400)
        scale_title.move_to(scale_box.get_top() + DOWN * 0.3)

        scale_text = Text(
            "Large d_k → large dot products → softmax saturates → vanishing gradients",
            font_size=12,
            color=SLATE_400
        )
        scale_text.move_to(scale_box.get_center())

        self.play(FadeIn(scale_box, scale_title, scale_text), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class MultiHeadAttentionScene(Scene):
    """
    Scene 6: Multi-Head Attention
    """

    def construct(self):
        # Title
        title = Text("Multi-Head Attention", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Concept
        concept = Text(
            "Multiple attention heads → attend to different aspects",
            font_size=18,
            color=PURPLE_400
        )
        concept.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(concept), run_time=1)

        # Create multi-head visualization
        mha = MultiHeadAttention(n_heads=8, seq_len=4, head_size=0.8, spacing=0.2)
        mha.move_to([0, -0.5, 0])

        self.play(FadeIn(mha), run_time=2)
        self.wait(1)

        # Configuration
        config = VGroup()
        config_items = [
            "d_model = 512",
            "n_heads = 8",
            "d_head = 64 (512 / 8)",
        ]

        for i, item in enumerate(config_items):
            t = Text(item, font_size=14, color=SLATE_400)
            t.to_corner(DL)
            t.shift(UP * (1 - i * 0.4))
            config.add(t)

        self.play(FadeIn(config), run_time=1)

        # Formula
        formula = MathTex(
            r"\text{MultiHead}(Q,K,V) = \text{Concat}(h_1, ..., h_8) W^O",
            font_size=20
        )
        formula.set_color(SLATE_300)
        formula.to_edge(DOWN)

        self.play(Write(formula), run_time=1.5)
        self.wait(2)

        # What each head learns
        head_roles = VGroup()
        roles = [
            "Head 1: Local patterns",
            "Head 2: Long-range dependencies",
            "Head 3: Frequency patterns",
            "..."
        ]

        for i, role in enumerate(roles):
            t = Text(role, font_size=12, color=SLATE_500)
            t.to_corner(DR)
            t.shift(UP * (1.5 - i * 0.3))
            head_roles.add(t)

        self.play(FadeIn(head_roles, lag_ratio=0.2), run_time=1.5)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class TransformerFoundationsSummary(Scene):
    """
    Summary scene for transformer foundations
    """

    def construct(self):
        # Title
        title = Text("Transformer Foundations Summary", font_size=36, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Key components
        components = VGroup()
        comp_data = [
            ("Conv Subsampling", "4× temporal reduction", CYAN_500),
            ("RoPE", "Relative position encoding", PURPLE_500),
            ("Scaled Attention", "QK^T/√d_k → softmax → V", ATTENTION_COLOR),
            ("Multi-Head", "8 parallel attention heads", EMERALD_500),
        ]

        for i, (name, desc, color) in enumerate(comp_data):
            box = RoundedRectangle(
                width=5,
                height=0.8,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color
            )
            box.move_to([0, 1 - i * 1.1, 0])

            name_text = Text(name, font_size=16, color=SLATE_50)
            name_text.move_to(box.get_center() + LEFT * 1.5)

            desc_text = Text(desc, font_size=12, color=SLATE_400)
            desc_text.move_to(box.get_center() + RIGHT * 0.5)

            components.add(box, name_text, desc_text)

        self.play(FadeIn(components, lag_ratio=0.2), run_time=2)
        self.wait(2)

        # Next section
        next_text = Text(
            "Next: The Conformer Block →",
            font_size=18,
            color=EMERALD_400
        )
        next_text.to_edge(DOWN)

        self.play(FadeIn(next_text), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


# Scene sequence
if __name__ == "__main__":
    scenes = [
        TransformerIntroScene,
        ConvSubsamplingScene,
        PositionEncodingIntroScene,
        RoPEScene,
        AttentionIntroScene,
        ScaledDotProductScene,
        MultiHeadAttentionScene,
        TransformerFoundationsSummary,
    ]

    print("Video 2: Transformer Foundations Scenes")
    print("=" * 40)
    for i, scene in enumerate(scenes):
        print(f"{i+1}. {scene.__name__}")
