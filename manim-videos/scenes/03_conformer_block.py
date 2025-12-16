"""
Video 3: The Conformer Block
Duration: ~10-12 minutes

Scenes:
1. ConformerOverviewScene - Full block structure
2. FeedForwardScene - FFN with SwiGLU
3. ConvModuleScene - Convolution module
4. ConformerStackScene - Multiple blocks
"""

from manim import *
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from custom.colors import *
from custom.transformer_components import *


class ConformerIntroScene(Scene):
    """
    Opening scene for Conformer block
    """

    def construct(self):
        # Title
        title = Text("Part 3: The Conformer Block", font_size=48, color=SLATE_50)
        subtitle = Text("Combining Attention and Convolution", font_size=24, color=EMERALD_400)
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1)
        self.wait(2)

        # Key insight
        insight = Text(
            "Conformer = Attention (global patterns) + Convolution (local patterns)",
            font_size=18,
            color=SLATE_400
        )
        insight.move_to([0, -1, 0])

        self.play(FadeIn(insight), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class ConformerOverviewScene(Scene):
    """
    Scene 1: Full Conformer block structure
    """

    def construct(self):
        # Title
        title = Text("Conformer Block Architecture", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Create conformer block visualization
        conformer = ConformerBlock(width=3.5, height=5)
        conformer.move_to([0, -0.5, 0])

        self.play(FadeIn(conformer), run_time=2)
        self.wait(1)

        # Annotations for each component
        annotations = [
            ("First half of FFN", FFN_COLOR, UP * 2),
            ("Global context via attention", ATTENTION_COLOR, UP * 1),
            ("Local patterns via convolution", CONV_COLOR, ORIGIN),
            ("Second half of FFN", FFN_COLOR, DOWN * 1),
            ("Stabilizes training", SLATE_500, DOWN * 2),
        ]

        for i, (text, color, offset) in enumerate(annotations):
            ann = Text(text, font_size=12, color=color)
            ann.move_to([4.5, -0.5 + offset[1], 0])

            line = DashedLine(
                start=[2, -0.5 + offset[1], 0],
                end=[3.5, -0.5 + offset[1], 0],
                color=SLATE_600
            )

            self.play(FadeIn(ann, line), run_time=0.5)

        self.wait(2)

        # Macaron structure explanation
        macaron = VGroup()
        mac_title = Text("Macaron Structure", font_size=16, color=AMBER_400)
        mac_title.to_corner(DL).shift(UP * 1.5)

        mac_text = Text(
            "FFN split in half → sandwiches attention + conv",
            font_size=12,
            color=SLATE_400
        )
        mac_text.next_to(mac_title, DOWN, buff=0.2)

        mac_formula = MathTex(
            r"\text{output} = x + \frac{1}{2}\text{FFN} + \text{MHSA} + \text{Conv} + \frac{1}{2}\text{FFN}",
            font_size=14
        )
        mac_formula.set_color(SLATE_300)
        mac_formula.next_to(mac_text, DOWN, buff=0.2)

        macaron.add(mac_title, mac_text, mac_formula)

        self.play(FadeIn(macaron), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class FeedForwardScene(Scene):
    """
    Scene 2: Feed-Forward Network with SwiGLU
    """

    def construct(self):
        # Title
        title = Text("Feed-Forward Network (SwiGLU)", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Network diagram
        layers = VGroup()

        # Input
        input_layer = RoundedRectangle(
            width=2.5,
            height=0.8,
            corner_radius=0.1,
            fill_color=SLATE_600,
            fill_opacity=0.3,
            stroke_color=SLATE_500
        )
        input_layer.move_to([0, 2, 0])
        input_label = Text("Input (512)", font_size=14, color=SLATE_50)
        input_label.move_to(input_layer.get_center())
        layers.add(input_layer, input_label)

        # Linear expansion
        expand_layer = RoundedRectangle(
            width=4,
            height=0.8,
            corner_radius=0.1,
            fill_color=FFN_COLOR,
            fill_opacity=0.3,
            stroke_color=FFN_COLOR
        )
        expand_layer.move_to([0, 0.5, 0])
        expand_label = Text("Linear (2048)", font_size=14, color=SLATE_50)
        expand_label.move_to(expand_layer.get_center())
        layers.add(expand_layer, expand_label)

        # Gate branch
        gate_layer = RoundedRectangle(
            width=2,
            height=0.8,
            corner_radius=0.1,
            fill_color=AMBER_500,
            fill_opacity=0.3,
            stroke_color=AMBER_400
        )
        gate_layer.move_to([3, 0.5, 0])
        gate_label = Text("Gate", font_size=14, color=SLATE_50)
        gate_label.move_to(gate_layer.get_center())
        layers.add(gate_layer, gate_label)

        # SwiGLU activation
        swiglu = RoundedRectangle(
            width=3,
            height=0.8,
            corner_radius=0.1,
            fill_color=EMERALD_500,
            fill_opacity=0.3,
            stroke_color=EMERALD_400
        )
        swiglu.move_to([0, -1, 0])
        swiglu_label = Text("SwiGLU", font_size=14, color=SLATE_50)
        swiglu_label.move_to(swiglu.get_center())
        layers.add(swiglu, swiglu_label)

        # Output
        output_layer = RoundedRectangle(
            width=2.5,
            height=0.8,
            corner_radius=0.1,
            fill_color=SLATE_600,
            fill_opacity=0.3,
            stroke_color=SLATE_500
        )
        output_layer.move_to([0, -2.5, 0])
        output_label = Text("Output (512)", font_size=14, color=SLATE_50)
        output_label.move_to(output_layer.get_center())
        layers.add(output_layer, output_label)

        # Arrows
        arrows = VGroup()
        arrows.add(Arrow(start=[0, 1.5, 0], end=[0, 1, 0], color=SLATE_600, buff=0.1))
        arrows.add(Arrow(start=[0, 0, 0], end=[0, -0.5, 0], color=SLATE_600, buff=0.1))
        arrows.add(Arrow(start=[0, -1.5, 0], end=[0, -2, 0], color=SLATE_600, buff=0.1))
        arrows.add(Arrow(start=[1.5, 0.5, 0], end=[2.5, 0.5, 0], color=SLATE_600, buff=0.1))
        arrows.add(Arrow(start=[3, 0, 0], end=[1, -1, 0], color=AMBER_400, buff=0.1))

        self.play(FadeIn(layers, arrows), run_time=2)
        self.wait(1)

        # SwiGLU formula
        formula_box = RoundedRectangle(
            width=8,
            height=2,
            corner_radius=0.1,
            fill_color=SLATE_800,
            fill_opacity=0.8,
            stroke_color=SLATE_700
        )
        formula_box.to_corner(DR)

        formula_title = Text("SwiGLU Activation", font_size=14, color=EMERALD_400)
        formula_title.move_to(formula_box.get_top() + DOWN * 0.3)

        swiglu_formula = MathTex(
            r"\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes xW_2",
            font_size=18
        )
        swiglu_formula.set_color(SLATE_300)
        swiglu_formula.move_to(formula_box.get_center())

        swish_formula = MathTex(
            r"\text{Swish}(x) = x \cdot \sigma(x)",
            font_size=14
        )
        swish_formula.set_color(SLATE_400)
        swish_formula.move_to(formula_box.get_center() + DOWN * 0.5)

        self.play(FadeIn(formula_box, formula_title, swiglu_formula, swish_formula), run_time=1.5)
        self.wait(2)

        # Swish activation graph
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 3, 1],
            x_length=4,
            y_length=2.5,
            axis_config={"color": SLATE_600}
        )
        axes.to_corner(DL)

        # Swish function
        swish_graph = axes.plot(
            lambda x: x * (1 / (1 + np.exp(-x))),
            x_range=[-3, 3],
            color=EMERALD_400
        )

        # ReLU for comparison
        relu_graph = axes.plot(
            lambda x: max(0, x),
            x_range=[-3, 3],
            color=SLATE_500,
            stroke_width=1
        )

        graph_label = Text("Swish vs ReLU", font_size=12, color=SLATE_400)
        graph_label.next_to(axes, UP, buff=0.1)

        self.play(Create(axes), Create(swish_graph), Create(relu_graph), FadeIn(graph_label), run_time=2)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class ConvModuleScene(Scene):
    """
    Scene 3: Convolution Module
    """

    def construct(self):
        # Title
        title = Text("Convolution Module", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Purpose
        purpose = Text(
            "Captures local patterns that attention misses",
            font_size=18,
            color=CONV_COLOR
        )
        purpose.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(purpose), run_time=1)

        # Module structure
        modules = VGroup()

        module_data = [
            ("LayerNorm", SLATE_500, 2.5),
            ("Pointwise Conv", CONV_COLOR, 1.5),
            ("GLU Activation", AMBER_500, 0.5),
            ("Depthwise Conv", CONV_COLOR, -0.5),
            ("BatchNorm + Swish", SLATE_500, -1.5),
            ("Pointwise Conv", CONV_COLOR, -2.5),
        ]

        for name, color, y_pos in module_data:
            box = RoundedRectangle(
                width=4,
                height=0.7,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color
            )
            box.move_to([-2, y_pos, 0])

            label = Text(name, font_size=12, color=SLATE_50)
            label.move_to(box.get_center())

            modules.add(box, label)

        # Arrows
        for i in range(len(module_data) - 1):
            arrow = Arrow(
                start=[-2, module_data[i][2] - 0.4, 0],
                end=[-2, module_data[i + 1][2] + 0.4, 0],
                color=SLATE_600,
                stroke_width=2,
                buff=0
            )
            modules.add(arrow)

        self.play(FadeIn(modules), run_time=2)

        # Kernel visualization
        kernel_viz = ConvolutionModule(kernel_size=31)
        kernel_viz.move_to([3, 0, 0])

        self.play(FadeIn(kernel_viz), run_time=1)

        # Key points
        points = VGroup()
        point_texts = [
            "• Kernel size: 31 (captures ~300ms context)",
            "• Depthwise: each channel processed independently",
            "• Efficient: O(T × d × k) instead of O(T × d²)"
        ]

        for i, text in enumerate(point_texts):
            t = Text(text, font_size=12, color=SLATE_400)
            t.to_corner(DR)
            t.shift(UP * (1.5 - i * 0.4))
            points.add(t)

        self.play(FadeIn(points, lag_ratio=0.2), run_time=1.5)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class ConformerStackScene(Scene):
    """
    Scene 4: Stacking Conformer blocks
    """

    def construct(self):
        # Title
        title = Text("Conformer Encoder Stack", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Create stack
        stack = TransformerStack(n_layers=12, block_width=3, block_height=0.5, spacing=0.1)
        stack.move_to([0, 0, 0])

        self.play(FadeIn(stack), run_time=2)

        # Configuration
        config = VGroup()
        config_items = [
            "N_encoder = 12 layers",
            "d_model = 512",
            "n_heads = 8",
            "kernel_size = 31",
            "FFN expansion = 4×"
        ]

        for i, item in enumerate(config_items):
            t = Text(item, font_size=14, color=SLATE_400)
            t.to_corner(DR)
            t.shift(UP * (2 - i * 0.4))
            config.add(t)

        self.play(FadeIn(config), run_time=1)

        # Total parameters
        params = Text("Total Parameters: ~80M", font_size=18, color=EMERALD_400)
        params.to_edge(DOWN)

        self.play(FadeIn(params), run_time=1)
        self.wait(2)

        # Information flow
        info = Text(
            "Information flows up through all 12 blocks, building richer representations",
            font_size=14,
            color=SLATE_400
        )
        info.next_to(title, DOWN, buff=0.3)

        # Animate information flowing up
        dot = Dot(color=CYAN_400, radius=0.1)
        dot.move_to([0, -3, 0])

        self.play(FadeIn(info, dot), run_time=1)
        self.play(dot.animate.move_to([0, 2.5, 0]), run_time=2)
        self.play(FadeOut(dot), run_time=0.5)

        self.wait(1)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class ConformerSummaryScene(Scene):
    """
    Summary of Conformer block
    """

    def construct(self):
        # Title
        title = Text("Conformer Block Summary", font_size=36, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Key components
        components = VGroup()
        comp_data = [
            ("FFN (½)", "Non-linear transformation", FFN_COLOR),
            ("MHSA", "Global pattern recognition", ATTENTION_COLOR),
            ("Conv Module", "Local pattern recognition", CONV_COLOR),
            ("FFN (½)", "Non-linear transformation", FFN_COLOR),
            ("LayerNorm", "Training stability", SLATE_500),
        ]

        for i, (name, desc, color) in enumerate(comp_data):
            box = RoundedRectangle(
                width=8,
                height=0.7,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color
            )
            box.move_to([0, 1.5 - i * 0.9, 0])

            name_text = Text(name, font_size=14, color=SLATE_50)
            name_text.move_to(box.get_center() + LEFT * 2.5)

            desc_text = Text(desc, font_size=12, color=SLATE_400)
            desc_text.move_to(box.get_center() + RIGHT * 1)

            components.add(box, name_text, desc_text)

        self.play(FadeIn(components, lag_ratio=0.1), run_time=2)
        self.wait(1)

        # Why Conformer works
        why = VGroup()
        why_title = Text("Why Conformer?", font_size=16, color=AMBER_400)
        why_title.to_corner(DL).shift(UP * 1.5)

        reasons = [
            "✓ Attention: captures long-range dependencies",
            "✓ Convolution: captures local acoustic patterns",
            "✓ Best of both worlds for speech!"
        ]

        for i, reason in enumerate(reasons):
            color = EMERALD_400 if i == 2 else SLATE_400
            t = Text(reason, font_size=12, color=color)
            t.next_to(why_title, DOWN, buff=0.2 + i * 0.3)
            t.align_to(why_title, LEFT)
            why.add(t)

        why.add(why_title)

        self.play(FadeIn(why, lag_ratio=0.2), run_time=1.5)
        self.wait(2)

        # Next section
        next_text = Text(
            "Next: Training with CTC Loss →",
            font_size=18,
            color=ROSE_400
        )
        next_text.to_edge(DOWN)

        self.play(FadeIn(next_text), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


# Scene sequence
if __name__ == "__main__":
    scenes = [
        ConformerIntroScene,
        ConformerOverviewScene,
        FeedForwardScene,
        ConvModuleScene,
        ConformerStackScene,
        ConformerSummaryScene,
    ]

    print("Video 3: Conformer Block Scenes")
    print("=" * 40)
    for i, scene in enumerate(scenes):
        print(f"{i+1}. {scene.__name__}")
