"""
Video 5: Full Pipeline Integration
Duration: ~8-10 minutes

Scenes:
1. EndToEndScene - Complete architecture
2. ModelConfigScene - Model variants
3. TrainingScene - Training process
4. InferenceScene - Inference pipeline
5. FinalSummaryScene - Key takeaways
"""

from manim import *
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from custom.colors import *
from custom.dsp_components import *
from custom.transformer_components import *
from custom.ctc_components import *


class FinalIntroScene(Scene):
    """
    Opening scene for full pipeline
    """

    def construct(self):
        # Title
        title = Text("Part 5: Full Pipeline Integration", font_size=48, color=SLATE_50)
        subtitle = Text("Putting It All Together", font_size=24, color=CYAN_400)
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1)
        self.wait(2)

        # Components recap
        recap = Text(
            "Audio Frontend → Conformer Encoder → CTC Decoder → Text",
            font_size=18,
            color=SLATE_400
        )
        recap.move_to([0, -1, 0])

        self.play(FadeIn(recap), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class EndToEndScene(Scene):
    """
    Scene 1: Complete VoxFormer architecture
    """

    def construct(self):
        # Title
        title = Text("VoxFormer Architecture", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Create full pipeline
        components = []
        positions = [
            ("Raw Audio", SLATE_500, -5.5, 1, "16kHz waveform"),
            ("Audio Frontend", CYAN_500, -3.5, 1, "Mel spectrogram"),
            ("Conv Subsample", AMBER_500, -1.5, 1, "4× reduction"),
            ("Conformer ×12", EMERALD_500, 0.5, 1.5, "d=512, h=8"),
            ("CTC Head", ROSE_500, 2.5, 1, "Vocab logits"),
            ("Output", PURPLE_500, 4.5, 1, "Text"),
        ]

        pipeline = VGroup()

        for i, (name, color, x, height_mult, desc) in enumerate(positions):
            box = RoundedRectangle(
                width=1.8,
                height=1.2 * height_mult,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color,
                stroke_width=2
            )
            box.move_to([x, 0, 0])

            label = Text(name, font_size=11, color=SLATE_50)
            label.move_to(box.get_center() + UP * 0.2)

            desc_label = Text(desc, font_size=8, color=SLATE_400)
            desc_label.move_to(box.get_center() + DOWN * 0.3)

            pipeline.add(box, label, desc_label)

            # Arrow to next (except last)
            if i < len(positions) - 1:
                next_x = positions[i + 1][2]
                arrow = Arrow(
                    start=[x + 0.95, 0, 0],
                    end=[next_x - 0.95, 0, 0],
                    color=SLATE_600,
                    stroke_width=2,
                    buff=0
                )
                pipeline.add(arrow)

        self.play(FadeIn(pipeline), run_time=2)
        self.wait(1)

        # Data flow animation
        data_dot = Dot(color=CYAN_400, radius=0.15)
        data_dot.move_to([-5.5, 0, 0])

        self.play(FadeIn(data_dot), run_time=0.3)

        for i, (name, color, x, _, _) in enumerate(positions[1:]):
            self.play(
                data_dot.animate.move_to([x, 0, 0]),
                data_dot.animate.set_color(color),
                run_time=0.5
            )

        self.play(FadeOut(data_dot), run_time=0.3)

        # Shape annotations
        shapes = VGroup()
        shape_data = [
            ("(T,)", -5.5),
            ("(T', 80)", -3.5),
            ("(T'/4, 512)", -1.5),
            ("(T'/4, 512)", 0.5),
            ("(T'/4, |V|)", 2.5),
            ("text", 4.5),
        ]

        for text, x in shape_data:
            shape = Text(text, font_size=10, color=SLATE_500)
            shape.move_to([x, -1.3, 0])
            shapes.add(shape)

        self.play(FadeIn(shapes), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class ModelConfigScene(Scene):
    """
    Scene 2: Model configurations
    """

    def construct(self):
        # Title
        title = Text("Model Configurations", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Create table
        table_data = [
            ["Config", "d_model", "heads", "enc", "dec", "FFN", "Params"],
            ["Tiny", "256", "4", "6", "4", "1024", "~15M"],
            ["Base", "512", "8", "12", "6", "2048", "~80M"],
            ["Large", "768", "12", "18", "8", "3072", "~200M"],
        ]

        table = VGroup()
        cell_width = 1.3
        cell_height = 0.6

        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                # Cell background
                is_header = i == 0
                is_base = i == 2  # Highlight Base config

                fill_color = SLATE_700 if is_header else (CYAN_500 if is_base else SLATE_800)
                fill_opacity = 0.5 if is_header else (0.3 if is_base else 0.2)

                rect = Rectangle(
                    width=cell_width,
                    height=cell_height,
                    fill_color=fill_color,
                    fill_opacity=fill_opacity,
                    stroke_color=SLATE_600,
                    stroke_width=1
                )
                rect.move_to([
                    (j - len(row) / 2 + 0.5) * cell_width,
                    (1.5 - i) * cell_height + 0.5,
                    0
                ])

                # Cell text
                text_color = SLATE_50 if is_header else (CYAN_400 if is_base else SLATE_300)
                font_size = 12 if is_header else 11

                text = Text(cell, font_size=font_size, color=text_color)
                text.move_to(rect.get_center())

                table.add(rect, text)

        self.play(FadeIn(table), run_time=2)

        # Highlight target config
        target = Text("← Target Configuration", font_size=14, color=CYAN_400)
        target.move_to([5, 0.5 + 0.5 - 0.6, 0])

        self.play(FadeIn(target), run_time=0.5)
        self.wait(1)

        # Trade-offs
        tradeoffs = VGroup()
        tradeoff_data = [
            ("Tiny:", "Fast inference, limited accuracy", SLATE_400),
            ("Base:", "Best accuracy/speed tradeoff", CYAN_400),
            ("Large:", "Highest accuracy, slower", SLATE_400),
        ]

        for i, (name, desc, color) in enumerate(tradeoff_data):
            name_text = Text(name, font_size=14, color=color)
            desc_text = Text(desc, font_size=12, color=SLATE_500)

            name_text.move_to([-3, -1.5 - i * 0.4, 0])
            desc_text.next_to(name_text, RIGHT, buff=0.2)

            tradeoffs.add(name_text, desc_text)

        self.play(FadeIn(tradeoffs), run_time=1.5)
        self.wait(2)

        # Performance metrics
        metrics = VGroup()
        metric_box = RoundedRectangle(
            width=5,
            height=2,
            corner_radius=0.1,
            fill_color=SLATE_800,
            fill_opacity=0.5,
            stroke_color=EMERALD_400
        )
        metric_box.to_corner(DR)

        metric_title = Text("VoxFormer-Base Targets", font_size=14, color=EMERALD_400)
        metric_title.move_to(metric_box.get_top() + DOWN * 0.3)

        metric_items = [
            "WER: < 5% (clean speech)",
            "Latency: < 100ms (real-time)",
            "Memory: < 500MB (inference)",
        ]

        for i, item in enumerate(metric_items):
            t = Text(item, font_size=11, color=SLATE_400)
            t.move_to(metric_box.get_center() + UP * (0.3 - i * 0.4))
            metrics.add(t)

        metrics.add(metric_box, metric_title)

        self.play(FadeIn(metrics), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class TrainingScene(Scene):
    """
    Scene 3: Training process
    """

    def construct(self):
        # Title
        title = Text("Training Process", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Training loop diagram
        loop_elements = VGroup()

        # Main loop steps
        steps = [
            ("Load Batch", CYAN_500, [-4, 1, 0]),
            ("Forward Pass", EMERALD_500, [-1, 1, 0]),
            ("CTC Loss", ROSE_500, [2, 1, 0]),
            ("Backward", PURPLE_500, [2, -1, 0]),
            ("Update", AMBER_500, [-1, -1, 0]),
        ]

        for name, color, pos in steps:
            box = RoundedRectangle(
                width=2,
                height=0.8,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color
            )
            box.move_to(pos)

            label = Text(name, font_size=12, color=SLATE_50)
            label.move_to(box.get_center())

            loop_elements.add(box, label)

        # Arrows forming loop
        arrows = VGroup()
        arrow_paths = [
            ([-3, 1, 0], [0, 1, 0]),
            ([0, 1, 0], [1, 1, 0]),
            ([3, 0.5, 0], [3, -0.5, 0]),
            ([1, -1, 0], [0, -1, 0]),
            ([-2, -0.5, 0], [-3, 0.5, 0]),
        ]

        for start, end in arrow_paths:
            arrow = Arrow(
                start=start,
                end=end,
                color=SLATE_600,
                stroke_width=2
            )
            arrows.add(arrow)

        loop_elements.add(arrows)

        self.play(FadeIn(loop_elements), run_time=2)
        self.wait(1)

        # Training hyperparameters
        params_box = RoundedRectangle(
            width=4,
            height=3,
            corner_radius=0.1,
            fill_color=SLATE_800,
            fill_opacity=0.5,
            stroke_color=SLATE_700
        )
        params_box.to_corner(DL)

        params_title = Text("Hyperparameters", font_size=14, color=AMBER_400)
        params_title.move_to(params_box.get_top() + DOWN * 0.3)

        params = [
            "Optimizer: AdamW",
            "Learning Rate: 1e-4",
            "Warmup: 10k steps",
            "Batch Size: 32",
            "Max Epochs: 100",
            "Gradient Clip: 1.0",
        ]

        params_group = VGroup()
        for i, param in enumerate(params):
            t = Text(param, font_size=10, color=SLATE_400)
            t.move_to(params_box.get_center() + UP * (0.8 - i * 0.35))
            params_group.add(t)

        self.play(FadeIn(params_box, params_title, params_group), run_time=1)

        # Loss curve
        loss_axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=2,
            axis_config={"color": SLATE_600}
        )
        loss_axes.to_corner(DR).shift(UP * 0.5)

        loss_curve = loss_axes.plot(
            lambda x: 4 * np.exp(-x / 30) + 0.5 + 0.2 * np.random.rand(),
            x_range=[0, 100],
            color=ROSE_400
        )

        loss_label = Text("Training Loss", font_size=12, color=ROSE_400)
        loss_label.next_to(loss_axes, UP, buff=0.1)

        epoch_label = Text("Epochs", font_size=10, color=SLATE_500)
        epoch_label.next_to(loss_axes, DOWN, buff=0.1)

        self.play(
            Create(loss_axes),
            Create(loss_curve),
            FadeIn(loss_label, epoch_label),
            run_time=2
        )
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class InferenceScene(Scene):
    """
    Scene 4: Inference pipeline
    """

    def construct(self):
        # Title
        title = Text("Inference Pipeline", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Simplified inference flow
        flow = VGroup()

        flow_steps = [
            ("🎤 Audio In", CYAN_500),
            ("Preprocess", SLATE_500),
            ("Encoder", EMERALD_500),
            ("CTC Decode", AMBER_500),
            ("📝 Text Out", PURPLE_500),
        ]

        for i, (name, color) in enumerate(flow_steps):
            box = RoundedRectangle(
                width=2,
                height=0.8,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.2,
                stroke_color=color
            )
            box.move_to([(i - 2) * 2.5, 1, 0])

            label = Text(name, font_size=12, color=SLATE_50)
            label.move_to(box.get_center())

            flow.add(box, label)

            if i < len(flow_steps) - 1:
                arrow = Arrow(
                    start=[(i - 2) * 2.5 + 1.1, 1, 0],
                    end=[(i - 1) * 2.5 - 1.1, 1, 0],
                    color=SLATE_600,
                    stroke_width=2
                )
                flow.add(arrow)

        self.play(FadeIn(flow), run_time=2)

        # Performance metrics
        metrics = VGroup()
        metric_data = [
            ("Latency", "< 100ms", "Real-time capable"),
            ("Throughput", "100+ QPS", "Production ready"),
            ("Memory", "< 500MB", "Edge deployable"),
        ]

        for i, (name, value, desc) in enumerate(metric_data):
            box = RoundedRectangle(
                width=3,
                height=1.2,
                corner_radius=0.1,
                fill_color=EMERALD_500,
                fill_opacity=0.1,
                stroke_color=EMERALD_400
            )
            box.move_to([(i - 1) * 3.5, -1, 0])

            name_text = Text(name, font_size=14, color=EMERALD_400)
            name_text.move_to(box.get_center() + UP * 0.3)

            value_text = Text(value, font_size=18, color=SLATE_50)
            value_text.move_to(box.get_center())

            desc_text = Text(desc, font_size=10, color=SLATE_500)
            desc_text.move_to(box.get_center() + DOWN * 0.35)

            metrics.add(box, name_text, value_text, desc_text)

        self.play(FadeIn(metrics), run_time=1.5)
        self.wait(2)

        # Optimizations
        opt_title = Text("Optimizations", font_size=16, color=AMBER_400)
        opt_title.to_corner(DL).shift(UP * 1)

        optimizations = [
            "• Quantization (INT8)",
            "• KV-cache for streaming",
            "• Batch processing",
            "• ONNX export"
        ]

        opt_group = VGroup()
        for i, opt in enumerate(optimizations):
            t = Text(opt, font_size=12, color=SLATE_400)
            t.next_to(opt_title, DOWN, buff=0.2 + i * 0.3)
            t.align_to(opt_title, LEFT)
            opt_group.add(t)

        self.play(FadeIn(opt_title, opt_group), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class FinalSummaryScene(Scene):
    """
    Scene 5: Final summary and key takeaways
    """

    def construct(self):
        # Title
        title = Text("VoxFormer: Key Takeaways", font_size=36, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Key points
        points = VGroup()
        point_data = [
            ("Audio Frontend", "Mel spectrograms capture acoustic information", CYAN_400),
            ("Conv Subsampling", "4× reduction for efficient attention", AMBER_400),
            ("RoPE", "Relative position encoding without overhead", PURPLE_400),
            ("Conformer", "Attention + Convolution = global + local", EMERALD_400),
            ("CTC Loss", "End-to-end training without alignment", ROSE_400),
        ]

        for i, (key, value, color) in enumerate(point_data):
            check = Text("✓", font_size=18, color=color)
            check.move_to([-5, 1 - i * 0.8, 0])

            key_text = Text(key + ":", font_size=14, color=color)
            key_text.next_to(check, RIGHT, buff=0.2)

            value_text = Text(value, font_size=12, color=SLATE_400)
            value_text.next_to(key_text, RIGHT, buff=0.3)

            points.add(check, key_text, value_text)

        self.play(FadeIn(points, lag_ratio=0.15), run_time=2.5)
        self.wait(1)

        # Final architecture summary
        arch_box = RoundedRectangle(
            width=10,
            height=1.5,
            corner_radius=0.1,
            fill_color=CYAN_500,
            fill_opacity=0.1,
            stroke_color=CYAN_400
        )
        arch_box.move_to([0, -2.5, 0])

        arch_summary = Text(
            "Audio → Mel → Conv↓ → Conformer×12 → CTC → Text",
            font_size=18,
            color=SLATE_50
        )
        arch_summary.move_to(arch_box.get_center())

        params_text = Text(
            "~80M parameters | Real-time inference | Game-dev optimized",
            font_size=12,
            color=SLATE_400
        )
        params_text.next_to(arch_summary, DOWN, buff=0.2)

        self.play(FadeIn(arch_box, arch_summary, params_text), run_time=1.5)
        self.wait(2)

        # Thank you
        self.play(FadeOut(points, arch_box, arch_summary, params_text), run_time=1)

        thanks = Text("Thank You!", font_size=48, color=CYAN_400)
        thanks.move_to([0, 0.5, 0])

        subtitle = Text(
            "VoxFormer: Speech-to-Text for 3D Game AI",
            font_size=20,
            color=SLATE_400
        )
        subtitle.next_to(thanks, DOWN, buff=0.5)

        self.play(Write(thanks), run_time=1)
        self.play(FadeIn(subtitle), run_time=0.5)
        self.wait(3)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=2)


# Scene sequence
if __name__ == "__main__":
    scenes = [
        FinalIntroScene,
        EndToEndScene,
        ModelConfigScene,
        TrainingScene,
        InferenceScene,
        FinalSummaryScene,
    ]

    print("Video 5: Full Pipeline Scenes")
    print("=" * 40)
    for i, scene in enumerate(scenes):
        print(f"{i+1}. {scene.__name__}")
