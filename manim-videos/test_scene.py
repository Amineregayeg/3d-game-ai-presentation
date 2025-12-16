"""
Simple test scene to verify Manim works
"""
from manim import *
import numpy as np

class SimpleTestScene(Scene):
    """Test scene without LaTeX"""

    def construct(self):
        # Title
        title = Text("VoxFormer STT", font_size=64, color="#06b6d4")
        subtitle = Text("Speech-to-Text Transformer", font_size=32, color="#a855f7")
        subtitle.next_to(title, DOWN, buff=0.5)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle), run_time=1)
        self.wait(1)

        # Pipeline diagram
        self.play(FadeOut(title, subtitle))

        # Create pipeline boxes
        boxes = []
        labels = ["Audio", "Mel", "Encoder", "CTC", "Text"]
        colors = ["#06b6d4", "#f59e0b", "#10b981", "#f43f5e", "#a855f7"]

        for i, (label, color) in enumerate(zip(labels, colors)):
            box = RoundedRectangle(
                width=1.8,
                height=1,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.3,
                stroke_color=color,
                stroke_width=2
            )
            box.move_to([(i - 2) * 2.5, 0, 0])

            text = Text(label, font_size=20, color=WHITE)
            text.move_to(box.get_center())

            boxes.append(VGroup(box, text))

        # Animate boxes appearing
        for i, box in enumerate(boxes):
            self.play(FadeIn(box), run_time=0.3)
            if i < len(boxes) - 1:
                arrow = Arrow(
                    start=box.get_right() + RIGHT * 0.1,
                    end=boxes[i+1].get_left() + LEFT * 0.1,
                    color="#64748b",
                    stroke_width=2
                )
                self.play(Create(arrow), run_time=0.2)

        self.wait(1)

        # Animate data flow
        dot = Dot(color="#06b6d4", radius=0.15)
        dot.move_to(boxes[0].get_center())
        self.play(FadeIn(dot), run_time=0.3)

        for box in boxes[1:]:
            self.play(dot.animate.move_to(box.get_center()), run_time=0.4)

        self.play(FadeOut(dot), run_time=0.3)

        # Final message
        complete = Text("Complete Pipeline!", font_size=36, color="#10b981")
        complete.next_to(VGroup(*boxes), DOWN, buff=0.8)
        self.play(FadeIn(complete), run_time=0.5)
        self.wait(2)

        self.play(FadeOut(*self.mobjects), run_time=1)


class WaveformScene(Scene):
    """Animated waveform visualization"""

    def construct(self):
        title = Text("Audio Waveform", font_size=40, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Create axes
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=4,
            axis_config={"color": "#64748b"},
        )
        axes.shift(DOWN * 0.5)

        self.play(Create(axes), run_time=1)

        # Create animated waveform
        def wave_func(t):
            return np.sin(2 * np.pi * 2 * t) * (0.5 + 0.5 * np.sin(np.pi * t))

        wave = axes.plot(wave_func, x_range=[0, 4], color="#06b6d4")
        self.play(Create(wave), run_time=2)

        # Add sample points
        samples = VGroup()
        for i in range(20):
            t = i * 0.2
            y = wave_func(t)
            point = Dot(axes.c2p(t, y), radius=0.06, color="#10b981")
            samples.add(point)

        self.play(FadeIn(samples, lag_ratio=0.05), run_time=1.5)

        # Label
        label = Text("16,000 samples/second", font_size=20, color="#f59e0b")
        label.to_corner(DR)
        self.play(FadeIn(label), run_time=0.5)

        self.wait(2)
        self.play(FadeOut(*self.mobjects), run_time=1)
