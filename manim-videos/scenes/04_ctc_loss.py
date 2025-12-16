"""
Video 4: Training with CTC Loss
Duration: ~8-10 minutes

Scenes:
1. CTCProblemScene - The alignment challenge
2. CTCAlgorithmScene - Forward-backward algorithm
3. CTCGradientScene - Gradient computation
4. CTCDecodingScene - Inference
"""

from manim import *
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from custom.colors import *
from custom.ctc_components import *


class CTCIntroScene(Scene):
    """
    Opening scene for CTC Loss
    """

    def construct(self):
        # Title
        title = Text("Part 4: Training with CTC Loss", font_size=48, color=SLATE_50)
        subtitle = Text("Connectionist Temporal Classification", font_size=24, color=ROSE_400)
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1)
        self.wait(2)

        # The challenge
        challenge = Text(
            "Challenge: How do we train without frame-level alignment?",
            font_size=18,
            color=SLATE_400
        )
        challenge.move_to([0, -1, 0])

        self.play(FadeIn(challenge), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class CTCProblemScene(Scene):
    """
    Scene 1: The alignment problem
    """

    def construct(self):
        # Title
        title = Text("The Alignment Problem", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Audio representation
        audio_label = Text("Audio Input (T frames)", font_size=16, color=CYAN_400)
        audio_label.move_to([-4, 1.5, 0])

        audio_frames = VGroup()
        n_frames = 10
        for i in range(n_frames):
            frame = Rectangle(
                width=0.6,
                height=1,
                fill_color=CYAN_500,
                fill_opacity=0.3,
                stroke_color=CYAN_400,
                stroke_width=1
            )
            frame.move_to([(i - n_frames / 2 + 0.5) * 0.7, 1.5, 0])
            audio_frames.add(frame)

        self.play(FadeIn(audio_label, audio_frames), run_time=1)

        # Target text
        target_label = Text("Target: 'CAT'", font_size=16, color=EMERALD_400)
        target_label.move_to([-4.5, -0.5, 0])

        target_text = VGroup()
        for i, char in enumerate("CAT"):
            box = RoundedRectangle(
                width=1,
                height=0.8,
                corner_radius=0.1,
                fill_color=EMERALD_500,
                fill_opacity=0.3,
                stroke_color=EMERALD_400
            )
            box.move_to([(i - 1) * 1.5, -0.5, 0])
            label = Text(char, font_size=18, color=SLATE_50)
            label.move_to(box.get_center())
            target_text.add(VGroup(box, label))

        self.play(FadeIn(target_label, target_text), run_time=1)

        # Question marks showing unknown alignment
        question = Text("Which frames → which letters?", font_size=16, color=ROSE_400)
        question.move_to([0, 0.5, 0])

        arrows = VGroup()
        for i in range(n_frames):
            arrow = DashedLine(
                start=[(i - n_frames / 2 + 0.5) * 0.7, 1, 0],
                end=[np.random.uniform(-1.5, 1.5), 0, 0],
                color=ROSE_400,
                dash_length=0.1
            )
            arrows.add(arrow)

        self.play(FadeIn(question), run_time=0.5)
        self.play(FadeIn(arrows), run_time=1)
        self.wait(1)

        # Clear and show multiple valid alignments
        self.play(FadeOut(arrows, question), run_time=0.5)

        alignment_title = Text("Many valid alignments!", font_size=18, color=AMBER_400)
        alignment_title.move_to([0, 0.5, 0])
        self.play(FadeIn(alignment_title), run_time=0.5)

        # Show alignment examples
        alignments_viz = AlignmentVisualization(target="CAT", n_frames=10)
        alignments_viz.scale(0.7)
        alignments_viz.move_to([0, -1.5, 0])

        self.play(FadeIn(alignments_viz), run_time=2)
        self.wait(2)

        # Solution
        solution = Text(
            "Solution: CTC sums probabilities over ALL valid alignments!",
            font_size=16,
            color=EMERALD_400
        )
        solution.to_edge(DOWN)

        self.play(FadeIn(solution), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class CTCBlankTokenScene(Scene):
    """
    Scene 2: The blank token
    """

    def construct(self):
        # Title
        title = Text("The Blank Token (ε)", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Explanation
        explanation = Text(
            "CTC uses a special 'blank' token to handle variable-length outputs",
            font_size=16,
            color=SLATE_400
        )
        explanation.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(explanation), run_time=1)

        # Expanded label sequence
        exp_title = Text("Label expansion with blanks:", font_size=16, color=CYAN_400)
        exp_title.move_to([-3, 1, 0])

        original = Text("'CAT'", font_size=24, color=EMERALD_400)
        original.move_to([-3, 0.3, 0])

        arrow = Arrow(
            start=[-3, -0.1, 0],
            end=[-3, -0.6, 0],
            color=SLATE_600
        )

        # Expanded sequence
        expanded = VGroup()
        exp_labels = ['ε', 'C', 'ε', 'A', 'ε', 'T', 'ε']
        for i, label in enumerate(exp_labels):
            is_blank = label == 'ε'
            box = RoundedRectangle(
                width=0.8,
                height=0.6,
                corner_radius=0.05,
                fill_color=SLATE_700 if is_blank else EMERALD_500,
                fill_opacity=0.5 if is_blank else 0.8,
                stroke_color=SLATE_600 if is_blank else EMERALD_400
            )
            box.move_to([-4 + i * 1, -1.2, 0])
            text = Text(label, font_size=14, color=SLATE_400 if is_blank else SLATE_50)
            text.move_to(box.get_center())
            expanded.add(VGroup(box, text))

        self.play(
            FadeIn(exp_title, original),
            run_time=1
        )
        self.play(Create(arrow), run_time=0.5)
        self.play(FadeIn(expanded, lag_ratio=0.1), run_time=1.5)

        # Purpose of blanks
        purposes = VGroup()
        purpose_texts = [
            "• Separates repeated characters: 'EE' → εEεEε",
            "• Allows model to emit 'nothing' at certain frames",
            "• Enables variable output length"
        ]

        for i, text in enumerate(purpose_texts):
            t = Text(text, font_size=14, color=SLATE_400)
            t.move_to([0, -2.2 - i * 0.4, 0])
            purposes.add(t)

        self.play(FadeIn(purposes, lag_ratio=0.3), run_time=1.5)
        self.wait(2)

        # Collapse operation
        collapse_title = Text("Collapse function B⁻¹:", font_size=16, color=AMBER_400)
        collapse_title.to_corner(DR).shift(UP * 2)

        collapse_steps = VGroup()
        steps = [
            "1. Remove blanks",
            "2. Merge consecutive duplicates",
        ]
        for i, step in enumerate(steps):
            t = Text(step, font_size=12, color=SLATE_400)
            t.next_to(collapse_title, DOWN, buff=0.2 + i * 0.3)
            t.align_to(collapse_title, LEFT)
            collapse_steps.add(t)

        self.play(FadeIn(collapse_title, collapse_steps), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class CTCAlgorithmScene(Scene):
    """
    Scene 3: Forward-Backward Algorithm
    """

    def construct(self):
        # Title
        title = Text("Forward-Backward Algorithm", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Create trellis
        trellis = CTCTrellis(n_frames=8, labels="HI", cell_size=0.5)
        trellis.scale(0.9)
        trellis.move_to([-0.5, 0, 0])

        self.play(FadeIn(trellis), run_time=2)

        # Forward pass explanation
        forward_title = Text("Forward Pass (α)", font_size=18, color=CYAN_400)
        forward_title.to_corner(UL).shift(DOWN * 1.2)

        forward_formula = MathTex(
            r"\alpha_t(s) = \sum_{s'} \alpha_{t-1}(s') \cdot y_t^{l_s}",
            font_size=16
        )
        forward_formula.set_color(SLATE_300)
        forward_formula.next_to(forward_title, DOWN, buff=0.2)

        self.play(FadeIn(forward_title, forward_formula), run_time=1)

        # Animate forward pass (highlight cells left to right)
        n_frames = 8
        n_labels = 5  # εHεIε

        for t in range(n_frames):
            highlights = VGroup()
            for s in range(n_labels):
                if t == 0 and s > 1:
                    continue  # Can only start at first two states
                if s > 2 * t + 1:
                    continue  # Can't skip labels

                x = (t + 0.5) * 0.5 - (n_frames * 0.5) / 2
                y = (n_labels - s - 0.5) * 0.5 - (n_labels * 0.5) / 2

                highlight = Circle(
                    radius=0.15,
                    fill_color=CYAN_400,
                    fill_opacity=0.5,
                    stroke_width=0
                )
                highlight.move_to(trellis.get_center() + np.array([x, y, 0]))
                highlights.add(highlight)

            self.play(FadeIn(highlights), run_time=0.3)

        self.wait(1)

        # Backward pass
        backward_title = Text("Backward Pass (β)", font_size=18, color=PURPLE_400)
        backward_title.to_corner(UR).shift(DOWN * 1.2 + LEFT * 0.5)

        backward_formula = MathTex(
            r"\beta_t(s) = \sum_{s'} \beta_{t+1}(s') \cdot y_{t+1}^{l_{s'}}",
            font_size=16
        )
        backward_formula.set_color(SLATE_300)
        backward_formula.next_to(backward_title, DOWN, buff=0.2)

        self.play(FadeIn(backward_title, backward_formula), run_time=1)

        # Loss formula
        loss_formula = MathTex(
            r"\mathcal{L} = -\log P(Y|X) = -\log \sum_s \alpha_T(s) \cdot \beta_T(s)",
            font_size=18
        )
        loss_formula.set_color(ROSE_400)
        loss_formula.to_edge(DOWN)

        self.play(Write(loss_formula), run_time=1.5)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class CTCGradientScene(Scene):
    """
    Scene 4: Gradient computation
    """

    def construct(self):
        # Title
        title = Text("CTC Gradient for Training", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Gradient formula
        grad_formula = MathTex(
            r"\frac{\partial \mathcal{L}}{\partial y_t^k} = y_t^k - \frac{1}{P(Y|X)} \sum_{s: l_s = k} \alpha_t(s) \beta_t(s)",
            font_size=24
        )
        grad_formula.set_color(SLATE_50)
        grad_formula.move_to([0, 1, 0])

        self.play(Write(grad_formula), run_time=2)
        self.wait(1)

        # Interpretation
        interp = VGroup()
        interp_texts = [
            ("Prediction:", "y_t^k", "Model's probability for label k at time t", CYAN_400),
            ("Target:", "α·β term", "Posterior probability from forward-backward", EMERALD_400),
            ("Gradient:", "difference", "Push prediction toward target", ROSE_400),
        ]

        for i, (name, math_str, desc, color) in enumerate(interp_texts):
            row = VGroup()
            name_text = Text(name, font_size=14, color=color)
            desc_text = Text(desc, font_size=12, color=SLATE_400)

            name_text.move_to([-4, -0.5 - i * 0.7, 0])
            desc_text.next_to(name_text, RIGHT, buff=0.3)

            row.add(name_text, desc_text)
            interp.add(row)

        self.play(FadeIn(interp, lag_ratio=0.3), run_time=1.5)
        self.wait(1)

        # Training loop visualization
        loop_box = RoundedRectangle(
            width=8,
            height=2.5,
            corner_radius=0.1,
            fill_color=SLATE_800,
            fill_opacity=0.5,
            stroke_color=SLATE_700
        )
        loop_box.move_to([0, -2.5, 0])

        loop_title = Text("Training Loop", font_size=16, color=AMBER_400)
        loop_title.move_to(loop_box.get_top() + DOWN * 0.3)

        loop_steps = VGroup()
        steps = [
            "1. Forward pass: get y_t^k for all t, k",
            "2. Compute α, β via dynamic programming",
            "3. Compute CTC loss and gradients",
            "4. Backpropagate through encoder",
            "5. Update weights with optimizer"
        ]

        for i, step in enumerate(steps):
            t = Text(step, font_size=11, color=SLATE_400)
            t.move_to(loop_box.get_center() + UP * (0.6 - i * 0.35))
            loop_steps.add(t)

        self.play(FadeIn(loop_box, loop_title, loop_steps), run_time=1.5)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class CTCDecodingScene(Scene):
    """
    Scene 5: Decoding during inference
    """

    def construct(self):
        # Title
        title = Text("CTC Decoding", font_size=32, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Greedy decoding
        greedy_title = Text("Greedy Decoding", font_size=20, color=CYAN_400)
        greedy_title.move_to([-3, 1.5, 0])

        greedy_steps = VGroup()
        steps = [
            "1. At each frame, pick argmax",
            "2. Remove blank tokens",
            "3. Collapse consecutive duplicates"
        ]

        for i, step in enumerate(steps):
            t = Text(step, font_size=14, color=SLATE_400)
            t.move_to([-3, 0.8 - i * 0.4, 0])
            greedy_steps.add(t)

        self.play(FadeIn(greedy_title, greedy_steps), run_time=1.5)

        # Example
        decoder = CTCDecoder(
            output_sequence=['ε', 'H', 'H', 'ε', 'E', 'ε', 'L', 'L', 'L', 'O', 'ε']
        )
        decoder.scale(0.8)
        decoder.move_to([0, -1.5, 0])

        self.play(FadeIn(decoder), run_time=2)
        self.wait(1)

        # Beam search
        beam_title = Text("Beam Search (Better)", font_size=20, color=EMERALD_400)
        beam_title.to_corner(DR).shift(UP * 2.5 + LEFT * 0.5)

        beam_desc = Text(
            "Maintains top-k candidates\n→ Better results, slower",
            font_size=12,
            color=SLATE_400
        )
        beam_desc.next_to(beam_title, DOWN, buff=0.2)

        self.play(FadeIn(beam_title, beam_desc), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class CTCSummaryScene(Scene):
    """
    Summary of CTC Loss
    """

    def construct(self):
        # Title
        title = Text("CTC Loss Summary", font_size=36, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Key points
        points = VGroup()
        point_data = [
            ("Problem", "Unknown alignment between audio and text", ROSE_400),
            ("Solution", "Sum over all valid alignments", EMERALD_400),
            ("Blank Token", "Enables variable-length outputs", AMBER_400),
            ("Algorithm", "Forward-backward dynamic programming", CYAN_400),
            ("Decoding", "Greedy or beam search", PURPLE_400),
        ]

        for i, (key, value, color) in enumerate(point_data):
            box = RoundedRectangle(
                width=9,
                height=0.7,
                corner_radius=0.1,
                fill_color=color,
                fill_opacity=0.1,
                stroke_color=color,
                stroke_width=1
            )
            box.move_to([0, 1.5 - i * 0.9, 0])

            key_text = Text(key + ":", font_size=14, color=color)
            key_text.move_to(box.get_center() + LEFT * 3)

            value_text = Text(value, font_size=12, color=SLATE_400)
            value_text.move_to(box.get_center() + RIGHT * 0.5)

            points.add(box, key_text, value_text)

        self.play(FadeIn(points, lag_ratio=0.1), run_time=2)
        self.wait(2)

        # CTC advantage
        advantage = Text(
            "CTC enables end-to-end training without manual alignments!",
            font_size=16,
            color=EMERALD_400
        )
        advantage.to_edge(DOWN).shift(UP * 0.5)

        self.play(FadeIn(advantage), run_time=1)
        self.wait(1)

        # Next section
        next_text = Text(
            "Next: Full Pipeline Integration →",
            font_size=18,
            color=CYAN_400
        )
        next_text.to_edge(DOWN)

        self.play(FadeIn(next_text), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


# Scene sequence
if __name__ == "__main__":
    scenes = [
        CTCIntroScene,
        CTCProblemScene,
        CTCBlankTokenScene,
        CTCAlgorithmScene,
        CTCGradientScene,
        CTCDecodingScene,
        CTCSummaryScene,
    ]

    print("Video 4: CTC Loss Scenes")
    print("=" * 40)
    for i, scene in enumerate(scenes):
        print(f"{i+1}. {scene.__name__}")
