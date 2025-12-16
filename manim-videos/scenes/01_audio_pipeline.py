"""
Video 1: The Audio Pipeline
Duration: ~8-10 minutes

Scenes:
1. AudioWaveformScene - Introduction to sound waves
2. STFTScene - Short-Time Fourier Transform
3. MelFilterBankScene - Mel-scale filter banks
4. LogCompressionScene - Log compression and normalization
5. AudioPipelineSummary - Complete pipeline overview
"""

from manim import *
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from custom.colors import *
from custom.dsp_components import *


class IntroScene(Scene):
    """
    Opening scene introducing the audio pipeline concept
    """

    def construct(self):
        # Title
        title = Text("Part 1: The Audio Pipeline", font_size=48, color=SLATE_50)
        subtitle = Text("From Sound Waves to Numbers", font_size=24, color=CYAN_400)
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1)
        self.wait(2)

        # Transition text
        question = Text(
            "How do we convert speech into a format neural networks can understand?",
            font_size=20,
            color=SLATE_400
        )
        question.move_to([0, -1, 0])

        self.play(FadeIn(question), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(title, subtitle, question), run_time=1)


class AudioWaveformScene(Scene):
    """
    Scene 1: The Waveform
    Shows a real audio waveform with time and amplitude
    """

    def construct(self):
        # Section title
        section_title = Text("The Sound Wave", font_size=36, color=SLATE_50)
        section_title.to_edge(UP)
        self.play(Write(section_title), run_time=1)

        # Create axes
        axes = Axes(
            x_range=[0, 2, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=4,
            axis_config={
                "color": SLATE_600,
                "include_tip": True,
                "tip_length": 0.2
            },
            x_axis_config={"numbers_to_include": [0, 0.5, 1, 1.5, 2]},
            y_axis_config={"numbers_to_include": [-1, 0, 1]},
        )
        axes.shift(DOWN * 0.5)

        # Axis labels
        x_label = Text("Time (seconds)", font_size=16, color=SLATE_400)
        x_label.next_to(axes.x_axis, DOWN, buff=0.3)

        y_label = Text("Amplitude", font_size=16, color=SLATE_400)
        y_label.rotate(PI / 2)
        y_label.next_to(axes.y_axis, LEFT, buff=0.3)

        self.play(Create(axes), FadeIn(x_label, y_label), run_time=2)

        # Create and animate waveform
        def speech_wave(t):
            """Generate speech-like waveform"""
            envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 2 * t)
            carrier = np.sin(2 * np.pi * 220 * t)
            harmonics = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 660 * t)
            return envelope * (carrier + harmonics)

        # Simple sine wave first
        sine_wave = axes.plot(
            lambda t: np.sin(2 * np.pi * 2 * t),
            x_range=[0, 2],
            color=CYAN_400,
            stroke_width=2
        )

        sine_label = Text("Simple Sine Wave (2 Hz)", font_size=14, color=CYAN_400)
        sine_label.next_to(axes, UP, buff=0.2)

        self.play(Create(sine_wave), FadeIn(sine_label), run_time=2)
        self.wait(1)

        # Transform to complex speech-like wave
        speech_label = Text("Speech Waveform (Complex)", font_size=14, color=CYAN_400)
        speech_label.next_to(axes, UP, buff=0.2)

        # Create a more complex waveform visualization
        # Since we can't plot high frequencies directly, show envelope
        complex_wave = axes.plot(
            lambda t: (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 8 * t),
            x_range=[0, 2],
            color=CYAN_400,
            stroke_width=2
        )

        self.play(
            Transform(sine_wave, complex_wave),
            Transform(sine_label, speech_label),
            run_time=2
        )
        self.wait(1)

        # Show sampling rate info
        sample_info = VGroup()
        sr_text = Text("Sampling Rate: 16,000 Hz", font_size=18, color=AMBER_400)
        sr_text.to_corner(DR)
        sr_text.shift(UP * 0.5)

        samples_text = Text("32,000 samples for 2 seconds", font_size=14, color=SLATE_400)
        samples_text.next_to(sr_text, DOWN, buff=0.1)

        sample_info.add(sr_text, samples_text)

        self.play(FadeIn(sample_info), run_time=1)
        self.wait(2)

        # Show discrete samples
        sample_points = VGroup()
        n_samples = 30  # Show subset
        for i in range(n_samples):
            t = i * 2 / n_samples
            y = (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 8 * t)
            point = Dot(
                axes.c2p(t, y),
                radius=0.05,
                color=EMERALD_400
            )
            sample_points.add(point)

        sampling_label = Text("Discrete Samples", font_size=14, color=EMERALD_400)
        sampling_label.to_corner(UL)
        sampling_label.shift(DOWN * 1)

        self.play(
            FadeIn(sample_points, lag_ratio=0.05),
            FadeIn(sampling_label),
            run_time=2
        )
        self.wait(2)

        # Key insight
        insight = Text(
            "Challenge: Raw samples don't reveal frequency content",
            font_size=18,
            color=ROSE_400
        )
        insight.to_edge(DOWN)

        self.play(FadeIn(insight), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class STFTScene(Scene):
    """
    Scene 2: STFT (Short-Time Fourier Transform)
    Shows windowing and frequency analysis
    """

    def construct(self):
        # Section title
        section_title = Text("Short-Time Fourier Transform (STFT)", font_size=32, color=SLATE_50)
        section_title.to_edge(UP)
        self.play(Write(section_title), run_time=1)

        # Create waveform
        axes = Axes(
            x_range=[0, 1, 0.25],
            y_range=[-1, 1, 0.5],
            x_length=12,
            y_length=2,
            axis_config={"color": SLATE_600}
        )
        axes.shift(UP * 1.5)

        waveform = axes.plot(
            lambda t: 0.7 * np.sin(2 * np.pi * 8 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)),
            x_range=[0, 1],
            color=CYAN_400,
            stroke_width=2
        )

        time_label = Text("Time →", font_size=14, color=SLATE_400)
        time_label.next_to(axes, DOWN, buff=0.1)

        self.play(Create(axes), Create(waveform), FadeIn(time_label), run_time=2)

        # STFT parameters
        params = VGroup()
        param_texts = [
            "Window: 25ms (400 samples)",
            "Hop: 10ms (160 samples)",
            "FFT Size: 512"
        ]
        for i, text in enumerate(param_texts):
            t = Text(text, font_size=14, color=SLATE_400)
            t.to_corner(UL)
            t.shift(DOWN * (1.2 + i * 0.3))
            params.add(t)

        self.play(FadeIn(params), run_time=1)

        # Create sliding window
        window_width = 1.5
        window = Rectangle(
            width=window_width,
            height=2.5,
            fill_color=AMBER_400,
            fill_opacity=0.2,
            stroke_color=AMBER_400,
            stroke_width=2
        )
        window.move_to(axes.c2p(0.125, 0))

        window_label = Text("Window (25ms)", font_size=12, color=AMBER_400)
        window_label.next_to(window, UP, buff=0.1)

        self.play(FadeIn(window, window_label), run_time=1)
        self.wait(1)

        # Animate window sliding
        hop_count = 5
        hop_width = 0.1

        # Create frequency display area
        freq_axes = Axes(
            x_range=[0, 8000, 2000],
            y_range=[0, 1, 0.25],
            x_length=5,
            y_length=2,
            axis_config={"color": SLATE_600}
        )
        freq_axes.to_corner(DR)
        freq_axes.shift(LEFT * 0.5 + UP * 0.5)

        freq_label = Text("Frequency (Hz)", font_size=12, color=SLATE_400)
        freq_label.next_to(freq_axes, DOWN, buff=0.1)

        self.play(Create(freq_axes), FadeIn(freq_label), run_time=1)

        # FFT visualization (bars)
        n_bars = 20
        bars = VGroup()
        for i in range(n_bars):
            bar = Rectangle(
                width=0.2,
                height=0.1,
                fill_color=CYAN_500,
                fill_opacity=0.8,
                stroke_width=0
            )
            x = freq_axes.c2p(i * 400, 0)[0]
            bar.move_to([x, freq_axes.c2p(0, 0)[1] + 0.05, 0])
            bars.add(bar)

        self.add(bars)

        # Slide window and update FFT
        for hop in range(hop_count):
            new_pos = axes.c2p(0.125 + hop * hop_width, 0)

            # Generate random FFT magnitudes (simulated)
            magnitudes = np.random.rand(n_bars) ** 2
            magnitudes[0:5] *= 2  # Lower frequencies stronger
            magnitudes = np.clip(magnitudes, 0.05, 1)

            # Animate window movement and FFT update
            animations = [window.animate.move_to(new_pos)]
            animations.append(window_label.animate.move_to(new_pos + UP * 1.4))

            for i, (bar, mag) in enumerate(zip(bars, magnitudes)):
                new_height = mag * 1.5
                x = freq_axes.c2p(i * 400, 0)[0]
                y = freq_axes.c2p(0, 0)[1] + new_height / 2
                animations.append(bar.animate.stretch_to_fit_height(new_height).move_to([x, y, 0]))

            self.play(*animations, run_time=0.8)

        self.wait(1)

        # Show spectrogram building
        spec_title = Text("→ Stack frames to build Spectrogram", font_size=16, color=EMERALD_400)
        spec_title.next_to(freq_axes, LEFT, buff=0.5)
        spec_title.shift(UP * 0.5)

        self.play(FadeIn(spec_title), run_time=1)

        # Create mini spectrogram
        spectrogram = SpectrogramDisplay(
            n_frames=30,
            n_freqs=20,
            width=4,
            height=2
        )
        spectrogram.move_to([-3, -1.5, 0])

        spec_label = Text("Spectrogram (Time × Frequency)", font_size=12, color=SLATE_400)
        spec_label.next_to(spectrogram, DOWN, buff=0.2)

        self.play(FadeIn(spectrogram, spec_label), run_time=2)
        self.wait(2)

        # STFT formula
        formula = MathTex(
            r"X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j2\pi kn/N}",
            font_size=24
        )
        formula.set_color(SLATE_300)
        formula.to_edge(DOWN)

        self.play(Write(formula), run_time=2)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class MelFilterBankScene(Scene):
    """
    Scene 3: Mel Filter Bank
    Shows conversion to mel-scale
    """

    def construct(self):
        # Section title
        section_title = Text("Mel Filter Bank", font_size=32, color=SLATE_50)
        section_title.to_edge(UP)
        self.play(Write(section_title), run_time=1)

        # Explanation
        explanation = Text(
            "Human hearing is more sensitive to low frequencies",
            font_size=18,
            color=SLATE_400
        )
        explanation.next_to(section_title, DOWN, buff=0.3)
        self.play(FadeIn(explanation), run_time=1)

        # Create frequency comparison
        # Linear scale
        lin_axes = Axes(
            x_range=[0, 8000, 2000],
            y_range=[0, 1, 0.5],
            x_length=5,
            y_length=2,
            axis_config={"color": SLATE_600}
        )
        lin_axes.move_to([-3, 0.5, 0])

        lin_label = Text("Linear Frequency Scale", font_size=14, color=SLATE_400)
        lin_label.next_to(lin_axes, DOWN, buff=0.2)

        # Mel scale
        mel_axes = Axes(
            x_range=[0, 8000, 2000],
            y_range=[0, 1, 0.5],
            x_length=5,
            y_length=2,
            axis_config={"color": SLATE_600}
        )
        mel_axes.move_to([3, 0.5, 0])

        mel_label = Text("Mel Frequency Scale", font_size=14, color=EMERALD_400)
        mel_label.next_to(mel_axes, DOWN, buff=0.2)

        self.play(
            Create(lin_axes), Create(mel_axes),
            FadeIn(lin_label, mel_label),
            run_time=2
        )

        # Show linear spacing
        lin_marks = VGroup()
        for f in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
            mark = Dot(lin_axes.c2p(f, 0.5), radius=0.08, color=CYAN_400)
            lin_marks.add(mark)

        self.play(FadeIn(lin_marks), run_time=1)

        # Show mel spacing (non-linear)
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        mel_freqs = np.linspace(hz_to_mel(0), hz_to_mel(8000), 9)[1:]
        hz_points = 700 * (10 ** (mel_freqs / 2595) - 1)

        mel_marks = VGroup()
        for f in hz_points:
            mark = Dot(mel_axes.c2p(f, 0.5), radius=0.08, color=EMERALD_400)
            mel_marks.add(mark)

        self.play(FadeIn(mel_marks), run_time=1)
        self.wait(1)

        # Annotation
        annotation = Text(
            "More filters at low frequencies where humans discriminate better",
            font_size=14,
            color=AMBER_400
        )
        annotation.move_to([0, -0.8, 0])
        self.play(FadeIn(annotation), run_time=1)
        self.wait(1)

        # Mel formula
        mel_formula = MathTex(
            r"\text{mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)",
            font_size=24
        )
        mel_formula.set_color(EMERALD_400)
        mel_formula.move_to([0, -1.8, 0])

        self.play(Write(mel_formula), run_time=1.5)
        self.wait(1)

        # Clear and show filter bank
        self.play(
            FadeOut(lin_axes, mel_axes, lin_marks, mel_marks, lin_label, mel_label, annotation),
            mel_formula.animate.to_corner(DL),
            run_time=1
        )

        # Create mel filter bank visualization
        filter_bank = MelFilterBank(
            n_mels=80,
            n_display=15,
            width=10,
            height=3
        )
        filter_bank.move_to([0, 0, 0])

        filter_title = Text("80 Triangular Mel Filters", font_size=18, color=SLATE_50)
        filter_title.next_to(filter_bank, UP, buff=0.3)

        self.play(FadeIn(filter_bank, filter_title), run_time=2)
        self.wait(2)

        # Show output shape
        output_info = VGroup()
        info1 = Text("Input: Linear Spectrogram (257 freq bins)", font_size=14, color=SLATE_400)
        info2 = Text("Output: Mel Spectrogram (80 mel bins)", font_size=14, color=EMERALD_400)
        info1.to_corner(DR).shift(UP * 1.5)
        info2.next_to(info1, DOWN, buff=0.2)
        output_info.add(info1, info2)

        self.play(FadeIn(output_info), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class LogCompressionScene(Scene):
    """
    Scene 4: Log Compression and Normalization
    Final preprocessing step
    """

    def construct(self):
        # Section title
        section_title = Text("Log Compression & Normalization", font_size=32, color=SLATE_50)
        section_title.to_edge(UP)
        self.play(Write(section_title), run_time=1)

        # Explanation
        explanation = Text(
            "Human perception of loudness is logarithmic",
            font_size=18,
            color=SLATE_400
        )
        explanation.next_to(section_title, DOWN, buff=0.3)
        self.play(FadeIn(explanation), run_time=1)

        # Show linear vs log scale perception
        # Create bars showing dynamic range
        linear_bars = VGroup()
        log_bars = VGroup()

        values = [1, 10, 100, 1000, 10000]
        log_values = [np.log10(v) for v in values]
        max_log = max(log_values)

        bar_width = 1.5
        max_height = 3

        for i, (val, log_val) in enumerate(zip(values, log_values)):
            # Linear bar (would be huge range)
            lin_height = min(val / 1000 * max_height, max_height)
            lin_bar = Rectangle(
                width=bar_width * 0.8,
                height=lin_height,
                fill_color=CYAN_500,
                fill_opacity=0.6,
                stroke_width=0
            )
            lin_bar.move_to([-4 + i * bar_width, lin_height / 2 - 1, 0])
            linear_bars.add(lin_bar)

            # Log bar (compressed range)
            log_height = log_val / max_log * max_height
            log_bar = Rectangle(
                width=bar_width * 0.8,
                height=log_height,
                fill_color=EMERALD_500,
                fill_opacity=0.6,
                stroke_width=0
            )
            log_bar.move_to([2 + i * bar_width, log_height / 2 - 1, 0])
            log_bars.add(log_bar)

        # Labels
        lin_title = Text("Linear Scale", font_size=16, color=CYAN_400)
        lin_title.move_to([-2, -2, 0])

        log_title = Text("Log Scale", font_size=16, color=EMERALD_400)
        log_title.move_to([4, -2, 0])

        self.play(
            FadeIn(linear_bars, log_bars),
            FadeIn(lin_title, log_title),
            run_time=2
        )
        self.wait(1)

        # Value labels
        for i, val in enumerate(values):
            label = Text(str(val), font_size=10, color=SLATE_400)
            label.move_to([-4 + i * bar_width, -1.3, 0])
            self.add(label)

        self.wait(1)

        # Formula
        formula = MathTex(
            r"\text{log\_mel} = \log(\max(\text{mel}, 10^{-10}))",
            font_size=24
        )
        formula.set_color(SLATE_300)
        formula.move_to([0, 2, 0])

        self.play(Write(formula), run_time=1.5)
        self.wait(1)

        # Clear and show normalization
        self.play(FadeOut(linear_bars, log_bars, lin_title, log_title, formula), run_time=1)

        # Normalization explanation
        norm_title = Text("Global Mean-Variance Normalization", font_size=20, color=AMBER_400)
        norm_title.move_to([0, 1, 0])

        norm_formula = MathTex(
            r"x_{\text{norm}} = \frac{x - \mu}{\sigma}",
            font_size=32
        )
        norm_formula.set_color(SLATE_50)
        norm_formula.next_to(norm_title, DOWN, buff=0.5)

        benefits = VGroup()
        benefit_texts = [
            "• Zero mean: centers data around 0",
            "• Unit variance: consistent scale for neural network",
            "• Improves training stability"
        ]
        for i, text in enumerate(benefit_texts):
            t = Text(text, font_size=16, color=SLATE_400)
            t.move_to([0, -0.5 - i * 0.4, 0])
            benefits.add(t)

        self.play(FadeIn(norm_title), run_time=1)
        self.play(Write(norm_formula), run_time=1.5)
        self.play(FadeIn(benefits, lag_ratio=0.3), run_time=1.5)
        self.wait(2)

        # Final output shape
        output = VGroup()
        output_title = Text("Final Audio Features", font_size=20, color=EMERALD_400)
        output_shape = Text("Shape: (T_frames, 80)", font_size=16, color=SLATE_50)
        output_desc = Text("Ready for Transformer encoder!", font_size=14, color=SLATE_400)

        output_title.move_to([0, -2.2, 0])
        output_shape.next_to(output_title, DOWN, buff=0.2)
        output_desc.next_to(output_shape, DOWN, buff=0.1)
        output.add(output_title, output_shape, output_desc)

        self.play(FadeIn(output), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


class AudioPipelineSummary(Scene):
    """
    Scene 5: Complete pipeline overview
    Summary of all audio preprocessing steps
    """

    def construct(self):
        # Title
        title = Text("Audio Pipeline Summary", font_size=36, color=SLATE_50)
        title.to_edge(UP)
        self.play(Write(title), run_time=1)

        # Create pipeline blocks
        blocks = []
        block_data = [
            ("Raw Audio", "16kHz Waveform", SLATE_500, "(T samples)"),
            ("STFT", "25ms window, 10ms hop", CYAN_500, "(T', 257)"),
            ("Mel Filter", "80 triangular filters", EMERALD_500, "(T', 80)"),
            ("Log + Norm", "Compression & scaling", AMBER_500, "(T', 80)"),
        ]

        block_width = 2.8
        block_height = 1.2
        spacing = 0.4
        start_x = -5

        for i, (name, desc, color, shape) in enumerate(block_data):
            block = ProcessingBlock(
                label=name,
                width=block_width,
                height=block_height,
                color=color,
                sublabel=desc
            )
            block.move_to([start_x + i * (block_width + spacing), 0.5, 0])

            # Shape annotation
            shape_label = Text(shape, font_size=12, color=SLATE_500)
            shape_label.next_to(block, DOWN, buff=0.1)

            blocks.append((block, shape_label))

            self.play(FadeIn(block, shape_label), run_time=0.5)

            # Add arrow (except after last)
            if i < len(block_data) - 1:
                arrow = Arrow(
                    start=block.get_right() + RIGHT * 0.1,
                    end=block.get_right() + RIGHT * (spacing - 0.1),
                    color=SLATE_600,
                    stroke_width=3,
                    buff=0
                )
                self.play(Create(arrow), run_time=0.3)

        self.wait(1)

        # Key parameters box
        params_box = RoundedRectangle(
            width=10,
            height=2,
            corner_radius=0.1,
            fill_color=SLATE_800,
            fill_opacity=0.5,
            stroke_color=SLATE_700
        )
        params_box.move_to([0, -2, 0])

        params_title = Text("Key Parameters", font_size=16, color=SLATE_50)
        params_title.move_to(params_box.get_top() + DOWN * 0.3)

        params = VGroup()
        param_items = [
            ("Sample Rate:", "16,000 Hz"),
            ("Window:", "25ms (400 samples)"),
            ("Hop:", "10ms (160 samples)"),
            ("Mel Bins:", "80"),
            ("Reduction:", "4× via Conv subsampling"),
        ]

        for i, (key, val) in enumerate(param_items):
            col = i % 3
            row = i // 3
            key_text = Text(key, font_size=12, color=SLATE_400)
            val_text = Text(val, font_size=12, color=CYAN_400)
            val_text.next_to(key_text, RIGHT, buff=0.1)

            item = VGroup(key_text, val_text)
            item.move_to([-3.5 + col * 3.5, -1.7 - row * 0.4, 0])
            params.add(item)

        self.play(FadeIn(params_box, params_title, params), run_time=1)
        self.wait(2)

        # Transition to next section
        next_text = Text(
            "Next: Transformer Foundations →",
            font_size=18,
            color=PURPLE_400
        )
        next_text.to_edge(DOWN)

        self.play(FadeIn(next_text), run_time=1)
        self.wait(2)

        # Fade out
        self.play(FadeOut(*self.mobjects), run_time=1)


# Scene sequence for rendering
if __name__ == "__main__":
    scenes = [
        IntroScene,
        AudioWaveformScene,
        STFTScene,
        MelFilterBankScene,
        LogCompressionScene,
        AudioPipelineSummary,
    ]

    print("Video 1: Audio Pipeline Scenes")
    print("=" * 40)
    for i, scene in enumerate(scenes):
        print(f"{i+1}. {scene.__name__}")
