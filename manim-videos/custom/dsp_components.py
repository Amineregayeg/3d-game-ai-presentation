"""
Digital Signal Processing components for audio visualization
Includes: Waveforms, STFT, Mel filterbanks, Spectrograms
"""

from manim import *
import numpy as np
from .colors import *


class AudioWaveform(VGroup):
    """
    Animated audio waveform visualization

    Creates a dynamic waveform that can be animated and manipulated
    """

    def __init__(
        self,
        duration=2.0,
        sample_rate=16000,
        frequency=440,
        amplitude=1.0,
        width=10,
        height=2,
        color=WAVEFORM_COLOR,
        stroke_width=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.duration = duration
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.amplitude = amplitude
        self.width = width
        self.height = height

        # Generate waveform data
        t = np.linspace(0, duration, int(duration * sample_rate))
        self.samples = amplitude * np.sin(2 * np.pi * frequency * t)

        # Create the waveform graph
        self.waveform = self._create_waveform(color, stroke_width)
        self.add(self.waveform)

    def _create_waveform(self, color, stroke_width):
        """Create the waveform as a VMobject"""
        # Downsample for visualization
        display_samples = 1000
        indices = np.linspace(0, len(self.samples) - 1, display_samples).astype(int)
        y_values = self.samples[indices]

        # Create points
        points = []
        for i, y in enumerate(y_values):
            x = (i / display_samples) * self.width - self.width / 2
            y_scaled = y * self.height / 2
            points.append([x, y_scaled, 0])

        # Create the curve
        curve = VMobject()
        curve.set_points_smoothly([np.array(p) for p in points])
        curve.set_stroke(color=color, width=stroke_width)

        return curve

    def get_segment(self, start_frac, end_frac):
        """Get a segment of the waveform"""
        return self.waveform.get_subcurve(start_frac, end_frac)


class SpeechWaveform(AudioWaveform):
    """
    More complex speech-like waveform with multiple frequencies
    """

    def __init__(self, **kwargs):
        # Override to create speech-like pattern
        super().__init__(**kwargs)

    def _create_waveform(self, color, stroke_width):
        """Create speech-like waveform with varying amplitude and frequency"""
        t = np.linspace(0, self.duration, int(self.duration * self.sample_rate))

        # Speech-like modulation
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # Syllable-like envelope
        carrier = np.sin(2 * np.pi * self.frequency * t)

        # Add harmonics
        harmonics = (
            0.5 * np.sin(2 * np.pi * self.frequency * 2 * t) +
            0.25 * np.sin(2 * np.pi * self.frequency * 3 * t) +
            0.125 * np.sin(2 * np.pi * self.frequency * 4 * t)
        )

        # Add noise
        noise = 0.1 * np.random.randn(len(t))

        self.samples = self.amplitude * envelope * (carrier + harmonics) + noise

        # Downsample for visualization
        display_samples = 2000
        indices = np.linspace(0, len(self.samples) - 1, display_samples).astype(int)
        y_values = self.samples[indices]

        # Create points
        points = []
        for i, y in enumerate(y_values):
            x = (i / display_samples) * self.width - self.width / 2
            y_scaled = y * self.height / 2
            points.append([x, y_scaled, 0])

        curve = VMobject()
        curve.set_points_smoothly([np.array(p) for p in points])
        curve.set_stroke(color=color, width=stroke_width)

        return curve


class STFTWindow(VGroup):
    """
    Visualization of STFT sliding window
    """

    def __init__(
        self,
        width=1.0,
        height=2.0,
        color=STFT_WINDOW_COLOR,
        fill_opacity=0.3,
        stroke_width=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Window rectangle
        self.window_rect = Rectangle(
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=fill_opacity,
            stroke_color=color,
            stroke_width=stroke_width
        )
        self.add(self.window_rect)

        # Hann window shape overlay
        self.hann_curve = self._create_hann_window(width, height, color)
        self.add(self.hann_curve)

    def _create_hann_window(self, width, height, color):
        """Create Hann window shape"""
        n_points = 100
        x_vals = np.linspace(-width/2, width/2, n_points)
        # Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
        n = np.linspace(0, 1, n_points)
        y_vals = 0.5 * (1 - np.cos(2 * np.pi * n)) * height / 2

        points = [[x, y - height/4, 0] for x, y in zip(x_vals, y_vals)]

        curve = VMobject()
        curve.set_points_smoothly([np.array(p) for p in points])
        curve.set_stroke(color=SLATE_50, width=2)

        return curve


class FrequencySpectrum(VGroup):
    """
    Bar chart visualization of frequency spectrum (FFT result)
    """

    def __init__(
        self,
        n_bins=64,
        width=4,
        height=2,
        color_low=SLATE_700,
        color_high=CYAN_400,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_bins = n_bins
        self.width = width
        self.height = height
        self.bars = []

        bar_width = width / n_bins * 0.8

        # Generate random spectrum data (will be updated)
        magnitudes = np.random.rand(n_bins) ** 2  # Squared for more realistic look

        for i in range(n_bins):
            bar_height = magnitudes[i] * height
            bar = Rectangle(
                width=bar_width,
                height=max(bar_height, 0.05),
                fill_opacity=0.8,
                stroke_width=0
            )

            # Position bar
            x_pos = (i / n_bins) * width - width / 2 + bar_width / 2
            bar.move_to([x_pos, bar_height / 2 - height / 2, 0])

            # Color based on magnitude
            t = magnitudes[i]
            bar.set_fill(interpolate_color(color_low, color_high, t))

            self.bars.append(bar)
            self.add(bar)

    def update_spectrum(self, magnitudes):
        """Update bar heights based on new magnitude data"""
        for i, (bar, mag) in enumerate(zip(self.bars, magnitudes)):
            new_height = max(mag * self.height, 0.05)
            bar.stretch_to_fit_height(new_height)
            bar.move_to([bar.get_center()[0], new_height / 2 - self.height / 2, 0])
            bar.set_fill(interpolate_color(SLATE_700, CYAN_400, mag))


class MelFilterBank(VGroup):
    """
    Visualization of triangular Mel filter banks
    """

    def __init__(
        self,
        n_mels=80,
        n_display=20,  # Show subset for clarity
        width=8,
        height=2,
        f_min=0,
        f_max=8000,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_mels = n_mels
        self.n_display = n_display
        self.width = width
        self.height = height

        # Create frequency axis
        self.freq_axis = self._create_freq_axis(f_min, f_max)
        self.add(self.freq_axis)

        # Create mel filters
        self.filters = self._create_mel_filters(n_display, f_min, f_max)
        self.add(self.filters)

    def _hz_to_mel(self, hz):
        """Convert Hz to Mel scale"""
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        """Convert Mel to Hz"""
        return 700 * (10 ** (mel / 2595) - 1)

    def _create_freq_axis(self, f_min, f_max):
        """Create frequency axis with labels"""
        axis = VGroup()

        # Axis line
        line = Line(
            start=[-self.width/2, -self.height/2, 0],
            end=[self.width/2, -self.height/2, 0],
            color=SLATE_500
        )
        axis.add(line)

        # Frequency labels
        freqs = [0, 1000, 2000, 4000, 8000]
        for f in freqs:
            x = (f / f_max) * self.width - self.width / 2
            tick = Line(
                start=[x, -self.height/2, 0],
                end=[x, -self.height/2 - 0.1, 0],
                color=SLATE_500
            )
            label = Text(f"{f//1000}k" if f >= 1000 else str(f), font_size=16, color=SLATE_400)
            label.next_to(tick, DOWN, buff=0.05)
            axis.add(tick, label)

        # Axis label
        axis_label = Text("Frequency (Hz)", font_size=18, color=SLATE_400)
        axis_label.next_to(line, DOWN, buff=0.4)
        axis.add(axis_label)

        return axis

    def _create_mel_filters(self, n_filters, f_min, f_max):
        """Create triangular Mel filters"""
        filters = VGroup()

        # Calculate mel points
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Colors for filters (gradient from emerald to cyan)
        colors = [
            interpolate_color(EMERALD_400, CYAN_400, i / n_filters)
            for i in range(n_filters)
        ]

        for i in range(n_filters):
            f_left = hz_points[i]
            f_center = hz_points[i + 1]
            f_right = hz_points[i + 2]

            # Convert to x positions
            x_left = (f_left / f_max) * self.width - self.width / 2
            x_center = (f_center / f_max) * self.width - self.width / 2
            x_right = (f_right / f_max) * self.width - self.width / 2

            # Create triangle
            triangle = Polygon(
                [x_left, -self.height/2, 0],
                [x_center, self.height/2, 0],
                [x_right, -self.height/2, 0],
                fill_color=colors[i],
                fill_opacity=0.3,
                stroke_color=colors[i],
                stroke_width=1.5
            )
            filters.add(triangle)

        return filters


class SpectrogramDisplay(VGroup):
    """
    Heatmap visualization of spectrogram
    """

    def __init__(
        self,
        n_frames=100,
        n_freqs=80,
        width=8,
        height=4,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_frames = n_frames
        self.n_freqs = n_freqs
        self.width = width
        self.height = height

        # Create background
        self.background = Rectangle(
            width=width,
            height=height,
            fill_color=SLATE_900,
            fill_opacity=1,
            stroke_color=SLATE_700,
            stroke_width=1
        )
        self.add(self.background)

        # Create pixel grid (simplified - uses rectangles)
        self.pixels = self._create_pixel_grid()
        self.add(self.pixels)

        # Add axis labels
        self.axes = self._create_axes()
        self.add(self.axes)

    def _create_pixel_grid(self):
        """Create a grid of colored rectangles for the spectrogram"""
        pixels = VGroup()

        # For performance, use fewer visual elements
        visual_frames = min(self.n_frames, 50)
        visual_freqs = min(self.n_freqs, 40)

        pixel_width = self.width / visual_frames
        pixel_height = self.height / visual_freqs

        # Generate sample spectrogram data
        data = self._generate_sample_spectrogram(visual_frames, visual_freqs)

        for i in range(visual_frames):
            for j in range(visual_freqs):
                value = data[j, i]

                x = (i / visual_frames) * self.width - self.width / 2 + pixel_width / 2
                y = (j / visual_freqs) * self.height - self.height / 2 + pixel_height / 2

                pixel = Rectangle(
                    width=pixel_width,
                    height=pixel_height,
                    fill_color=self._value_to_color(value),
                    fill_opacity=1,
                    stroke_width=0
                )
                pixel.move_to([x, y, 0])
                pixels.add(pixel)

        return pixels

    def _generate_sample_spectrogram(self, n_frames, n_freqs):
        """Generate sample spectrogram data"""
        # Create a pattern that looks like speech
        data = np.zeros((n_freqs, n_frames))

        for f in range(n_freqs):
            # Fundamental frequency and harmonics
            freq_factor = (n_freqs - f) / n_freqs

            # Speech-like pattern with formants
            formant1 = np.exp(-((f - n_freqs * 0.1) ** 2) / (n_freqs * 0.02))
            formant2 = np.exp(-((f - n_freqs * 0.3) ** 2) / (n_freqs * 0.05))
            formant3 = np.exp(-((f - n_freqs * 0.5) ** 2) / (n_freqs * 0.08))

            base = formant1 + 0.7 * formant2 + 0.4 * formant3

            # Time variation
            for t in range(n_frames):
                envelope = 0.5 + 0.5 * np.sin(2 * np.pi * t / n_frames * 3)
                noise = 0.1 * np.random.rand()
                data[f, t] = envelope * base * 0.8 + noise

        return np.clip(data, 0, 1)

    def _value_to_color(self, value):
        """Convert normalized value to color"""
        gradient = get_spectrogram_gradient()
        n = len(gradient) - 1
        idx = value * n
        low_idx = int(np.floor(idx))
        high_idx = min(low_idx + 1, n)
        t = idx - low_idx
        return interpolate_color(gradient[low_idx], gradient[high_idx], t)

    def _create_axes(self):
        """Create time and frequency axes"""
        axes = VGroup()

        # Time axis
        time_label = Text("Time", font_size=18, color=SLATE_400)
        time_label.next_to(self.background, DOWN, buff=0.3)
        axes.add(time_label)

        # Frequency axis
        freq_label = Text("Freq", font_size=18, color=SLATE_400)
        freq_label.rotate(PI / 2)
        freq_label.next_to(self.background, LEFT, buff=0.3)
        axes.add(freq_label)

        return axes


class ProcessingBlock(VGroup):
    """
    Reusable processing block visualization
    """

    def __init__(
        self,
        label,
        width=2.5,
        height=1,
        color=CYAN_500,
        sublabel=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Main rectangle
        self.rect = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.1,
            fill_color=color,
            fill_opacity=0.2,
            stroke_color=color,
            stroke_width=2
        )
        self.add(self.rect)

        # Main label
        self.label = Text(label, font_size=20, color=SLATE_50)
        self.label.move_to(self.rect.get_center())
        if sublabel:
            self.label.shift(UP * 0.15)
        self.add(self.label)

        # Sublabel
        if sublabel:
            self.sublabel = Text(sublabel, font_size=14, color=SLATE_400)
            self.sublabel.next_to(self.label, DOWN, buff=0.1)
            self.add(self.sublabel)


class DataFlowArrow(VGroup):
    """
    Animated arrow showing data flow between components
    """

    def __init__(
        self,
        start,
        end,
        color=SLATE_500,
        label=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.arrow = Arrow(
            start=start,
            end=end,
            color=color,
            buff=0.1,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.15
        )
        self.add(self.arrow)

        if label:
            self.label = Text(label, font_size=14, color=SLATE_400)
            self.label.move_to(self.arrow.get_center())
            self.label.shift(UP * 0.3)
            self.add(self.label)
