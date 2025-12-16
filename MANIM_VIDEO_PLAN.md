# VoxFormer STT Manim Video Production Plan

## Overview
This document outlines a comprehensive video series using Manim to visualize and explain the VoxFormer Speech-to-Text Transformer architecture. The videos will transform the technical documentation into engaging visual explanations, perfect for technical presentations, tutorials, and educational content.

---

## Video Series Structure

### 5-Part Video Series (45-60 minutes total)

```
Part 1: The Audio Pipeline           [8-10 min]
  ├─ From Sound Waves to Numbers
  └─ STFT, Mel-Spectrograms, and Feature Extraction

Part 2: Transformer Foundations      [10-12 min]
  ├─ Attention Mechanisms
  └─ Position Encoding with RoPE

Part 3: The Conformer Block          [10-12 min]
  ├─ Multi-Head Attention
  ├─ Convolution Modules
  └─ Feed-Forward Networks

Part 4: Training with CTC Loss       [8-10 min]
  ├─ CTC Algorithm Visualization
  └─ Forward-Backward Algorithm

Part 5: Full Pipeline Integration    [8-10 min]
  ├─ End-to-End Architecture
  └─ Inference and Decoding
```

---

## Video 1: The Audio Pipeline (8-10 minutes)

### Scene 1: The Waveform
**Duration**: 1-2 min
**Purpose**: Introduce the problem - converting sound to numbers

```python
# Manim Scene Structure
class AudioWaveformScene(Scene):
    """
    Shows a real audio waveform (sine wave demo) oscillating
    with time and amplitude labels.
    """
    def construct(self):
        # Draw axes
        axes = Axes(...)

        # Animate waveform generation
        waveform = FunctionGraph(...)
        self.play(Create(axes), run_time=2)
        self.play(Create(waveform), run_time=3)

        # Show sampling at 16kHz
        sample_points = [dots for discrete time intervals]
        self.play(FadeIn(sample_points), run_time=2)
```

**Key Concepts**:
- Raw audio signal at 16 kHz sampling rate
- Each sample represents amplitude at that moment
- Magnitude of variation in real-time speech

---

### Scene 2: STFT (Short-Time Fourier Transform)
**Duration**: 2-3 min
**Purpose**: Transform time-domain signal to frequency-domain

```python
class STFTVisualization(Scene):
    """
    Shows the STFT process: slicing audio into windows,
    computing FFT on each, resulting in spectrogram
    """
    def construct(self):
        # Show waveform
        waveform = FunctionGraph(...)
        self.add(waveform)

        # Animate window sliding (25ms windows, 10ms hop)
        for i in range(num_windows):
            window = Rectangle(...)  # Show window box
            self.play(window.animate.shift(...), run_time=0.5)

            # Show FFT computation
            fft_result = BarChart(frequencies)
            self.play(Transform(window, fft_result), run_time=1)

            # Accumulate in spectrogram
            self.play(...)

        # Final: Full spectrogram image
        spec_img = ImageMobject("spectrogram.png")
        self.play(FadeOut(intermediates), FadeIn(spec_img))
```

**Visualizations**:
- Waveform with sliding 25ms window
- Window function (Hann window) multiplication
- FFT computation showing 512 frequency bins
- Building up the spectrogram frame-by-frame
- Timeline: T samples → T' frames (T' = (T - 400) / 160 + 1)

**Equations to Display**:
```
Window size: 25ms (400 samples @ 16kHz)
Hop length: 10ms (160 samples @ 16kHz)
FFT size: 512
X[m, k] = Σ_{n=0}^{N-1} x[n + mH] · w[n] · e^{-j2πkn/N}
```

---

### Scene 3: Mel Filter Bank
**Duration**: 1.5-2 min
**Purpose**: Convert to mel-scale (human auditory perception)

```python
class MelFilterBankScene(Scene):
    """
    Shows triangular filters applied in mel-scale,
    more filters at low frequencies (where humans are more sensitive)
    """
    def construct(self):
        # Show linear frequency scale vs mel scale
        lin_freqs = [...frequencies...]
        mel_freqs = [mel_conversion(f) for f in lin_freqs]

        # Linear scale
        axes_lin = Axes(x_min=0, x_max=8000)
        plot_lin = axes_lin.plot(lin_freqs)

        # Mel scale (nonlinear spacing)
        axes_mel = Axes()
        plot_mel = axes_mel.plot(mel_freqs)

        self.add(axes_lin, plot_lin)
        self.play(Transform(plot_lin, plot_mel))  # Show warping

        # Draw 80 triangular filters
        for i in range(80):
            triangle = Polygon(...)  # Triangular filter
            self.play(FadeIn(triangle), run_time=0.05)

        # Show filter bank matrix application
        spectrogram = ImageMobject("linear_spec.png")
        filters = Matrix([80 x 257])  # Filter bank matrix
        result = Matrix([80 x T'])     # Mel-spectrogram

        self.play(...)  # Matrix multiplication animation
```

**Visualizations**:
- Comparison: Linear frequency vs Mel scale
- 80 triangular overlapping filters
- Filter bank matrix (80 × 257)
- Before/after: Linear spectrogram → Mel-spectrogram
- Show more filters concentrated at low frequencies

**Equations**:
```
mel(f) = 2595 · log₁₀(1 + f/700)    [HTK formula]
f(mel) = 700 · (10^(mel/2595) - 1)

Each filter: H_m[k] = triangular response
            = 0 if k < left or k > right
            = (k - left)/(center - left) if left ≤ k ≤ center
            = (right - k)/(right - center) if center < k ≤ right
```

---

### Scene 4: Log Compression & Normalization
**Duration**: 1 min
**Purpose**: Final preprocessing step

```python
class LogCompressionScene(Scene):
    """
    Shows logarithmic compression of mel-spectrogram
    to match human hearing (logarithmic perception of loudness)
    """
    def construct(self):
        # Before: Raw mel-spectrogram (heatmap)
        mel_spec = ImageMobject("mel_spectrogram.png")
        self.add(mel_spec)

        # Apply log
        log_label = Tex("log(max(M, 10^{-10}))")
        self.add(log_label)

        # Show transformation (brighter areas compress)
        log_spec = ImageMobject("log_mel_spectrogram.png")
        self.play(Transform(mel_spec, log_spec), run_time=2)

        # Normalize: subtract global mean, divide by variance
        self.play(  # Show normalization effect
            mel_spec.animate.shift(DOWN * 2),
            run_time=1
        )
        norm_spec = ImageMobject("normalized_log_mel.png")
        self.play(FadeOut(mel_spec), FadeIn(norm_spec))
```

**Result**:
- Input shape: (T_frames, 80)
- Output shape: (T_frames, 80) with log-compressed, normalized values
- Ready for transformer encoder

---

## Video 2: Transformer Foundations (10-12 minutes)

### Scene 1: Convolutional Subsampling
**Duration**: 2 min
**Purpose**: Reduce temporal resolution before attention

```python
class ConvSubsamplingScene(Scene):
    """
    Shows 2× subsampling via Conv2D twice (4× total reduction)
    Mel-spectrogram (T, 80) → (T/4, d_model)
    """
    def construct(self):
        # Input mel-spectrogram as matrix
        spec_matrix = Matrix([[entry for entry in spec]])
        self.add(spec_matrix)

        label_in = Tex("(T, 80)")
        self.add(label_in)

        # Conv Layer 1: 2× subsampling
        conv1_label = Tex("Conv2D(kernel=3×3, stride=2×2)")
        self.play(FadeIn(conv1_label))

        # Show result: T/2, 80 → T/2, 256
        result1 = Matrix(height=spec_matrix.height//2)
        self.play(Transform(spec_matrix, result1), run_time=1.5)
        self.add(Tex("(T/2, 256)"))

        # Conv Layer 2: 2× subsampling again
        conv2_label = Tex("Conv2D(kernel=3×3, stride=2×2)")
        self.play(FadeIn(conv2_label), run_time=1)

        result2 = Matrix(height=result1.height//2)
        self.play(Transform(result1, result2), run_time=1.5)

        # Final: Linear projection to d_model (512)
        final_shape = Tex("(T/4, 512)")
        self.play(...)
```

**Key Points**:
- Each 2D convolution reduces spatial dims by 2×
- Double convolution = 4× temporal reduction
- Reduces computational cost for attention (O(T²) → O(T²/16))
- Maintains frequency information while reducing sequence length

---

### Scene 2: Rotary Position Embeddings (RoPE)
**Duration**: 3-4 min
**Purpose**: Encode position information via rotation

```python
class RotaryPositionEmbeddingScene(ThreeDScene):
    """
    Shows how RoPE works: rotating vectors in 2D subspaces
    to encode position information
    """
    def construct(self):
        # 1. Show sinusoidal position encoding (baseline)
        self.add_fixed_in_frame_mobjects(Tex("Traditional Sinusoidal"))

        pos_0 = np.array([1, 0, 0])  # Position 0 vector
        pos_1 = np.array([np.cos(θ₀), np.sin(θ₀), 0])  # Position 1

        sphere = Sphere(radius=1)
        self.add(sphere)

        # 2. Introduce RoPE concept
        self.add_fixed_in_frame_mobjects(Tex("RoPE: Rotation-based"))

        # 3. Show 2D rotation in detail
        self.set_camera_orientation(phi=0, theta=0)

        vec_q = Vector([1, 0, 0], color=BLUE)  # Query
        vec_k = Vector([1, 0, 0], color=RED)   # Key

        self.add(vec_q, vec_k)

        # Rotate by angle proportional to position
        angle = 2 * np.pi * 0 / d_model  # Position 0
        rotated_q = Vector(rotate_2d(vec_q, angle), color=BLUE)
        rotated_k = Vector(rotate_2d(vec_k, angle), color=RED)

        self.play(Transform(vec_q, rotated_q), Transform(vec_k, rotated_k))

        # 4. Show multiple dimensions
        self.play(...)  # Animate rotating different dimensional pairs

        # 5. Show the key property: relative position dependency
        # attention_score(i, j) depends only on (i - j)
        distance_label = Tex("Attention depends on relative distance only!")
        self.play(FadeIn(distance_label))
```

**Visualizations**:
- 2D vector rotation in subspace
- Color-coded dimension pairs
- Rotation angles: θᵢ = 10000^(-2i/d)
- Comparison: sinusoidal vs RoPE properties

**Equations**:
```
θᵢ = 10000^(-2i/d)    (frequency for dimension i)

R_θ,m = [cos(mθᵢ)  -sin(mθᵢ)]
        [sin(mθᵢ)   cos(mθᵢ)]

q_m' = R_θ,m · q_m
k_n' = R_θ,n · k_n

Key property: <q_m', k_n'> depends only on (m - n)
```

---

### Scene 3: Introduction to Attention
**Duration**: 2-3 min
**Purpose**: Explain scaled dot-product attention

```python
class AttentionMechanismScene(Scene):
    """
    Shows the attention formula step-by-step:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    def construct(self):
        # 1. Input sequence of vectors
        seq_len = 5
        vectors = [
            Vector(...) for _ in range(seq_len)
        ]

        # 2. Show Query, Key, Value projections
        q_matrix = Matrix([["q₁"], ["q₂"], ..., ["q₅"]])
        k_matrix = Matrix([["k₁"], ["k₂"], ..., ["k₅"]])
        v_matrix = Matrix([["v₁"], ["v₂"], ..., ["v₅"]])

        # 3. Compute scores: QK^T
        scores = Matrix([[...]])  # 5×5 matrix of scores

        # 4. Scale by √d_k
        scaled_scores = Matrix([[...]])  # Smaller values for stability

        # 5. Apply softmax
        weights = Matrix([[...]])  # Rows sum to 1

        # 6. Weight values
        output = weighted_sum(weights, v_matrix)

        # Animation flow:
        self.play(FadeIn(q_matrix, k_matrix, v_matrix))
        self.play(  # Compute and show scores
            ...
        )
        # ... continue for each step
```

**Key Visualizations**:
- Input sequence representation
- Q, K, V matrices
- Attention score matrix (heatmap showing which positions attend to which)
- Softmax transformation (darker colors = higher weight)
- Output as weighted combination of values

---

## Video 3: The Conformer Block (10-12 minutes)

### Scene 1: Conformer Block Overview
**Duration**: 1-2 min
**Purpose**: Show the overall structure

```python
class ConformerBlockOverviewScene(Scene):
    """
    Shows the Conformer block structure:
    FFN(1/2) → MHSA → Conv → FFN(1/2) → LN
    """
    def construct(self):
        # Define block components as rectangles
        ffn1 = Rectangle(width=2, height=1, fill_opacity=0.5)
        mhsa = Rectangle(width=2, height=1, fill_opacity=0.5)
        conv = Rectangle(width=2, height=1, fill_opacity=0.5)
        ffn2 = Rectangle(width=2, height=1, fill_opacity=0.5)
        ln = Rectangle(width=2, height=1, fill_opacity=0.5)

        # Stack vertically with arrows
        self.add(ffn1)
        arrow1 = Arrow(ffn1.get_bottom(), mhsa.get_top())
        self.add(arrow1)
        self.add(mhsa)

        # ... continue with other blocks

        # Add labels
        labels = [
            "FFN (½ dim)",
            "Multi-Head Self-Attention",
            "Convolution Module",
            "FFN (½ dim)",
            "Layer Normalization"
        ]

        # Show that it's repeated N times
        self.add(Tex("Repeat N = 12 times"))
```

---

### Scene 2: Multi-Head Self-Attention in Detail
**Duration**: 3-4 min
**Purpose**: Deep dive into MHSA

```python
class MultiHeadAttentionDetailScene(Scene):
    """
    Shows the flow: X → Q,K,V → Split heads → Attention → Concat → Output
    """
    def construct(self):
        # 1. Input X: (B, T, 512)
        input_x = Matrix(...)
        self.add(input_x)

        # 2. Linear projections: Q, K, V
        # Each projects to (B, T, 512)

        # 3. Reshape to multi-head format
        # (B, T, 512) → (B, 8, T, 64)
        # 8 heads, 64-dimensional

        # 4. Show 8 heads as separate columns
        heads = [
            Matrix(...) for _ in range(8)
        ]

        # 5. Apply attention on each head
        for i, head in enumerate(heads):
            self.play(
                head.animate.scale(1.1).set_color(BLUE),
                run_time=0.2
            )
            # Show attention computation
            attention_result = ...
            self.play(Transform(head, attention_result))

        # 6. Concatenate heads
        concat = Tex("Concatenate all heads")
        self.play(FadeIn(concat))

        # 7. Output projection
        w_o = Matrix([[...]])  # Output projection matrix
        output = w_o.multiply(concat_result)
```

---

### Scene 3: Convolution Module
**Duration**: 2-3 min
**Purpose**: Show depthwise convolution for local context

```python
class ConvolutionModuleScene(Scene):
    """
    Shows convolution applied along time dimension for local modeling
    """
    def construct(self):
        # Input: (B, T, d_model) = (1, T, 512)

        # Point-wise conv: expand dimensions
        self.play(Tex("Expand: (T, 512) → (T, 1024)"))

        # GLU activation
        self.play(Tex("Gate Linear Unit (GLU)"))

        # Depthwise convolution: kernel_size=31, groups=512
        # Each channel convolved independently
        conv_kernel = Rectangle(width=0.5, height=1)
        self.play(...)  # Show kernel sliding over sequence

        # Point-wise conv: project back
        self.play(Tex("Project: (T, 1024) → (T, 512)"))

        # Show the local receptive field
        receptive_field = Tex("Receptive field: 31 time steps")
        self.add(receptive_field)
```

---

### Scene 4: Feed-Forward Network
**Duration**: 1-2 min
**Purpose**: Show FFN with SwiGLU activation

```python
class FFNScene(Scene):
    """
    Dense → SwiGLU → Dense
    (T, 512) → (T, 2048) → (T, 512)
    """
    def construct(self):
        # Linear projection: expand
        self.play(Tex("Linear: (512) → (2048)"))

        # SwiGLU activation
        self.play(Tex("SwiGLU Activation"))
        self.play(Tex("x * swish(gate) where gate is computed"))

        # Linear projection: project back
        self.play(Tex("Linear: (2048) → (512)"))

        # Show the non-linearity
        swiglu_plot = self.get_graph(
            lambda x: x * (x / (1 + np.exp(-x))),
            x_range=[-5, 5]
        )
        self.play(FadeIn(swiglu_plot))
```

---

## Video 4: Training with CTC Loss (8-10 minutes)

### Scene 1: The CTC Problem
**Duration**: 1.5 min
**Purpose**: Explain why CTC is needed

```python
class CTCProblemScene(Scene):
    """
    Shows the challenge: sequence-to-sequence with unknown alignment
    """
    def construct(self):
        # Audio: "Hello"
        waveform = FunctionGraph(...)
        self.add(waveform)
        self.play(...)

        # Encoder output: (T_audio/4) = 200 frames
        # Decoder output: must produce text tokens

        # Problem: which frames correspond to which letters?
        letters = ["H", "e", "l", "l", "o"]

        # Show misalignment possibilities
        self.play(Tex("Many possible alignments!"))

        # CTC solution: marginalize over all alignments
        self.play(Tex("CTC sums over all possible alignments"))
```

---

### Scene 2: CTC Algorithm Step-by-Step
**Duration**: 3-4 min
**Purpose**: Visualize forward-backward algorithm

```python
class CTCForwardBackwardScene(Scene):
    """
    Shows the dynamic programming solution

    Path types:
    1. Direct transitions: label_i → label_i+1
    2. Stays: label_i → label_i (blank inserted between)
    """
    def construct(self):
        # Build trellis
        n_frames = 10
        n_labels = 5

        # Create grid
        positions = [
            (i, j) for i in range(n_frames) for j in range(n_labels)
        ]
        nodes = [Circle(radius=0.3).move_to(pos) for pos in positions]

        self.play(FadeIn(*nodes))

        # Forward algorithm
        self.play(Tex("Forward Pass: α[t][j] = ..."))

        # Animate forward pass
        for t in range(n_frames):
            for j in range(n_labels):
                # Compute α[t][j] from previous states
                prev_states = [α[t-1][j-1], α[t-1][j], α[t-1][j+1]]

                # Highlight incoming edges
                for prev in prev_states:
                    edge = Line(prev.get_center(), nodes[t][j].get_center())
                    self.play(FadeIn(edge), run_time=0.1)

                # Update node
                self.play(nodes[t][j].animate.set_color(BLUE), run_time=0.1)

        # Backward algorithm
        self.play(Tex("Backward Pass: β[t][j] = ..."))
        # Similar animation going backwards

        # Loss computation
        self.play(Tex("Loss = -log(α[T][end] + α[T][end-1])"))
```

**Key Concepts**:
- Trellis structure
- Forward pass (left to right)
- Backward pass (right to left)
- Path probability computation
- Gradient flow for training

---

### Scene 3: CTC Decoding
**Duration**: 1.5-2 min
**Purpose**: Show inference process

```python
class CTCDecodingScene(Scene):
    """
    Greedy decoding: pick highest probability token at each frame
    """
    def construct(self):
        # Logits from model: (T, vocab_size)
        logits = Matrix([[...]])

        # Find argmax at each timestep
        self.play(Tex("Greedy Decoding: argmax_k logits[t, k]"))

        # Remove blanks
        self.play(Tex("Remove <blank> tokens"))

        # Merge consecutive duplicates
        self.play(Tex("Merge consecutive duplicates"))

        # Result
        self.play(Tex("Output: 'HELLO'"))
```

---

## Video 5: Full Pipeline Integration (8-10 minutes)

### Scene 1: End-to-End Architecture
**Duration**: 3-4 min
**Purpose**: Show complete flow

```python
class EndToEndArchitectureScene(Scene):
    """
    Full pipeline visualization
    """
    def construct(self):
        # 1. Raw audio waveform (left)
        audio = FunctionGraph(...)

        # 2. Audio Frontend
        frontend = Rectangle(fill_opacity=0.3)
        self.add(frontend, Tex("Audio Frontend"))

        # 3. Encoder
        encoder = Rectangle(fill_opacity=0.3)
        self.add(encoder, Tex("Encoder Stack (12 blocks)"))

        # 4. CTC Head (auxiliary)
        ctc = Rectangle(fill_opacity=0.3)
        self.add(ctc, Tex("CTC Head"))

        # 5. Decoder
        decoder = Rectangle(fill_opacity=0.3)
        self.add(decoder, Tex("Decoder (6 blocks)"))

        # 6. Output logits → Text
        output = Tex("Output: 'HELLO WORLD'")

        # Animate data flow
        self.play(
            audio.animate.shift(UP * 5),
            FadeIn(frontend),
            run_time=1
        )

        # ... continue flow
```

---

### Scene 2: Model Comparison
**Duration**: 2-3 min
**Purpose**: Show different configurations

```python
class ModelConfigurationScene(Scene):
    """
    Compare Tiny, Base, Large configurations
    """
    def construct(self):
        configs = [
            ("Tiny", 256, 4, 6, 4, "~15M"),
            ("Base", 512, 8, 12, 6, "~80M"),
            ("Large", 768, 12, 18, 8, "~200M"),
        ]

        # Create table
        table = Table(
            [["Config", "d_model", "n_heads", "n_enc", "n_dec", "Params"],
             *configs],
            include_outer_lines=True
        )

        self.play(FadeIn(table))

        # Highlight Base (our target)
        base_row = table.get_rows()[2]  # Base config
        self.play(base_row.animate.set_color(BLUE))

        # Show speed/accuracy tradeoff
        self.play(Tex("Speed vs Accuracy Tradeoff"))
```

---

### Scene 3: Training Loop Visualization
**Duration**: 2-3 min
**Purpose**: Show iterative training process

```python
class TrainingLoopScene(Scene):
    """
    Shows training iterations: forward → loss → backward → update
    """
    def construct(self):
        # Initialize loss tracking
        loss_values = []

        for epoch in range(5):
            # Forward pass
            self.play(Tex(f"Epoch {epoch+1}: Forward Pass"))

            # Compute loss
            loss = self.compute_loss()
            loss_values.append(loss)

            # Backward pass
            self.play(Tex("Backward Pass (Gradients)"))

            # Update weights
            self.play(Tex("Update Weights"))

            # Plot loss
            self.plot_loss(loss_values)
```

---

### Scene 4: Key Takeaways
**Duration**: 1-2 min
**Purpose**: Summary and conclusions

```python
class SummaryScene(Scene):
    """
    Key insights from VoxFormer
    """
    def construct(self):
        takeaways = [
            "✓ Conformer blocks combine attention and convolution",
            "✓ RoPE enables efficient relative position encoding",
            "✓ CTC loss handles alignment automatically",
            "✓ 4× temporal reduction via subsampling",
            "✓ 80M parameters, real-time inference"
        ]

        for point in takeaways:
            bullet = Tex(point)
            self.play(FadeIn(bullet))
            self.wait(1)
```

---

## Production Implementation Roadmap

### Phase 1: Setup & Configuration (Week 1)
- [ ] Install Manim, FFmpeg, LaTeX
- [ ] Create `/manim-videos/` directory structure
- [ ] Set up custom config (`custom_config.yml`)
- [ ] Create custom Manim objects for DSP components

**Custom Objects to Create**:
```python
# custom/dsp_components.py
class STFTVisualization(VMobject):
    """Animated STFT window sliding"""

class MelFilterBankVisualization(VMobject):
    """80 triangular filters on frequency axis"""

class SpectrogramHeatmap(VMobject):
    """Animated spectrogram display"""

class TransformerBlock(VMobject):
    """Visual representation of Conformer block"""

class AttentionHeatmap(VMobject):
    """Show attention weights over sequence"""

class CTCTrellis(VMobject):
    """Dynamic programming trellis for CTC"""
```

### Phase 2: Video 1 - Audio Pipeline (Weeks 2-3)
- [ ] Scene: Waveform basics
- [ ] Scene: STFT visualization
- [ ] Scene: Mel filterbank
- [ ] Scene: Log compression
- [ ] Polish and render

**Renders**:
- 4K @ 30fps: ~5-10 min per scene
- Total: ~60 min render time for Part 1

### Phase 3: Video 2 - Foundations (Weeks 4-5)
- [ ] Scene: Convolutional subsampling
- [ ] Scene: RoPE (3D visualization)
- [ ] Scene: Attention mechanism
- [ ] Polish and combine

### Phase 4: Video 3 - Conformer (Weeks 6-7)
- [ ] Scene: Conformer block overview
- [ ] Scene: MHSA in detail
- [ ] Scene: Convolution module
- [ ] Scene: FFN with SwiGLU

### Phase 5: Video 4 - CTC Loss (Week 8)
- [ ] Scene: CTC problem
- [ ] Scene: Forward-backward algorithm
- [ ] Scene: Decoding

### Phase 6: Video 5 - Integration (Week 9)
- [ ] Scene: End-to-end flow
- [ ] Scene: Model comparison
- [ ] Scene: Training loop
- [ ] Scene: Summary

### Phase 7: Post-Production (Week 10)
- [ ] Add music/sound design
- [ ] Synchronize with narration
- [ ] Color correction
- [ ] Final quality check

---

## Technical Specifications

### Rendering Configuration
```yaml
# custom_config.yml
resolution: "4K"  # 3840 × 2160
frame_rate: 30
pixel_height: 2160
pixel_width: 3840

# For preview during development
quality: "low_quality"  # Lower resolution for iteration
```

### File Structure
```
/manim-videos/
├── scenes/
│   ├── 01_audio_pipeline.py
│   ├── 02_foundations.py
│   ├── 03_conformer.py
│   ├── 04_ctc_loss.py
│   └── 05_integration.py
├── custom/
│   ├── dsp_components.py
│   ├── transformer_components.py
│   └── ctc_components.py
├── assets/
│   ├── audio_samples/
│   ├── spectrograms/
│   └── reference_images/
├── output/
│   └── (generated videos)
├── manim_imports_ext.py
└── custom_config.yml
```

---

## Narration & Timing Strategy

### Pacing Guidelines
1. **Visual Introduction** (20-30% of scene)
   - Show the concept visually without heavy text

2. **Narration + Animation Sync** (50-60%)
   - Narrate as animated equations appear

3. **Mathematical Details** (10-15%)
   - Show formulas with visual examples

4. **Recap/Transition** (5-10%)
   - Brief summary before next scene

### Example Narration
```
[Visual: waveform oscillating]
"Sound is created by vibrations in air - vibrations that
we can represent as a mathematical wave. Here, we're
sampling this wave at 16,000 times per second, the
standard for speech recognition."

[Visual: STFT window sliding]
"To analyze the frequency content, we break the signal
into small overlapping windows - 25 milliseconds each -
and compute the Fourier Transform of each window."

[Visual: Spectrogram building up]
"The result is a spectrogram - a 2D image where time
flows left to right, and the vertical axis shows
frequency content. Darker colors mean more energy at
that frequency-time combination."
```

---

## Quality Assurance Checklist

### Per-Scene Requirements
- [ ] Animations smooth and don't jump
- [ ] Equations displayed correctly with proper LaTeX formatting
- [ ] All colors match brand palette (emerald/cyan)
- [ ] Text size consistent and readable
- [ ] No rendering artifacts
- [ ] Timing matches narration (±0.5 seconds)

### Per-Video Requirements
- [ ] All scenes transition smoothly
- [ ] Consistent visual style across scenes
- [ ] Mathematical accuracy verified against documentation
- [ ] Timing: ±1 minute from target duration
- [ ] Color grading uniform

---

## Estimated Time & Resources

### Development Time
- Scene Development: ~2-3 hours per scene × 20 scenes = 40-60 hours
- Custom Object Creation: ~20 hours
- Integration & Testing: ~15 hours
- **Total: ~75-95 hours of development**

### Rendering Time (4K @ 30fps)
- Average: 5-10 minutes per scene
- Total: ~100-200 minutes = 2-3 hours
- Can be parallelized across machines

### Total Production Time
- **Assuming parallel rendering: 10-12 weeks**
- **Development can start immediately**
- **Rendering parallelization: ~8 hours continuous rendering time**

---

## Integration with Existing Project

### Next Steps
1. **Add to repository**:
   ```bash
   git clone https://github.com/3b1b/manim.git /path/to/project/manim
   mkdir -p /path/to/project/manim-videos
   ```

2. **Create initial scene**:
   - Start with simplest scene (AudioWaveformScene)
   - Test rendering pipeline
   - Establish custom style

3. **Link to presentations**:
   - Add `/manim` route in Next.js linking to video playlist
   - Create `/manim/<video-id>` pages showing embedded videos
   - Add "See Animated Explanation" links from `/technical` slides

4. **Documentation**:
   - Update CLAUDE.md with Manim section
   - Add developer guide for adding new scenes
   - Document custom Manim components

---

## Success Criteria

✓ Production-quality 4K videos at 30fps
✓ Accurate representation of technical concepts
✓ Professional visual style matching brand
✓ 5-part series covering complete architecture
✓ ~50-60 minutes total content
✓ Synchronized with narration
✓ Suitable for educational platforms (YouTube, etc.)

