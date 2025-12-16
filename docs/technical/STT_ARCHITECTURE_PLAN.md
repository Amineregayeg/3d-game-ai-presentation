# VoxFormer: Custom Speech-to-Text Transformer Architecture
## Technical Planning Document v1.0

---

## Executive Summary

This document outlines the architecture and implementation plan for **VoxFormer**, a custom-built Speech-to-Text transformer designed from the ground up using low-level deep learning primitives. The architecture combines state-of-the-art techniques from Conformer, Whisper, and modern LLM research to achieve high-accuracy, real-time speech recognition optimized for game development voice commands.

---

## 1. Architecture Overview

### 1.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VoxFormer Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────┐    ┌─────────────────────┐    │
│  │  Raw Audio  │───▶│  Audio Frontend      │───▶│  Feature Projection │    │
│  │  Waveform   │    │  (Mel-Spectrogram)   │    │  (Conv Subsampling) │    │
│  └─────────────┘    └──────────────────────┘    └──────────┬──────────┘    │
│                                                             │               │
│                     ┌───────────────────────────────────────▼───────┐      │
│                     │           CONFORMER ENCODER STACK             │      │
│                     │  ┌─────────────────────────────────────────┐  │      │
│                     │  │  ConformerBlock × N_enc (6-12 layers)   │  │      │
│                     │  │  ┌─────────────────────────────────┐    │  │      │
│                     │  │  │ 1. Feed-Forward Module (½)      │    │  │      │
│                     │  │  │ 2. Multi-Head Self-Attention    │    │  │      │
│                     │  │  │ 3. Convolution Module           │    │  │      │
│                     │  │  │ 4. Feed-Forward Module (½)      │    │  │      │
│                     │  │  │ 5. Layer Normalization          │    │  │      │
│                     │  │  └─────────────────────────────────┘    │  │      │
│                     │  └─────────────────────────────────────────┘  │      │
│                     └───────────────────────────────────────────────┘      │
│                                          │                                  │
│                    ┌─────────────────────┴─────────────────────┐           │
│                    │                                           │           │
│                    ▼                                           ▼           │
│         ┌─────────────────┐                      ┌─────────────────────┐   │
│         │   CTC Head      │                      │  TRANSFORMER DECODER│   │
│         │  (Auxiliary)    │                      │  ┌───────────────┐  │   │
│         └─────────────────┘                      │  │ DecoderBlock  │  │   │
│                                                  │  │ × N_dec (4-6) │  │   │
│                                                  │  └───────────────┘  │   │
│                                                  └──────────┬──────────┘   │
│                                                             │              │
│                                                  ┌──────────▼──────────┐   │
│                                                  │   Token Prediction  │   │
│                                                  │   (Output Logits)   │   │
│                                                  └─────────────────────┘   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Model Configurations

| Configuration | d_model | n_heads | n_enc | n_dec | FFN_dim | Conv_kernel | Params |
|--------------|---------|---------|-------|-------|---------|-------------|--------|
| VoxFormer-Tiny | 256 | 4 | 6 | 4 | 1024 | 15 | ~15M |
| VoxFormer-Base | 512 | 8 | 12 | 6 | 2048 | 31 | ~80M |
| VoxFormer-Large | 768 | 12 | 18 | 8 | 3072 | 31 | ~200M |

**Target Configuration**: VoxFormer-Base for optimal accuracy/latency tradeoff.

---

## 2. Low-Level Component Specifications

### 2.1 Audio Frontend Module

The audio frontend converts raw waveform to mel-spectrogram features with learnable parameters.

```
┌─────────────────────────────────────────────────────────────┐
│                    AUDIO FRONTEND                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Raw waveform x ∈ ℝ^T (16kHz sample rate)           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. STFT (Short-Time Fourier Transform)             │   │
│  │     - Window size: 25ms (400 samples)               │   │
│  │     - Hop length: 10ms (160 samples)                │   │
│  │     - FFT size: 512                                 │   │
│  │     - Window: Hann                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Mel Filter Bank                                 │   │
│  │     - n_mels: 80                                    │   │
│  │     - f_min: 0 Hz                                   │   │
│  │     - f_max: 8000 Hz                                │   │
│  │     - Mel scale: HTK                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. Log Compression + Normalization                 │   │
│  │     - log(max(mel, 1e-10))                         │   │
│  │     - Global mean/variance normalization            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: Mel-spectrogram M ∈ ℝ^(T' × 80)                   │
│  where T' = ⌊(T - 400) / 160⌋ + 1                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.1.1 Implementation: Custom STFT

```python
# Pseudo-code for low-level STFT implementation
class CustomSTFT:
    """
    Low-level STFT implementation without using torch.stft

    Mathematical Foundation:
    X[m, k] = Σ_{n=0}^{N-1} x[n + mH] · w[n] · e^{-j2πkn/N}

    Where:
    - m: frame index
    - k: frequency bin index
    - H: hop length
    - N: FFT size
    - w[n]: window function
    """

    def __init__(self, n_fft=512, hop_length=160, win_length=400):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Pre-compute Hann window
        # w[n] = 0.5 * (1 - cos(2πn / (N-1)))
        self.window = self._hann_window(win_length)

        # Pre-compute DFT matrix for efficiency
        # W_N[k,n] = e^{-j2πkn/N}
        self.dft_matrix = self._compute_dft_matrix(n_fft)

    def _hann_window(self, length):
        n = torch.arange(length, dtype=torch.float32)
        return 0.5 * (1 - torch.cos(2 * math.pi * n / (length - 1)))

    def _compute_dft_matrix(self, n):
        # Real and imaginary components separately for numerical stability
        k = torch.arange(n).unsqueeze(1)
        n_idx = torch.arange(n).unsqueeze(0)
        angles = -2 * math.pi * k * n_idx / n
        dft_real = torch.cos(angles)
        dft_imag = torch.sin(angles)
        return dft_real, dft_imag

    def forward(self, waveform):
        # Frame the signal
        frames = self._frame_signal(waveform)  # (batch, n_frames, win_length)

        # Apply window
        windowed = frames * self.window

        # Zero-pad to n_fft
        padded = F.pad(windowed, (0, self.n_fft - self.win_length))

        # Apply DFT (manual matrix multiplication)
        real_part = torch.matmul(padded, self.dft_matrix[0].T)
        imag_part = torch.matmul(padded, self.dft_matrix[1].T)

        # Compute magnitude spectrum (only positive frequencies)
        n_freqs = self.n_fft // 2 + 1
        magnitude = torch.sqrt(real_part[..., :n_freqs]**2 + imag_part[..., :n_freqs]**2)

        return magnitude  # (batch, n_frames, n_freqs)
```

#### 2.1.2 Implementation: Mel Filter Bank

```python
class MelFilterBank:
    """
    Low-level Mel filter bank implementation

    Mel Scale Conversion:
    mel(f) = 2595 * log10(1 + f/700)  [HTK formula]
    f(mel) = 700 * (10^(mel/2595) - 1)

    Filter Construction:
    - Triangular filters evenly spaced in mel scale
    - Each filter: H_m[k] = triangular response centered at mel frequency m
    """

    def __init__(self, n_mels=80, n_fft=512, sample_rate=16000, f_min=0, f_max=8000):
        self.n_mels = n_mels
        self.n_freqs = n_fft // 2 + 1

        # Create mel filter bank matrix
        self.filter_bank = self._create_mel_filters(
            n_mels, n_fft, sample_rate, f_min, f_max
        )

    def _hz_to_mel(self, hz):
        return 2595 * torch.log10(1 + hz / 700)

    def _mel_to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def _create_mel_filters(self, n_mels, n_fft, sr, f_min, f_max):
        # Convert frequency bounds to mel scale
        mel_min = self._hz_to_mel(torch.tensor(f_min))
        mel_max = self._hz_to_mel(torch.tensor(f_max))

        # Create n_mels + 2 points evenly spaced in mel scale
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        # Convert to FFT bin indices
        bin_points = torch.floor((n_fft + 1) * hz_points / sr).long()

        # Create triangular filters
        n_freqs = n_fft // 2 + 1
        filter_bank = torch.zeros(n_mels, n_freqs)

        for m in range(n_mels):
            f_left = bin_points[m]
            f_center = bin_points[m + 1]
            f_right = bin_points[m + 2]

            # Rising slope
            for k in range(f_left, f_center):
                filter_bank[m, k] = (k - f_left) / (f_center - f_left)

            # Falling slope
            for k in range(f_center, f_right):
                filter_bank[m, k] = (f_right - k) / (f_right - f_center)

        return filter_bank

    def forward(self, spectrogram):
        # spectrogram: (batch, n_frames, n_freqs)
        # Apply mel filters via matrix multiplication
        mel_spec = torch.matmul(spectrogram, self.filter_bank.T)
        return mel_spec  # (batch, n_frames, n_mels)
```

### 2.2 Convolutional Subsampling Module

Reduces temporal resolution while projecting to model dimension.

```
┌─────────────────────────────────────────────────────────────┐
│              CONVOLUTIONAL SUBSAMPLING                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Mel-spectrogram M ∈ ℝ^(T × 80)                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Conv2D Layer 1                                     │   │
│  │  - in_channels: 1                                   │   │
│  │  - out_channels: d_model // 2                       │   │
│  │  - kernel: (3, 3), stride: (2, 2), padding: (1, 1)  │   │
│  │  - Activation: SiLU (Swish)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Conv2D Layer 2                                     │   │
│  │  - in_channels: d_model // 2                        │   │
│  │  - out_channels: d_model                            │   │
│  │  - kernel: (3, 3), stride: (2, 2), padding: (1, 1)  │   │
│  │  - Activation: SiLU (Swish)                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Reshape + Linear Projection                        │   │
│  │  - Flatten frequency dimension                      │   │
│  │  - Project to d_model                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: X ∈ ℝ^(T/4 × d_model)                             │
│  Temporal reduction factor: 4× (each conv: 2×)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Positional Encoding: Rotary Position Embeddings (RoPE)

We use RoPE instead of sinusoidal embeddings for better relative position modeling.

```
┌─────────────────────────────────────────────────────────────┐
│              ROTARY POSITION EMBEDDINGS (RoPE)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Mathematical Foundation:                                   │
│                                                             │
│  For position m and dimension pair (2i, 2i+1):             │
│                                                             │
│  R_θ,m = [cos(mθ_i)  -sin(mθ_i)]                          │
│          [sin(mθ_i)   cos(mθ_i)]                          │
│                                                             │
│  where θ_i = 10000^(-2i/d)                                 │
│                                                             │
│  Application to Query/Key:                                  │
│  q_m' = R_θ,m · q_m                                        │
│  k_n' = R_θ,n · k_n                                        │
│                                                             │
│  Property: <q_m', k_n'> depends only on (m - n)            │
│  This enables relative position awareness                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class RotaryPositionEmbedding:
    """
    Low-level implementation of Rotary Position Embeddings

    Key insight: RoPE encodes position by rotating vector pairs
    in 2D subspaces, making attention naturally position-aware.
    """

    def __init__(self, dim, max_seq_len=8192, base=10000):
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies: θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute cos/sin for all positions
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len):
        # Position indices
        t = torch.arange(seq_len, dtype=torch.float32)

        # Outer product: position × frequency
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)

        # Duplicate for paired dimensions
        freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)

        self.cos_cached = freqs.cos()
        self.sin_cached = freqs.sin()

    def _rotate_half(self, x):
        """Rotate half the hidden dims of x"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, seq_len):
        """
        Apply rotary embeddings to queries and keys

        Args:
            q: queries (batch, n_heads, seq_len, head_dim)
            k: keys (batch, n_heads, seq_len, head_dim)
            seq_len: sequence length
        """
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Apply rotation: x' = x * cos + rotate_half(x) * sin
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed
```

### 2.4 Multi-Head Self-Attention (Custom Implementation)

```
┌─────────────────────────────────────────────────────────────┐
│           MULTI-HEAD SELF-ATTENTION (CUSTOM)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: X ∈ ℝ^(B × T × d_model)                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Linear Projections (no bias for efficiency)        │   │
│  │  Q = X · W_Q    W_Q ∈ ℝ^(d_model × d_model)        │   │
│  │  K = X · W_K    W_K ∈ ℝ^(d_model × d_model)        │   │
│  │  V = X · W_V    W_V ∈ ℝ^(d_model × d_model)        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Reshape to Multi-Head Format                       │   │
│  │  Q, K, V ∈ ℝ^(B × n_heads × T × d_head)            │   │
│  │  where d_head = d_model / n_heads                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Apply RoPE to Q and K                              │   │
│  │  Q', K' = RoPE(Q, K, positions)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Scaled Dot-Product Attention                       │   │
│  │                                                     │   │
│  │  scores = (Q' · K'^T) / √d_head                    │   │
│  │  weights = softmax(scores + mask)                   │   │
│  │  output = weights · V                               │   │
│  │                                                     │   │
│  │  (Use Flash Attention kernel for O(T) memory)       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Concatenate Heads + Output Projection              │   │
│  │  output = Concat(head_1, ..., head_h) · W_O        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: Y ∈ ℝ^(B × T × d_model)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class MultiHeadSelfAttention:
    """
    Custom Multi-Head Self-Attention Implementation

    Features:
    - Fused QKV projection for efficiency
    - RoPE integration
    - Flash Attention compatible tiling
    - Optional Grouped Query Attention (GQA)
    """

    def __init__(self, d_model, n_heads, dropout=0.1, use_gqa=False, n_kv_heads=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Grouped Query Attention: fewer K/V heads
        self.n_kv_heads = n_kv_heads if use_gqa else n_heads
        self.n_rep = n_heads // self.n_kv_heads

        # Fused QKV projection
        self.qkv_dim = (n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.W_qkv = nn.Linear(d_model, self.qkv_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # Fused QKV projection
        qkv = self.W_qkv(x)

        # Split into Q, K, V
        q, k, v = self._split_qkv(qkv, B, T)

        # Apply RoPE
        q, k = self.rope(q, k, T)

        # Repeat K/V for GQA if needed
        if self.n_rep > 1:
            k = self._repeat_kv(k, self.n_rep)
            v = self._repeat_kv(v, self.n_rep)

        # Scaled dot-product attention (can be replaced with Flash Attention)
        attn_output = self._scaled_dot_product_attention(q, k, v, mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.W_o(attn_output)

        return output

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Manual implementation of scaled dot-product attention

        Attention(Q, K, V) = softmax(QK^T / √d_k) V
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask (for causal attention in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax with numerical stability
        scores_max = scores.max(dim=-1, keepdim=True).values
        scores = scores - scores_max
        attn_weights = torch.exp(scores)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        output = torch.matmul(attn_weights, v)

        return output
```

### 2.5 Conformer Convolution Module

The key innovation of Conformer: combining convolution with attention for local + global context.

```
┌─────────────────────────────────────────────────────────────┐
│              CONFORMER CONVOLUTION MODULE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: X ∈ ℝ^(B × T × d_model)                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Layer Normalization                             │   │
│  │     X_norm = LayerNorm(X)                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Pointwise Conv (expand) + GLU                   │   │
│  │     - Conv1D: d_model → 2 * d_model                 │   │
│  │     - GLU activation: 2 * d_model → d_model         │   │
│  │       GLU(x) = x_1 ⊙ σ(x_2)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. Depthwise Convolution                           │   │
│  │     - Conv1D: groups = d_model                      │   │
│  │     - kernel_size: 31 (15 left + 1 center + 15 right)│   │
│  │     - Causal padding for streaming (optional)       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  4. Batch Normalization + SiLU Activation           │   │
│  │     X = SiLU(BatchNorm(X))                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  5. Pointwise Conv (project back)                   │   │
│  │     - Conv1D: d_model → d_model                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  6. Dropout + Residual                              │   │
│  │     Output = Dropout(Conv_out) + X                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class ConformerConvModule:
    """
    Conformer Convolution Module - captures local acoustic patterns

    Key Design Choices:
    - Depthwise separable convolution for efficiency
    - GLU gating for information flow control
    - Large kernel (31) for wide temporal receptive field
    """

    def __init__(self, d_model, kernel_size=31, dropout=0.1, causal=False):
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.causal = causal

        # Layer norm
        self.layer_norm = RMSNorm(d_model)

        # Pointwise expansion with GLU
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

        # Depthwise convolution
        # Padding: (kernel_size - 1) // 2 for same output length
        if causal:
            self.padding = (kernel_size - 1, 0)  # Left padding only
        else:
            self.padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            groups=d_model,  # Depthwise: each channel separately
            padding=self.padding if not causal else 0
        )

        # Batch norm + activation
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Pointwise projection
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x

        # Layer norm
        x = self.layer_norm(x)

        # Transpose for Conv1d: (B, T, C) → (B, C, T)
        x = x.transpose(1, 2)

        # Pointwise + GLU
        x = self.pointwise_conv1(x)
        x = self._glu(x)

        # Causal padding if needed
        if self.causal:
            x = F.pad(x, self.padding)

        # Depthwise conv
        x = self.depthwise_conv(x)

        # BatchNorm + SiLU
        x = self.batch_norm(x)
        x = F.silu(x)

        # Pointwise projection
        x = self.pointwise_conv2(x)

        # Transpose back: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)

        # Dropout + residual
        x = self.dropout(x) + residual

        return x

    def _glu(self, x):
        """Gated Linear Unit: GLU(x) = x_1 ⊙ σ(x_2)"""
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)
```

### 2.6 Feed-Forward Module with SwiGLU

Modern FFN using SwiGLU activation (superior to ReLU/GELU).

```
┌─────────────────────────────────────────────────────────────┐
│              FEED-FORWARD MODULE (SwiGLU)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Standard FFN:                                              │
│  FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2                  │
│                                                             │
│  SwiGLU FFN (superior):                                     │
│  FFN(x) = W_2 · (SiLU(W_gate · x) ⊙ (W_up · x))           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. RMS Normalization                               │   │
│  │     x_norm = RMSNorm(x)                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Gate and Up Projections (parallel)              │   │
│  │     gate = W_gate · x_norm    (d → d_ff)            │   │
│  │     up = W_up · x_norm        (d → d_ff)            │   │
│  │                                                     │   │
│  │  3. SwiGLU Activation                               │   │
│  │     hidden = SiLU(gate) ⊙ up                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  4. Down Projection                                 │   │
│  │     output = W_down · hidden   (d_ff → d)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  5. Dropout + Residual                              │   │
│  │     final = Dropout(output) + x                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Note: d_ff = d_model * expansion_factor (typically 4×)     │
│  For SwiGLU: use 8/3 × d_model to match parameter count    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class SwiGLUFeedForward:
    """
    Feed-Forward Network with SwiGLU Activation

    SwiGLU: SiLU(Wx) ⊙ (Vx), where ⊙ is element-wise product

    Benefits over standard FFN:
    - Better gradient flow
    - Improved training stability
    - Superior performance in language modeling tasks
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1, bias=False):
        # Default: 8/3 × d_model to match standard 4× expansion parameter count
        d_ff = d_ff or int(8 * d_model / 3)
        # Round to multiple of 256 for efficiency
        d_ff = ((d_ff + 255) // 256) * 256

        self.d_model = d_model
        self.d_ff = d_ff

        # Fused gate + up projection
        self.W_gate_up = nn.Linear(d_model, 2 * d_ff, bias=bias)
        self.W_down = nn.Linear(d_ff, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Fused gate and up projection
        gate_up = self.W_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # SwiGLU
        hidden = F.silu(gate) * up

        # Down projection
        output = self.W_down(hidden)
        output = self.dropout(output)

        return output + residual
```

### 2.7 RMS Normalization

```python
class RMSNorm:
    """
    Root Mean Square Layer Normalization

    Simpler and more efficient than LayerNorm:
    - No mean subtraction (no centering)
    - Only scale, no shift

    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = √(mean(x²))
    """

    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_norm = x / rms

        return x_norm * self.weight
```

### 2.8 Complete Conformer Encoder Block

```
┌─────────────────────────────────────────────────────────────┐
│              CONFORMER ENCODER BLOCK                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Conformer Block = FFN(½) → MHSA → Conv → FFN(½) → LN      │
│                                                             │
│  Input: X ∈ ℝ^(B × T × d_model)                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. First Half Feed-Forward                         │   │
│  │     X = X + 0.5 * FFN(X)                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Multi-Head Self-Attention                       │   │
│  │     X = X + MHSA(X)                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. Convolution Module                              │   │
│  │     X = X + Conv(X)                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  4. Second Half Feed-Forward                        │   │
│  │     X = X + 0.5 * FFN(X)                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  5. Final Layer Norm                                │   │
│  │     X = LayerNorm(X)                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: X ∈ ℝ^(B × T × d_model)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class ConformerEncoderBlock:
    """
    Complete Conformer Encoder Block

    The Conformer architecture achieves SOTA on speech recognition by:
    1. Self-attention captures global context
    2. Convolution captures local acoustic patterns
    3. Macaron-style FFN (two half-strength FFN) improves gradient flow
    """

    def __init__(self, d_model, n_heads, d_ff, conv_kernel_size=31, dropout=0.1):
        # First half FFN
        self.ffn1 = SwiGLUFeedForward(d_model, d_ff, dropout)

        # Multi-head self-attention
        self.mhsa = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.mhsa_norm = RMSNorm(d_model)

        # Convolution module
        self.conv = ConformerConvModule(d_model, conv_kernel_size, dropout)

        # Second half FFN
        self.ffn2 = SwiGLUFeedForward(d_model, d_ff, dropout)

        # Final layer norm
        self.final_norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # First half FFN (Macaron-style: 0.5× residual weight)
        x = x + 0.5 * self.ffn1(x)

        # Multi-head self-attention with pre-norm
        x_norm = self.mhsa_norm(x)
        x = x + self.dropout(self.mhsa(x_norm, mask))

        # Convolution module
        x = self.conv(x)

        # Second half FFN
        x = x + 0.5 * self.ffn2(x)

        # Final normalization
        x = self.final_norm(x)

        return x
```

### 2.9 Transformer Decoder Block

```
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER DECODER BLOCK                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Y ∈ ℝ^(B × L × d_model) (target sequence)          │
│         enc_out ∈ ℝ^(B × T × d_model) (encoder output)     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Masked Self-Attention (Causal)                  │   │
│  │     Y = Y + MaskedMHSA(Y)                           │   │
│  │     (prevents attending to future tokens)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Cross-Attention                                 │   │
│  │     Y = Y + CrossAttention(Y, enc_out)              │   │
│  │     Q from decoder, K/V from encoder                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. Feed-Forward                                    │   │
│  │     Y = Y + FFN(Y)                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: Y ∈ ℝ^(B × L × d_model)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. CTC Loss Implementation

Connectionist Temporal Classification for auxiliary loss during training.

```
┌─────────────────────────────────────────────────────────────┐
│                    CTC LOSS                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CTC enables training without frame-level alignment:        │
│                                                             │
│  1. Add blank token (ε) to vocabulary                       │
│  2. Collapse consecutive duplicates                         │
│  3. Remove blanks                                           │
│                                                             │
│  Example:                                                   │
│  Prediction: [a, a, ε, b, b, b, ε, c]                      │
│  After collapse: [a, b, c]                                  │
│                                                             │
│  Loss: -log P(Y|X) = -log Σ_π P(π|X)                       │
│  where π are all valid alignments of Y                      │
│                                                             │
│  Forward-Backward Algorithm:                                │
│  - α[t,s]: probability of emitting first s labels by time t │
│  - β[t,s]: probability of emitting remaining labels from t  │
│  - P(Y|X) = Σ_s α[T,s] (sum over final states)             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class CTCLoss:
    """
    Custom CTC Loss Implementation using Forward-Backward Algorithm

    Key insight: CTC marginalizes over all possible alignments,
    enabling training without explicit frame-level labels.
    """

    def __init__(self, blank_id=0):
        self.blank = blank_id

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: (T, B, V) - log probabilities from encoder
            targets: (B, S) - target sequences
            input_lengths: (B,) - lengths of each input
            target_lengths: (B,) - lengths of each target

        Returns:
            loss: scalar CTC loss
        """
        batch_size = log_probs.size(1)
        losses = []

        for b in range(batch_size):
            T = input_lengths[b]
            S = target_lengths[b]

            # Get log probs and targets for this sample
            lp = log_probs[:T, b, :]  # (T, V)
            target = targets[b, :S]   # (S,)

            # Create extended target with blanks: [ε, y1, ε, y2, ε, ..., yS, ε]
            ext_target = self._extend_target(target)
            L = len(ext_target)  # 2S + 1

            # Forward pass
            alpha = self._forward(lp, ext_target)

            # Total probability (sum of valid end states)
            log_prob = self._log_sum_exp(alpha[-1, -1], alpha[-1, -2])

            losses.append(-log_prob)

        return torch.stack(losses).mean()

    def _extend_target(self, target):
        """Insert blanks between and around target labels"""
        extended = [self.blank]
        for label in target:
            extended.extend([label.item(), self.blank])
        return extended

    def _forward(self, log_probs, ext_target):
        """
        Forward algorithm in log space

        α[t,s] = log P(emit first s labels by time t)
        """
        T = log_probs.size(0)
        L = len(ext_target)

        # Initialize in log space (use -inf for impossible states)
        alpha = torch.full((T, L), float('-inf'), device=log_probs.device)

        # Initial conditions
        alpha[0, 0] = log_probs[0, ext_target[0]]
        if L > 1:
            alpha[0, 1] = log_probs[0, ext_target[1]]

        # Fill forward
        for t in range(1, T):
            for s in range(L):
                label = ext_target[s]

                # Same state
                prev = alpha[t-1, s]

                # Previous state
                if s > 0:
                    prev = self._log_sum_exp(prev, alpha[t-1, s-1])

                # Skip blank (if current and s-2 are same non-blank, can't skip)
                if s > 1 and ext_target[s] != ext_target[s-2]:
                    prev = self._log_sum_exp(prev, alpha[t-1, s-2])

                alpha[t, s] = prev + log_probs[t, label]

        return alpha

    @staticmethod
    def _log_sum_exp(a, b):
        """Numerically stable log(exp(a) + exp(b))"""
        if a == float('-inf'):
            return b
        if b == float('-inf'):
            return a
        max_val = max(a, b)
        return max_val + torch.log(torch.exp(a - max_val) + torch.exp(b - max_val))
```

---

## 4. Complete Model Architecture

```python
class VoxFormer:
    """
    VoxFormer: Custom Speech-to-Text Transformer

    Architecture: Conformer Encoder + Transformer Decoder
    Training: Hybrid CTC + Cross-Entropy Loss

    Model Configuration (Base):
    - d_model: 512
    - n_heads: 8
    - n_encoder_layers: 12
    - n_decoder_layers: 6
    - d_ff: 2048
    - conv_kernel: 31
    - vocab_size: 5000 (BPE)
    """

    def __init__(self, config):
        self.config = config

        # Audio Frontend
        self.audio_frontend = AudioFrontend(
            n_mels=config.n_mels,
            sample_rate=config.sample_rate
        )

        # Convolutional Subsampling
        self.conv_subsample = ConvolutionalSubsampling(
            input_dim=config.n_mels,
            output_dim=config.d_model
        )

        # Conformer Encoder Stack
        self.encoder_layers = nn.ModuleList([
            ConformerEncoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                conv_kernel_size=config.conv_kernel,
                dropout=config.dropout
            )
            for _ in range(config.n_encoder_layers)
        ])

        # CTC Head (auxiliary)
        self.ctc_head = nn.Linear(config.d_model, config.vocab_size)

        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer Decoder Stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.n_decoder_layers)
        ])

        # Output Head
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (embedding and output head share weights)
        self.output_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def encode(self, audio, audio_lengths=None):
        """
        Encode audio to hidden representations

        Args:
            audio: Raw waveform (B, T_audio) or mel-spectrogram (B, T, n_mels)
            audio_lengths: Original lengths for masking

        Returns:
            encoder_output: (B, T', d_model)
            encoder_mask: (B, T')
        """
        # Audio frontend (if raw waveform)
        if audio.dim() == 2:
            x = self.audio_frontend(audio)  # (B, T, n_mels)
        else:
            x = audio

        # Convolutional subsampling (4× temporal reduction)
        x = self.conv_subsample(x)  # (B, T/4, d_model)

        # Create attention mask if lengths provided
        if audio_lengths is not None:
            encoder_lengths = audio_lengths // 4  # Account for subsampling
            encoder_mask = self._create_mask(encoder_lengths, x.size(1))
        else:
            encoder_mask = None

        # Conformer encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask=encoder_mask)

        return x, encoder_mask

    def decode(self, encoder_output, encoder_mask, target_tokens, target_mask=None):
        """
        Decode target sequence given encoder output

        Args:
            encoder_output: (B, T', d_model)
            encoder_mask: (B, T')
            target_tokens: (B, L) target token ids
            target_mask: (B, L) optional mask

        Returns:
            logits: (B, L, vocab_size)
        """
        # Token embedding
        x = self.token_embedding(target_tokens)  # (B, L, d_model)

        # Create causal mask for decoder self-attention
        L = target_tokens.size(1)
        causal_mask = self._create_causal_mask(L, x.device)

        # Decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, causal_mask, encoder_mask)

        # Output projection
        logits = self.output_head(x)  # (B, L, vocab_size)

        return logits

    def forward(self, audio, audio_lengths, target_tokens, target_lengths):
        """
        Full forward pass for training

        Returns:
            ce_loss: Cross-entropy loss for decoder
            ctc_loss: CTC loss for encoder (auxiliary)
        """
        # Encode
        encoder_output, encoder_mask = self.encode(audio, audio_lengths)

        # CTC loss (auxiliary)
        ctc_logits = self.ctc_head(encoder_output)  # (B, T', vocab)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T', B, vocab)

        # Prepare decoder input (shift right, prepend <sos>)
        decoder_input = self._shift_right(target_tokens)

        # Decode
        decoder_logits = self.decode(encoder_output, encoder_mask, decoder_input)

        return decoder_logits, ctc_log_probs

    @torch.no_grad()
    def transcribe(self, audio, beam_size=5, max_length=256):
        """
        Inference: Transcribe audio to text using beam search
        """
        # Encode
        encoder_output, encoder_mask = self.encode(audio)

        # Beam search decoding
        transcription = self._beam_search(
            encoder_output, encoder_mask, beam_size, max_length
        )

        return transcription
```

---

## 5. Training Strategy

### 5.1 Loss Function

```
Total Loss = λ_ce × L_CE + λ_ctc × L_CTC

where:
- L_CE: Cross-entropy loss for decoder output
- L_CTC: CTC loss for encoder output (auxiliary)
- λ_ce = 0.7, λ_ctc = 0.3 (typical values)
```

### 5.2 Training Configuration

```yaml
# training_config.yaml
training:
  # Optimization
  optimizer: AdamW
  learning_rate: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.98]
  eps: 1e-9

  # Learning rate schedule
  scheduler: warmup_cosine
  warmup_steps: 10000
  total_steps: 500000
  min_lr: 1e-6

  # Batch and gradient
  batch_size: 32  # per GPU
  gradient_accumulation: 4
  max_grad_norm: 1.0

  # Loss weights
  ce_weight: 0.7
  ctc_weight: 0.3

  # Mixed precision
  fp16: true

  # Regularization
  dropout: 0.1
  label_smoothing: 0.1

data:
  # Audio settings
  sample_rate: 16000
  max_audio_length: 30  # seconds
  min_audio_length: 0.5  # seconds

  # Augmentation
  spec_augment:
    freq_mask_param: 27
    time_mask_param: 100
    n_freq_masks: 2
    n_time_masks: 2

  speed_perturb:
    rates: [0.9, 1.0, 1.1]

  noise_augment:
    snr_range: [10, 20]
    probability: 0.3
```

### 5.3 Data Augmentation: SpecAugment

```
┌─────────────────────────────────────────────────────────────┐
│                    SPECAUGMENT                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Frequency Masking                                       │
│     - Randomly mask f consecutive frequency bins           │
│     - f ~ Uniform(0, F)  where F = 27                      │
│     - Apply n_freq_masks = 2 times                         │
│                                                             │
│  2. Time Masking                                            │
│     - Randomly mask t consecutive time steps               │
│     - t ~ Uniform(0, T)  where T = min(100, 0.2×seq_len)   │
│     - Apply n_time_masks = 2 times                         │
│                                                             │
│  Visual:                                                    │
│  ┌──────────────────────────────┐                          │
│  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ← freq mask              │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░│                          │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░│                          │
│  │░░░░░░▓▓▓▓▓░░░░░░░░░░░░░░░░░│ ← freq mask              │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░│                          │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░│                          │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░│                          │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░│                          │
│  └──────┼─┼──────────┼──┼──────┘                          │
│         └─┴── time masks ┴──┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Roadmap

### Phase 1: Core Components (Weeks 1-2)
- [ ] Custom STFT implementation
- [ ] Mel filter bank
- [ ] Convolutional subsampling
- [ ] RMSNorm
- [ ] RoPE (Rotary Position Embeddings)

### Phase 2: Attention Mechanisms (Weeks 3-4)
- [ ] Multi-head self-attention
- [ ] Cross-attention
- [ ] Causal masking
- [ ] Flash Attention optimization (optional)

### Phase 3: Transformer Blocks (Weeks 5-6)
- [ ] SwiGLU feed-forward
- [ ] Conformer convolution module
- [ ] Complete Conformer encoder block
- [ ] Transformer decoder block

### Phase 4: Model Assembly (Week 7)
- [ ] Full VoxFormer model
- [ ] CTC head
- [ ] Output head with weight tying
- [ ] Beam search decoder

### Phase 5: Training Infrastructure (Weeks 8-9)
- [ ] Data pipeline (audio loading, batching)
- [ ] SpecAugment
- [ ] Hybrid CTC + CE loss
- [ ] Learning rate scheduler
- [ ] Gradient checkpointing

### Phase 6: Training & Evaluation (Weeks 10-12)
- [ ] Train on LibriSpeech/Common Voice
- [ ] Implement WER/CER metrics
- [ ] Hyperparameter tuning
- [ ] Model optimization (quantization, pruning)

---

## 7. Datasets

### Primary Training Data
| Dataset | Hours | Description |
|---------|-------|-------------|
| LibriSpeech | 960 | Clean English audiobooks |
| Common Voice | 2000+ | Crowdsourced multilingual |
| GigaSpeech | 10000 | YouTube, podcasts |
| VoxPopuli | 400K | EU Parliament speeches |

### Domain-Specific Fine-tuning
- Game-related vocabulary dataset (to be created)
- 3D modeling terminology
- Action/command phrases

---

## 8. Performance Targets

| Metric | Target (LibriSpeech test-clean) |
|--------|--------------------------------|
| WER | < 3% |
| RTF (Real-Time Factor) | < 0.1 |
| Model Size | < 100M params |
| Latency (30s audio) | < 500ms |

---

## 9. File Structure

```
voxformer/
├── config/
│   ├── model_config.yaml
│   └── training_config.yaml
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── stft.py           # Custom STFT
│   │   ├── mel.py            # Mel filter bank
│   │   └── augment.py        # SpecAugment
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py      # Multi-head attention
│   │   ├── conformer.py      # Conformer blocks
│   │   ├── decoder.py        # Transformer decoder
│   │   ├── embedding.py      # RoPE, token embeddings
│   │   ├── ffn.py           # SwiGLU feed-forward
│   │   ├── norm.py          # RMSNorm
│   │   └── voxformer.py     # Main model
│   ├── loss/
│   │   ├── __init__.py
│   │   ├── ctc.py           # CTC loss
│   │   └── hybrid.py        # Hybrid CTC + CE
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset classes
│   │   └── tokenizer.py     # BPE tokenizer
│   └── utils/
│       ├── __init__.py
│       ├── decode.py        # Beam search
│       └── metrics.py       # WER, CER
├── train.py
├── evaluate.py
└── transcribe.py
```

---

## 10. References

1. **Conformer**: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)
2. **Whisper**: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (2022)
3. **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
4. **SwiGLU**: Shazeer, "GLU Variants Improve Transformer" (2020)
5. **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
6. **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR" (2019)
7. **CTC**: Graves et al., "Connectionist Temporal Classification" (2006)

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Authors: 3D Game AI Assistant Development Team*
