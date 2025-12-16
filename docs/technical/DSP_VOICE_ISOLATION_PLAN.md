# Advanced Digital Signal Processing & Voice Isolation Pipeline
## Technical Planning Document v1.0

---

## Executive Summary

This document outlines the comprehensive Digital Signal Processing (DSP) pipeline for voice isolation and audio preprocessing. All components are implemented at the low level using fundamental mathematical principles, without relying on high-level audio processing libraries. The pipeline ensures robust human voice extraction from noisy, reverberant, and multi-source audio environments.

---

## 1. DSP Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     VOICE ISOLATION & DSP PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐                                                               │
│  │  Raw Audio   │  Microphone Input (16-bit PCM, potentially noisy)            │
│  │   Input      │                                                               │
│  └──────┬───────┘                                                               │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 1: SIGNAL CONDITIONING                                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐   │  │
│  │  │  DC Offset  │→ │  Pre-       │→ │  Dithering  │→ │ Sample Rate    │   │  │
│  │  │  Removal    │  │  Emphasis   │  │  & Noise    │  │ Conversion     │   │  │
│  │  │             │  │  Filter     │  │  Shaping    │  │ (if needed)    │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 2: VOICE ACTIVITY DETECTION (VAD)                                 │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐   │  │
│  │  │  Energy-    │→ │  Zero-      │→ │  Spectral   │→ │ Neural VAD     │   │  │
│  │  │  Based VAD  │  │  Crossing   │  │  Entropy    │  │ (DNN-based)    │   │  │
│  │  │             │  │  Rate       │  │  Analysis   │  │                │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 3: NOISE ESTIMATION & REDUCTION                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐   │  │
│  │  │  Noise      │→ │  Spectral   │→ │  Wiener     │→ │ MMSE-STSA      │   │  │
│  │  │  Estimation │  │  Subtraction│  │  Filter     │  │ Estimator      │   │  │
│  │  │  (MCRA)     │  │             │  │             │  │                │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 4: ACOUSTIC ECHO CANCELLATION (AEC)                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                       │  │
│  │  │  Adaptive   │→ │  NLMS/RLS   │→ │  Double-    │                       │  │
│  │  │  Filter     │  │  Algorithm  │  │  Talk       │                       │  │
│  │  │  Bank       │  │             │  │  Detection  │                       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 5: VOICE ISOLATION / SOURCE SEPARATION                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐   │  │
│  │  │  Harmonic-  │→ │  Ideal      │→ │  Deep       │→ │ Post-Filter    │   │  │
│  │  │  Percussive │  │  Binary     │  │  Attractor  │  │ Masking        │   │  │
│  │  │  Separation │  │  Mask (IBM) │  │  Network    │  │                │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STAGE 6: DEREVERBERATION                                                │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                       │  │
│  │  │  WPE        │→ │  Spectral   │→ │  Late       │                       │  │
│  │  │  (Weighted  │  │  Enhancement│  │  Reverb     │                       │  │
│  │  │  Prediction)│  │             │  │  Suppression│                       │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                       │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│         │                                                                       │
│         ▼                                                                       │
│  ┌──────────────┐                                                               │
│  │  Clean Voice │  Isolated human voice ready for STT                          │
│  │   Output     │                                                               │
│  └──────────────┘                                                               │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Fundamental DSP Mathematics

### 2.1 Sampling & Quantization Theory

```
┌─────────────────────────────────────────────────────────────┐
│              SAMPLING FUNDAMENTALS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Nyquist-Shannon Sampling Theorem:                          │
│  ─────────────────────────────────                          │
│  A bandlimited signal x(t) with maximum frequency f_max     │
│  can be perfectly reconstructed from samples if:            │
│                                                             │
│       f_s > 2 × f_max                                       │
│                                                             │
│  For speech (f_max ≈ 8kHz): f_s = 16kHz is standard        │
│                                                             │
│  Quantization:                                              │
│  ─────────────                                              │
│  16-bit signed integer: range [-32768, 32767]               │
│  Dynamic range: 20 × log10(2^16) ≈ 96.33 dB                │
│                                                             │
│  Quantization noise power:                                  │
│       σ_q² = Δ² / 12                                        │
│  where Δ = full_scale / 2^n_bits                           │
│                                                             │
│  Dithering: Add small noise before quantization to          │
│  decorrelate quantization error (prevents distortion)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Discrete Fourier Transform (DFT) - Low Level

```python
class DiscreteFourierTransform:
    """
    Low-level DFT/IDFT implementation without FFT libraries

    DFT Definition:
    X[k] = Σ_{n=0}^{N-1} x[n] · e^{-j2πkn/N}

    IDFT Definition:
    x[n] = (1/N) · Σ_{k=0}^{N-1} X[k] · e^{j2πkn/N}

    Euler's Formula:
    e^{jθ} = cos(θ) + j·sin(θ)
    """

    def __init__(self, n_fft: int):
        self.n_fft = n_fft
        # Pre-compute twiddle factors for efficiency
        self._precompute_twiddle_factors()

    def _precompute_twiddle_factors(self):
        """
        Twiddle factors: W_N^{kn} = e^{-j2πkn/N}

        Store cos and sin components separately for real arithmetic
        """
        N = self.n_fft
        k = np.arange(N).reshape(-1, 1)  # Column vector
        n = np.arange(N).reshape(1, -1)  # Row vector

        angles = -2 * np.pi * k * n / N

        self.W_real = np.cos(angles).astype(np.float32)  # (N, N)
        self.W_imag = np.sin(angles).astype(np.float32)  # (N, N)

        # Inverse twiddle factors (positive exponent)
        self.W_inv_real = np.cos(-angles).astype(np.float32)
        self.W_inv_imag = np.sin(-angles).astype(np.float32)

    def dft(self, x: np.ndarray) -> tuple:
        """
        Compute DFT of real signal x

        Args:
            x: Real signal of length N

        Returns:
            X_real: Real part of DFT coefficients
            X_imag: Imaginary part of DFT coefficients
        """
        # Pad or truncate to n_fft
        if len(x) < self.n_fft:
            x = np.pad(x, (0, self.n_fft - len(x)))
        elif len(x) > self.n_fft:
            x = x[:self.n_fft]

        # Matrix multiplication: X = W · x
        # For real input x:
        # X_real = W_real · x
        # X_imag = W_imag · x
        X_real = np.dot(self.W_real, x)
        X_imag = np.dot(self.W_imag, x)

        return X_real, X_imag

    def idft(self, X_real: np.ndarray, X_imag: np.ndarray) -> np.ndarray:
        """
        Compute inverse DFT

        Args:
            X_real: Real part of spectrum
            X_imag: Imaginary part of spectrum

        Returns:
            x: Reconstructed time-domain signal
        """
        N = self.n_fft

        # x = (1/N) · W_inv · X
        # For complex X = X_real + j·X_imag:
        # x_real = (1/N) · (W_inv_real · X_real - W_inv_imag · X_imag)
        x_real = (np.dot(self.W_inv_real, X_real) -
                  np.dot(self.W_inv_imag, X_imag)) / N

        return x_real

    def magnitude(self, X_real: np.ndarray, X_imag: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrum: |X| = sqrt(X_real² + X_imag²)"""
        return np.sqrt(X_real**2 + X_imag**2)

    def phase(self, X_real: np.ndarray, X_imag: np.ndarray) -> np.ndarray:
        """Compute phase spectrum: φ = atan2(X_imag, X_real)"""
        return np.arctan2(X_imag, X_real)


class CooleyTukeyFFT:
    """
    Cooley-Tukey FFT Algorithm (Radix-2 DIT)

    Complexity: O(N log N) vs O(N²) for naive DFT

    The algorithm recursively splits the DFT into even and odd indices:
    X[k] = E[k] + W_N^k · O[k]       for k = 0, 1, ..., N/2 - 1
    X[k + N/2] = E[k] - W_N^k · O[k] for k = 0, 1, ..., N/2 - 1

    where E[k] = DFT of even-indexed samples
          O[k] = DFT of odd-indexed samples
    """

    def fft(self, x: np.ndarray) -> tuple:
        """
        Compute FFT using Cooley-Tukey algorithm

        Args:
            x: Input signal (length must be power of 2)

        Returns:
            X_real, X_imag: Real and imaginary parts of FFT
        """
        N = len(x)

        # Base case
        if N == 1:
            return np.array([x[0]]), np.array([0.0])

        # Check power of 2
        if N & (N - 1) != 0:
            # Pad to next power of 2
            next_pow2 = 1 << (N - 1).bit_length()
            x = np.pad(x, (0, next_pow2 - N))
            N = next_pow2

        # Recursive FFT
        return self._fft_recursive(x)

    def _fft_recursive(self, x: np.ndarray) -> tuple:
        N = len(x)

        if N == 1:
            return np.array([x[0]]), np.array([0.0])

        # Split into even and odd
        x_even = x[0::2]
        x_odd = x[1::2]

        # Recursive calls
        E_real, E_imag = self._fft_recursive(x_even)
        O_real, O_imag = self._fft_recursive(x_odd)

        # Twiddle factors for this stage
        k = np.arange(N // 2)
        angles = -2 * np.pi * k / N
        W_real = np.cos(angles)
        W_imag = np.sin(angles)

        # Complex multiplication: W · O = (W_r + jW_i)(O_r + jO_i)
        # = (W_r·O_r - W_i·O_i) + j(W_r·O_i + W_i·O_r)
        WO_real = W_real * O_real - W_imag * O_imag
        WO_imag = W_real * O_imag + W_imag * O_real

        # Butterfly operation
        X_real = np.concatenate([E_real + WO_real, E_real - WO_real])
        X_imag = np.concatenate([E_imag + WO_imag, E_imag - WO_imag])

        return X_real, X_imag
```

---

## 3. Digital Filter Design

### 3.1 FIR Filter (Finite Impulse Response)

```
┌─────────────────────────────────────────────────────────────┐
│              FIR FILTER DESIGN                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FIR Filter Equation:                                       │
│  y[n] = Σ_{k=0}^{M-1} h[k] · x[n-k]                        │
│                                                             │
│  where h[k] are the filter coefficients (impulse response)  │
│        M is the filter order                                │
│                                                             │
│  Properties:                                                │
│  - Always stable (no feedback)                              │
│  - Linear phase possible (symmetric coefficients)           │
│  - Higher order needed for sharp cutoffs                    │
│                                                             │
│  Design Methods:                                            │
│  1. Window Method (Hamming, Blackman, Kaiser)               │
│  2. Frequency Sampling                                      │
│  3. Parks-McClellan (Optimal Equiripple)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class FIRFilterDesign:
    """
    Low-level FIR filter design and implementation

    Design using windowed sinc method:
    1. Start with ideal (sinc) filter
    2. Apply window function to truncate
    3. Result: practical FIR filter
    """

    @staticmethod
    def ideal_lowpass_coeffs(cutoff_freq: float, sample_rate: float,
                              num_taps: int) -> np.ndarray:
        """
        Design ideal lowpass filter coefficients using sinc function

        Ideal lowpass impulse response:
        h_ideal[n] = (2·f_c/f_s) · sinc(2·f_c·(n - M/2) / f_s)

        where sinc(x) = sin(πx) / (πx)
        """
        M = num_taps - 1  # Filter order
        n = np.arange(num_taps)

        # Normalized cutoff frequency
        f_c_norm = cutoff_freq / sample_rate

        # Handle center tap (avoid division by zero)
        h = np.zeros(num_taps)
        center = M / 2

        for i in range(num_taps):
            if i == center:
                h[i] = 2 * f_c_norm
            else:
                x = 2 * np.pi * f_c_norm * (i - center)
                h[i] = np.sin(x) / (np.pi * (i - center))

        return h

    @staticmethod
    def apply_window(h: np.ndarray, window_type: str = 'hamming') -> np.ndarray:
        """
        Apply window function to filter coefficients

        Windows reduce spectral leakage (Gibbs phenomenon)
        """
        N = len(h)
        n = np.arange(N)

        if window_type == 'hamming':
            # Hamming: w[n] = 0.54 - 0.46·cos(2πn/(N-1))
            window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

        elif window_type == 'blackman':
            # Blackman: better sidelobe suppression
            window = (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) +
                      0.08 * np.cos(4 * np.pi * n / (N - 1)))

        elif window_type == 'kaiser':
            # Kaiser: adjustable parameter (beta = 5 typical)
            beta = 5.0
            window = FIRFilterDesign._kaiser_window(N, beta)

        elif window_type == 'hann':
            # Hann (Hanning): w[n] = 0.5·(1 - cos(2πn/(N-1)))
            window = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))

        else:
            window = np.ones(N)  # Rectangular (no window)

        return h * window

    @staticmethod
    def _kaiser_window(N: int, beta: float) -> np.ndarray:
        """
        Kaiser window: w[n] = I_0(β·√(1-(2n/N-1)²)) / I_0(β)

        I_0 is the modified Bessel function of the first kind, order 0
        """
        def bessel_i0(x):
            """Approximate I_0(x) using series expansion"""
            sum_val = 1.0
            term = 1.0
            for k in range(1, 25):  # 25 terms for good accuracy
                term *= (x / (2 * k)) ** 2
                sum_val += term
                if term < 1e-12:
                    break
            return sum_val

        n = np.arange(N)
        alpha = (N - 1) / 2
        arg = beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2)
        window = np.array([bessel_i0(a) for a in arg]) / bessel_i0(beta)

        return window

    @staticmethod
    def design_bandpass(f_low: float, f_high: float, sample_rate: float,
                        num_taps: int = 101) -> np.ndarray:
        """
        Design bandpass filter: combines highpass and lowpass

        Bandpass = Lowpass(f_high) - Lowpass(f_low)
        """
        h_low = FIRFilterDesign.ideal_lowpass_coeffs(f_low, sample_rate, num_taps)
        h_high = FIRFilterDesign.ideal_lowpass_coeffs(f_high, sample_rate, num_taps)

        # Bandpass = highpass - lowpass
        h_bp = h_high - h_low

        # Apply window
        h_bp = FIRFilterDesign.apply_window(h_bp, 'hamming')

        return h_bp


class FIRFilter:
    """
    FIR Filter implementation with convolution
    """

    def __init__(self, coefficients: np.ndarray):
        self.h = coefficients
        self.order = len(coefficients)
        self.buffer = np.zeros(self.order)

    def filter(self, x: np.ndarray) -> np.ndarray:
        """
        Apply FIR filter to input signal

        y[n] = Σ_{k=0}^{M-1} h[k] · x[n-k]
        """
        y = np.zeros(len(x))

        for n in range(len(x)):
            # Shift buffer and insert new sample
            self.buffer = np.roll(self.buffer, 1)
            self.buffer[0] = x[n]

            # Convolution (dot product)
            y[n] = np.dot(self.h, self.buffer)

        return y

    def filter_block(self, x: np.ndarray) -> np.ndarray:
        """
        Block filtering using convolution (more efficient)
        """
        return np.convolve(x, self.h, mode='same')
```

### 3.2 IIR Filter (Infinite Impulse Response)

```
┌─────────────────────────────────────────────────────────────┐
│              IIR FILTER DESIGN                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IIR Filter Equation (Direct Form I):                       │
│  y[n] = Σ_{k=0}^{M} b[k]·x[n-k] - Σ_{k=1}^{N} a[k]·y[n-k]  │
│                                                             │
│  Transfer Function:                                         │
│  H(z) = (b_0 + b_1·z^{-1} + ... + b_M·z^{-M})              │
│         ─────────────────────────────────────              │
│         (1 + a_1·z^{-1} + ... + a_N·z^{-N})                │
│                                                             │
│  Properties:                                                │
│  - Can be unstable (poles must be inside unit circle)       │
│  - Lower order than FIR for same performance                │
│  - Non-linear phase                                         │
│                                                             │
│  Common Types:                                              │
│  - Butterworth (maximally flat passband)                    │
│  - Chebyshev Type I (equiripple passband)                  │
│  - Chebyshev Type II (equiripple stopband)                 │
│  - Elliptic (equiripple both bands, sharpest cutoff)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class ButterworthFilter:
    """
    Butterworth IIR Filter Design

    Characteristics:
    - Maximally flat frequency response in passband
    - Monotonic rolloff
    - No ripple

    Magnitude Response:
    |H(jω)|² = 1 / (1 + (ω/ω_c)^{2N})
    where N is the filter order
    """

    @staticmethod
    def design_lowpass(cutoff_freq: float, sample_rate: float,
                       order: int = 4) -> tuple:
        """
        Design Butterworth lowpass filter coefficients

        Returns (b, a) coefficients for difference equation
        """
        # Prewarp cutoff frequency (bilinear transform)
        # ω_a = (2/T) · tan(ω_d · T/2)
        T = 1 / sample_rate
        omega_d = 2 * np.pi * cutoff_freq
        omega_a = (2 / T) * np.tan(omega_d * T / 2)

        # Butterworth poles (analog prototype)
        # Poles are evenly spaced on left half of unit circle
        poles_s = []
        for k in range(order):
            theta = np.pi * (2 * k + order + 1) / (2 * order)
            pole = omega_a * np.exp(1j * theta)
            poles_s.append(pole)

        # Bilinear transform: s = (2/T)(z-1)/(z+1)
        # Maps analog poles to digital poles
        poles_z = [(2/T + p) / (2/T - p) for p in poles_s]

        # Convert poles to polynomial coefficients
        # a(z) = Π(z - p_k)
        a = ButterworthFilter._poles_to_coeffs(poles_z)

        # Numerator: all zeros at z = -1 for lowpass
        zeros_z = [-1] * order
        b = ButterworthFilter._poles_to_coeffs(zeros_z)

        # Normalize for unity gain at DC
        gain = sum(a) / sum(b)
        b = [coef * gain for coef in b]

        return np.array(b), np.array(a)

    @staticmethod
    def _poles_to_coeffs(poles: list) -> list:
        """
        Convert poles/zeros to polynomial coefficients

        For poles p_1, p_2, ..., p_N:
        a(z) = (z - p_1)(z - p_2)...(z - p_N)
        """
        coeffs = [1.0]

        for pole in poles:
            # Multiply by (z - pole)
            new_coeffs = [0.0] * (len(coeffs) + 1)
            for i, c in enumerate(coeffs):
                new_coeffs[i] += c
                new_coeffs[i + 1] -= c * pole

            coeffs = new_coeffs

        # Return real parts (imaginary parts should cancel)
        return [np.real(c) for c in coeffs]


class IIRFilter:
    """
    IIR Filter implementation using Direct Form II Transposed

    More numerically stable than Direct Form I
    """

    def __init__(self, b: np.ndarray, a: np.ndarray):
        """
        Args:
            b: Numerator coefficients [b_0, b_1, ..., b_M]
            a: Denominator coefficients [1, a_1, ..., a_N]
        """
        self.b = b / a[0]  # Normalize
        self.a = a / a[0]
        self.order = max(len(b), len(a)) - 1

        # State variables (delay elements)
        self.state = np.zeros(self.order)

    def filter_sample(self, x: float) -> float:
        """
        Process single sample (for real-time processing)

        Direct Form II Transposed:
        y[n] = b_0·x[n] + s_0[n-1]
        s_0[n] = b_1·x[n] - a_1·y[n] + s_1[n-1]
        s_1[n] = b_2·x[n] - a_2·y[n] + s_2[n-1]
        ...
        """
        # Output
        y = self.b[0] * x + self.state[0]

        # Update state
        for i in range(self.order - 1):
            b_i = self.b[i + 1] if i + 1 < len(self.b) else 0
            a_i = self.a[i + 1] if i + 1 < len(self.a) else 0
            self.state[i] = b_i * x - a_i * y + self.state[i + 1]

        # Last state
        b_last = self.b[-1] if len(self.b) > self.order else 0
        a_last = self.a[-1] if len(self.a) > self.order else 0
        self.state[-1] = b_last * x - a_last * y

        return y

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Process array of samples"""
        y = np.zeros(len(x))
        for n in range(len(x)):
            y[n] = self.filter_sample(x[n])
        return y

    def reset(self):
        """Reset filter state"""
        self.state = np.zeros(self.order)
```

### 3.3 Adaptive Filters (NLMS & RLS)

```
┌─────────────────────────────────────────────────────────────┐
│              ADAPTIVE FILTERS                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Use case: Echo cancellation, noise cancellation            │
│  The filter adapts its coefficients to minimize error       │
│                                                             │
│  General Structure:                                         │
│  ┌─────────┐    x[n]     ┌─────────────┐                   │
│  │Reference│───────────▶│ Adaptive    │───▶ y[n] (estimate)│
│  │ Signal  │             │ Filter W    │                    │
│  └─────────┘             └─────────────┘                    │
│                                  │                          │
│  ┌─────────┐    d[n]            │                          │
│  │ Desired │─────────────▶ ⊕ ◀──┘                          │
│  │ Signal  │              -      │                          │
│  └─────────┘                     ▼                          │
│                              e[n] = d[n] - y[n]             │
│                                                             │
│  Algorithms:                                                │
│  - LMS: Simple, slow convergence                            │
│  - NLMS: Normalized LMS, faster convergence                 │
│  - RLS: Fastest convergence, higher complexity              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class NLMSAdaptiveFilter:
    """
    Normalized Least Mean Squares (NLMS) Adaptive Filter

    Update Equation:
    w[n+1] = w[n] + μ · e[n] · x[n] / (||x[n]||² + ε)

    where:
    - w: filter coefficients
    - μ: step size (0 < μ < 2 for stability)
    - e[n] = d[n] - y[n]: error signal
    - x[n]: input vector [x[n], x[n-1], ..., x[n-L+1]]
    - ε: small constant for numerical stability
    """

    def __init__(self, filter_length: int, step_size: float = 0.1,
                 regularization: float = 1e-6):
        self.L = filter_length
        self.mu = step_size
        self.eps = regularization

        # Filter coefficients (initially zero)
        self.w = np.zeros(filter_length)

        # Input buffer
        self.x_buffer = np.zeros(filter_length)

    def update(self, x: float, d: float) -> tuple:
        """
        Process one sample and update filter

        Args:
            x: Reference input sample
            d: Desired signal sample

        Returns:
            y: Filter output (estimate)
            e: Error signal
        """
        # Shift buffer and insert new sample
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x

        # Filter output: y = w^T · x
        y = np.dot(self.w, self.x_buffer)

        # Error signal
        e = d - y

        # NLMS update
        norm_sq = np.dot(self.x_buffer, self.x_buffer) + self.eps
        self.w = self.w + (self.mu * e / norm_sq) * self.x_buffer

        return y, e

    def process(self, x: np.ndarray, d: np.ndarray) -> tuple:
        """
        Process arrays of samples

        Returns:
            y: Filter output array
            e: Error signal array
        """
        y = np.zeros(len(x))
        e = np.zeros(len(x))

        for n in range(len(x)):
            y[n], e[n] = self.update(x[n], d[n])

        return y, e


class RLSAdaptiveFilter:
    """
    Recursive Least Squares (RLS) Adaptive Filter

    Minimizes weighted least squares cost:
    J[n] = Σ_{k=0}^{n} λ^{n-k} · |e[k]|²

    Update Equations:
    k[n] = (P[n-1]·x[n]) / (λ + x^T[n]·P[n-1]·x[n])
    e[n] = d[n] - w^T[n-1]·x[n]
    w[n] = w[n-1] + k[n]·e[n]
    P[n] = (P[n-1] - k[n]·x^T[n]·P[n-1]) / λ

    where:
    - λ: forgetting factor (0.95 < λ ≤ 1)
    - P: inverse correlation matrix
    - k: Kalman gain vector
    """

    def __init__(self, filter_length: int, forgetting_factor: float = 0.99,
                 delta: float = 0.01):
        self.L = filter_length
        self.lambda_ = forgetting_factor
        self.delta = delta

        # Filter coefficients
        self.w = np.zeros(filter_length)

        # Inverse correlation matrix (initialize as scaled identity)
        self.P = np.eye(filter_length) / delta

        # Input buffer
        self.x_buffer = np.zeros(filter_length)

    def update(self, x: float, d: float) -> tuple:
        """
        Process one sample and update filter using RLS
        """
        # Shift buffer
        self.x_buffer = np.roll(self.x_buffer, 1)
        self.x_buffer[0] = x

        x_vec = self.x_buffer.reshape(-1, 1)  # Column vector

        # Kalman gain: k = P·x / (λ + x^T·P·x)
        Px = self.P @ x_vec
        denom = self.lambda_ + (x_vec.T @ Px)[0, 0]
        k = Px / denom

        # A priori error
        y = np.dot(self.w, self.x_buffer)
        e = d - y

        # Update weights
        self.w = self.w + (k.flatten() * e)

        # Update inverse correlation matrix
        self.P = (self.P - k @ x_vec.T @ self.P) / self.lambda_

        return y, e

    def process(self, x: np.ndarray, d: np.ndarray) -> tuple:
        """Process arrays of samples"""
        y = np.zeros(len(x))
        e = np.zeros(len(x))

        for n in range(len(x)):
            y[n], e[n] = self.update(x[n], d[n])

        return y, e
```

---

## 4. Voice Activity Detection (VAD)

### 4.1 Energy-Based VAD

```
┌─────────────────────────────────────────────────────────────┐
│              ENERGY-BASED VAD                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Frame Energy:                                              │
│  E[m] = Σ_{n=0}^{N-1} |x[mH + n]|²                         │
│                                                             │
│  where m is frame index, H is hop size, N is frame length  │
│                                                             │
│  Decision:                                                  │
│  - Speech if E[m] > threshold                               │
│  - Threshold adapts based on noise estimate                 │
│                                                             │
│  Improvements:                                              │
│  - Use log energy: 10·log10(E + ε)                         │
│  - Adaptive threshold with hangover                         │
│  - Combine with zero-crossing rate                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class EnergyVAD:
    """
    Voice Activity Detection using Frame Energy

    Features:
    - Adaptive noise floor estimation
    - Hangover scheme to avoid clipping speech
    - Smoothing to reduce spurious detections
    """

    def __init__(self, frame_size: int = 400, hop_size: int = 160,
                 sample_rate: int = 16000):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

        # Adaptive threshold parameters
        self.noise_floor = 1e-6
        self.noise_alpha = 0.95  # Noise floor smoothing
        self.speech_threshold_db = 15  # dB above noise floor

        # Hangover to prevent speech cutoff
        self.hangover_frames = 10
        self.hangover_counter = 0

        # State
        self.is_speech = False

    def compute_frame_energy(self, frame: np.ndarray) -> float:
        """Compute log energy of a frame"""
        energy = np.sum(frame ** 2) / len(frame)
        return 10 * np.log10(energy + 1e-10)

    def update_noise_floor(self, energy_db: float):
        """Update noise floor estimate during non-speech"""
        self.noise_floor = (self.noise_alpha * self.noise_floor +
                            (1 - self.noise_alpha) * energy_db)

    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Determine if frame contains speech

        Returns:
            is_speech: True if speech detected
        """
        energy_db = self.compute_frame_energy(frame)

        # Adaptive threshold
        threshold = self.noise_floor + self.speech_threshold_db

        if energy_db > threshold:
            self.is_speech = True
            self.hangover_counter = self.hangover_frames
        else:
            # Hangover: keep speech active for a few frames
            if self.hangover_counter > 0:
                self.hangover_counter -= 1
            else:
                self.is_speech = False
                # Update noise floor only during confirmed silence
                self.update_noise_floor(energy_db)

        return self.is_speech

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process entire audio signal

        Returns:
            vad_mask: Boolean array, True for speech frames
        """
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        vad_mask = np.zeros(num_frames, dtype=bool)

        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size]
            vad_mask[i] = self.process_frame(frame)

        return vad_mask


class SpectralEntropyVAD:
    """
    VAD using Spectral Entropy

    Speech has more structured spectral content (lower entropy)
    Noise has more uniform spectrum (higher entropy)

    Spectral Entropy:
    H = -Σ P[k] · log2(P[k])

    where P[k] = |X[k]|² / Σ|X[k]|² is normalized power spectrum
    """

    def __init__(self, n_fft: int = 512, hop_size: int = 160):
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.dft = DiscreteFourierTransform(n_fft)

        # Threshold (speech typically has entropy < 0.7)
        self.entropy_threshold = 0.7

    def compute_spectral_entropy(self, frame: np.ndarray) -> float:
        """
        Compute normalized spectral entropy of a frame
        """
        # Apply window
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(len(frame)) / (len(frame) - 1))
        windowed = frame * window

        # Compute magnitude spectrum
        X_real, X_imag = self.dft.dft(windowed)
        magnitude = np.sqrt(X_real[:self.n_fft//2]**2 + X_imag[:self.n_fft//2]**2)

        # Normalize to probability distribution
        power = magnitude ** 2
        total_power = np.sum(power) + 1e-10
        P = power / total_power

        # Compute entropy
        P = P + 1e-10  # Avoid log(0)
        entropy = -np.sum(P * np.log2(P))

        # Normalize by maximum entropy (log2(N))
        max_entropy = np.log2(len(P))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def is_speech(self, frame: np.ndarray) -> bool:
        """Determine if frame contains speech based on spectral entropy"""
        entropy = self.compute_spectral_entropy(frame)
        return entropy < self.entropy_threshold
```

### 4.2 Neural VAD

```python
class NeuralVAD:
    """
    Deep Neural Network-based Voice Activity Detection

    Architecture: Small CNN + GRU for temporal modeling

    Features extracted per frame:
    - Log mel filterbank energies (40 bins)
    - Delta and delta-delta features
    - Total: 120 features per frame
    """

    def __init__(self, n_mels: int = 40, hidden_size: int = 64):
        self.n_mels = n_mels
        self.hidden_size = hidden_size

        # Feature extraction
        self.mel_filterbank = MelFilterBank(n_mels=n_mels)

        # Neural network layers (to be implemented with low-level ops)
        self.conv1 = Conv1DLayer(n_mels * 3, 32, kernel_size=5)  # 3 for static+delta+delta2
        self.conv2 = Conv1DLayer(32, 64, kernel_size=3)
        self.gru = GRULayer(64, hidden_size)
        self.fc = LinearLayer(hidden_size, 1)

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract log mel features with deltas"""
        # Compute mel spectrogram
        # (In practice, compute for context window)
        pass

    def forward(self, features: np.ndarray) -> float:
        """
        Forward pass through VAD network

        Args:
            features: (T, n_features) sequence of frame features

        Returns:
            probability: Speech probability [0, 1]
        """
        # Conv layers
        x = self.conv1.forward(features)
        x = relu(x)
        x = self.conv2.forward(x)
        x = relu(x)

        # GRU for temporal modeling
        x, _ = self.gru.forward(x)

        # Take last hidden state
        x = x[-1]

        # Output layer with sigmoid
        logit = self.fc.forward(x)
        probability = sigmoid(logit)

        return probability
```

---

## 5. Noise Reduction

### 5.1 Spectral Subtraction

```
┌─────────────────────────────────────────────────────────────┐
│              SPECTRAL SUBTRACTION                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Basic Principle:                                           │
│  |S(ω)|² ≈ |Y(ω)|² - |N(ω)|²                               │
│                                                             │
│  where:                                                     │
│  - Y(ω): Noisy speech spectrum                              │
│  - N(ω): Estimated noise spectrum                           │
│  - S(ω): Clean speech estimate                              │
│                                                             │
│  Improved (with oversubtraction and flooring):              │
│  |Ŝ(ω)|² = max(|Y(ω)|² - α·|N̂(ω)|², β·|N̂(ω)|²)           │
│                                                             │
│  where:                                                     │
│  - α > 1: Oversubtraction factor (reduces musical noise)    │
│  - β < 1: Spectral floor (prevents negative values)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class SpectralSubtraction:
    """
    Spectral Subtraction Noise Reduction

    Enhanced version with:
    - Oversubtraction to reduce musical noise
    - Spectral flooring to prevent artifacts
    - Multi-band processing for better quality
    """

    def __init__(self, n_fft: int = 512, hop_size: int = 128,
                 alpha: float = 2.0, beta: float = 0.02):
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.alpha = alpha  # Oversubtraction factor
        self.beta = beta    # Spectral floor

        self.fft = CooleyTukeyFFT()

        # Noise estimate (updated during non-speech)
        self.noise_spectrum = None
        self.noise_alpha = 0.98  # Smoothing factor

    def estimate_noise(self, magnitude_spectrum: np.ndarray):
        """Update noise estimate using exponential averaging"""
        if self.noise_spectrum is None:
            self.noise_spectrum = magnitude_spectrum.copy()
        else:
            self.noise_spectrum = (self.noise_alpha * self.noise_spectrum +
                                   (1 - self.noise_alpha) * magnitude_spectrum)

    def process_frame(self, frame: np.ndarray, is_speech: bool) -> np.ndarray:
        """
        Apply spectral subtraction to a single frame

        Args:
            frame: Time-domain frame
            is_speech: VAD decision for this frame

        Returns:
            enhanced_frame: Noise-reduced time-domain frame
        """
        # Window
        window = np.hanning(len(frame))
        windowed = frame * window

        # FFT
        X_real, X_imag = self.fft.fft(windowed)
        magnitude = np.sqrt(X_real**2 + X_imag**2)
        phase = np.arctan2(X_imag, X_real)

        # Update noise estimate during silence
        if not is_speech:
            self.estimate_noise(magnitude)

        # Skip processing if no noise estimate yet
        if self.noise_spectrum is None:
            return frame

        # Spectral subtraction with oversubtraction and flooring
        power_noisy = magnitude ** 2
        power_noise = self.noise_spectrum ** 2

        # Subtract with oversubtraction
        power_clean = power_noisy - self.alpha * power_noise

        # Apply spectral floor
        power_floor = self.beta * power_noise
        power_clean = np.maximum(power_clean, power_floor)

        # Reconstruct magnitude
        magnitude_clean = np.sqrt(power_clean)

        # Reconstruct complex spectrum (keep original phase)
        X_clean_real = magnitude_clean * np.cos(phase)
        X_clean_imag = magnitude_clean * np.sin(phase)

        # Inverse FFT
        dft = DiscreteFourierTransform(self.n_fft)
        frame_clean = dft.idft(X_clean_real, X_clean_imag)

        # Overlap-add windowing
        frame_clean = frame_clean[:len(frame)] * window

        return frame_clean


class WienerFilter:
    """
    Wiener Filter for Noise Reduction

    Optimal linear filter that minimizes mean squared error:
    H(ω) = |S(ω)|² / (|S(ω)|² + |N(ω)|²)
         = SNR(ω) / (1 + SNR(ω))

    This is equivalent to MMSE estimator under Gaussian assumptions
    """

    def __init__(self, n_fft: int = 512):
        self.n_fft = n_fft
        self.noise_power = None
        self.noise_alpha = 0.98

    def compute_gain(self, signal_power: np.ndarray,
                     noise_power: np.ndarray) -> np.ndarray:
        """
        Compute Wiener filter gain

        G(ω) = max(1 - noise_power/signal_power, 0)
             = max(SNR / (1 + SNR), gain_floor)
        """
        # A priori SNR estimate
        snr = np.maximum(signal_power / (noise_power + 1e-10) - 1, 0)

        # Wiener gain
        gain = snr / (1 + snr)

        # Apply gain floor to prevent musical noise
        gain = np.maximum(gain, 0.1)

        return gain

    def process_frame(self, noisy_spectrum: np.ndarray,
                      noise_estimate: np.ndarray) -> np.ndarray:
        """Apply Wiener filter to spectrum"""
        noisy_power = noisy_spectrum ** 2
        noise_power = noise_estimate ** 2

        gain = self.compute_gain(noisy_power, noise_power)

        return noisy_spectrum * gain
```

### 5.2 MMSE-STSA Estimator

```python
class MMSE_STSA:
    """
    Minimum Mean Square Error Short-Time Spectral Amplitude Estimator

    More sophisticated than spectral subtraction:
    - Models speech and noise as Gaussian
    - Computes optimal amplitude estimate

    Â = G(ξ, γ) · Y

    where:
    - γ = |Y|²/λ_n : A posteriori SNR
    - ξ = E{|S|²}/λ_n : A priori SNR
    - G: MMSE gain function involving Bessel functions
    """

    def __init__(self, n_fft: int = 512):
        self.n_fft = n_fft
        self.xi_prev = None  # Previous a priori SNR
        self.alpha = 0.98    # Decision-directed smoothing

    def compute_gain(self, xi: float, gamma: float) -> float:
        """
        Compute MMSE-STSA gain using modified Bessel functions

        G = (√π/2) · (√v/γ) · exp(-v/2) · [(1+v)I_0(v/2) + v·I_1(v/2)]

        where v = xi·γ/(1+xi)
        """
        v = xi * gamma / (1 + xi)

        # Bessel function approximations
        I0 = self._bessel_i0(v / 2)
        I1 = self._bessel_i1(v / 2)

        # Gain computation
        if v < 1e-5:
            return xi / (1 + xi)  # Limit case

        gain = (np.sqrt(np.pi * v) / (2 * gamma)) * np.exp(-v / 2) * \
               ((1 + v) * I0 + v * I1)

        return np.minimum(gain, 1.0)

    def _bessel_i0(self, x: float) -> float:
        """Modified Bessel function of first kind, order 0"""
        if x < 3.75:
            t = (x / 3.75) ** 2
            return 1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 +
                   t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
        else:
            t = 3.75 / x
            return (np.exp(x) / np.sqrt(x)) * (0.39894228 + t * (0.01328592 +
                   t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 +
                   t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 +
                   t * 0.00392377))))))))

    def _bessel_i1(self, x: float) -> float:
        """Modified Bessel function of first kind, order 1"""
        if x < 3.75:
            t = (x / 3.75) ** 2
            return x * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 +
                   t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))))
        else:
            t = 3.75 / x
            return (np.exp(x) / np.sqrt(x)) * (0.39894228 + t * (-0.03988024 +
                   t * (-0.00362018 + t * (0.00163801 + t * (-0.01031555 +
                   t * (0.02282967 + t * (-0.02895312 + t * (0.01787654 +
                   t * (-0.00420059)))))))))

    def process_frame(self, Y_mag: np.ndarray, noise_power: np.ndarray) -> np.ndarray:
        """
        Process a frame using MMSE-STSA

        Args:
            Y_mag: Magnitude spectrum of noisy signal
            noise_power: Estimated noise power spectrum

        Returns:
            S_mag: Enhanced magnitude spectrum
        """
        Y_power = Y_mag ** 2

        # A posteriori SNR
        gamma = Y_power / (noise_power + 1e-10)

        # A priori SNR (decision-directed approach)
        if self.xi_prev is None:
            xi = np.maximum(gamma - 1, 0)
        else:
            # Decision-directed: blend previous estimate with current
            xi = self.alpha * self.xi_prev + (1 - self.alpha) * np.maximum(gamma - 1, 0)

        # Compute MMSE gain for each frequency bin
        gain = np.array([self.compute_gain(x, g) for x, g in zip(xi, gamma)])

        # Store for next frame
        self.xi_prev = (gain ** 2) * gamma

        return Y_mag * gain
```

---

## 6. Noise Estimation: MCRA

```python
class MCRA_NoiseEstimator:
    """
    Minima Controlled Recursive Averaging (MCRA)

    Robust noise estimation that tracks noise floor
    even during speech activity.

    Key insight: Minimum of smoothed spectrum over time
    provides good noise floor estimate.
    """

    def __init__(self, n_freq: int, L: int = 96):
        """
        Args:
            n_freq: Number of frequency bins
            L: Minimum tracking window length (frames)
        """
        self.n_freq = n_freq
        self.L = L

        # Smoothed power spectrum
        self.S = np.zeros(n_freq)
        self.S_min = np.ones(n_freq) * 1e10

        # Noise power estimate
        self.noise_power = np.zeros(n_freq)

        # Speech presence probability
        self.p_speech = np.zeros(n_freq)

        # Parameters
        self.alpha_s = 0.8   # Spectrum smoothing
        self.alpha_d = 0.95  # Noise smoothing
        self.delta = 5.0     # Threshold for speech presence

        # Minimum buffer
        self.min_buffer = []
        self.frame_count = 0

    def update(self, Y_power: np.ndarray) -> np.ndarray:
        """
        Update noise estimate with new frame

        Args:
            Y_power: Power spectrum of current frame

        Returns:
            noise_power: Updated noise power estimate
        """
        self.frame_count += 1

        # Smooth power spectrum
        self.S = self.alpha_s * self.S + (1 - self.alpha_s) * Y_power

        # Track minimum
        self.S_min = np.minimum(self.S_min, self.S)

        # Speech presence indicator
        # Speech likely present if S >> S_min
        S_ratio = self.S / (self.S_min + 1e-10)
        I_speech = (S_ratio > self.delta).astype(float)

        # Update speech presence probability
        self.p_speech = self.alpha_s * self.p_speech + (1 - self.alpha_s) * I_speech

        # Noise update (only during noise-only periods)
        # α_tilde = α_d + (1 - α_d) * p_speech
        alpha_tilde = self.alpha_d + (1 - self.alpha_d) * self.p_speech

        self.noise_power = alpha_tilde * self.noise_power + \
                           (1 - alpha_tilde) * Y_power

        # Reset minimum tracking periodically
        if self.frame_count % self.L == 0:
            self.S_min = self.S.copy()

        return self.noise_power
```

---

## 7. Voice Isolation: Deep Attractor Network

```python
class DeepAttractorNetwork:
    """
    Deep Attractor Network for Speech Separation

    Architecture for isolating target speaker from mixture:
    1. BLSTM encoder: Maps mixture spectrogram to embedding space
    2. Attractor computation: Cluster embeddings to find sources
    3. Mask estimation: Derive soft masks for each source

    V = f(Y) ∈ R^{T×F×D}  (embedding)
    A_c = (1/|C|) Σ_{(t,f)∈C} V_{t,f}  (attractor)
    M_{t,f,c} = softmax(<V_{t,f}, A_c>)  (mask)
    """

    def __init__(self, n_freq: int = 257, embedding_dim: int = 40,
                 hidden_size: int = 300, n_layers: int = 4):
        self.n_freq = n_freq
        self.embedding_dim = embedding_dim

        # Bidirectional LSTM encoder
        self.blstm = BidirectionalLSTM(
            input_size=n_freq,
            hidden_size=hidden_size,
            n_layers=n_layers
        )

        # Embedding layer
        self.embedding_layer = LinearLayer(
            hidden_size * 2,  # Bidirectional
            n_freq * embedding_dim
        )

    def forward(self, mixture_spec: np.ndarray,
                target_mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass for separation

        Args:
            mixture_spec: (T, F) mixture magnitude spectrogram
            target_mask: (T, F) binary mask indicating target speaker
                         (for training; optional for inference)

        Returns:
            separation_mask: (T, F) soft mask for target speaker
        """
        T, F = mixture_spec.shape

        # Encode mixture
        # (T, F) -> BLSTM -> (T, H*2) -> Linear -> (T, F*D)
        hidden = self.blstm.forward(mixture_spec)
        embeddings = self.embedding_layer.forward(hidden)

        # Reshape to (T, F, D)
        V = embeddings.reshape(T, F, self.embedding_dim)

        # Compute attractor
        if target_mask is not None:
            # Training: use ground truth mask
            attractor = self._compute_attractor(V, target_mask)
        else:
            # Inference: use k-means clustering
            attractor = self._estimate_attractor(V)

        # Compute mask from embeddings and attractor
        mask = self._compute_mask(V, attractor)

        return mask

    def _compute_attractor(self, V: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
        """
        Compute attractor point as weighted average of embeddings

        A = Σ_{t,f} mask_{t,f} · V_{t,f} / Σ_{t,f} mask_{t,f}
        """
        # Expand mask for broadcasting
        mask_expanded = mask[:, :, np.newaxis]

        # Weighted sum
        weighted_sum = np.sum(V * mask_expanded, axis=(0, 1))
        weight_total = np.sum(mask) + 1e-10

        attractor = weighted_sum / weight_total

        return attractor

    def _estimate_attractor(self, V: np.ndarray) -> np.ndarray:
        """
        Estimate attractor using k-means clustering on embeddings
        (for inference when ground truth not available)
        """
        T, F, D = V.shape

        # Flatten to (T*F, D)
        V_flat = V.reshape(-1, D)

        # Simple k-means (k=2 for speech + noise)
        # Initialize with random points
        idx = np.random.choice(len(V_flat), 2, replace=False)
        centroids = V_flat[idx].copy()

        for _ in range(10):  # Iterations
            # Assign points to nearest centroid
            distances = np.array([
                np.sum((V_flat - c) ** 2, axis=1) for c in centroids
            ])
            labels = np.argmin(distances, axis=0)

            # Update centroids
            for k in range(2):
                if np.sum(labels == k) > 0:
                    centroids[k] = np.mean(V_flat[labels == k], axis=0)

        # Return centroid with higher energy (likely speech)
        # This is a heuristic; better methods exist
        return centroids[0]  # Simplified

    def _compute_mask(self, V: np.ndarray,
                      attractor: np.ndarray) -> np.ndarray:
        """
        Compute soft mask from embeddings and attractor

        M_{t,f} = σ(<V_{t,f}, A>)

        Using sigmoid for binary separation
        """
        T, F, D = V.shape

        # Dot product with attractor
        similarity = np.sum(V * attractor, axis=2)

        # Sigmoid activation
        mask = 1 / (1 + np.exp(-similarity))

        return mask


class BidirectionalLSTM:
    """
    Bidirectional LSTM implementation

    Processes sequence in both forward and backward directions,
    concatenating hidden states.
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 1):
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Forward and backward LSTM cells for each layer
        self.lstm_forward = [
            LSTMCell(input_size if i == 0 else hidden_size * 2, hidden_size)
            for i in range(n_layers)
        ]
        self.lstm_backward = [
            LSTMCell(input_size if i == 0 else hidden_size * 2, hidden_size)
            for i in range(n_layers)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: (T, input_size) input sequence

        Returns:
            output: (T, hidden_size * 2) bidirectional hidden states
        """
        T = len(x)
        current_input = x

        for layer in range(self.n_layers):
            # Forward pass
            h_forward = np.zeros((T, self.hidden_size))
            h, c = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
            for t in range(T):
                h, c = self.lstm_forward[layer].forward(current_input[t], h, c)
                h_forward[t] = h

            # Backward pass
            h_backward = np.zeros((T, self.hidden_size))
            h, c = np.zeros(self.hidden_size), np.zeros(self.hidden_size)
            for t in range(T - 1, -1, -1):
                h, c = self.lstm_backward[layer].forward(current_input[t], h, c)
                h_backward[t] = h

            # Concatenate
            current_input = np.concatenate([h_forward, h_backward], axis=1)

        return current_input


class LSTMCell:
    """
    Single LSTM Cell

    Gates:
    - Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    - Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    - Cell candidate: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
    - Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

    Updates:
    - c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
    - h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Combined weights for efficiency
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)

    def forward(self, x: np.ndarray, h_prev: np.ndarray,
                c_prev: np.ndarray) -> tuple:
        """Single LSTM step"""
        # Concatenate input and hidden state
        combined = np.concatenate([h_prev, x])

        # Compute all gates at once
        gates = np.dot(self.W, combined) + self.b

        # Split into individual gates
        H = self.hidden_size
        f = self._sigmoid(gates[:H])           # Forget
        i = self._sigmoid(gates[H:2*H])        # Input
        c_tilde = np.tanh(gates[2*H:3*H])      # Cell candidate
        o = self._sigmoid(gates[3*H:])         # Output

        # Update cell state and hidden state
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)

        return h, c

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

---

## 8. Acoustic Echo Cancellation (AEC)

```
┌─────────────────────────────────────────────────────────────┐
│              ACOUSTIC ECHO CANCELLATION                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Problem: Speaker output is picked up by microphone         │
│                                                             │
│  ┌─────────┐                    ┌─────────┐                │
│  │ Far-end │───▶ Speaker ───▶▶ │ Room    │───▶ Microphone  │
│  │ Signal  │                    │ Impulse │      │         │
│  │  x[n]   │                    │Response │      ▼         │
│  └─────────┘                    └─────────┘   y[n] = h*x + s│
│       │                                          │         │
│       │    ┌─────────────────┐                   │         │
│       └───▶│ Adaptive Filter │───▶ ŷ[n] ───▶ ⊕ ◀┘         │
│            │ (estimates h)   │              -    │         │
│            └─────────────────┘                   ▼         │
│                                              e[n] = s[n]   │
│                                              (clean signal)│
│                                                             │
│  Challenge: Double-talk (both near-end and far-end active) │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
class AcousticEchoCanceller:
    """
    Acoustic Echo Cancellation using Frequency-Domain Adaptive Filter

    Key features:
    - Partitioned block frequency domain (efficient for long filters)
    - Double-talk detection
    - Nonlinear processing for residual echo
    """

    def __init__(self, filter_length: int = 4096, block_size: int = 256,
                 sample_rate: int = 16000):
        self.filter_length = filter_length
        self.block_size = block_size
        self.n_blocks = filter_length // block_size

        # FFT size (2× block for overlap-save)
        self.fft_size = 2 * block_size

        # Adaptive filter in frequency domain
        self.W = np.zeros((self.n_blocks, self.fft_size // 2 + 1), dtype=complex)

        # Reference signal buffer
        self.x_buffer = np.zeros(filter_length)

        # Far-end frequency domain buffer
        self.X_buffer = np.zeros((self.n_blocks, self.fft_size // 2 + 1), dtype=complex)

        # Adaptation parameters
        self.mu = 0.5           # Step size
        self.power_smooth = 0.9  # Power estimation smoothing
        self.X_power = np.ones(self.fft_size // 2 + 1) * 1e-6

        # Double-talk detector
        self.dtd = DoubleTalkDetector()

    def process_block(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Process one block of samples

        Args:
            mic: Microphone signal (block_size samples)
            ref: Reference/far-end signal (block_size samples)

        Returns:
            output: Echo-cancelled signal
        """
        N = self.block_size
        N2 = self.fft_size

        # Update reference buffer
        self.x_buffer = np.roll(self.x_buffer, -N)
        self.x_buffer[-N:] = ref

        # Shift X buffer and compute new block's FFT
        self.X_buffer = np.roll(self.X_buffer, 1, axis=0)
        x_block = np.concatenate([np.zeros(N), ref])  # Overlap-save padding
        self.X_buffer[0] = np.fft.rfft(x_block)

        # Update reference power estimate
        self.X_power = (self.power_smooth * self.X_power +
                        (1 - self.power_smooth) * np.abs(self.X_buffer[0]) ** 2)

        # Compute echo estimate: ŷ = IFFT(Σ W_k * X_k)
        Y_hat = np.zeros(N2 // 2 + 1, dtype=complex)
        for k in range(self.n_blocks):
            Y_hat += self.W[k] * self.X_buffer[k]

        y_hat = np.fft.irfft(Y_hat)
        y_hat = y_hat[N:]  # Take second half (overlap-save)

        # Error signal (residual)
        error = mic - y_hat

        # Double-talk detection
        is_doubletalk = self.dtd.detect(mic, ref, error)

        # Update filter (only when no double-talk)
        if not is_doubletalk:
            self._update_filter(error)

        return error

    def _update_filter(self, error: np.ndarray):
        """
        Update adaptive filter using normalized LMS in frequency domain
        """
        N = self.block_size
        N2 = self.fft_size

        # Error in frequency domain
        e_padded = np.concatenate([np.zeros(N), error])
        E = np.fft.rfft(e_padded)

        # Normalized step size per frequency
        mu_normalized = self.mu / (self.X_power + 1e-10)

        # Update each partition
        for k in range(self.n_blocks):
            # Gradient in frequency domain
            gradient = mu_normalized * np.conj(self.X_buffer[k]) * E

            # Constrained update (ensures linear convolution)
            w_time = np.fft.irfft(self.W[k] + gradient)
            w_time[N:] = 0  # Constrain to first half
            self.W[k] = np.fft.rfft(w_time)


class DoubleTalkDetector:
    """
    Double-Talk Detection using Geigel Algorithm + Coherence

    Double-talk occurs when both near-end and far-end speakers
    are active simultaneously. Must pause adaptation to prevent
    filter divergence.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

        # Geigel detector state
        self.max_ref = 0
        self.ref_buffer = []
        self.buffer_len = 1024

    def detect(self, mic: np.ndarray, ref: np.ndarray,
               error: np.ndarray) -> bool:
        """
        Detect double-talk condition

        Returns True if double-talk detected
        """
        # Geigel algorithm: compare mic level to max reference level
        self.ref_buffer.extend(ref.tolist())
        if len(self.ref_buffer) > self.buffer_len:
            self.ref_buffer = self.ref_buffer[-self.buffer_len:]

        max_ref = max(np.abs(self.ref_buffer)) + 1e-10
        max_mic = max(np.abs(mic))

        # If mic >> ref, likely double-talk
        if max_mic > self.threshold * max_ref:
            return True

        # Also check if error is growing (divergence indicator)
        error_power = np.mean(error ** 2)
        mic_power = np.mean(mic ** 2) + 1e-10

        if error_power > 1.5 * mic_power:
            return True

        return False
```

---

## 9. Pre-Emphasis Filter

```python
class PreEmphasisFilter:
    """
    Pre-emphasis filter to boost high frequencies

    High frequencies in speech have lower energy but contain
    important information. Pre-emphasis compensates for this.

    y[n] = x[n] - α·x[n-1]

    Typical α = 0.97 (boosts frequencies above ~1kHz)

    This is equivalent to a highpass filter with:
    H(z) = 1 - α·z^{-1}
    """

    def __init__(self, coeff: float = 0.97):
        self.alpha = coeff
        self.prev_sample = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis to signal"""
        y = np.zeros(len(x))

        for n in range(len(x)):
            y[n] = x[n] - self.alpha * self.prev_sample
            self.prev_sample = x[n]

        return y

    def process_batch(self, x: np.ndarray) -> np.ndarray:
        """Vectorized version for efficiency"""
        return np.append(x[0], x[1:] - self.alpha * x[:-1])


class DeEmphasisFilter:
    """
    De-emphasis filter (inverse of pre-emphasis)

    y[n] = x[n] + α·y[n-1]

    Applied after processing to restore natural spectral balance
    """

    def __init__(self, coeff: float = 0.97):
        self.alpha = coeff
        self.prev_output = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Apply de-emphasis to signal"""
        y = np.zeros(len(x))

        for n in range(len(x)):
            y[n] = x[n] + self.alpha * self.prev_output
            self.prev_output = y[n]

        return y
```

---

## 10. Complete DSP Pipeline

```python
class VoiceIsolationPipeline:
    """
    Complete Voice Isolation and DSP Pipeline

    Integrates all components for robust voice extraction
    """

    def __init__(self, sample_rate: int = 16000, frame_size: int = 512,
                 hop_size: int = 128):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_fft = frame_size

        # Stage 1: Signal conditioning
        self.pre_emphasis = PreEmphasisFilter(0.97)

        # Stage 2: Voice Activity Detection
        self.vad = EnergyVAD(frame_size, hop_size, sample_rate)
        self.spectral_vad = SpectralEntropyVAD(self.n_fft, hop_size)

        # Stage 3: Noise estimation and reduction
        self.noise_estimator = MCRA_NoiseEstimator(self.n_fft // 2 + 1)
        self.mmse = MMSE_STSA(self.n_fft)

        # Stage 4: Voice isolation (neural)
        self.separator = DeepAttractorNetwork(
            n_freq=self.n_fft // 2 + 1,
            embedding_dim=40,
            hidden_size=300
        )

        # Post-processing
        self.de_emphasis = DeEmphasisFilter(0.97)

        # FFT
        self.fft = CooleyTukeyFFT()

    def process(self, audio: np.ndarray,
                use_neural_separation: bool = True) -> np.ndarray:
        """
        Full pipeline processing

        Args:
            audio: Raw audio waveform
            use_neural_separation: Whether to use DAN for separation

        Returns:
            clean_voice: Isolated human voice signal
        """
        # === Stage 1: Signal Conditioning ===
        audio = self._remove_dc_offset(audio)
        audio = self.pre_emphasis.process_batch(audio)

        # === Stage 2: Frame Processing ===
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        enhanced_frames = []

        # Compute full spectrogram for neural separation
        spectrogram = self._compute_spectrogram(audio)

        # === Stage 3: Noise Reduction ===
        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size]

            # VAD
            is_speech = self.vad.process_frame(frame)

            # Get spectrum
            magnitude = spectrogram[i]

            # Update noise estimate
            noise_power = self.noise_estimator.update(magnitude ** 2)

            # MMSE enhancement
            enhanced_mag = self.mmse.process_frame(magnitude, noise_power)

            enhanced_frames.append(enhanced_mag)

        enhanced_spectrogram = np.array(enhanced_frames)

        # === Stage 4: Voice Isolation (Neural) ===
        if use_neural_separation:
            separation_mask = self.separator.forward(enhanced_spectrogram)
            enhanced_spectrogram = enhanced_spectrogram * separation_mask

        # === Stage 5: Reconstruct Waveform ===
        # Use original phase from noisy signal
        phase = self._compute_phase(audio)
        clean_audio = self._reconstruct_waveform(enhanced_spectrogram, phase)

        # === Stage 6: De-emphasis ===
        clean_audio = self.de_emphasis.process(clean_audio)

        return clean_audio

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from signal"""
        return audio - np.mean(audio)

    def _compute_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrogram"""
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        spectrogram = np.zeros((num_frames, self.n_fft // 2 + 1))

        window = np.hanning(self.frame_size)

        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size] * window

            X_real, X_imag = self.fft.fft(frame)
            spectrogram[i] = np.sqrt(X_real[:self.n_fft//2+1]**2 +
                                     X_imag[:self.n_fft//2+1]**2)

        return spectrogram

    def _compute_phase(self, audio: np.ndarray) -> np.ndarray:
        """Compute phase spectrogram"""
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        phase = np.zeros((num_frames, self.n_fft // 2 + 1))

        window = np.hanning(self.frame_size)

        for i in range(num_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size] * window

            X_real, X_imag = self.fft.fft(frame)
            phase[i] = np.arctan2(X_imag[:self.n_fft//2+1],
                                  X_real[:self.n_fft//2+1])

        return phase

    def _reconstruct_waveform(self, magnitude: np.ndarray,
                              phase: np.ndarray) -> np.ndarray:
        """Reconstruct waveform from magnitude and phase using overlap-add"""
        num_frames = len(magnitude)
        output_length = (num_frames - 1) * self.hop_size + self.frame_size
        output = np.zeros(output_length)

        window = np.hanning(self.frame_size)
        dft = DiscreteFourierTransform(self.n_fft)

        for i in range(num_frames):
            # Reconstruct complex spectrum
            X_real = magnitude[i] * np.cos(phase[i])
            X_imag = magnitude[i] * np.sin(phase[i])

            # Mirror for full spectrum
            X_real_full = np.concatenate([X_real, X_real[-2:0:-1]])
            X_imag_full = np.concatenate([X_imag, -X_imag[-2:0:-1]])

            # IDFT
            frame = dft.idft(X_real_full, X_imag_full)
            frame = frame[:self.frame_size] * window

            # Overlap-add
            start = i * self.hop_size
            output[start:start + self.frame_size] += frame

        # Normalize by window overlap
        norm = np.zeros(output_length)
        for i in range(num_frames):
            start = i * self.hop_size
            norm[start:start + self.frame_size] += window ** 2
        output = output / (norm + 1e-10)

        return output
```

---

## 11. File Structure

```
voxformer/
├── src/
│   ├── dsp/
│   │   ├── __init__.py
│   │   ├── transforms.py        # DFT, FFT implementations
│   │   ├── filters/
│   │   │   ├── __init__.py
│   │   │   ├── fir.py          # FIR filter design & implementation
│   │   │   ├── iir.py          # IIR filter design & implementation
│   │   │   └── adaptive.py     # NLMS, RLS adaptive filters
│   │   ├── vad/
│   │   │   ├── __init__.py
│   │   │   ├── energy_vad.py   # Energy-based VAD
│   │   │   ├── spectral_vad.py # Spectral entropy VAD
│   │   │   └── neural_vad.py   # DNN-based VAD
│   │   ├── enhancement/
│   │   │   ├── __init__.py
│   │   │   ├── spectral_subtraction.py
│   │   │   ├── wiener.py       # Wiener filter
│   │   │   ├── mmse_stsa.py    # MMSE estimator
│   │   │   └── noise_estimation.py  # MCRA
│   │   ├── separation/
│   │   │   ├── __init__.py
│   │   │   ├── deep_attractor.py  # DAN for separation
│   │   │   └── beamforming.py     # Multi-mic (future)
│   │   ├── aec/
│   │   │   ├── __init__.py
│   │   │   ├── echo_canceller.py
│   │   │   └── double_talk.py
│   │   └── pipeline.py          # Complete DSP pipeline
│   └── ...
└── ...
```

---

## 12. Implementation Roadmap

### Phase 1: Fundamentals (Week 1)
- [ ] DFT/FFT implementation
- [ ] Window functions (Hann, Hamming, Blackman, Kaiser)
- [ ] Pre-emphasis / De-emphasis filters

### Phase 2: Digital Filters (Week 2)
- [ ] FIR filter design (windowed sinc)
- [ ] IIR Butterworth filter
- [ ] Bandpass voice filter (300Hz - 3400Hz)

### Phase 3: VAD (Week 3)
- [ ] Energy-based VAD
- [ ] Spectral entropy VAD
- [ ] VAD state machine with hangover

### Phase 4: Noise Reduction (Week 4-5)
- [ ] MCRA noise estimator
- [ ] Spectral subtraction
- [ ] Wiener filter
- [ ] MMSE-STSA estimator

### Phase 5: Advanced Components (Week 6-7)
- [ ] Adaptive filter (NLMS)
- [ ] Acoustic echo canceller
- [ ] Double-talk detection

### Phase 6: Neural Separation (Week 8-9)
- [ ] LSTM/GRU implementation
- [ ] Deep Attractor Network
- [ ] Training pipeline for separator

### Phase 7: Integration (Week 10)
- [ ] Complete pipeline integration
- [ ] Real-time processing optimization
- [ ] Testing with various noise conditions

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Authors: 3D Game AI Assistant Development Team*
