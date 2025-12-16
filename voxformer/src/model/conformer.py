"""
Conformer Module

Implementation of the Conformer block for speech recognition, combining:
- Multi-Head Self-Attention with Rotary Position Embeddings (RoPE)
- Depthwise Separable Convolution
- Macaron-style Feed-Forward Networks

Architecture per block (Macaron-style):
    x → FFN(½) → MHSA → Conv → FFN(½) → LayerNorm → output

Reference: "Conformer: Convolution-augmented Transformer for Speech Recognition"
           https://arxiv.org/abs/2005.08100
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for relative position encoding.

    RoPE encodes position information by rotating query and key vectors
    in 2D subspaces, enabling efficient relative position modeling.

    Args:
        dim: Dimension of each attention head
        max_seq_len: Maximum sequence length (default: 8192)
        base: Base for frequency computation (default: 10000)
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute sin/cos cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build sin/cos cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin for given sequence length.

        Args:
            x: Input tensor (for device placement)
            seq_len: Current sequence length

        Returns:
            Tuple of (cos, sin) each of shape (seq_len, dim)
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_cache[:seq_len].to(x.dtype),
            self.sin_cache[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor (B, num_heads, T, head_dim)
        k: Key tensor (B, num_heads, T, head_dim)
        cos: Cosine embeddings (T, head_dim)
        sin: Sine embeddings (T, head_dim)

    Returns:
        Tuple of rotated (q, k) tensors
    """
    # Expand cos/sin for batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Position Embeddings.

    Features:
    - Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
    - RoPE for relative position encoding
    - Optional Flash Attention support (via PyTorch 2.0 SDPA)

    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        dropout: Attention dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length for RoPE (default: 8192)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Rotary position embeddings
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model)
            attention_mask: Optional mask of shape (B, T) or (B, 1, 1, T)
            key_value_cache: Optional KV cache for inference (K, V tensors)

        Returns:
            Tuple of:
            - Output tensor of shape (B, T, d_model)
            - Updated KV cache (for inference)
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache for streaming inference
        if key_value_cache is not None:
            k_cache, v_cache = key_value_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v) if key_value_cache is not None else None

        # Scaled dot-product attention (uses Flash Attention if available)
        if attention_mask is not None:
            # Convert boolean mask to attention bias
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=x.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min

        # Use PyTorch 2.0 SDPA for efficiency
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape and project output
        attn_output = rearrange(attn_output, "b h t d -> b t (h d)")
        output = self.out_proj(attn_output)

        return output, new_cache


class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module with depthwise separable convolution.

    Architecture:
        LayerNorm → Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv → Dropout

    The depthwise convolution captures local context while being parameter-efficient.

    Args:
        d_model: Model dimension (default: 512)
        kernel_size: Convolution kernel size (default: 31)
        dropout: Dropout probability (default: 0.1)
        causal: Whether to use causal convolution for streaming (default: False)
    """

    def __init__(
        self,
        d_model: int = 512,
        kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.causal = causal

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise expansion (1x1 conv)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, 2 * d_model, kernel_size=1, bias=False
        )

        # Depthwise convolution
        padding = (kernel_size - 1) if causal else (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,  # Depthwise
            bias=False,
        )

        self.batch_norm = nn.BatchNorm1d(d_model)

        # Pointwise projection (1x1 conv)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for convolution module.

        Args:
            x: Input tensor of shape (B, T, d_model)

        Returns:
            Output tensor of shape (B, T, d_model)
        """
        x = self.layer_norm(x)

        # Transpose for conv1d: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)

        # Pointwise expansion with GLU activation
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # Split and gate: 2D -> D

        # Depthwise convolution
        x = self.depthwise_conv(x)

        # Remove right padding for causal convolution
        if self.causal:
            x = x[:, :, : -(self.kernel_size - 1)]

        # Batch norm and Swish activation
        x = self.batch_norm(x)
        x = F.silu(x)  # Swish = x * sigmoid(x)

        # Pointwise projection
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # Transpose back: (B, D, T) -> (B, T, D)
        x = x.transpose(1, 2)

        return x


class FeedForwardModule(nn.Module):
    """
    Macaron-style Feed-Forward Module with SwiGLU activation.

    Architecture:
        LayerNorm → Linear → SwiGLU → Dropout → Linear → Dropout

    SwiGLU provides better gradient flow than ReLU/GELU for speech tasks.

    Args:
        d_model: Model dimension (default: 512)
        d_ff: Feed-forward dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.layer_norm = nn.LayerNorm(d_model)

        # SwiGLU requires 2x expansion for gate
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward module.

        Args:
            x: Input tensor of shape (B, T, d_model)

        Returns:
            Output tensor of shape (B, T, d_model)
        """
        x = self.layer_norm(x)

        # SwiGLU: (Swish(xW1) ⊙ xW2) W3
        gate = F.silu(self.w1(x))  # Swish activation
        x = gate * self.w2(x)  # Element-wise gate
        x = self.dropout(x)
        x = self.w3(x)
        x = self.dropout(x)

        return x


class ConformerBlock(nn.Module):
    """
    Single Conformer Block with Macaron-style architecture.

    Architecture:
        x → FFN(½) → + → MHSA → + → Conv → + → FFN(½) → + → LayerNorm → output
           ↑___________↑______↑_________↑_________↑

    The "½" indicates half-step residual (multiply by 0.5 before adding).

    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 2048)
        kernel_size: Convolution kernel size (default: 31)
        dropout: Dropout probability (default: 0.1)
        causal: Whether to use causal convolution (default: False)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model

        # First half-step FFN
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)

        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)

        # Convolution module
        self.conv = ConvolutionModule(d_model, kernel_size, dropout, causal)

        # Second half-step FFN
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)

        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for Conformer block.

        Args:
            x: Input tensor of shape (B, T, d_model)
            attention_mask: Optional mask of shape (B, T)
            key_value_cache: Optional KV cache for streaming

        Returns:
            Tuple of:
            - Output tensor of shape (B, T, d_model)
            - Updated KV cache
        """
        # First FFN (half-step residual)
        x = x + 0.5 * self.ff1(x)

        # Self-attention
        attn_out, new_cache = self.self_attn(x, attention_mask, key_value_cache)
        x = x + attn_out

        # Convolution
        x = x + self.conv(x)

        # Second FFN (half-step residual)
        x = x + 0.5 * self.ff2(x)

        # Final layer norm
        x = self.layer_norm(x)

        return x, new_cache
