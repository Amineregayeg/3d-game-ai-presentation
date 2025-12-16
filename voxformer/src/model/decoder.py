"""
Transformer Decoder Module

Implementation of the autoregressive Transformer decoder for VoxFormer.
The decoder generates text tokens conditioned on encoder outputs.

Key features:
- Masked self-attention (causal)
- Cross-attention to encoder outputs
- RoPE position embeddings
- KV-cache for efficient streaming inference

Architecture per layer:
    x → MaskedSelfAttn → CrossAttn → FFN → LayerNorm → output

Args total: ~20M parameters (4 layers, 512 dim, 8 heads)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from einops import rearrange

from src.model.conformer import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    FeedForwardModule,
)


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for decoder.

    Uses RoPE and applies causal mask to prevent attending to future tokens.

    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 2048)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary_emb = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for causal self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model)
            cache: Optional KV cache (K, V) from previous steps
            position_offset: Position offset for RoPE when using cache

        Returns:
            Tuple of output tensor and updated (K, V) cache
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary embeddings with position offset
        seq_len = T + position_offset
        cos, sin = self.rotary_emb(x, seq_len)
        cos = cos[position_offset:seq_len]
        sin = sin[position_offset:seq_len]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v)

        # Get causal mask for current sequence
        total_len = k.shape[2]
        if total_len <= self.causal_mask.shape[0]:
            causal_mask = self.causal_mask[:T, :total_len]
        else:
            causal_mask = torch.triu(
                torch.ones(T, total_len, device=x.device), diagonal=total_len - T + 1
            ).bool()

        # Convert to attention bias
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
        attn_mask = attn_mask * torch.finfo(x.dtype).min

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape and project
        attn_output = rearrange(attn_output, "b h t d -> b t (h d)")
        output = self.out_proj(attn_output)

        return output, new_cache


class CrossAttention(nn.Module):
    """
    Cross-attention layer for attending to encoder outputs.

    The decoder queries attend to encoder key-values, allowing
    the model to condition text generation on audio features.

    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for cross-attention.

        Args:
            x: Decoder input of shape (B, T_dec, d_model)
            encoder_output: Encoder output of shape (B, T_enc, d_model)
            encoder_mask: Optional encoder mask of shape (B, T_enc)
            cache: Optional KV cache from encoder (reused across decode steps)

        Returns:
            Tuple of output tensor and (K, V) cache
        """
        B, T_dec, _ = x.shape

        # Query from decoder
        q = self.q_proj(x)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)

        # Key, Value from encoder (cached after first decode step)
        if cache is not None:
            k, v = cache
        else:
            kv = self.kv_proj(encoder_output)
            kv = rearrange(kv, "b t (two h d) -> two b h t d", two=2, h=self.num_heads)
            k, v = kv[0], kv[1]

        new_cache = (k, v)

        # Prepare attention mask from encoder mask
        attn_mask = None
        if encoder_mask is not None:
            # (B, T_enc) -> (B, 1, 1, T_enc)
            attn_mask = encoder_mask.unsqueeze(1).unsqueeze(2).to(dtype=x.dtype)
            attn_mask = (1.0 - attn_mask) * torch.finfo(x.dtype).min

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        # Reshape and project
        attn_output = rearrange(attn_output, "b h t d -> b t (h d)")
        output = self.out_proj(attn_output)

        return output, new_cache


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.

    Architecture:
        x → LayerNorm → CausalSelfAttn → + → LayerNorm → CrossAttn → + → LayerNorm → FFN → + → output
           ↑__________________________|   ↑__________________|   ↑______________|

    Args:
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 2048)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        # Pre-norm layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Attention layers
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout, max_seq_len)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)

        # Feed-forward
        self.ffn = FeedForwardModule(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        self_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_attn_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for decoder layer.

        Args:
            x: Input tensor of shape (B, T, d_model)
            encoder_output: Encoder output of shape (B, T_enc, d_model)
            encoder_mask: Optional encoder mask
            self_attn_cache: KV cache for self-attention
            cross_attn_cache: KV cache for cross-attention
            position_offset: Position offset for RoPE

        Returns:
            Tuple of:
            - Output tensor
            - Updated self-attention cache
            - Updated cross-attention cache
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x, new_self_cache = self.self_attn(x, self_attn_cache, position_offset)
        x = self.dropout(x) + residual

        # Cross-attention with pre-norm
        residual = x
        x = self.norm2(x)
        x, new_cross_cache = self.cross_attn(x, encoder_output, encoder_mask, cross_attn_cache)
        x = self.dropout(x) + residual

        # Feed-forward with pre-norm
        residual = x
        x = self.norm3(x)
        x = self.ffn(x) + residual

        return x, new_self_cache, new_cross_cache


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for VoxFormer.

    Generates text tokens autoregressively conditioned on encoder output.
    Includes token embedding, positional encoding via RoPE, and output projection.

    Args:
        vocab_size: Vocabulary size (default: 2000 for BPE)
        d_model: Model dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of decoder layers (default: 4)
        d_ff: Feed-forward dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 2048)
        pad_token_id: Padding token ID (default: 0)
        bos_token_id: Beginning of sequence token ID (default: 1)
        eos_token_id: End of sequence token ID (default: 2)
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Embedding scale (as in original Transformer)
        self.embed_scale = math.sqrt(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection (tied with embeddings optionally)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, list]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, list]]]:
        """
        Forward pass for decoder.

        Args:
            input_ids: Token IDs of shape (B, T)
            encoder_output: Encoder output of shape (B, T_enc, d_model)
            encoder_mask: Optional encoder mask of shape (B, T_enc)
            cache: Optional cache dict with 'self_attn' and 'cross_attn' lists
            use_cache: Whether to return cache (for inference)

        Returns:
            Tuple of:
            - Logits of shape (B, T, vocab_size)
            - Updated cache (or None if use_cache=False)
        """
        B, T = input_ids.shape

        # Get position offset from cache
        position_offset = 0
        if cache is not None and cache.get("self_attn"):
            position_offset = cache["self_attn"][0][0].shape[2]

        # Token embeddings
        x = self.token_embedding(input_ids) * self.embed_scale
        x = self.dropout(x)

        # Initialize new cache if needed
        should_cache = use_cache or cache is not None
        new_cache = {"self_attn": [], "cross_attn": []} if should_cache else None

        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            self_cache = cache["self_attn"][i] if cache and cache.get("self_attn") else None
            cross_cache = cache["cross_attn"][i] if cache and cache.get("cross_attn") else None

            x, new_self_cache, new_cross_cache = layer(
                x,
                encoder_output,
                encoder_mask,
                self_cache,
                cross_cache,
                position_offset,
            )

            if new_cache is not None:
                new_cache["self_attn"].append(new_self_cache)
                new_cache["cross_attn"].append(new_cross_cache)

        # Final norm and projection
        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits, new_cache

    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: int = 256,
        beam_size: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            encoder_output: Encoder output of shape (B, T_enc, d_model)
            encoder_mask: Optional encoder mask
            max_length: Maximum generation length
            beam_size: Beam size (1 = greedy)
            temperature: Sampling temperature

        Returns:
            Generated token IDs of shape (B, gen_len)
        """
        B = encoder_output.shape[0]
        device = encoder_output.device

        if beam_size == 1:
            return self._greedy_decode(
                encoder_output, encoder_mask, max_length, temperature
            )
        else:
            return self._beam_search(
                encoder_output, encoder_mask, max_length, beam_size
            )

    def _greedy_decode(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        max_length: int,
        temperature: float,
    ) -> torch.Tensor:
        """Greedy decoding for real-time inference."""
        B = encoder_output.shape[0]
        device = encoder_output.device

        # Start with BOS token
        generated = torch.full(
            (B, 1), self.bos_token_id, dtype=torch.long, device=device
        )

        # Cache will be populated after first forward pass
        cache = None
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
            # First step: process full sequence, subsequent steps: last token only
            if cache is None:
                input_tokens = generated
            else:
                input_tokens = generated[:, -1:]

            # Forward pass
            logits, cache = self.forward(
                input_tokens,
                encoder_output,
                encoder_mask,
                cache,
                use_cache=True,
            )

            # Get next token
            next_logits = logits[:, -1, :] / temperature
            next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Update finished
            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)

            # Append token
            generated = torch.cat([generated, next_token], dim=1)

            # Early stopping
            if finished.all():
                break

        return generated

    def _beam_search(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        max_length: int,
        beam_size: int,
    ) -> torch.Tensor:
        """Beam search for higher quality (used after utterance ends)."""
        B = encoder_output.shape[0]
        device = encoder_output.device

        # Expand encoder output for beams
        encoder_output = encoder_output.repeat_interleave(beam_size, dim=0)
        if encoder_mask is not None:
            encoder_mask = encoder_mask.repeat_interleave(beam_size, dim=0)

        # Initialize beams
        generated = torch.full(
            (B * beam_size, 1), self.bos_token_id, dtype=torch.long, device=device
        )
        scores = torch.zeros(B * beam_size, device=device)

        cache = None
        finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
            # Get logits
            logits, cache = self.forward(
                generated[:, -1:] if cache else generated,
                encoder_output,
                encoder_mask,
                cache,
                use_cache=True,
            )

            # Get log probabilities
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            if step == 0:
                # First step: only keep top beam_size from first beam
                log_probs = log_probs.view(B, beam_size, -1)[:, 0, :]
                topk_scores, topk_ids = log_probs.topk(beam_size, dim=-1)
                scores = topk_scores.view(-1)
                generated = torch.cat([
                    generated.view(B, beam_size, -1)[:, 0, :].repeat_interleave(beam_size, dim=0),
                    topk_ids.view(-1, 1)
                ], dim=1)
            else:
                # Subsequent steps
                log_probs = log_probs + scores.unsqueeze(-1)
                log_probs = log_probs.view(B, beam_size * self.vocab_size)

                topk_scores, topk_ids = log_probs.topk(beam_size, dim=-1)

                # Get beam and token indices
                beam_ids = topk_ids // self.vocab_size
                token_ids = topk_ids % self.vocab_size

                # Reorder beams
                batch_offsets = torch.arange(B, device=device).unsqueeze(1) * beam_size
                beam_ids = beam_ids + batch_offsets
                beam_ids = beam_ids.view(-1)

                generated = torch.cat([
                    generated[beam_ids],
                    token_ids.view(-1, 1)
                ], dim=1)
                scores = topk_scores.view(-1)

                # Update cache for reordered beams
                if cache:
                    for key in cache:
                        cache[key] = [
                            (k.index_select(0, beam_ids), v.index_select(0, beam_ids))
                            for k, v in cache[key]
                        ]

            # Check finished
            finished = finished | (generated[:, -1] == self.eos_token_id)
            if finished.all():
                break

        # Select best beam for each batch
        scores = scores.view(B, beam_size)
        best_beams = scores.argmax(dim=-1)

        batch_offsets = torch.arange(B, device=device) * beam_size
        best_ids = best_beams + batch_offsets

        return generated[best_ids]
