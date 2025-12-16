"""
Decoder fix: Add repetition penalty and improve generation
"""

import torch
import torch.nn.functional as F


def generate_with_repetition_penalty(
    self,
    encoder_output: torch.Tensor,
    encoder_mask=None,
    max_length: int = 256,
    beam_size: int = 1,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
) -> torch.Tensor:
    """
    Generate tokens with repetition penalty.

    Args:
        encoder_output: Encoder output (B, T_enc, d_model)
        encoder_mask: Optional encoder mask
        max_length: Maximum generation length
        beam_size: Beam size (1 = greedy)
        temperature: Sampling temperature
        repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
        no_repeat_ngram_size: Prevent repeating n-grams of this size

    Returns:
        Generated token IDs (B, gen_len)
    """
    B = encoder_output.shape[0]
    device = encoder_output.device

    # Start with BOS token
    generated = torch.full(
        (B, 1), self.bos_token_id, dtype=torch.long, device=device
    )

    cache = None
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_length - 1):
        # Forward pass
        if cache is None:
            input_tokens = generated
        else:
            input_tokens = generated[:, -1:]

        logits, cache = self.forward(
            input_tokens,
            encoder_output,
            encoder_mask,
            cache,
            use_cache=True,
        )

        next_logits = logits[:, -1, :] / temperature

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for i in range(B):
                for prev_token in generated[i].tolist():
                    if prev_token < next_logits.shape[-1]:
                        # Reduce probability of repeated tokens
                        if next_logits[i, prev_token] > 0:
                            next_logits[i, prev_token] /= repetition_penalty
                        else:
                            next_logits[i, prev_token] *= repetition_penalty

        # No repeat n-gram blocking
        if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
            for i in range(B):
                # Get recent n-1 tokens
                recent = generated[i, -(no_repeat_ngram_size-1):].tolist()
                # Check all previous n-grams
                for j in range(generated.shape[1] - no_repeat_ngram_size + 1):
                    prev_ngram = generated[i, j:j+no_repeat_ngram_size-1].tolist()
                    if prev_ngram == recent:
                        # Block the token that would complete this n-gram
                        blocked_token = generated[i, j+no_repeat_ngram_size-1].item()
                        next_logits[i, blocked_token] = float('-inf')

        # Get next token
        next_token = next_logits.argmax(dim=-1, keepdim=True)

        # Update finished
        finished = finished | (next_token.squeeze(-1) == self.eos_token_id)

        # Append token
        generated = torch.cat([generated, next_token], dim=1)

        # Early stopping
        if finished.all():
            break

    return generated


# Monkey-patch instructions for the decoder
PATCH_CODE = '''
# Add this method to TransformerDecoder class in src/model/decoder.py

def generate_with_penalty(
    self,
    encoder_output,
    encoder_mask=None,
    max_length=256,
    beam_size=1,
    temperature=1.0,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
):
    """Generate with repetition penalty."""
    B = encoder_output.shape[0]
    device = encoder_output.device

    generated = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
    cache = None
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(max_length - 1):
        if cache is None:
            input_tokens = generated
        else:
            input_tokens = generated[:, -1:]

        logits, cache = self.forward(input_tokens, encoder_output, encoder_mask, cache, use_cache=True)
        next_logits = logits[:, -1, :] / temperature

        # Repetition penalty
        if repetition_penalty != 1.0:
            for i in range(B):
                for prev_token in generated[i].tolist():
                    if prev_token < next_logits.shape[-1]:
                        if next_logits[i, prev_token] > 0:
                            next_logits[i, prev_token] /= repetition_penalty
                        else:
                            next_logits[i, prev_token] *= repetition_penalty

        # No-repeat n-gram
        if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
            for i in range(B):
                recent = generated[i, -(no_repeat_ngram_size-1):].tolist()
                for j in range(generated.shape[1] - no_repeat_ngram_size + 1):
                    prev_ngram = generated[i, j:j+no_repeat_ngram_size-1].tolist()
                    if prev_ngram == recent:
                        blocked = generated[i, j+no_repeat_ngram_size-1].item()
                        next_logits[i, blocked] = float('-inf')

        next_token = next_logits.argmax(dim=-1, keepdim=True)
        finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
        generated = torch.cat([generated, next_token], dim=1)

        if finished.all():
            break

    return generated
'''

if __name__ == "__main__":
    print("Decoder fix module loaded")
    print("Use generate_with_repetition_penalty() for inference")
