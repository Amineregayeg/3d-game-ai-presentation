"""
Decoder Fix: Add repetition penalty and improve EOS handling

This patch fixes the decoder's generate function to prevent repetition loops.
Apply this to src/model/decoder.py
"""

# Add this to the _greedy_decode function after getting next_logits:

REPETITION_PENALTY_CODE = '''
    def _greedy_decode(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor],
        max_length: int,
        temperature: float,
        repetition_penalty: float = 1.2,  # NEW
        no_repeat_ngram_size: int = 3,     # NEW
    ) -> torch.Tensor:
        """Greedy decoding with repetition penalty."""
        B = encoder_output.shape[0]
        device = encoder_output.device

        # Start with BOS token
        generated = torch.full(
            (B, 1), self.bos_token_id, dtype=torch.long, device=device
        )

        cache = None
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
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

            # === REPETITION PENALTY (NEW) ===
            if repetition_penalty != 1.0:
                for i in range(B):
                    for prev_token in generated[i].unique():
                        # Penalize tokens that already appeared
                        if next_logits[i, prev_token] > 0:
                            next_logits[i, prev_token] /= repetition_penalty
                        else:
                            next_logits[i, prev_token] *= repetition_penalty

            # === NO REPEAT N-GRAM (NEW) ===
            if no_repeat_ngram_size > 0 and step >= no_repeat_ngram_size - 1:
                for i in range(B):
                    # Get the last (n-1) tokens
                    ngram_prefix = tuple(generated[i, -(no_repeat_ngram_size-1):].tolist())
                    # Check all previous n-grams
                    for j in range(generated.shape[1] - no_repeat_ngram_size + 1):
                        prev_ngram = tuple(generated[i, j:j+no_repeat_ngram_size-1].tolist())
                        if prev_ngram == ngram_prefix:
                            # The next token would complete a repeated n-gram
                            banned_token = generated[i, j + no_repeat_ngram_size - 1].item()
                            next_logits[i, banned_token] = -float('inf')

            # === BOOST EOS PROBABILITY (NEW) ===
            # After generating enough tokens, boost EOS to help model stop
            if step > 10:
                eos_boost = min(0.1 * (step - 10), 2.0)  # Gradual boost
                next_logits[:, self.eos_token_id] += eos_boost

            next_token = next_logits.argmax(dim=-1, keepdim=True)

            # Update finished
            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)

            # Append token
            generated = torch.cat([generated, next_token], dim=1)

            # Early stopping
            if finished.all():
                break

        return generated
'''

# Also update the generate() method to pass the new parameters:
GENERATE_UPDATE = '''
    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: int = 256,
        beam_size: int = 1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,  # NEW
        no_repeat_ngram_size: int = 3,     # NEW
    ) -> torch.Tensor:
        if beam_size == 1:
            return self._greedy_decode(
                encoder_output, encoder_mask, max_length, temperature,
                repetition_penalty, no_repeat_ngram_size  # NEW
            )
        else:
            return self._beam_search(
                encoder_output, encoder_mask, max_length, beam_size
            )
'''

print("Apply these changes to src/model/decoder.py to fix repetition issues")
print("\n1. Replace _greedy_decode with the updated version")
print("2. Update generate() to accept new parameters")
print("\nKey fixes:")
print("- Repetition penalty: penalizes already-generated tokens")
print("- No-repeat n-gram: prevents exact n-gram repetitions")
print("- EOS boost: gradually increases EOS probability")
