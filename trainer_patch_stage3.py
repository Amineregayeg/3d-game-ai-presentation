"""
Trainer patch for Stage 3: Scheduled Sampling support

This patch adds scheduled sampling to the training loop.
Scheduled sampling gradually replaces ground-truth tokens with
model predictions during training, which helps prevent the
repetition issue at inference time.

Apply this patch to src/training/trainer.py
"""

import torch
import random

# Add this to the Trainer class __init__:
INIT_PATCH = '''
        # Scheduled sampling parameters
        self.scheduled_sampling = training_config.get("scheduled_sampling", False)
        self.ss_start = training_config.get("scheduled_sampling_start", 0.0)
        self.ss_end = training_config.get("scheduled_sampling_end", 0.3)
        self.ss_warmup = training_config.get("scheduled_sampling_warmup", 2000)
'''

# Add this method to Trainer class:
METHOD_PATCH = '''
    def get_scheduled_sampling_ratio(self) -> float:
        """Get current scheduled sampling ratio based on global step."""
        if not self.scheduled_sampling:
            return 0.0

        if self.global_step < self.ss_warmup:
            # Linear warmup
            progress = self.global_step / self.ss_warmup
            return self.ss_start + (self.ss_end - self.ss_start) * progress
        else:
            return self.ss_end

    def apply_scheduled_sampling(self, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply scheduled sampling to target sequence.

        With probability `ratio`, replace each token with model's prediction
        from previous step.

        Args:
            target_ids: Ground truth target IDs (B, T)

        Returns:
            Modified target IDs with some tokens replaced by predictions
        """
        ratio = self.get_scheduled_sampling_ratio()
        if ratio == 0.0:
            return target_ids

        B, T = target_ids.shape
        device = target_ids.device

        # Clone to avoid modifying original
        modified = target_ids.clone()

        # Get model predictions for the full sequence
        with torch.no_grad():
            outputs = self.model(
                waveform=None,  # Not needed, use cached encoder output
                target_ids=target_ids,
            )
            predictions = outputs["decoder_logits"].argmax(dim=-1)  # (B, T)

        # Replace tokens with predictions based on ratio
        for b in range(B):
            for t in range(1, T):  # Skip BOS token
                if random.random() < ratio:
                    modified[b, t] = predictions[b, t-1]

        return modified
'''

# Modify train_step to use scheduled sampling:
TRAIN_STEP_PATCH = '''
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step with scheduled sampling support."""
        waveform = batch["waveform"].to(self.device)
        waveform_lengths = batch["waveform_lengths"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        target_lengths = batch["target_lengths"].to(self.device)

        # Apply scheduled sampling
        if self.scheduled_sampling and self.global_step > self.ss_warmup // 2:
            target_ids = self.apply_scheduled_sampling(target_ids)

        self.optimizer.zero_grad()

        with autocast(enabled=self.mixed_precision):
            outputs = self.model(
                waveform=waveform,
                waveform_lengths=waveform_lengths,
                target_ids=target_ids,
            )

            loss_dict = self.loss_fn(
                ctc_logits=outputs["ctc_logits"],
                decoder_logits=outputs.get("decoder_logits"),
                targets=target_ids,
                input_lengths=outputs["encoder_mask"].sum(dim=1).long(),
                target_lengths=target_lengths,
            )

        # Backward pass with gradient scaling
        self.scaler.scale(loss_dict["total"]).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Scheduler step
        self.scheduler.step()

        self.global_step += 1

        return {k: v.item() for k, v in loss_dict.items()}
'''

if __name__ == "__main__":
    print("Stage 3 Trainer Patches")
    print("="*50)
    print("\n1. Add to __init__:")
    print(INIT_PATCH)
    print("\n2. Add methods:")
    print(METHOD_PATCH)
    print("\n3. Modify train_step:")
    print(TRAIN_STEP_PATCH)
