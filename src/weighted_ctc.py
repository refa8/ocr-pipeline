"""
src/weighted_ctc.py
Weighted CTC Loss for rare character recognition.

Strategy: Instead of modifying log_probs (which inflates loss values),
we apply per-sample reweighting AFTER computing standard CTC loss.
This keeps loss in normal range (5-30) while still boosting rare chars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import math


def compute_char_weights(label_file: str, char_to_idx: dict,
                         smoothing: float = 1.0,
                         max_weight: float = 5.0) -> torch.Tensor:
    """
    Compute per-character inverse-frequency weights from training labels.
    """
    counts = Counter()
    total = 0

    with open(label_file, encoding="utf-8") as f:
        for line in f:
            if "	" not in line:
                continue
            text = line.strip().split("	", 1)[1]
            for ch in text:
                if ch in char_to_idx:
                    counts[char_to_idx[ch]] += 1
                    total += 1

    vocab_size = len(char_to_idx) + 1
    weights = torch.ones(vocab_size)
    weights[0] = 1.0  # blank always 1

    for idx in range(1, vocab_size):
        count = counts.get(idx, 0) + smoothing
        freq = count / (total + smoothing * vocab_size)
        weight = 1.0 / (freq * vocab_size)
        weights[idx] = min(weight, max_weight)

    # Normalize: mean weight = 1 so overall loss scale is preserved
    weights[1:] = weights[1:] / weights[1:].mean()

    print(f"  Char weights computed over {total} characters")
    print(f"  Weight range: [{weights[1:].min():.3f}, {weights[1:].max():.3f}]")
    return weights


class WeightedCTCLoss(nn.Module):
    """
    Weighted CTC: applies per-target-character weighting to the loss.
    Each sample loss is scaled by the mean weight of its target characters.
    This keeps loss values in the normal CTC range while boosting rare chars.
    """

    def __init__(self, char_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("char_weights", char_weights)
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Standard CTC loss per sample (on CPU as required by PyTorch)
        device = log_probs.device
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)
        per_sample_loss = self.ctc(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )  # shape: (B,)

        # Compute per-sample weight = mean weight of its target characters

        weights = self.char_weights
        B = per_sample_loss.shape[0]
        sample_weights = torch.ones(B, device=device)



        offset = 0
        for b in range(B):
            length = target_lengths[b].item()
            if length > 0:
                chars = targets[offset: offset + length].to(device)
                char_w = weights[chars.clamp(0, len(weights)-1)]
                sample_weights[b] = char_w.mean()
            offset += length

        # Weighted mean loss — stays in normal CTC range
        weighted_loss = (per_sample_loss * sample_weights).mean()
        return weighted_loss


print("✓ weighted_ctc.py saved")
