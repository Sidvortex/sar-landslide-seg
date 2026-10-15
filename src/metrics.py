"""Pixel metrics for the positive (changed / landslide) class, accumulated over batches."""
from __future__ import annotations

import torch


class BinaryStats:
    def __init__(self, threshold: float = 0.5):
        self.t = threshold
        self.tp = self.fp = self.fn = self.tn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.sigmoid(logits.float()) > self.t
        tgt = target > 0.5
        self.tp += int((pred & tgt).sum())
        self.fp += int((pred & ~tgt).sum())
        self.fn += int((~pred & tgt).sum())
        self.tn += int((~pred & ~tgt).sum())

    def compute(self) -> dict:
        eps = 1e-9
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        return {
            "iou": self.tp / (self.tp + self.fp + self.fn + eps),
            "f1": 2 * precision * recall / (precision + recall + eps),
            "precision": precision,
            "recall": recall,
            "accuracy": (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps),
        }
