"""Within-domain matryoshka InfoNCE loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatryoshkaInfoNCE(nn.Module):
    """InfoNCE loss computed at multiple matryoshka dimensions.

    For a batch of (anchor, positive, negative) triplets, computes InfoNCE
    at each dimension in matryoshka_dims. The final loss is the mean across dims.
    """

    def __init__(
        self,
        matryoshka_dims: list[int] = (32, 64, 128, 256, 384),
        temperature: float = 0.07,
    ):
        super().__init__()
        self.matryoshka_dims = list(matryoshka_dims)
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute matryoshka InfoNCE.

        Args:
            anchor: (B, D) full embeddings for anchors.
            positive: (B, D) full embeddings for positives.
            negative: (B, D) full embeddings for negatives.

        Returns:
            Scalar loss.
        """
        total_loss = torch.tensor(0.0, device=anchor.device)

        for dim in self.matryoshka_dims:
            a = F.normalize(anchor[:, :dim], dim=-1)
            p = F.normalize(positive[:, :dim], dim=-1)
            n = F.normalize(negative[:, :dim], dim=-1)

            # Positive similarity
            pos_sim = (a * p).sum(dim=-1) / self.temperature  # (B,)

            # Use all negatives in batch (in-batch negatives)
            # Similarity of each anchor with all negatives
            neg_sim = torch.mm(a, n.t()) / self.temperature  # (B, B)

            # InfoNCE: log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # Numerator: pos_sim
            # Denominator: logsumexp over pos + all negatives
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+B)
            labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
            loss = F.cross_entropy(logits, labels)
            total_loss = total_loss + loss

        return total_loss / len(self.matryoshka_dims)
