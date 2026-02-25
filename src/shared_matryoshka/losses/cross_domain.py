"""Cross-domain prefix losses: InfoNCE and MSE variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixInfoNCE(nn.Module):
    """InfoNCE on prefix dimensions for cross-domain identity alignment.

    Given anchor embeddings from domain A and positive/negative embeddings
    from domain B, computes InfoNCE using only the first prefix_dim dimensions.
    """

    def __init__(self, prefix_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.prefix_dim = prefix_dim
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prefix InfoNCE.

        Args:
            anchor: (B, D) embeddings from domain A.
            positive: (B, D) embeddings from domain B (same person).
            negatives: (B, N, D) embeddings from domain B (different people).

        Returns:
            Scalar loss.
        """
        # Slice to prefix
        a = F.normalize(anchor[:, : self.prefix_dim], dim=-1)  # (B, K)
        p = F.normalize(positive[:, : self.prefix_dim], dim=-1)  # (B, K)
        n = F.normalize(negatives[:, :, : self.prefix_dim], dim=-1)  # (B, N, K)

        # Positive similarity
        pos_sim = (a * p).sum(dim=-1, keepdim=True) / self.temperature  # (B, 1)

        # Negative similarities
        neg_sim = torch.bmm(n, a.unsqueeze(-1)).squeeze(-1) / self.temperature  # (B, N)

        # InfoNCE
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+N)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


class PrefixMSE(nn.Module):
    """MSE on prefix dimensions — ablation baseline (expected to cause collapse)."""

    def __init__(self, prefix_dim: int = 64):
        super().__init__()
        self.prefix_dim = prefix_dim

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MSE between prefix embeddings of same person across domains.

        Args:
            anchor: (B, D) embeddings from domain A.
            positive: (B, D) embeddings from domain B (same person).
            negatives: Ignored (kept for interface compatibility).

        Returns:
            Scalar loss.
        """
        a = anchor[:, : self.prefix_dim]
        p = positive[:, : self.prefix_dim]
        return F.mse_loss(a, p)
