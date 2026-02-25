"""Cross-domain prefix losses: InfoNCE and MSE variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixInfoNCE(nn.Module):
    """InfoNCE on prefix dimensions for cross-domain identity alignment.

    Uses in-batch negatives: for each (anchor_i, positive_i) pair,
    all other positives in the batch serve as negatives.
    Falls back to explicit negatives if provided.
    """

    def __init__(self, prefix_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.prefix_dim = prefix_dim
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute prefix InfoNCE.

        Args:
            anchor: (B, D) embeddings from domain A.
            positive: (B, D) embeddings from domain B (same person).
            negatives: Optional (B, N, D) explicit negatives. If None,
                      uses in-batch negatives (more memory efficient).

        Returns:
            Scalar loss.
        """
        # Slice to prefix
        a = F.normalize(anchor[:, : self.prefix_dim], dim=-1)  # (B, K)
        p = F.normalize(positive[:, : self.prefix_dim], dim=-1)  # (B, K)

        if negatives is not None:
            # Explicit negatives path
            n = F.normalize(negatives[:, :, : self.prefix_dim], dim=-1)  # (B, N, K)
            pos_sim = (a * p).sum(dim=-1, keepdim=True) / self.temperature  # (B, 1)
            neg_sim = torch.bmm(n, a.unsqueeze(-1)).squeeze(-1) / self.temperature  # (B, N)
            logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+N)
            labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
            return F.cross_entropy(logits, labels)
        else:
            # In-batch negatives: sim(a_i, p_j) for all i,j; diagonal = positive
            sim_matrix = torch.mm(a, p.t()) / self.temperature  # (B, B)
            labels = torch.arange(a.size(0), device=a.device)
            return F.cross_entropy(sim_matrix, labels)


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
