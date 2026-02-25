"""Domain adversarial loss with gradient reversal layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer (Ganin et al., 2016)."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_val: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_val)


class DomainClassifier(nn.Module):
    """Binary domain classifier (dating=0, hiring=1)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainAdversarialLoss(nn.Module):
    """Domain confusion loss on prefix embeddings via gradient reversal.

    Trains a domain classifier on prefix embeddings with gradient reversal,
    so the encoder learns to make the prefix domain-invariant.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.classifier = DomainClassifier(input_dim, hidden_dim)
        self.grl_lambda = grl_lambda

    def forward(
        self,
        embeddings: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adversarial loss.

        Args:
            embeddings: (B, D) prefix embeddings from both domains.
            domain_labels: (B,) binary domain labels (0=dating, 1=hiring).

        Returns:
            Domain classification loss (with gradient reversal applied to encoder).
        """
        # Apply gradient reversal
        reversed_embs = grad_reverse(embeddings, self.grl_lambda)
        logits = self.classifier(reversed_embs)
        return F.cross_entropy(logits, domain_labels)
