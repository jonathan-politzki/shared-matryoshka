"""Loss factory: config -> composed loss function."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..config import ExperimentConfig, Method
from .infonce import MatryoshkaInfoNCE
from .cross_domain import PrefixInfoNCE, PrefixMSE


class CombinedLoss(nn.Module):
    """Composes within-domain and cross-domain losses with configurable weights."""

    def __init__(
        self,
        within_domain_loss: nn.Module | None,
        cross_domain_loss: nn.Module | None,
        adversarial_loss: nn.Module | None = None,
        within_weight: float = 1.0,
        cross_weight: float = 1.0,
        adversarial_weight: float = 0.0,
    ):
        super().__init__()
        self.within_domain_loss = within_domain_loss
        self.cross_domain_loss = cross_domain_loss
        self.adversarial_loss = adversarial_loss
        self.within_weight = within_weight
        self.cross_weight = cross_weight
        self.adversarial_weight = adversarial_weight

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Compute all loss components.

        Accepts keyword arguments and routes them to the appropriate sub-losses.
        Returns dict with individual losses and 'total'.
        """
        losses = {}
        total = torch.tensor(0.0, device=self._get_device(kwargs))

        # Within-domain triplet loss
        if (
            self.within_domain_loss is not None
            and self.within_weight > 0
            and "anchor" in kwargs
            and "positive" in kwargs
            and "negative" in kwargs
        ):
            wl = self.within_domain_loss(kwargs["anchor"], kwargs["positive"], kwargs["negative"])
            losses["within_domain"] = wl
            total = total + self.within_weight * wl

        # Cross-domain prefix loss
        if (
            self.cross_domain_loss is not None
            and self.cross_weight > 0
            and "cross_anchor" in kwargs
            and "cross_positive" in kwargs
        ):
            cl = self.cross_domain_loss(
                kwargs["cross_anchor"],
                kwargs["cross_positive"],
                kwargs.get("cross_negatives"),
            )
            losses["cross_domain"] = cl
            total = total + self.cross_weight * cl

        # Adversarial loss (if applicable)
        if (
            self.adversarial_loss is not None
            and self.adversarial_weight > 0
            and "domain_embeddings" in kwargs
            and "domain_labels" in kwargs
        ):
            al = self.adversarial_loss(kwargs["domain_embeddings"], kwargs["domain_labels"])
            losses["adversarial"] = al
            total = total + self.adversarial_weight * al

        losses["total"] = total
        return losses

    def _get_device(self, kwargs: dict) -> torch.device:
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")


def build_loss(cfg: ExperimentConfig) -> CombinedLoss:
    """Build the appropriate loss function from config."""
    method = cfg.method

    # Within-domain loss (all methods except adversarial-only use this)
    within_loss = MatryoshkaInfoNCE(
        matryoshka_dims=cfg.model.matryoshka_dims,
        temperature=cfg.loss.temperature,
    )

    # Cross-domain loss depends on method
    cross_loss = None
    adversarial_loss = None

    if method == Method.V3_CONTRASTIVE:
        cross_loss = PrefixInfoNCE(
            prefix_dim=cfg.model.prefix_dim,
            temperature=cfg.loss.temperature,
        )
    elif method == Method.V3_MSE:
        cross_loss = PrefixMSE(prefix_dim=cfg.model.prefix_dim)
    elif method == Method.ADVERSARIAL:
        # Import here to avoid circular dependency
        from .adversarial import DomainAdversarialLoss

        cross_loss = PrefixInfoNCE(
            prefix_dim=cfg.model.prefix_dim,
            temperature=cfg.loss.temperature,
        )
        adversarial_loss = DomainAdversarialLoss(
            input_dim=cfg.model.prefix_dim,
            hidden_dim=cfg.model.adversarial_hidden_dim,
            grl_lambda=cfg.loss.grl_lambda,
        )
    elif method == Method.PROJECTION_HEADS:
        cross_loss = PrefixInfoNCE(
            prefix_dim=cfg.model.identity_head_dim,
            temperature=cfg.loss.temperature,
        )

    return CombinedLoss(
        within_domain_loss=within_loss,
        cross_domain_loss=cross_loss,
        adversarial_loss=adversarial_loss,
        within_weight=cfg.loss.within_domain_weight,
        cross_weight=cfg.loss.cross_domain_weight,
        adversarial_weight=cfg.loss.adversarial_weight,
    )
