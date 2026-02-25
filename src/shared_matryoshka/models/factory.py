"""Model factory: config -> model instance."""

from __future__ import annotations

from ..config import ExperimentConfig, Method
from .base import SharedEmbeddingModel
from .matryoshka import MatryoshkaModel
from .single_domain import SingleDomainModel


def build_model(cfg: ExperimentConfig) -> SharedEmbeddingModel:
    """Build the appropriate model from config."""
    method = cfg.method
    base = cfg.model.base_model
    prefix = cfg.model.prefix_dim
    embed = cfg.model.embedding_dim

    if method in (Method.V3_CONTRASTIVE, Method.V3_MSE, Method.V3_NO_PREFIX):
        return MatryoshkaModel(base, prefix_dim=prefix, embed_dim=embed)

    elif method == Method.SINGLE_DATING:
        return SingleDomainModel(base, domain="dating", prefix_dim=prefix, embed_dim=embed)

    elif method == Method.SINGLE_HIRING:
        return SingleDomainModel(base, domain="hiring", prefix_dim=prefix, embed_dim=embed)

    elif method == Method.PROJECTION_HEADS:
        from .projection_heads import ProjectionHeadsModel

        return ProjectionHeadsModel(
            base,
            identity_dim=cfg.model.identity_head_dim,
            embed_dim=embed,
        )

    elif method == Method.ADVERSARIAL:
        from .adversarial import AdversarialModel

        return AdversarialModel(base, prefix_dim=prefix, embed_dim=embed)

    else:
        raise ValueError(f"Unknown method: {method}")
