"""Projection heads baseline: shared backbone + identity/task projection heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from .base import SharedEmbeddingModel


class ProjectionHeadsModel(SharedEmbeddingModel):
    """Standard multi-task baseline with separate projection heads.

    Architecture: shared backbone -> identity head (cross-domain) + task head (per-domain).
    The identity head output is what gets compared across domains.
    """

    def __init__(
        self,
        base_model: str,
        identity_dim: int = 64,
        embed_dim: int = 384,
    ):
        super().__init__()
        self._prefix_dim = identity_dim
        self._embedding_dim = embed_dim
        self.backbone = SentenceTransformer(base_model)

        # Identity head: maps backbone output to domain-invariant space
        self.identity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, identity_dim),
        )

        # Task head: maps backbone output to task-specific space
        self.task_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @property
    def prefix_dim(self) -> int:
        return self._prefix_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _encode_backbone(self, texts: list[str]) -> torch.Tensor:
        emb = self.backbone.encode(
            texts, convert_to_tensor=True, normalize_embeddings=False
        )
        return emb.clone()

    def _heads_device(self) -> torch.device:
        return next(self.identity_head.parameters()).device

    @torch.no_grad()
    def encode(self, texts: list[str], domain: str | None = None) -> torch.Tensor:
        backbone_out = self._encode_backbone(texts).to(self._heads_device())
        task_out = self.task_head(backbone_out)
        return task_out

    @torch.no_grad()
    def encode_prefix(self, texts: list[str], domain: str | None = None) -> torch.Tensor:
        backbone_out = self._encode_backbone(texts).to(self._heads_device())
        return self.identity_head(backbone_out)

    def encode_at_dim(
        self, texts: list[str], domain: str | None = None, dim: int = 64
    ) -> torch.Tensor:
        # For projection heads, matryoshka slicing doesn't apply naturally.
        # Return identity head output if dim <= identity_dim, else task head.
        if dim <= self._prefix_dim:
            return self.encode_prefix(texts, domain)[:, :dim]
        return self.encode(texts, domain)[:, :dim]

    def forward_from_texts(self, texts: list[str]) -> torch.Tensor:
        """Forward pass with gradients. Returns concatenated [identity; task] embeddings.

        For loss computation, the caller slices as needed.
        """
        features = self.backbone.tokenize(texts)
        features = {k: v.to(self._get_device()) for k, v in features.items()}
        out = self.backbone.forward(features)
        backbone_out = out["sentence_embedding"]

        identity_out = self.identity_head(backbone_out)
        task_out = self.task_head(backbone_out)

        # Concatenate: [identity (prefix_dim) | task (embed_dim)]
        return torch.cat([identity_out, task_out], dim=-1)

    def _get_device(self) -> torch.device:
        return next(self.backbone.parameters()).device
