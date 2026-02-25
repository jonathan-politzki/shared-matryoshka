"""Single-domain baselines: dating-only and hiring-only models."""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from .base import SharedEmbeddingModel


class SingleDomainModel(SharedEmbeddingModel):
    """Baseline: fine-tune on one domain only.

    Same architecture as MatryoshkaModel, but trained on a single domain
    with no cross-domain loss. Provides dating/hiring ceiling baselines.
    """

    def __init__(
        self,
        base_model: str,
        domain: str,
        prefix_dim: int = 64,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.domain_name = domain
        self._prefix_dim = prefix_dim
        self._embedding_dim = embed_dim
        self.backbone = SentenceTransformer(base_model)

    @property
    def prefix_dim(self) -> int:
        return self._prefix_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _encode_raw(self, texts: list[str]) -> torch.Tensor:
        return self.backbone.encode(
            texts, convert_to_tensor=True, normalize_embeddings=False
        )

    def encode(self, texts: list[str], domain: str | None = None) -> torch.Tensor:
        return self._encode_raw(texts)

    def encode_prefix(self, texts: list[str], domain: str | None = None) -> torch.Tensor:
        return self._encode_raw(texts)[:, : self._prefix_dim]

    def encode_at_dim(
        self, texts: list[str], domain: str | None = None, dim: int = 64
    ) -> torch.Tensor:
        return self._encode_raw(texts)[:, :dim]

    def forward_from_texts(self, texts: list[str]) -> torch.Tensor:
        features = self.backbone.tokenize(texts)
        features = {k: v.to(self._get_device()) for k, v in features.items()}
        out = self.backbone.forward(features)
        return out["sentence_embedding"]

    def _get_device(self) -> torch.device:
        return next(self.backbone.parameters()).device
