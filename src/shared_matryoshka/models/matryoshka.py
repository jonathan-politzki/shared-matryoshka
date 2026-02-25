"""V3 models: single shared encoder with matryoshka prefix alignment.

Covers: v3_contrastive, v3_mse, v3_no_prefix (same architecture, different losses).
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from .base import SharedEmbeddingModel


class MatryoshkaModel(SharedEmbeddingModel):
    """Shared encoder for all V3 variants.

    Architecture: frozen-then-finetuned SentenceTransformer backbone.
    The prefix (first K dims) is trained to be domain-invariant via cross-domain loss.
    The remaining dims specialize per-domain.
    """

    def __init__(self, base_model: str, prefix_dim: int = 64, embed_dim: int = 384):
        super().__init__()
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
        """Encode texts through backbone, returning full embeddings on model device."""
        return self.backbone.encode(
            texts, convert_to_tensor=True, normalize_embeddings=False
        )

    def encode(self, texts: list[str], domain: str | None = None) -> torch.Tensor:
        return self._encode_raw(texts)

    def encode_prefix(self, texts: list[str], domain: str | None = None) -> torch.Tensor:
        emb = self._encode_raw(texts)
        return emb[:, : self._prefix_dim]

    def encode_at_dim(
        self, texts: list[str], domain: str | None = None, dim: int = 64
    ) -> torch.Tensor:
        emb = self._encode_raw(texts)
        return emb[:, :dim]

    def forward_from_texts(
        self, texts: list[str]
    ) -> torch.Tensor:
        """Forward pass that preserves gradients (unlike .encode which detaches).

        Uses the backbone's tokenizer + forward directly.
        """
        features = self.backbone.tokenize(texts)
        features = {k: v.to(self._get_device()) for k, v in features.items()}
        out = self.backbone.forward(features)
        return out["sentence_embedding"]

    def _get_device(self) -> torch.device:
        return next(self.backbone.parameters()).device
