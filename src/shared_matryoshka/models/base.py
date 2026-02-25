"""SharedEmbeddingModel ABC — all methods implement this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SharedEmbeddingModel(ABC, nn.Module):
    """Base class for all embedding models.

    Every method must implement encode(), encode_prefix(), and encode_at_dim().
    The evaluator is completely method-agnostic — it only calls these three methods.
    """

    @abstractmethod
    def encode(
        self, texts: list[str], domain: str | None = None
    ) -> torch.Tensor:
        """Encode texts into full-dimensional embeddings.

        Args:
            texts: List of input strings.
            domain: "dating" or "hiring" (optional, some models ignore this).

        Returns:
            Tensor of shape (batch_size, embedding_dim).
        """
        ...

    @abstractmethod
    def encode_prefix(
        self, texts: list[str], domain: str | None = None
    ) -> torch.Tensor:
        """Encode texts and return only the prefix (identity) subspace.

        Returns:
            Tensor of shape (batch_size, prefix_dim).
        """
        ...

    @abstractmethod
    def encode_at_dim(
        self, texts: list[str], domain: str | None = None, dim: int = 64
    ) -> torch.Tensor:
        """Encode texts and return the first `dim` dimensions (matryoshka slicing).

        Returns:
            Tensor of shape (batch_size, dim).
        """
        ...

    @property
    @abstractmethod
    def prefix_dim(self) -> int:
        """Dimensionality of the prefix/identity subspace."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Full embedding dimensionality."""
        ...
