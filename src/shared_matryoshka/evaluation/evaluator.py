"""Evaluator: runs all metrics on a trained model."""

from __future__ import annotations

import logging

import torch

from ..config import ExperimentConfig
from ..models.base import SharedEmbeddingModel
from . import metrics

logger = logging.getLogger("shared_matryoshka")


class Evaluator:
    """Method-agnostic evaluator. Calls model.encode() and computes all metrics."""

    def __init__(self, model: SharedEmbeddingModel, cfg: ExperimentConfig):
        self.model = model
        self.cfg = cfg
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        dating_texts: list[str],
        hiring_texts: list[str],
        person_ids: list[int],
        dating_triplets: list[tuple[int, int, int]],
        hiring_triplets: list[tuple[int, int, int]],
        batch_size: int = 128,
    ) -> dict[str, float]:
        """Run full evaluation suite.

        Args:
            dating_texts: Dating profile texts for validation set.
            hiring_texts: Hiring resume texts for validation set.
            person_ids: Person IDs aligned with texts.
            dating_triplets: Validation dating triplets (indices into texts).
            hiring_triplets: Validation hiring triplets (indices into texts).
            batch_size: Encoding batch size.

        Returns:
            Dict of metric_name -> value.
        """
        results = {}

        # Encode all texts
        logger.info("Encoding dating texts...")
        dating_embs = self._batch_encode(dating_texts, "dating", batch_size)
        logger.info("Encoding hiring texts...")
        hiring_embs = self._batch_encode(hiring_texts, "hiring", batch_size)

        prefix_dim = self.cfg.model.prefix_dim

        # 1. Zero-shot cross-domain transfer (PRIMARY)
        results["cross_domain_accuracy"] = metrics.cross_domain_accuracy(
            dating_embs, hiring_embs, person_ids, prefix_dim=prefix_dim
        )
        results["cross_domain_accuracy_full"] = metrics.cross_domain_accuracy(
            dating_embs, hiring_embs, person_ids, prefix_dim=None
        )
        results["cross_domain_margin"] = metrics.cross_domain_margin(
            dating_embs, hiring_embs, person_ids, prefix_dim=prefix_dim
        )

        # 2. Cross-format identity retrieval
        dating_prefix = dating_embs[:, :prefix_dim]
        hiring_prefix = hiring_embs[:, :prefix_dim]
        for k in self.cfg.eval.recall_k:
            results[f"recall@{k}"] = metrics.recall_at_k(
                dating_prefix, hiring_prefix, person_ids, person_ids, k=k
            )
        results["mrr"] = metrics.mrr(
            dating_prefix, hiring_prefix, person_ids, person_ids
        )

        # 3. Within-domain performance
        if dating_triplets:
            d_a = dating_embs[torch.tensor([t[0] for t in dating_triplets])]
            d_p = dating_embs[torch.tensor([t[1] for t in dating_triplets])]
            d_n = dating_embs[torch.tensor([t[2] for t in dating_triplets])]
            results["dating_triplet_accuracy"] = metrics.triplet_accuracy(d_a, d_p, d_n)

        if hiring_triplets:
            h_a = hiring_embs[torch.tensor([t[0] for t in hiring_triplets])]
            h_p = hiring_embs[torch.tensor([t[1] for t in hiring_triplets])]
            h_n = hiring_embs[torch.tensor([t[2] for t in hiring_triplets])]
            results["hiring_triplet_accuracy"] = metrics.triplet_accuracy(h_a, h_p, h_n)

        # 4. CKA across dimensions
        cka_results = metrics.cka_across_dims(
            dating_embs, hiring_embs, self.cfg.model.matryoshka_dims
        )
        for dim, val in cka_results.items():
            results[f"cka_dim_{dim}"] = val

        # 5. Prefix variance (collapse diagnostic)
        results["prefix_variance_dating"] = metrics.prefix_variance(dating_embs, prefix_dim)
        results["prefix_variance_hiring"] = metrics.prefix_variance(hiring_embs, prefix_dim)

        return results

    def _batch_encode(
        self, texts: list[str], domain: str, batch_size: int
    ) -> torch.Tensor:
        """Encode texts in batches."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            embs = self.model.encode(batch, domain=domain)
            all_embs.append(embs.cpu())
        return torch.cat(all_embs, dim=0)
