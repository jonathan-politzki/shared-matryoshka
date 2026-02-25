"""Custom training loop for all methods."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, Method
from ..losses.factory import CombinedLoss
from ..models.base import SharedEmbeddingModel

logger = logging.getLogger("shared_matryoshka")


def _collate_triplets(batch: list[dict]) -> dict:
    return {
        "anchors": [b["anchor"] for b in batch],
        "positives": [b["positive"] for b in batch],
        "negatives": [b["negative"] for b in batch],
    }


def _collate_cross_domain(batch: list[dict]) -> dict:
    return {
        "anchors": [b["anchor"] for b in batch],
        "positives": [b["positive"] for b in batch],
        "negatives_list": [b["negatives"] for b in batch],
    }


class Trainer:
    """Training loop that interleaves within-domain triplets + cross-domain pairs."""

    def __init__(
        self,
        model: SharedEmbeddingModel,
        loss_fn: CombinedLoss,
        cfg: ExperimentConfig,
        dating_triplet_loader: DataLoader | None = None,
        hiring_triplet_loader: DataLoader | None = None,
        cross_domain_loader: DataLoader | None = None,
        val_dating_loader: DataLoader | None = None,
        val_hiring_loader: DataLoader | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.dating_loader = dating_triplet_loader
        self.hiring_loader = hiring_triplet_loader
        self.cross_loader = cross_domain_loader
        self.val_dating_loader = val_dating_loader
        self.val_hiring_loader = val_hiring_loader

        self.device = self._detect_device()
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        # Linear warmup + cosine decay scheduler
        steps_per_epoch = self._estimate_steps_per_epoch()
        total_steps = max(steps_per_epoch * cfg.training.epochs, 1)
        warmup_steps = max(int(total_steps * cfg.training.warmup_ratio), 1)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.0, 0.5 * (1.0 + __import__("math").cos(3.14159 * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.global_step = 0

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _estimate_steps_per_epoch(self) -> int:
        counts = []
        if self.dating_loader:
            counts.append(len(self.dating_loader))
        if self.hiring_loader:
            counts.append(len(self.hiring_loader))
        if self.cross_loader:
            counts.append(len(self.cross_loader))
        return max(counts) if counts else 1

    def _encode_with_grad(self, texts: list[str]) -> torch.Tensor:
        """Encode texts preserving gradients for backprop."""
        return self.model.forward_from_texts(texts)

    def train(self) -> dict[str, list[float]]:
        """Run full training loop. Returns history of losses per epoch."""
        history = {"epoch": [], "total_loss": [], "within_loss": [], "cross_loss": []}

        for epoch in range(1, self.cfg.training.epochs + 1):
            self.model.train()
            epoch_losses = self._train_epoch(epoch)

            history["epoch"].append(epoch)
            history["total_loss"].append(epoch_losses.get("total", 0.0))
            history["within_loss"].append(epoch_losses.get("within_domain", 0.0))
            history["cross_loss"].append(epoch_losses.get("cross_domain", 0.0))

            logger.info(
                f"Epoch {epoch}/{self.cfg.training.epochs} — "
                f"total: {epoch_losses.get('total', 0):.4f}, "
                f"within: {epoch_losses.get('within_domain', 0):.4f}, "
                f"cross: {epoch_losses.get('cross_domain', 0):.4f}"
            )

        # Save checkpoint
        save_dir = Path(self.cfg.training.save_dir) / self.cfg.name
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        logger.info(f"Saved checkpoint to {save_dir / 'model.pt'}")

        return history

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for one epoch, interleaving all data sources."""
        running = {}
        n_steps = 0

        # Create iterators
        dating_iter = iter(self.dating_loader) if self.dating_loader else None
        hiring_iter = iter(self.hiring_loader) if self.hiring_loader else None
        cross_iter = iter(self.cross_loader) if self.cross_loader else None

        steps = self._estimate_steps_per_epoch()
        for step in range(steps):
            self.optimizer.zero_grad()
            loss_inputs = {}

            # Within-domain: alternate between dating and hiring
            triplet_iter = None
            if dating_iter and hiring_iter:
                triplet_iter = dating_iter if step % 2 == 0 else hiring_iter
            elif dating_iter:
                triplet_iter = dating_iter
            elif hiring_iter:
                triplet_iter = hiring_iter

            if triplet_iter is not None:
                try:
                    batch = next(triplet_iter)
                except StopIteration:
                    # Re-create iterator
                    if triplet_iter is dating_iter:
                        dating_iter = iter(self.dating_loader)
                        batch = next(dating_iter)
                    else:
                        hiring_iter = iter(self.hiring_loader)
                        batch = next(hiring_iter)

                anchor_emb = self._encode_with_grad(batch["anchors"])
                pos_emb = self._encode_with_grad(batch["positives"])
                neg_emb = self._encode_with_grad(batch["negatives"])
                loss_inputs["anchor"] = anchor_emb
                loss_inputs["positive"] = pos_emb
                loss_inputs["negative"] = neg_emb

            # Cross-domain pairs
            if cross_iter is not None:
                try:
                    cross_batch = next(cross_iter)
                except StopIteration:
                    cross_iter = iter(self.cross_loader)
                    cross_batch = next(cross_iter)

                ca_emb = self._encode_with_grad(cross_batch["anchors"])
                cp_emb = self._encode_with_grad(cross_batch["positives"])

                # Use in-batch negatives (memory efficient — no explicit neg encoding)
                loss_inputs["cross_anchor"] = ca_emb
                loss_inputs["cross_positive"] = cp_emb

                # For adversarial: provide domain labels
                if self.cfg.method == Method.ADVERSARIAL:
                    all_emb = torch.cat([ca_emb, cp_emb], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(ca_emb.size(0)),
                        torch.ones(cp_emb.size(0)),
                    ]).long().to(self.device)
                    loss_inputs["domain_embeddings"] = all_emb
                    loss_inputs["domain_labels"] = domain_labels

            if not loss_inputs:
                continue

            losses = self.loss_fn(**loss_inputs)
            total = losses["total"]
            total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            n_steps += 1

            # Accumulate losses
            for k, v in losses.items():
                if k not in running:
                    running[k] = 0.0
                running[k] += v.item()

            if self.global_step % self.cfg.training.log_every == 0:
                avg_total = running.get("total", 0) / max(n_steps, 1)
                logger.info(f"  step {self.global_step}: loss={avg_total:.4f}")

        # Average
        return {k: v / max(n_steps, 1) for k, v in running.items()}
