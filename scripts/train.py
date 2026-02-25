#!/usr/bin/env python3
"""Train a model from config."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from torch.utils.data import DataLoader

from shared_matryoshka.config import load_config, Method
from shared_matryoshka.data.datasets import (
    load_generated_data,
    TripletDataset,
    CrossDomainDataset,
)
from shared_matryoshka.losses.factory import build_loss
from shared_matryoshka.models.factory import build_model
from shared_matryoshka.training.trainer import Trainer, _collate_triplets, _collate_cross_domain
from shared_matryoshka.utils import seed_everything, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    log = setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log.info(f"Training method: {cfg.method.value} ({cfg.name})")

    # Load data
    data = load_generated_data(args.data_dir)
    train_ids = set(data["train_ids"])

    dating_texts = data["dating_texts"]
    hiring_texts = data["hiring_texts"]

    # Filter triplets to training set
    def filter_triplets(triplets, ids):
        return [(a, p, n) for a, p, n in triplets if a in ids and p in ids and n in ids]

    def filter_cross_pairs(pairs, ids):
        return [
            (a, p, [n for n in negs if n in ids])
            for a, p, negs in pairs
            if a in ids
        ]

    # Build data loaders based on method
    bs = cfg.training.batch_size
    dating_loader = None
    hiring_loader = None
    cross_loader = None

    method = cfg.method
    needs_dating = method in (
        Method.V3_CONTRASTIVE, Method.V3_MSE, Method.V3_NO_PREFIX,
        Method.SINGLE_DATING, Method.PROJECTION_HEADS, Method.ADVERSARIAL,
    )
    needs_hiring = method in (
        Method.V3_CONTRASTIVE, Method.V3_MSE, Method.V3_NO_PREFIX,
        Method.SINGLE_HIRING, Method.PROJECTION_HEADS, Method.ADVERSARIAL,
    )
    needs_cross = method in (
        Method.V3_CONTRASTIVE, Method.V3_MSE,
        Method.PROJECTION_HEADS, Method.ADVERSARIAL,
    )

    if needs_dating:
        dt = filter_triplets(data["dating_triplets"], train_ids)
        dating_ds = TripletDataset(dating_texts, dt)
        dating_loader = DataLoader(
            dating_ds, batch_size=bs, shuffle=True, collate_fn=_collate_triplets
        )
        log.info(f"Dating triplets: {len(dt)}")

    if needs_hiring:
        ht = filter_triplets(data["hiring_triplets"], train_ids)
        hiring_ds = TripletDataset(hiring_texts, ht)
        hiring_loader = DataLoader(
            hiring_ds, batch_size=bs, shuffle=True, collate_fn=_collate_triplets
        )
        log.info(f"Hiring triplets: {len(ht)}")

    if needs_cross:
        cp = filter_cross_pairs(data["cross_domain_pairs"], train_ids)
        cross_ds = CrossDomainDataset(dating_texts, hiring_texts, cp)
        cross_loader = DataLoader(
            cross_ds, batch_size=bs, shuffle=True, collate_fn=_collate_cross_domain
        )
        log.info(f"Cross-domain pairs: {len(cp)}")

    # Build model and loss
    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        cfg=cfg,
        dating_triplet_loader=dating_loader,
        hiring_triplet_loader=hiring_loader,
        cross_domain_loader=cross_loader,
    )

    history = trainer.train()

    # Save history
    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{cfg.name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
