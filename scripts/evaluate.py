#!/usr/bin/env python3
"""Evaluate a trained model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from shared_matryoshka.config import load_config
from shared_matryoshka.data.datasets import load_generated_data
from shared_matryoshka.evaluation.evaluator import Evaluator
from shared_matryoshka.models.factory import build_model
from shared_matryoshka.utils import seed_everything, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log.info(f"Evaluating method: {cfg.method.value} ({cfg.name})")

    # Load data
    data = load_generated_data(args.data_dir)
    val_ids = data["val_ids"]

    dating_texts = [data["dating_texts"][i] for i in val_ids]
    hiring_texts = [data["hiring_texts"][i] for i in val_ids]
    person_ids = val_ids

    # Build validation triplets (indices relative to val set)
    val_id_set = set(val_ids)
    val_id_to_local = {gid: i for i, gid in enumerate(val_ids)}

    def remap_triplets(triplets):
        remapped = []
        for a, p, n in triplets:
            if a in val_id_set and p in val_id_set and n in val_id_set:
                remapped.append((val_id_to_local[a], val_id_to_local[p], val_id_to_local[n]))
        return remapped

    dating_triplets = remap_triplets(data["dating_triplets"])
    hiring_triplets = remap_triplets(data["hiring_triplets"])

    # Build and load model
    model = build_model(cfg)
    ckpt_path = args.checkpoint or str(
        Path(cfg.training.save_dir) / cfg.name / "model.pt"
    )
    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        log.info(f"Loaded checkpoint: {ckpt_path}")
    else:
        log.warning(f"No checkpoint found at {ckpt_path}, evaluating untrained model")

    # Evaluate
    evaluator = Evaluator(model, cfg)
    results = evaluator.evaluate(
        dating_texts=dating_texts,
        hiring_texts=hiring_texts,
        person_ids=person_ids,
        dating_triplets=dating_triplets,
        hiring_triplets=hiring_triplets,
    )

    # Save results
    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{cfg.name}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved to {out_path}")
    for k, v in sorted(results.items()):
        log.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
