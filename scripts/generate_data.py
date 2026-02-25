#!/usr/bin/env python3
"""Generate synthetic training data."""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared_matryoshka.config import load_config, ExperimentConfig
from shared_matryoshka.data.generators import generate_people, render_people
from shared_matryoshka.data.compatibility import (
    dating_compatibility,
    hiring_compatibility,
    mine_triplets,
    mine_cross_domain_pairs,
)
from shared_matryoshka.data.datasets import save_generated_data, train_val_split
from shared_matryoshka.utils import seed_everything, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    log = setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)
    output_dir = args.output_dir or cfg.data.output_dir

    log.info(f"Generating {cfg.data.n_people} people with seed={cfg.data.seed}")
    people = generate_people(cfg.data.n_people, seed=cfg.data.seed)

    log.info("Rendering dating profiles and hiring resumes...")
    dating_texts, hiring_texts = render_people(people)

    log.info(f"Mining {cfg.data.triplets_per_domain} dating triplets...")
    dating_triplets = mine_triplets(
        people, dating_compatibility, cfg.data.triplets_per_domain, seed=cfg.data.seed
    )

    log.info(f"Mining {cfg.data.triplets_per_domain} hiring triplets...")
    hiring_triplets = mine_triplets(
        people, hiring_compatibility, cfg.data.triplets_per_domain, seed=cfg.data.seed + 1
    )

    log.info(f"Mining {cfg.data.cross_domain_pairs} cross-domain pairs...")
    cross_pairs = mine_cross_domain_pairs(
        people, cfg.data.cross_domain_pairs, seed=cfg.data.seed + 2
    )

    log.info("Splitting train/val...")
    all_ids = list(range(len(people)))
    train_ids, val_ids = train_val_split(all_ids, cfg.data.val_frac, seed=cfg.data.seed)
    log.info(f"  Train: {len(train_ids)}, Val: {len(val_ids)}")

    log.info(f"Saving to {output_dir}/")
    people_dicts = [dataclasses.asdict(p) for p in people]
    save_generated_data(
        output_dir,
        people_dicts,
        dating_texts,
        hiring_texts,
        dating_triplets,
        hiring_triplets,
        cross_pairs,
        train_ids,
        val_ids,
    )

    # Print samples
    log.info("\n--- Sample dating profile ---")
    log.info(dating_texts[0])
    log.info("\n--- Sample hiring resume (same person) ---")
    log.info(hiring_texts[0])
    log.info("\nData generation complete.")


if __name__ == "__main__":
    main()
