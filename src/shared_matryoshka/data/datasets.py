"""PyTorch Datasets + train/val splitting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """Within-domain triplet dataset: (anchor_text, positive_text, negative_text)."""

    def __init__(self, texts: list[str], triplets: list[tuple[int, int, int]]):
        self.texts = texts
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, str]:
        a, p, n = self.triplets[idx]
        return {
            "anchor": self.texts[a],
            "positive": self.texts[p],
            "negative": self.texts[n],
        }


class CrossDomainDataset(Dataset):
    """Cross-domain identity dataset.

    Each sample: anchor from domain A, positive (same person) from domain B,
    negatives (other people) from domain B.
    """

    def __init__(
        self,
        texts_a: list[str],
        texts_b: list[str],
        pairs: list[tuple[int, int, list[int]]],
    ):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        anchor_id, pos_id, neg_ids = self.pairs[idx]
        return {
            "anchor": self.texts_a[anchor_id],
            "positive": self.texts_b[pos_id],
            "negatives": [self.texts_b[n] for n in neg_ids],
        }


def train_val_split(
    indices: list[int],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Deterministic train/val split of indices."""
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_frac)
    return shuffled[n_val:], shuffled[:n_val]


def save_generated_data(
    output_dir: str | Path,
    people_dicts: list[dict],
    dating_texts: list[str],
    hiring_texts: list[str],
    dating_triplets: list[tuple[int, int, int]],
    hiring_triplets: list[tuple[int, int, int]],
    cross_domain_pairs: list[tuple[int, int, list[int]]],
    train_ids: list[int],
    val_ids: list[int],
) -> None:
    """Save all generated data to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "people.json", "w") as f:
        json.dump(people_dicts, f, indent=2)
    with open(out / "dating_texts.json", "w") as f:
        json.dump(dating_texts, f, indent=2)
    with open(out / "hiring_texts.json", "w") as f:
        json.dump(hiring_texts, f, indent=2)
    with open(out / "dating_triplets.json", "w") as f:
        json.dump(dating_triplets, f)
    with open(out / "hiring_triplets.json", "w") as f:
        json.dump(hiring_triplets, f)
    with open(out / "cross_domain_pairs.json", "w") as f:
        json.dump(cross_domain_pairs, f)
    with open(out / "train_ids.json", "w") as f:
        json.dump(train_ids, f)
    with open(out / "val_ids.json", "w") as f:
        json.dump(val_ids, f)


def load_generated_data(data_dir: str | Path) -> dict[str, Any]:
    """Load generated data from disk."""
    d = Path(data_dir)
    with open(d / "people.json") as f:
        people = json.load(f)
    with open(d / "dating_texts.json") as f:
        dating_texts = json.load(f)
    with open(d / "hiring_texts.json") as f:
        hiring_texts = json.load(f)
    with open(d / "dating_triplets.json") as f:
        dating_triplets = [tuple(t) for t in json.load(f)]
    with open(d / "hiring_triplets.json") as f:
        hiring_triplets = [tuple(t) for t in json.load(f)]
    with open(d / "cross_domain_pairs.json") as f:
        cross_domain_pairs = [(p[0], p[1], p[2]) for p in json.load(f)]
    with open(d / "train_ids.json") as f:
        train_ids = json.load(f)
    with open(d / "val_ids.json") as f:
        val_ids = json.load(f)

    return {
        "people": people,
        "dating_texts": dating_texts,
        "hiring_texts": hiring_texts,
        "dating_triplets": dating_triplets,
        "hiring_triplets": hiring_triplets,
        "cross_domain_pairs": cross_domain_pairs,
        "train_ids": train_ids,
        "val_ids": val_ids,
    }
