"""Pydantic config models + YAML loading."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class Method(str, Enum):
    V3_CONTRASTIVE = "v3_contrastive"
    V3_MSE = "v3_mse"
    V3_NO_PREFIX = "v3_no_prefix"
    SINGLE_DATING = "single_dating"
    SINGLE_HIRING = "single_hiring"
    PROJECTION_HEADS = "projection_heads"
    ADVERSARIAL = "adversarial"


class DataConfig(BaseModel):
    n_people: int = 1000
    val_frac: float = 0.15
    seed: int = 42
    output_dir: str = "data"
    triplets_per_domain: int = 5000
    cross_domain_pairs: int = 5000


class ModelConfig(BaseModel):
    base_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    prefix_dim: int = 64
    matryoshka_dims: list[int] = [32, 64, 128, 256, 384]
    identity_head_dim: int = 64
    adversarial_hidden_dim: int = 128


class LossConfig(BaseModel):
    temperature: float = 0.07
    prefix_weight: float = 1.0
    within_domain_weight: float = 1.0
    cross_domain_weight: float = 1.0
    adversarial_weight: float = 0.1
    grl_lambda: float = 1.0


class TrainingConfig(BaseModel):
    epochs: int = 20
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    eval_every: int = 5
    save_dir: str = "checkpoints"
    log_every: int = 50


class EvalConfig(BaseModel):
    recall_k: list[int] = [1, 5, 10]
    results_dir: str = "results"


class ExperimentConfig(BaseModel):
    name: str = "v3_contrastive"
    method: Method = Method.V3_CONTRASTIVE
    seed: int = 42
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    eval: EvalConfig = EvalConfig()


def load_config(path: str | Path) -> ExperimentConfig:
    """Load config from YAML, merging with defaults."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig(**raw)


def load_configs_for_run(config_dir: str | Path = "configs") -> list[ExperimentConfig]:
    """Load all YAML configs from a directory."""
    config_dir = Path(config_dir)
    configs = []
    for p in sorted(config_dir.glob("*.yaml")):
        if p.name == "base.yaml":
            continue
        configs.append(load_config(p))
    return configs
