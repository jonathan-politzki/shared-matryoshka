# shared-matryoshka

Cross-domain prefix alignment of human representations via shared matryoshka embeddings.

**Paper**: "Shared Matryoshka Embeddings: Cross-Domain Prefix Alignment of Human Representations"

**Core idea**: Repurpose the matryoshka embedding prefix (first K dimensions) as a domain-invariant identity subspace via cross-domain contrastive training. Test: match someone on a dating platform using only their resume.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Generate synthetic data (1000 people, ~5k triplets/domain)
python3 scripts/generate_data.py --config configs/base.yaml

# Train a single method
python3 scripts/train.py --config configs/v3_contrastive.yaml

# Evaluate
python3 scripts/evaluate.py --config configs/v3_contrastive.yaml

# Run all 7 conditions end-to-end
python3 scripts/run_all.py

# Run tests
pytest tests/ -v
```

## Experimental Conditions

| Method | Description |
|--------|-------------|
| `v3_contrastive` | **Proposed method** — InfoNCE prefix alignment |
| `v3_mse` | Ablation — MSE prefix (expect collapse) |
| `v3_no_prefix` | Ablation — joint training, no prefix loss |
| `single_dating` | Baseline — dating-only model |
| `single_hiring` | Baseline — hiring-only model |
| `projection_heads` | Baseline — shared backbone + identity/task heads |
| `adversarial` | Baseline — gradient reversal domain adaptation |

## Project Structure

```
configs/         # YAML experiment configs
src/shared_matryoshka/
  config.py      # Pydantic config models
  utils.py       # Seeding, device, logging
  data/          # Person generation, compatibility scoring, datasets
  models/        # SharedEmbeddingModel ABC + all architectures
  losses/        # InfoNCE, cross-domain prefix losses, adversarial
  training/      # Custom training loop
  evaluation/    # Metrics, evaluator, table formatting
scripts/         # Entry points: generate_data, train, evaluate, run_all
tests/           # pytest suite
paper/           # LaTeX source
```

## Key Metrics

1. **Cross-domain transfer** (primary) — match dating profiles to hiring resumes via prefix
2. **Identity retrieval** — Recall@K, MRR across domains
3. **Within-domain accuracy** — dating/hiring triplet accuracy (must not degrade)
4. **CKA across dimensions** — alignment at each matryoshka dim
5. **Prefix variance** — collapse diagnostic

## Base Model

`BAAI/bge-small-en-v1.5` (33M params, 384 dims). Prefix K=64, matryoshka dims {32, 64, 128, 256, 384}.
