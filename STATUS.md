# Project Status

## What's Done

### Phase 1: Data Pipeline
- Rich synthetic person generation (1000 people) with coherent attributes
  - Big 5 personality (0-100, Gaussian distribution)
  - MBTI correlated with Big 5
  - Career ecosystem with coherent skill/industry mapping
  - Religion & politics with intensity levels (1-5)
  - Seniority derived from age/experience
- **8 dating profile templates** + **8 hiring resume templates** for lexical diversity
- Compatibility scoring with dealbreaker penalties
- Semi-hard triplet mining (5000/domain)
- Cross-domain identity pairs (5000)
- Hard negative generation via attribute flipping (religion, kids, smoking, politics, relationship style, lifestyle)
- Deterministic generation with seed control

### Phase 2: Models
- `SharedEmbeddingModel` ABC — all methods implement `encode()`, `encode_prefix()`, `encode_at_dim()`
- `MatryoshkaModel` — shared encoder for v3_contrastive, v3_mse, v3_no_prefix
- `SingleDomainModel` — dating-only and hiring-only baselines
- `ProjectionHeadsModel` — shared backbone + identity/task projection heads
- `AdversarialModel` — gradient reversal on prefix for domain invariance
- Model factory: config → model instance

### Phase 3: Losses
- `MatryoshkaInfoNCE` — within-domain loss at multiple matryoshka dimensions
- `PrefixInfoNCE` — cross-domain prefix alignment via InfoNCE
- `PrefixMSE` — ablation (expected collapse)
- `DomainAdversarialLoss` — gradient reversal layer + domain classifier
- Loss factory with `CombinedLoss` composer

### Phase 4: Training
- Custom training loop interleaving within-domain triplets + cross-domain pairs
- Linear warmup + cosine decay scheduler
- Gradient clipping, checkpointing
- Support for all 7 experimental conditions

### Phase 5: Evaluation
- 5 metric categories: cross-domain transfer, identity retrieval, within-domain accuracy, CKA, prefix variance
- Method-agnostic evaluator
- Console + LaTeX table formatting
- `run_all.py` end-to-end orchestration script

### Infrastructure
- 44 passing tests (data, models, evaluation)
- 8 YAML config files (7 conditions + base)
- Makefile for common operations
- `pyproject.toml` with editable install

---

## What Still Needs Work

### High Priority
- [ ] **End-to-end training validation** — smoke-test full training loop on GPU (Lambda instance). Local MPS works but is slow.
- [ ] **LLM-powered profile paraphrasing** — the old `contrastive-test` repo had Gemini-based paraphrasing that converts template profiles into natural first-person bios. This is the single biggest data quality upgrade.
- [ ] **Hard negative integration** — `generate_hard_negative()` exists but isn't wired into the training data pipeline yet. Needs to generate attribute-flip negatives and include them in triplets.

### Medium Priority
- [ ] **Training hyperparameter tuning** — current defaults are reasonable but untested at scale
- [ ] **Early stopping** — trainer doesn't yet support validation-based early stopping
- [ ] **Evaluation on val set during training** — `eval_every` config exists but evaluator isn't called mid-training
- [ ] **Results visualization** — loss curves, CKA heatmaps, t-SNE of prefix space
- [ ] **Paper LaTeX** — `paper/` directory is empty, needs the tex file

### Lower Priority
- [ ] **Multi-GPU support** — currently single device
- [ ] **Mixed precision training** — would speed up GPU training
- [ ] **Wandb/tensorboard logging** — currently stdout only
- [ ] **CI/CD** — GitHub Actions for tests
- [ ] **Docker/Lambda deployment script** — for reproducible training on cloud
