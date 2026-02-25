# Architecture

Deep dive into the codebase structure, data flow, and design decisions.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_all.py                                │
│   generate_data.py  →  train.py (×7)  →  evaluate.py (×7)      │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────┐      ┌──────────────┐      ┌────────────┐
    │  Data    │      │   Training   │      │ Evaluation │
    │ Pipeline │      │    Loop      │      │  Pipeline  │
    └─────────┘      └──────────────┘      └────────────┘
```

## Data Pipeline

```
PersonSchema (structured attributes)
         │
         ├── generate_people(n, seed) ──→ List[PersonSchema]
         │      • Gaussian Big 5 (0-100)
         │      • MBTI correlated with Big 5
         │      • Career → skills/industry coherence
         │      • Seniority from age/experience
         │
         ├── render_dating_profile(person) ──→ str
         │      • 8 templates (rotated by person_id)
         │      • Templates: intro-first, personality-led, values-first,
         │        casual, interests-led, brief, narrative, psychology-forward
         │
         ├── render_hiring_resume(person) ──→ str
         │      • 8 templates (offset by +3 from dating)
         │      • Templates: resume-style, skills-first, narrative bio,
         │        culture-fit, LinkedIn, brief, ambition-led, team-fit
         │
         ├── mine_triplets(people, score_fn, n) ──→ [(anchor, pos, neg)]
         │      • Semi-hard mining: sample 20 candidates per anchor
         │      • Positive = highest score, negative from lower half
         │
         └── mine_cross_domain_pairs(people, n) ──→ [(id, id, [neg_ids])]
                • Same person across domains = positive
                • Random other people = negatives (15 per pair)
```

### Compatibility Scoring

```
identity_score(a, b) ──→ [0, 1]
├── 0.35 × personality_similarity  (Big 5 cosine)
├── 0.25 × values_overlap          (Jaccard on core_values)
├── 0.15 × mbti_similarity         (letter matching)
├── 0.15 × style_match             (communication + attachment)
└── 0.10 × religion/politics       (intensity-weighted)

dating_compatibility = identity_score + goal + lifestyle + interests
                       + relationship_style + kids - dealbreakers

hiring_compatibility = identity_score + work_style + team
                       + skills (peak at 40% overlap) + industry
```

## Model Architecture

All models implement the `SharedEmbeddingModel` ABC:

```python
class SharedEmbeddingModel(ABC):
    encode(texts, domain) → Tensor[B, D]         # Full embeddings
    encode_prefix(texts, domain) → Tensor[B, K]  # Identity subspace
    encode_at_dim(texts, domain, dim) → Tensor[B, dim]  # Matryoshka slice
```

### Model Variants

```
                    ┌─────────────────────────┐
                    │   BGE-small-en-v1.5     │
                    │   (33M params, 384d)     │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
    ┌────▼────┐           ┌─────▼─────┐           ┌────▼────────┐
    │Matryoshka│           │Projection │           │Adversarial  │
    │  Model   │           │  Heads    │           │   Model     │
    │          │           │           │           │             │
    │ dim 0:K  │           │ ┌───────┐ │           │ dim 0:K ◄──┤ GRL
    │ = prefix │           │ │Identity│ │           │ = prefix    │
    │ (shared  │           │ │  Head  │ │           │             │
    │  across  │           │ ├───────┤ │           └─────────────┘
    │  domains)│           │ │ Task   │ │
    └──────────┘           │ │ Head   │ │
                           │ └───────┘ │
    Used by:               └───────────┘
    • v3_contrastive
    • v3_mse                Used by:
    • v3_no_prefix          • projection_heads
    • single_dating
    • single_hiring
```

### Embedding Structure (Matryoshka)

```
Full embedding: 384 dimensions
├── [0:32]   ─── Coarsest matryoshka level
├── [0:64]   ─── PREFIX (identity subspace, K=64)  ◄── Cross-domain alignment
├── [0:128]  ─── Medium granularity
├── [0:256]  ─── Fine granularity
└── [0:384]  ─── Full embedding (domain-specific details)

The KEY INSIGHT: dims 0-63 encode WHO someone is.
                 dims 64-383 encode domain-specific context.
```

## Loss Architecture

```
CombinedLoss
├── within_domain_loss: MatryoshkaInfoNCE
│     • Applied at dims {32, 64, 128, 256, 384}
│     • In-batch negatives
│     • Loss = mean across dimensions
│
├── cross_domain_loss: PrefixInfoNCE | PrefixMSE
│     • PrefixInfoNCE: contrastive on dims 0:K across domains
│     • PrefixMSE: MSE on dims 0:K (ablation — expect collapse)
│
└── adversarial_loss: DomainAdversarialLoss (optional)
      • Gradient Reversal Layer on prefix
      • Binary domain classifier (dating=0, hiring=1)
      • Trains encoder to confuse classifier
```

### Loss Configuration by Method

```
Method              Within  Cross-domain  Adversarial  Prefix Loss
─────────────────   ──────  ────────────  ───────────  ────────────
v3_contrastive       ✓       InfoNCE        —          InfoNCE 0:K
v3_mse               ✓       MSE            —          MSE 0:K
v3_no_prefix         ✓        —             —           —
single_dating        ✓        —             —           —
single_hiring        ✓        —             —           —
projection_heads     ✓       InfoNCE        —          InfoNCE (head)
adversarial          ✓       InfoNCE        GRL         InfoNCE 0:K
```

## Training Loop

```
for epoch in 1..N:
    for step in 1..steps_per_epoch:

        ┌─ Within-domain (alternating) ─────────────────────┐
        │  step % 2 == 0 → dating triplet batch             │
        │  step % 2 == 1 → hiring triplet batch             │
        │                                                     │
        │  anchor_emb  = model.forward_from_texts(anchors)   │
        │  pos_emb     = model.forward_from_texts(positives) │
        │  neg_emb     = model.forward_from_texts(negatives) │
        └─────────────────────────────────────────────────────┘

        ┌─ Cross-domain ────────────────────────────────────┐
        │  cross_anchor = model.forward(dating_texts)        │
        │  cross_pos    = model.forward(hiring_texts_same)   │
        │  cross_neg    = model.forward(hiring_texts_other)  │
        └────────────────────────────────────────────────────┘

        loss = CombinedLoss(
            anchor, pos, neg,           # within-domain
            cross_anchor, cross_pos,    # cross-domain
            cross_neg                   # cross-domain negatives
        )
        loss.backward()
        clip_grad_norm()
        optimizer.step()
        scheduler.step()
```

## Evaluation Pipeline

```
Evaluator.evaluate()
│
├── Encode all val dating texts → dating_embs [N, 384]
├── Encode all val hiring texts → hiring_embs [N, 384]
│
├── 1. Cross-Domain Transfer (PRIMARY)
│   ├── accuracy: argmax(sim(dating[0:K], hiring[0:K]))
│   └── margin: correct_sim - best_wrong_sim
│
├── 2. Identity Retrieval
│   ├── recall@1, @5, @10 on prefix
│   └── MRR on prefix
│
├── 3. Within-Domain Accuracy
│   ├── dating triplet accuracy (full dims)
│   └── hiring triplet accuracy (full dims)
│
├── 4. CKA Across Dimensions
│   └── CKA(dating[0:d], hiring[0:d]) for d in {32,64,128,256,384}
│       → expect "elbow" at K=64
│
└── 5. Collapse Diagnostic
    └── trace(cov(prefix)) → should be > 0
        → v3_mse expected to collapse to ~0
```

## Config System

```yaml
# configs/v3_contrastive.yaml
name: v3_contrastive          # Experiment name
method: v3_contrastive        # → model factory + loss factory

model:
  base_model: BAAI/bge-small-en-v1.5
  prefix_dim: 64              # K = identity subspace size
  matryoshka_dims: [32, 64, 128, 256, 384]

loss:
  temperature: 0.07           # InfoNCE temperature
  prefix_weight: 1.0          # Weight on cross-domain loss
  within_domain_weight: 1.0   # Weight on triplet loss
```

Pydantic validates all configs at load time. No Hydra — 8 fixed conditions don't need it.

## File Map

```
src/shared_matryoshka/
├── config.py              Pydantic models, YAML loader
├── utils.py               seed_everything, get_device, setup_logging
├── data/
│   ├── schema.py          PersonSchema dataclass
│   ├── generators.py      Person gen + 16 text templates + hard negatives
│   ├── compatibility.py   Scoring functions + triplet mining
│   └── datasets.py        PyTorch Datasets + save/load
├── models/
│   ├── base.py            SharedEmbeddingModel ABC
│   ├── matryoshka.py      V3 shared encoder
│   ├── single_domain.py   Single-domain baselines
│   ├── projection_heads.py  Shared backbone + heads
│   ├── adversarial.py     GRL-based model
│   └── factory.py         config.method → model
├── losses/
│   ├── infonce.py         MatryoshkaInfoNCE
│   ├── cross_domain.py    PrefixInfoNCE + PrefixMSE
│   ├── adversarial.py     GRL + domain classifier
│   └── factory.py         config → CombinedLoss
├── training/
│   └── trainer.py         Training loop
└── evaluation/
    ├── metrics.py         Pure functions
    ├── evaluator.py       Orchestrator
    └── tables.py          Console + LaTeX formatting
```
