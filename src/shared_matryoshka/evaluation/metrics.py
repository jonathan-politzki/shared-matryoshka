"""Pure metric functions: recall@k, MRR, CKA, prefix variance, accuracy."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def recall_at_k(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    query_ids: list[int],
    gallery_ids: list[int],
    k: int = 1,
) -> float:
    """Recall@K for identity retrieval.

    For each query, check if the correct identity appears in the top-K nearest neighbors.

    Args:
        query_embs: (N, D) normalized query embeddings.
        gallery_embs: (M, D) normalized gallery embeddings.
        query_ids: Person IDs for queries.
        gallery_ids: Person IDs for gallery.
        k: Number of top results to check.

    Returns:
        Fraction of queries where the correct ID is in the top-K.
    """
    q = F.normalize(query_embs, dim=-1)
    g = F.normalize(gallery_embs, dim=-1)
    sims = torch.mm(q, g.t())  # (N, M)
    topk_indices = sims.topk(k, dim=1).indices  # (N, K)

    gallery_ids_arr = np.array(gallery_ids)
    hits = 0
    for i, qid in enumerate(query_ids):
        topk_gids = gallery_ids_arr[topk_indices[i].cpu().numpy()]
        if qid in topk_gids:
            hits += 1

    return hits / len(query_ids)


def mrr(
    query_embs: torch.Tensor,
    gallery_embs: torch.Tensor,
    query_ids: list[int],
    gallery_ids: list[int],
) -> float:
    """Mean Reciprocal Rank for identity retrieval."""
    q = F.normalize(query_embs, dim=-1)
    g = F.normalize(gallery_embs, dim=-1)
    sims = torch.mm(q, g.t())  # (N, M)
    sorted_indices = sims.argsort(dim=1, descending=True)

    gallery_ids_arr = np.array(gallery_ids)
    rr_sum = 0.0
    for i, qid in enumerate(query_ids):
        ranking = gallery_ids_arr[sorted_indices[i].cpu().numpy()]
        matches = np.where(ranking == qid)[0]
        if len(matches) > 0:
            rr_sum += 1.0 / (matches[0] + 1)

    return rr_sum / len(query_ids)


def triplet_accuracy(
    anchor_embs: torch.Tensor,
    positive_embs: torch.Tensor,
    negative_embs: torch.Tensor,
) -> float:
    """Fraction of triplets where anchor is closer to positive than negative."""
    a = F.normalize(anchor_embs, dim=-1)
    p = F.normalize(positive_embs, dim=-1)
    n = F.normalize(negative_embs, dim=-1)

    pos_sim = (a * p).sum(dim=-1)
    neg_sim = (a * n).sum(dim=-1)
    correct = (pos_sim > neg_sim).float().mean()
    return correct.item()


def cross_domain_accuracy(
    dating_embs: torch.Tensor,
    hiring_embs: torch.Tensor,
    person_ids: list[int],
    prefix_dim: int | None = None,
) -> float:
    """Zero-shot cross-domain accuracy.

    For each dating profile, find nearest hiring resume via prefix embeddings.
    Success = same person_id.

    Args:
        dating_embs: (N, D) dating embeddings.
        hiring_embs: (N, D) hiring embeddings.
        person_ids: Person IDs aligned with both matrices.
        prefix_dim: If set, use only first prefix_dim dimensions.
    """
    if prefix_dim is not None:
        d = dating_embs[:, :prefix_dim]
        h = hiring_embs[:, :prefix_dim]
    else:
        d = dating_embs
        h = hiring_embs

    d = F.normalize(d, dim=-1)
    h = F.normalize(h, dim=-1)

    sims = torch.mm(d, h.t())  # (N, N)
    predictions = sims.argmax(dim=1).cpu().numpy()

    ids = np.array(person_ids)
    correct = sum(ids[i] == ids[predictions[i]] for i in range(len(ids)))
    return correct / len(ids)


def cross_domain_margin(
    dating_embs: torch.Tensor,
    hiring_embs: torch.Tensor,
    person_ids: list[int],
    prefix_dim: int | None = None,
) -> float:
    """Average margin between correct and best-wrong cross-domain match."""
    if prefix_dim is not None:
        d = dating_embs[:, :prefix_dim]
        h = hiring_embs[:, :prefix_dim]
    else:
        d = dating_embs
        h = hiring_embs

    d = F.normalize(d, dim=-1)
    h = F.normalize(h, dim=-1)

    sims = torch.mm(d, h.t())  # (N, N)
    ids = np.array(person_ids)

    margins = []
    for i in range(len(ids)):
        correct_sim = sims[i, i].item()
        # Mask out correct match
        mask = torch.ones(sims.size(1), dtype=torch.bool)
        mask[i] = False
        best_wrong = sims[i, mask].max().item()
        margins.append(correct_sim - best_wrong)

    return float(np.mean(margins))


def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear Centered Kernel Alignment between two representation matrices.

    Args:
        X: (N, D1) matrix.
        Y: (N, D2) matrix.

    Returns:
        CKA similarity in [0, 1].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    XtX = torch.mm(X.t(), X)
    YtY = torch.mm(Y.t(), Y)
    XtY = torch.mm(X.t(), Y)

    hsic_xy = (XtY ** 2).sum()
    hsic_xx = (XtX ** 2).sum()
    hsic_yy = (YtY ** 2).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


def cka_across_dims(
    dating_embs: torch.Tensor,
    hiring_embs: torch.Tensor,
    dims: list[int],
) -> dict[int, float]:
    """CKA between dating/hiring matrices at each matryoshka dimension."""
    results = {}
    for d in dims:
        results[d] = cka(dating_embs[:, :d], hiring_embs[:, :d])
    return results


def prefix_variance(embs: torch.Tensor, prefix_dim: int) -> float:
    """Trace of prefix covariance matrix. Collapse → 0."""
    prefix = embs[:, :prefix_dim]
    prefix = prefix - prefix.mean(dim=0, keepdim=True)
    cov = torch.mm(prefix.t(), prefix) / (prefix.size(0) - 1)
    return cov.trace().item()
