"""Tests for evaluation metrics."""

import pytest
import torch

from shared_matryoshka.evaluation.metrics import (
    recall_at_k,
    mrr,
    triplet_accuracy,
    cross_domain_accuracy,
    cross_domain_margin,
    cka,
    prefix_variance,
)


class TestRecall:
    def test_perfect_retrieval(self):
        # Same embeddings -> perfect recall
        embs = torch.randn(10, 64)
        ids = list(range(10))
        assert recall_at_k(embs, embs, ids, ids, k=1) == 1.0

    def test_recall_at_5_ge_recall_at_1(self):
        q = torch.randn(10, 64)
        g = torch.randn(10, 64)
        ids = list(range(10))
        r1 = recall_at_k(q, g, ids, ids, k=1)
        r5 = recall_at_k(q, g, ids, ids, k=5)
        assert r5 >= r1


class TestMRR:
    def test_perfect_retrieval(self):
        embs = torch.randn(10, 64)
        ids = list(range(10))
        assert mrr(embs, embs, ids, ids) == 1.0

    def test_mrr_in_range(self):
        q = torch.randn(20, 64)
        g = torch.randn(20, 64)
        ids = list(range(20))
        result = mrr(q, g, ids, ids)
        assert 0 <= result <= 1


class TestTripletAccuracy:
    def test_obvious_triplets(self):
        # anchor = positive, so anchor-positive distance is 0
        anchor = torch.randn(10, 64)
        positive = anchor.clone()
        negative = torch.randn(10, 64)
        acc = triplet_accuracy(anchor, positive, negative)
        assert acc == 1.0

    def test_range(self):
        a = torch.randn(100, 64)
        p = torch.randn(100, 64)
        n = torch.randn(100, 64)
        acc = triplet_accuracy(a, p, n)
        assert 0 <= acc <= 1


class TestCrossDomainAccuracy:
    def test_perfect_alignment(self):
        embs = torch.randn(10, 128)
        ids = list(range(10))
        acc = cross_domain_accuracy(embs, embs, ids, prefix_dim=64)
        assert acc == 1.0

    def test_range(self):
        d = torch.randn(20, 128)
        h = torch.randn(20, 128)
        ids = list(range(20))
        acc = cross_domain_accuracy(d, h, ids, prefix_dim=64)
        assert 0 <= acc <= 1


class TestCrossDomainMargin:
    def test_perfect_alignment_positive_margin(self):
        embs = torch.randn(10, 128)
        ids = list(range(10))
        margin = cross_domain_margin(embs, embs, ids, prefix_dim=64)
        assert margin > 0


class TestCKA:
    def test_identical_matrices(self):
        X = torch.randn(50, 32)
        val = cka(X, X)
        assert abs(val - 1.0) < 1e-4

    def test_range(self):
        X = torch.randn(50, 32)
        Y = torch.randn(50, 32)
        val = cka(X, Y)
        assert 0 <= val <= 1.0 + 1e-6


class TestPrefixVariance:
    def test_nonzero_for_random(self):
        embs = torch.randn(100, 128)
        var = prefix_variance(embs, prefix_dim=64)
        assert var > 0

    def test_zero_for_constant(self):
        # All embeddings identical -> zero variance
        embs = torch.ones(100, 128) * 0.5
        var = prefix_variance(embs, prefix_dim=64)
        assert abs(var) < 1e-6
