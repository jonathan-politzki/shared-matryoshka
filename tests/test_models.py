"""Tests for model architectures."""

import pytest
import torch

from shared_matryoshka.config import ExperimentConfig, Method
from shared_matryoshka.models.factory import build_model


@pytest.fixture
def sample_texts():
    return [
        "A creative person who loves hiking and cooking.",
        "An organized professional with 5 years of experience.",
        "Outgoing individual seeking long-term relationship.",
    ]


class TestMatryoshkaModel:
    def test_build_v3_contrastive(self):
        cfg = ExperimentConfig(method=Method.V3_CONTRASTIVE)
        model = build_model(cfg)
        assert model.prefix_dim == 64
        assert model.embedding_dim == 384

    def test_encode_shape(self, sample_texts):
        cfg = ExperimentConfig(method=Method.V3_CONTRASTIVE)
        model = build_model(cfg)
        embs = model.encode(sample_texts)
        assert embs.shape == (3, 384)

    def test_encode_prefix_shape(self, sample_texts):
        cfg = ExperimentConfig(method=Method.V3_CONTRASTIVE)
        model = build_model(cfg)
        embs = model.encode_prefix(sample_texts)
        assert embs.shape == (3, 64)

    def test_encode_at_dim(self, sample_texts):
        cfg = ExperimentConfig(method=Method.V3_CONTRASTIVE)
        model = build_model(cfg)
        for dim in [32, 64, 128, 256, 384]:
            embs = model.encode_at_dim(sample_texts, dim=dim)
            assert embs.shape == (3, dim)

    def test_forward_preserves_grad(self, sample_texts):
        cfg = ExperimentConfig(method=Method.V3_CONTRASTIVE)
        model = build_model(cfg)
        embs = model.forward_from_texts(sample_texts)
        assert embs.requires_grad


class TestSingleDomainModel:
    def test_build_dating(self):
        cfg = ExperimentConfig(method=Method.SINGLE_DATING)
        model = build_model(cfg)
        assert model.domain_name == "dating"

    def test_build_hiring(self):
        cfg = ExperimentConfig(method=Method.SINGLE_HIRING)
        model = build_model(cfg)
        assert model.domain_name == "hiring"

    def test_encode_shape(self, sample_texts):
        cfg = ExperimentConfig(method=Method.SINGLE_DATING)
        model = build_model(cfg)
        embs = model.encode(sample_texts)
        assert embs.shape == (3, 384)


class TestProjectionHeadsModel:
    def test_build(self):
        cfg = ExperimentConfig(method=Method.PROJECTION_HEADS)
        model = build_model(cfg)
        assert model.prefix_dim == 64
        assert hasattr(model, "identity_head")
        assert hasattr(model, "task_head")

    def test_encode_prefix_shape(self, sample_texts):
        cfg = ExperimentConfig(method=Method.PROJECTION_HEADS)
        model = build_model(cfg)
        embs = model.encode_prefix(sample_texts)
        assert embs.shape == (3, 64)

    def test_encode_task_shape(self, sample_texts):
        cfg = ExperimentConfig(method=Method.PROJECTION_HEADS)
        model = build_model(cfg)
        embs = model.encode(sample_texts)
        assert embs.shape == (3, 384)


class TestAdversarialModel:
    def test_build(self):
        cfg = ExperimentConfig(method=Method.ADVERSARIAL)
        model = build_model(cfg)
        assert model.prefix_dim == 64

    def test_encode_shape(self, sample_texts):
        cfg = ExperimentConfig(method=Method.ADVERSARIAL)
        model = build_model(cfg)
        embs = model.encode(sample_texts)
        assert embs.shape == (3, 384)


class TestModelFactory:
    def test_all_methods_build(self):
        for method in Method:
            cfg = ExperimentConfig(method=method)
            model = build_model(cfg)
            assert model is not None
            assert model.prefix_dim > 0
            assert model.embedding_dim > 0
