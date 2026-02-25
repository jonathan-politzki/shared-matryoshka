"""Tests for data generation pipeline."""

import pytest

from shared_matryoshka.data.generators import generate_people, render_dating_profile, render_hiring_resume
from shared_matryoshka.data.compatibility import (
    identity_score,
    dating_compatibility,
    hiring_compatibility,
    mine_triplets,
    mine_cross_domain_pairs,
)
from shared_matryoshka.data.datasets import train_val_split


class TestGenerators:
    def test_deterministic_generation(self):
        p1 = generate_people(10, seed=42)
        p2 = generate_people(10, seed=42)
        for a, b in zip(p1, p2):
            assert a.person_id == b.person_id
            assert a.openness == b.openness
            assert a.core_values == b.core_values

    def test_different_seeds_differ(self):
        p1 = generate_people(10, seed=42)
        p2 = generate_people(10, seed=99)
        # At least some traits should differ
        diffs = sum(a.openness != b.openness for a, b in zip(p1, p2))
        assert diffs > 0

    def test_generate_correct_count(self):
        people = generate_people(50, seed=42)
        assert len(people) == 50

    def test_person_ids_sequential(self):
        people = generate_people(20, seed=42)
        assert [p.person_id for p in people] == list(range(20))

    def test_traits_in_valid_range(self):
        people = generate_people(100, seed=42)
        for p in people:
            assert 0 <= p.openness <= 100
            assert 0 <= p.conscientiousness <= 100
            assert 0 <= p.extraversion <= 100
            assert 0 <= p.agreeableness <= 100
            assert 0 <= p.neuroticism <= 100

    def test_dating_rendering_nonempty(self):
        people = generate_people(5, seed=42)
        for p in people:
            text = render_dating_profile(p)
            assert len(text) > 50
            assert p.location in text

    def test_hiring_rendering_nonempty(self):
        people = generate_people(5, seed=42)
        for p in people:
            text = render_hiring_resume(p)
            assert len(text) > 50
            assert p.industry in text


class TestCompatibility:
    def test_identity_score_self(self):
        people = generate_people(5, seed=42)
        for p in people:
            score = identity_score(p, p)
            assert score > 0.8  # Self-similarity should be high

    def test_identity_score_range(self):
        people = generate_people(20, seed=42)
        for i in range(10):
            score = identity_score(people[i], people[i + 10])
            assert 0 <= score <= 1

    def test_dating_compatibility_range(self):
        people = generate_people(20, seed=42)
        for i in range(10):
            score = dating_compatibility(people[i], people[i + 10])
            assert 0 <= score <= 1

    def test_hiring_compatibility_range(self):
        people = generate_people(20, seed=42)
        for i in range(10):
            score = hiring_compatibility(people[i], people[i + 10])
            assert 0 <= score <= 1

    def test_mine_triplets(self):
        people = generate_people(50, seed=42)
        triplets = mine_triplets(people, dating_compatibility, 100, seed=42)
        assert len(triplets) == 100
        for a, p, n in triplets:
            assert a != p
            assert a != n
            assert 0 <= a < 50
            assert 0 <= p < 50
            assert 0 <= n < 50

    def test_mine_triplets_deterministic(self):
        people = generate_people(50, seed=42)
        t1 = mine_triplets(people, dating_compatibility, 50, seed=42)
        t2 = mine_triplets(people, dating_compatibility, 50, seed=42)
        assert t1 == t2

    def test_mine_cross_domain_pairs(self):
        people = generate_people(50, seed=42)
        pairs = mine_cross_domain_pairs(people, 100, seed=42)
        assert len(pairs) == 100
        for anchor, pos, negs in pairs:
            assert anchor == pos  # Same person
            assert anchor not in negs
            assert len(negs) > 0


class TestDatasets:
    def test_train_val_split_sizes(self):
        ids = list(range(100))
        train, val = train_val_split(ids, val_frac=0.2, seed=42)
        assert len(train) + len(val) == 100
        assert len(val) == 20

    def test_train_val_split_no_overlap(self):
        ids = list(range(100))
        train, val = train_val_split(ids, val_frac=0.15, seed=42)
        assert set(train) & set(val) == set()

    def test_train_val_split_deterministic(self):
        ids = list(range(100))
        t1, v1 = train_val_split(ids, val_frac=0.15, seed=42)
        t2, v2 = train_val_split(ids, val_frac=0.15, seed=42)
        assert t1 == t2
        assert v1 == v2
