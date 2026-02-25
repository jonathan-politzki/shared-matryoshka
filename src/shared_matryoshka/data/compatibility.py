"""Compatibility scoring + triplet mining for training data.

Scoring functions use the 0-100 Big 5 scale and intensity-weighted values.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

import numpy as np

from .schema import PersonSchema


def personality_similarity(a: PersonSchema, b: PersonSchema) -> float:
    """Cosine similarity of Big 5 vectors (0-100 scale)."""
    va = np.array([a.openness, a.conscientiousness, a.extraversion,
                    a.agreeableness, a.neuroticism], dtype=float)
    vb = np.array([b.openness, b.conscientiousness, b.extraversion,
                    b.agreeableness, b.neuroticism], dtype=float)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm < 1e-8:
        return 0.0
    return float(dot / norm)


def values_overlap(a: PersonSchema, b: PersonSchema) -> float:
    """Jaccard overlap of core values."""
    sa, sb = set(a.core_values), set(b.core_values)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def style_match(a: PersonSchema, b: PersonSchema) -> float:
    """Weighted match on communication + attachment style."""
    score = 0.0
    if a.communication_style == b.communication_style:
        score += 0.5
    if a.attachment_style == b.attachment_style:
        score += 0.5
    return score


def mbti_similarity(a: PersonSchema, b: PersonSchema) -> float:
    """MBTI letter overlap (0-4 matching letters / 4)."""
    matches = sum(ca == cb for ca, cb in zip(a.myers_briggs, b.myers_briggs))
    return matches / 4.0


def identity_score(a: PersonSchema, b: PersonSchema) -> float:
    """Combined identity similarity (domain-invariant traits only).

    This is what the prefix subspace should capture.
    Weights: personality (0.35) + values (0.25) + MBTI (0.15) +
             style (0.15) + religion/politics alignment (0.10)
    """
    base = (
        0.35 * personality_similarity(a, b)
        + 0.25 * values_overlap(a, b)
        + 0.15 * mbti_similarity(a, b)
        + 0.15 * style_match(a, b)
    )

    # Religion alignment (weighted by intensity)
    rel_match = 0.0
    if a.religion == b.religion:
        rel_match = min(a.religion_intensity, b.religion_intensity) / 5.0
    elif a.religion_intensity <= 2 and b.religion_intensity <= 2:
        rel_match = 0.5  # Both don't care much

    # Politics alignment (weighted by intensity)
    pol_match = 0.0
    if a.politics == b.politics:
        pol_match = min(a.politics_intensity, b.politics_intensity) / 5.0
    elif a.politics_intensity <= 2 and b.politics_intensity <= 2:
        pol_match = 0.5

    base += 0.05 * rel_match + 0.05 * pol_match
    return min(1.0, max(0.0, base))


def dating_compatibility(a: PersonSchema, b: PersonSchema) -> float:
    """Dating compatibility score combining identity + dating-specific traits."""
    base = identity_score(a, b)

    # Relationship goal alignment
    goal_bonus = 0.12 if a.relationship_goal == b.relationship_goal else 0.0

    # Lifestyle overlap
    la, lb = set(a.lifestyle), set(b.lifestyle)
    lifestyle_sim = len(la & lb) / max(len(la | lb), 1) * 0.08

    # Interest overlap
    ia, ib = set(a.interests), set(b.interests)
    interest_sim = len(ia & ib) / max(len(ia | ib), 1) * 0.08

    # Relationship style compatibility
    rs_bonus = 0.05 if a.relationship_style == b.relationship_style else 0.0

    # Kids alignment
    kids_bonus = 0.05 if a.kids_wanted == b.kids_wanted else 0.0

    # Smoking compatibility
    smoking_penalty = 0.0
    if a.smoking != b.smoking:
        smoking_penalty = 0.03

    # Dealbreaker check — hard penalty
    dealbreaker_penalty = 0.0
    for db in a.dealbreakers:
        if db == "smoking" and b.smoking == "regular smoker":
            dealbreaker_penalty += 0.15
        elif db == "different religion" and a.religion != b.religion and a.religion_intensity >= 3:
            dealbreaker_penalty += 0.15
        elif db == "different politics" and a.politics != b.politics and a.politics_intensity >= 3:
            dealbreaker_penalty += 0.10
        elif db == "doesn't want kids" and b.kids_wanted == "none" and a.kids_wanted != "none":
            dealbreaker_penalty += 0.15

    return min(1.0, max(0.0,
        base + goal_bonus + lifestyle_sim + interest_sim + rs_bonus + kids_bonus
        - smoking_penalty - dealbreaker_penalty
    ))


def hiring_compatibility(a: PersonSchema, b: PersonSchema) -> float:
    """Hiring compatibility (would these people work well together?)."""
    base = identity_score(a, b)

    # Work style compatibility
    work_bonus = 0.08 if a.work_style == b.work_style else 0.0

    # Team preference
    team_bonus = 0.05 if a.team_preference == b.team_preference else 0.0

    # Skill complementarity (some overlap is good, too much means redundancy)
    sa, sb = set(a.skills), set(b.skills)
    if sa | sb:
        overlap = len(sa & sb) / len(sa | sb)
        skill_score = 0.08 * (1.0 - abs(overlap - 0.4) / 0.6)
    else:
        skill_score = 0.0

    # Industry match
    industry_bonus = 0.04 if a.industry == b.industry else 0.0

    return min(1.0, max(0.0, base + work_bonus + team_bonus + skill_score + industry_bonus))


def mine_triplets(
    people: Sequence[PersonSchema],
    score_fn: Callable,
    n_triplets: int,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Mine (anchor, positive, negative) triplets using score_fn.

    Semi-hard mining: positive has high score, negative has lower score,
    but not necessarily the absolute hardest negative.
    """
    rng = random.Random(seed)
    n = len(people)
    triplets = []

    for _ in range(n_triplets):
        anchor_idx = rng.randint(0, n - 1)
        candidates = rng.sample(range(n), min(20, n))
        candidates = [c for c in candidates if c != anchor_idx]
        if len(candidates) < 2:
            continue

        scores = [(c, score_fn(people[anchor_idx], people[c])) for c in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)

        pos_idx = scores[0][0]
        # Semi-hard: pick negative from lower half
        neg_pool = scores[len(scores) // 2:]
        neg_idx = rng.choice(neg_pool)[0]

        triplets.append((anchor_idx, pos_idx, neg_idx))

    return triplets


def mine_cross_domain_pairs(
    people: Sequence[PersonSchema],
    n_pairs: int,
    seed: int = 42,
) -> list[tuple[int, int, list[int]]]:
    """Mine cross-domain identity pairs: (person_i, person_i, [negative_ids]).

    Each pair: same person, different domain. Negatives are other people.
    Returns (anchor_id, positive_id (same), negative_ids).
    """
    rng = random.Random(seed)
    n = len(people)
    pairs = []

    for _ in range(n_pairs):
        anchor_idx = rng.randint(0, n - 1)
        neg_pool = [j for j in range(n) if j != anchor_idx]
        neg_ids = rng.sample(neg_pool, min(15, len(neg_pool)))
        pairs.append((anchor_idx, anchor_idx, neg_ids))

    return pairs
