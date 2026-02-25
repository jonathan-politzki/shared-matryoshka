"""PersonSchema dataclass — rich domain-invariant + domain-specific attributes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PersonSchema:
    """A synthetic person with both dating and hiring attributes.

    Domain-invariant traits (what makes someone *them*):
        Big 5, MBTI, values, communication style, attachment style

    Domain-specific traits differ in framing but derive from the same person:
        Dating: relationship goals, lifestyle, dealbreakers
        Hiring: skills, work experience, team preferences
    """

    person_id: int

    # ── Demographics ─────────────────────────────────────────────────────
    age: int = 28
    sex: str = "male"  # "male" or "female"
    location: str = ""
    height_inches: float = 70.0
    ethnicity: str = ""

    # ── Core personality (domain-invariant) ──────────────────────────────
    # Big 5: 0-100 scale (more granular than 0-1)
    openness: int = 50
    conscientiousness: int = 50
    extraversion: int = 50
    agreeableness: int = 50
    neuroticism: int = 50

    myers_briggs: str = "INTJ"
    core_values: list[str] = field(default_factory=list)
    communication_style: str = ""  # "direct", "diplomatic", "analytical", "expressive"
    attachment_style: str = ""  # "secure", "anxious", "avoidant", "fearful"

    # Career & ambitions (shared across domains, framed differently)
    career: str = ""
    ambitions: str = ""
    education: str = ""
    income_bracket: str = ""  # "modest", "medium", "high", "very_high"

    # ── Dating-specific ──────────────────────────────────────────────────
    relationship_goal: str = ""
    lifestyle: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    dealbreakers: list[str] = field(default_factory=list)

    # Values with intensity (not just binary)
    religion: str = ""
    religion_intensity: int = 1  # 1-5
    politics: str = ""
    politics_intensity: int = 1  # 1-5
    smoking: str = "non-smoker"
    relationship_style: str = "monogamous"

    kids_current: int = 0
    kids_wanted: str = ""  # "none", "1-2", "3+"

    # ── Hiring-specific ──────────────────────────────────────────────────
    skills: list[str] = field(default_factory=list)
    experience_years: int = 0
    work_style: str = ""  # "remote", "hybrid", "in-office", "flexible"
    industry: str = ""
    role_level: str = ""  # "junior", "mid", "senior", "lead", "executive"
    team_preference: str = ""  # "small", "medium", "large"

    # ── Generated text (filled by rendering functions) ───────────────────
    dating_text: str = ""
    hiring_text: str = ""
