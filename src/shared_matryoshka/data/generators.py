"""Rich person generation + multi-template text rendering for dating and hiring domains.

Key improvements over naive template-fill:
1. Coherent attribute generation (skills match industry, seniority matches experience)
2. Big 5 on 0-100 scale with Gaussian distribution (not uniform)
3. MBTI types with personality-consistent generation
4. Multiple rendering templates per domain (8 dating, 8 hiring) for lexical diversity
5. Religion/politics with intensity levels (not just labels)
6. Hard negative support via attribute-flip cloning
"""

from __future__ import annotations

import copy
import math
import random
from typing import Sequence

from .schema import PersonSchema

# ── Attribute pools ──────────────────────────────────────────────────────────

LOCATIONS = [
    "Austin, TX", "San Francisco, CA", "Brooklyn, NY", "Chicago, IL",
    "Miami, FL", "Denver, CO", "Seattle, WA", "Nashville, TN",
    "Boston, MA", "Portland, OR", "Los Angeles, CA", "Atlanta, GA",
    "Minneapolis, MN", "Washington, DC", "San Diego, CA", "Dallas, TX",
    "Philadelphia, PA", "Phoenix, AZ",
]

ETHNICITIES = ["White", "Black", "Asian", "Hispanic", "Middle Eastern", "Mixed"]

CORE_VALUES = [
    "growth", "honesty", "loyalty", "creativity", "stability",
    "adventure", "compassion", "independence", "community", "excellence",
    "humor", "authenticity", "ambition", "balance", "curiosity",
    "faith", "family", "justice", "freedom", "service",
]

COMMUNICATION_STYLES = ["direct", "diplomatic", "analytical", "expressive"]
ATTACHMENT_STYLES = ["secure", "anxious", "avoidant", "fearful"]

MYERS_BRIGGS_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP",
]

MBTI_DESCRIPTORS = {
    "INTJ": "strategic and independent", "INTP": "analytical and curious",
    "ENTJ": "decisive and ambitious", "ENTP": "innovative and quick-witted",
    "INFJ": "compassionate and visionary", "INFP": "idealistic and empathetic",
    "ENFJ": "charismatic and inspiring", "ENFP": "enthusiastic and creative",
    "ISTJ": "reliable and responsible", "ISFJ": "warm and dependable",
    "ESTJ": "organized and efficient", "ESFJ": "sociable and caring",
    "ISTP": "practical and observant", "ISFP": "gentle and creative",
    "ESTP": "energetic and pragmatic", "ESFP": "spontaneous and fun-loving",
}

RELIGIONS = [
    "Christian", "Catholic", "Jewish", "Muslim", "Hindu",
    "Buddhist", "Spiritual", "Non-religious", "Agnostic",
]

POLITICS = ["Liberal", "Moderate", "Conservative", "Libertarian", "Apolitical"]

RELATIONSHIP_GOALS = ["long-term", "casual", "marriage", "exploring"]
RELATIONSHIP_STYLES = ["monogamous", "open-minded", "polyamorous"]
SMOKING_VALUES = ["non-smoker", "social smoker", "regular smoker"]

LIFESTYLES = [
    "outdoorsy", "urban", "homebody", "traveler", "fitness-oriented",
    "artsy", "foodie", "minimalist", "social butterfly", "bookworm",
    "wellness-focused", "nightlife", "family-oriented", "tech-savvy",
]

INTERESTS = [
    "hiking", "cooking", "reading", "gaming", "music", "photography",
    "yoga", "travel", "movies", "dancing", "volunteering", "gardening",
    "coding", "sports", "art", "writing", "meditation", "board games",
    "wine tasting", "running", "rock climbing", "surfing", "investing",
    "podcasts", "theater", "fashion", "camping", "cycling",
]

EDUCATION_LEVELS = [
    "High school", "Some college", "Bachelor's degree",
    "Master's degree", "PhD", "Professional degree (MD, JD)",
]

# ── Career ecosystem (coherent skills/industry mapping) ──────────────────────

CAREER_ECOSYSTEM = {
    "Software engineer": {
        "industry": "technology",
        "skills": ["Python", "system design", "algorithms", "code review", "cloud infrastructure"],
    },
    "Product manager": {
        "industry": "technology",
        "skills": ["roadmapping", "stakeholder management", "data analysis", "prioritization", "user research"],
    },
    "Data scientist": {
        "industry": "technology",
        "skills": ["machine learning", "statistics", "Python", "data visualization", "SQL"],
    },
    "Corporate lawyer": {
        "industry": "legal",
        "skills": ["contract law", "negotiation", "legal research", "client management", "compliance"],
    },
    "Teacher": {
        "industry": "education",
        "skills": ["curriculum design", "classroom management", "communication", "patience", "mentoring"],
    },
    "Nurse": {
        "industry": "healthcare",
        "skills": ["patient care", "clinical judgment", "teamwork", "stress management", "empathy"],
    },
    "Physician": {
        "industry": "healthcare",
        "skills": ["diagnosis", "patient care", "clinical research", "decision-making", "leadership"],
    },
    "Financial advisor": {
        "industry": "finance",
        "skills": ["portfolio management", "client relationships", "risk analysis", "financial modeling", "compliance"],
    },
    "Marketing director": {
        "industry": "media",
        "skills": ["brand strategy", "analytics", "team leadership", "creativity", "budget management"],
    },
    "Entrepreneur": {
        "industry": "technology",
        "skills": ["leadership", "fundraising", "vision", "adaptability", "strategic planning"],
    },
    "Social worker": {
        "industry": "nonprofit",
        "skills": ["case management", "advocacy", "empathy", "crisis intervention", "documentation"],
    },
    "Freelance designer": {
        "industry": "media",
        "skills": ["visual design", "UX research", "client communication", "creativity", "project management"],
    },
    "Real estate developer": {
        "industry": "real estate",
        "skills": ["deal sourcing", "negotiation", "project management", "financial modeling", "networking"],
    },
    "Military officer": {
        "industry": "government",
        "skills": ["leadership", "strategy", "discipline", "decision-making", "team building"],
    },
    "Restaurant owner": {
        "industry": "hospitality",
        "skills": ["operations", "staff management", "customer service", "budgeting", "food safety"],
    },
    "Veterinarian": {
        "industry": "healthcare",
        "skills": ["animal care", "diagnosis", "surgery", "client communication", "empathy"],
    },
}

CAREERS = list(CAREER_ECOSYSTEM.keys())

CAREER_GOALS = [
    "build a company that impacts millions", "reach VP level by 40",
    "open my own practice", "write a bestselling book",
    "become a partner at the firm", "start a nonprofit",
    "retire early and travel", "build generational wealth",
    "become a thought leader in my field", "mentor the next generation",
]

FAMILY_GOALS = [
    "be a present parent", "raise kind children",
    "build a loving home", "create a strong family unit",
]

PERSONAL_GOALS = [
    "stay fit and healthy", "travel to 50 countries",
    "learn three languages", "run a marathon",
    "write a memoir", "build a cabin in the mountains",
]

WORK_STYLES = ["remote", "hybrid", "in-office", "flexible"]
TEAM_PREFERENCES = ["small (1-5)", "medium (5-20)", "large (20+)"]


def _big5_level(score: int) -> str:
    if score >= 65:
        return "high"
    elif score >= 35:
        return "medium"
    return "low"


def _clamp(val: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, val))


def _sample(pool: list[str], k: int, rng: random.Random) -> list[str]:
    return rng.sample(pool, min(k, len(pool)))


# ── Person generation ────────────────────────────────────────────────────────

def generate_people(n: int, seed: int = 42) -> list[PersonSchema]:
    """Generate n synthetic people with coherent, realistic attributes."""
    rng = random.Random(seed)
    people = []

    for i in range(n):
        sex = rng.choice(["male", "female"])
        age = rng.randint(22, 45)

        # Height with realistic distribution
        if sex == "male":
            height = round(rng.gauss(70, 3), 1)
            height = max(64, min(78, height))
        else:
            height = round(rng.gauss(65, 2.5), 1)
            height = max(59, min(72, height))

        # Big 5 with Gaussian distribution (more realistic than uniform)
        openness = _clamp(int(rng.gauss(55, 18)))
        conscientiousness = _clamp(int(rng.gauss(55, 18)))
        extraversion = _clamp(int(rng.gauss(50, 20)))
        agreeableness = _clamp(int(rng.gauss(55, 18)))
        neuroticism = _clamp(int(rng.gauss(40, 18)))

        # MBTI — weakly correlated with Big 5
        if extraversion >= 50:
            mbti_e = rng.choice(["E"] * 3 + ["I"])
        else:
            mbti_e = rng.choice(["I"] * 3 + ["E"])
        if openness >= 50:
            mbti_n = rng.choice(["N"] * 3 + ["S"])
        else:
            mbti_n = rng.choice(["S"] * 3 + ["N"])
        if agreeableness >= 50:
            mbti_f = rng.choice(["F"] * 3 + ["T"])
        else:
            mbti_f = rng.choice(["T"] * 3 + ["F"])
        mbti_j = rng.choice(["J", "P"])
        mbti = mbti_e + mbti_n + mbti_f + mbti_j
        if mbti not in MBTI_DESCRIPTORS:
            mbti = rng.choice(MYERS_BRIGGS_TYPES)

        # Career — coherent with personality
        career = rng.choice(CAREERS)
        eco = CAREER_ECOSYSTEM[career]
        industry = eco["industry"]
        skills = list(eco["skills"])  # copy

        # Seniority from age/experience
        experience_years = max(0, age - 22 + rng.randint(-2, 2))
        if age >= 40 or experience_years >= 15:
            role_level = rng.choice(["senior", "lead", "executive"])
        elif age >= 32 or experience_years >= 8:
            role_level = rng.choice(["mid", "senior"])
        else:
            role_level = rng.choice(["junior", "mid"])

        # Income brackets correlated with seniority
        income_map = {
            "junior": ["modest", "medium"],
            "mid": ["medium", "high"],
            "senior": ["high", "very_high"],
            "lead": ["high", "very_high"],
            "executive": ["very_high"],
        }
        income_bracket = rng.choice(income_map.get(role_level, ["medium"]))

        # Ambitions
        cg = rng.choice(CAREER_GOALS)
        fg = rng.choice(FAMILY_GOALS)
        pg = rng.choice(PERSONAL_GOALS)
        ambitions = rng.choice([f"{cg}, {fg}", f"{cg}, {pg}", f"{cg}, {fg}, {pg}"])

        # Values/religion/politics with intensity
        religion = rng.choice(RELIGIONS)
        religion_intensity = rng.randint(1, 5)
        politics = rng.choice(POLITICS)
        politics_intensity = rng.randint(1, 5)

        # Smoking — weighted realistic distribution
        smoking = rng.choices(SMOKING_VALUES, weights=[70, 15, 15])[0]

        # Kids
        kids_current = rng.choices([0, 0, 0, 1, 2], weights=[50, 20, 10, 15, 5])[0]
        kids_wanted = rng.choice(["none", "1-2", "1-2", "3+"])

        person = PersonSchema(
            person_id=i,
            age=age,
            sex=sex,
            location=rng.choice(LOCATIONS),
            height_inches=height,
            ethnicity=rng.choice(ETHNICITIES),
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism,
            myers_briggs=mbti,
            core_values=_sample(CORE_VALUES, rng.randint(2, 4), rng),
            communication_style=rng.choice(COMMUNICATION_STYLES),
            attachment_style=rng.choice(ATTACHMENT_STYLES),
            career=career,
            ambitions=ambitions,
            education=rng.choice(EDUCATION_LEVELS),
            income_bracket=income_bracket,
            relationship_goal=rng.choice(RELATIONSHIP_GOALS),
            lifestyle=_sample(LIFESTYLES, rng.randint(2, 4), rng),
            interests=_sample(INTERESTS, rng.randint(3, 6), rng),
            dealbreakers=_sample(
                ["smoking", "no ambition", "dishonesty", "different life goals",
                 "lack of humor", "inflexibility", "jealousy", "different religion",
                 "different politics", "doesn't want kids"],
                rng.randint(1, 3), rng
            ),
            religion=religion,
            religion_intensity=religion_intensity,
            politics=politics,
            politics_intensity=politics_intensity,
            smoking=smoking,
            relationship_style=rng.choices(RELATIONSHIP_STYLES, weights=[75, 15, 10])[0],
            kids_current=kids_current,
            kids_wanted=kids_wanted,
            skills=skills,
            experience_years=experience_years,
            work_style=rng.choice(WORK_STYLES),
            industry=industry,
            role_level=role_level,
            team_preference=rng.choice(TEAM_PREFERENCES),
        )
        people.append(person)

    return people


# ── Text rendering: DATING ───────────────────────────────────────────────────
# 8 templates that vary sentence structure, emphasis, and tone

def _big5_dating_traits(p: PersonSchema) -> list[str]:
    """Convert Big 5 scores to natural dating personality descriptions."""
    traits = []
    if p.openness >= 65:
        traits.append("creative and open-minded")
    elif p.openness < 35:
        traits.append("practical and grounded")
    if p.extraversion >= 65:
        traits.append("outgoing and energetic")
    elif p.extraversion < 35:
        traits.append("thoughtful and introspective")
    if p.agreeableness >= 65:
        traits.append("warm and empathetic")
    elif p.agreeableness < 35:
        traits.append("straightforward and independent-minded")
    if p.conscientiousness >= 65:
        traits.append("organized and driven")
    elif p.conscientiousness < 35:
        traits.append("spontaneous and go-with-the-flow")
    if p.neuroticism >= 65:
        traits.append("passionate and intense")
    elif p.neuroticism < 35:
        traits.append("calm and steady")
    return traits


def _religion_text(p: PersonSchema) -> str:
    if p.religion_intensity <= 1:
        return "not religious"
    elif p.religion_intensity <= 2:
        return "doesn't care much about religion"
    elif p.religion_intensity <= 3:
        return f"moderately {p.religion}"
    elif p.religion_intensity <= 4:
        return f"{p.religion} — faith is important"
    return f"devout {p.religion}"


def _politics_text(p: PersonSchema) -> str:
    if p.politics_intensity <= 2:
        return ""
    return f"politically {p.politics.lower()}"


def _kids_text(p: PersonSchema) -> str:
    if p.kids_wanted == "none":
        return "doesn't want kids"
    elif p.kids_wanted == "1-2":
        return "wants a couple of kids"
    return "wants a big family"


def _dating_template_0(p: PersonSchema) -> str:
    """Standard intro-first template."""
    traits = _big5_dating_traits(p)
    lines = [f"{p.age}, {p.location}."]
    if traits:
        lines.append(f"I'm {', '.join(traits)}.")
    lines.append(f"{MBTI_DESCRIPTORS.get(p.myers_briggs, '')} ({p.myers_briggs}).")
    lines.append(f"I value {', '.join(p.core_values)}.")
    lines.append(f"Communication style: {p.communication_style}. Attachment: {p.attachment_style}.")
    lines.append(f"Looking for {p.relationship_goal}. {p.relationship_style.title()}.")
    lines.append(f"Into: {', '.join(p.interests)}.")
    rel = _religion_text(p)
    if rel:
        lines.append(f"{rel}.")
    pol = _politics_text(p)
    if pol:
        lines.append(f"{pol}.")
    lines.append(f"{_kids_text(p)}.")
    if p.dealbreakers:
        lines.append(f"Dealbreakers: {', '.join(p.dealbreakers)}.")
    return " ".join(lines)


def _dating_template_1(p: PersonSchema) -> str:
    """Personality-led template."""
    traits = _big5_dating_traits(p)
    mbti = MBTI_DESCRIPTORS.get(p.myers_briggs, "")
    lines = [f"I'd describe myself as {mbti} — {', '.join(traits) if traits else 'complex'}."]
    lines.append(f"I'm a {p.age}-year-old in {p.location}, {p.career.lower()}.")
    lines.append(f"My core values: {', '.join(p.core_values)}.")
    lines.append(f"I communicate in a {p.communication_style} way and tend toward {p.attachment_style} attachment.")
    lines.append(f"Lifestyle: {', '.join(p.lifestyle)}. I spend my time on {', '.join(p.interests)}.")
    lines.append(f"Looking for: {p.relationship_goal} ({p.relationship_style}).")
    rel = _religion_text(p)
    pol = _politics_text(p)
    extras = [x for x in [rel, pol, _kids_text(p)] if x]
    if extras:
        lines.append(f"{'. '.join(extras).capitalize()}.")
    if p.dealbreakers:
        lines.append(f"Can't deal with: {', '.join(p.dealbreakers)}.")
    return " ".join(lines)


def _dating_template_2(p: PersonSchema) -> str:
    """Values-first template."""
    lines = [f"What matters most to me: {', '.join(p.core_values)}."]
    lines.append(f"I'm {_religion_text(p)}, {_kids_text(p)}, and looking for {p.relationship_goal}.")
    traits = _big5_dating_traits(p)
    if traits:
        lines.append(f"People describe me as {', '.join(traits)}.")
    lines.append(f"{p.age}, {p.location}. {p.career}.")
    lines.append(f"I'm into {', '.join(p.interests[:3])} and {', '.join(p.interests[3:])}." if len(p.interests) > 3 else f"I enjoy {', '.join(p.interests)}.")
    lines.append(f"Communication: {p.communication_style}. {p.attachment_style.title()} attachment style.")
    pol = _politics_text(p)
    if pol:
        lines.append(f"I'm {pol}.")
    if p.dealbreakers:
        lines.append(f"Non-negotiables: {', '.join(p.dealbreakers)}.")
    return " ".join(lines)


def _dating_template_3(p: PersonSchema) -> str:
    """Casual conversational template."""
    traits = _big5_dating_traits(p)
    lines = [f"Hey! {p.age}, living in {p.location}."]
    lines.append(f"I work as a {p.career.lower()} and love {', '.join(p.interests[:3])}.")
    if traits:
        lines.append(f"Friends would say I'm {', '.join(traits[:2])}.")
    lines.append(f"Big on {', '.join(p.core_values[:2])}.")
    lines.append(f"I'm {p.relationship_style} and looking for something {p.relationship_goal}.")
    rel = _religion_text(p)
    if rel and p.religion_intensity >= 3:
        lines.append(f"Faith: {rel}.")
    lines.append(f"{_kids_text(p).capitalize()}.")
    if p.dealbreakers:
        lines.append(f"Please don't if: {', '.join(p.dealbreakers)}.")
    return " ".join(lines)


def _dating_template_4(p: PersonSchema) -> str:
    """Interests-led template."""
    lines = [f"You'll find me {', '.join(p.lifestyle[:2])} on weekends."]
    lines.append(f"Passionate about {', '.join(p.interests)}.")
    lines.append(f"I'm {p.age}, based in {p.location}. {p.career}.")
    traits = _big5_dating_traits(p)
    if traits:
        lines.append(f"Personality: {', '.join(traits)}. {MBTI_DESCRIPTORS.get(p.myers_briggs, '')}.")
    lines.append(f"I value {', '.join(p.core_values)} above all.")
    lines.append(f"Seeking {p.relationship_goal}. {p.relationship_style.title()}.")
    extras = [_religion_text(p), _politics_text(p), _kids_text(p)]
    extras = [x for x in extras if x]
    if extras:
        lines.append(f"{'. '.join(x.capitalize() for x in extras)}.")
    return " ".join(lines)


def _dating_template_5(p: PersonSchema) -> str:
    """Brief/minimal template."""
    traits = _big5_dating_traits(p)
    trait_str = f", {', '.join(traits[:2])}" if traits else ""
    lines = [f"{p.age}. {p.location}. {p.career}{trait_str}."]
    lines.append(f"Values: {', '.join(p.core_values)}. {_religion_text(p).capitalize()}.")
    lines.append(f"Into {', '.join(p.interests[:3])}. {_kids_text(p).capitalize()}.")
    lines.append(f"Want: {p.relationship_goal}, {p.relationship_style}.")
    return " ".join(lines)


def _dating_template_6(p: PersonSchema) -> str:
    """Narrative/storytelling template."""
    sex_word = "guy" if p.sex == "male" else "woman"
    traits = _big5_dating_traits(p)
    lines = [f"I'm the kind of {sex_word} who {', '.join(p.lifestyle[:2])}."]
    if traits:
        lines.append(f"At my core, I'm {' and '.join(traits[:2])}.")
    lines.append(f"I've built my career as a {p.career.lower()} ({p.experience_years} years), driven by {', '.join(p.core_values[:2])}.")
    lines.append(f"On any given day you'll find me into {', '.join(p.interests[:3])}.")
    lines.append(f"I'm {p.age}, call {p.location} home.")
    lines.append(f"{_religion_text(p).capitalize()}. {_kids_text(p).capitalize()}.")
    lines.append(f"Here for {p.relationship_goal}.")
    if p.dealbreakers:
        lines.append(f"Hard pass on: {', '.join(p.dealbreakers)}.")
    return " ".join(lines)


def _dating_template_7(p: PersonSchema) -> str:
    """MBTI/psychology-forward template."""
    mbti = p.myers_briggs
    desc = MBTI_DESCRIPTORS.get(mbti, "")
    lines = [f"{mbti} here — {desc}."]
    lines.append(f"{p.attachment_style.title()} attachment, {p.communication_style} communicator.")
    traits = _big5_dating_traits(p)
    if traits:
        lines.append(f"Also: {', '.join(traits)}.")
    lines.append(f"{p.age}, {p.location}. {p.career}.")
    lines.append(f"I care deeply about {', '.join(p.core_values)}.")
    lines.append(f"Hobbies: {', '.join(p.interests)}.")
    lines.append(f"Seeking {p.relationship_goal}. {_kids_text(p).capitalize()}.")
    rel = _religion_text(p)
    if p.religion_intensity >= 3:
        lines.append(f"{rel.capitalize()}.")
    return " ".join(lines)


DATING_TEMPLATES = [
    _dating_template_0, _dating_template_1, _dating_template_2, _dating_template_3,
    _dating_template_4, _dating_template_5, _dating_template_6, _dating_template_7,
]


# ── Text rendering: HIRING ──────────────────────────────────────────────────
# 8 templates that vary how the same person is presented professionally

def _big5_hiring_traits(p: PersonSchema) -> list[str]:
    """Convert Big 5 scores to professional trait descriptions."""
    traits = []
    if p.openness >= 65:
        traits.append("innovative thinker who embraces new approaches")
    elif p.openness < 35:
        traits.append("methodical professional who values proven methods")
    if p.extraversion >= 65:
        traits.append("strong communicator who thrives in team settings")
    elif p.extraversion < 35:
        traits.append("focused individual contributor who excels in deep work")
    if p.agreeableness >= 65:
        traits.append("collaborative team player")
    elif p.agreeableness < 35:
        traits.append("decisive leader who challenges assumptions")
    if p.conscientiousness >= 65:
        traits.append("detail-oriented and highly reliable")
    elif p.conscientiousness < 35:
        traits.append("adaptable and comfortable with ambiguity")
    if p.neuroticism >= 65:
        traits.append("deeply invested in work quality")
    elif p.neuroticism < 35:
        traits.append("composed under pressure")
    return traits


def _hiring_template_0(p: PersonSchema) -> str:
    """Standard resume-style."""
    traits = _big5_hiring_traits(p)
    lines = [f"{p.role_level.title()} {p.career} in {p.industry}."]
    lines.append(f"{p.experience_years} years of experience. {p.education}. Based in {p.location}.")
    if traits:
        lines.append(f"Professional profile: {', '.join(traits)}.")
    lines.append(f"Work style: {p.work_style}. Prefers {p.team_preference} teams.")
    lines.append(f"Communication approach: {p.communication_style}.")
    lines.append(f"Core professional values: {', '.join(p.core_values)}.")
    lines.append(f"Key skills: {', '.join(p.skills)}.")
    lines.append(f"Ambitions: {p.ambitions}.")
    return " ".join(lines)


def _hiring_template_1(p: PersonSchema) -> str:
    """Skills-first template."""
    lines = [f"Core competencies: {', '.join(p.skills)}."]
    lines.append(f"{p.experience_years} years as {p.career.lower()} in {p.industry}. {p.role_level.title()} level.")
    lines.append(f"{p.education}. {p.location}.")
    traits = _big5_hiring_traits(p)
    if traits:
        lines.append(f"Work personality: {', '.join(traits)}.")
    lines.append(f"Thrives in {p.work_style} environments with {p.team_preference} teams.")
    lines.append(f"Driven by: {', '.join(p.core_values)}.")
    mbti = MBTI_DESCRIPTORS.get(p.myers_briggs, "")
    if mbti:
        lines.append(f"Personality type: {p.myers_briggs} — {mbti}.")
    return " ".join(lines)


def _hiring_template_2(p: PersonSchema) -> str:
    """Narrative professional bio."""
    sex_word = "professional" if p.sex == "male" else "professional"
    traits = _big5_hiring_traits(p)
    lines = [f"A {p.role_level} {sex_word} with {p.experience_years} years in {p.industry}."]
    lines.append(f"Currently working as {p.career.lower()} in {p.location}.")
    if traits:
        lines.append(f"Known for being {' and '.join(traits[:2])}.")
    lines.append(f"Skilled in {', '.join(p.skills[:3])} with additional experience in {', '.join(p.skills[3:])}." if len(p.skills) > 3 else f"Skilled in {', '.join(p.skills)}.")
    lines.append(f"Education: {p.education}. Values {', '.join(p.core_values[:2])} in the workplace.")
    lines.append(f"Prefers {p.work_style} work in {p.team_preference} teams.")
    return " ".join(lines)


def _hiring_template_3(p: PersonSchema) -> str:
    """Culture-fit focused."""
    traits = _big5_hiring_traits(p)
    mbti = MBTI_DESCRIPTORS.get(p.myers_briggs, "")
    lines = [f"Work personality: {mbti} ({p.myers_briggs})."]
    if traits:
        lines.append(f"Strengths: {', '.join(traits)}.")
    lines.append(f"Communication style: {p.communication_style}. Values: {', '.join(p.core_values)}.")
    lines.append(f"Background: {p.role_level} {p.career.lower()}, {p.experience_years}yr experience in {p.industry}.")
    lines.append(f"Skills: {', '.join(p.skills)}. {p.education}.")
    lines.append(f"Looking for {p.work_style} roles with {p.team_preference} teams.")
    lines.append(f"Goals: {p.ambitions}.")
    return " ".join(lines)


def _hiring_template_4(p: PersonSchema) -> str:
    """LinkedIn summary style."""
    traits = _big5_hiring_traits(p)
    trait_str = traits[0] if traits else "experienced professional"
    lines = [f"{p.role_level.title()} {p.career} | {trait_str} | {p.location}."]
    lines.append(f"With {p.experience_years} years in {p.industry}, I bring expertise in {', '.join(p.skills[:3])}.")
    lines.append(f"{p.education}. I'm {p.communication_style} in my approach and thrive in {p.work_style} settings.")
    lines.append(f"I value {', '.join(p.core_values)} and work best in {p.team_preference} teams.")
    if len(traits) > 1:
        lines.append(f"Colleagues describe me as {' and '.join(traits[1:3])}.")
    lines.append(f"Career direction: {p.ambitions}.")
    return " ".join(lines)


def _hiring_template_5(p: PersonSchema) -> str:
    """Brief/executive summary."""
    lines = [f"{p.career}. {p.role_level.title()}. {p.experience_years} years. {p.industry}."]
    lines.append(f"Skills: {', '.join(p.skills)}. {p.education}.")
    traits = _big5_hiring_traits(p)
    if traits:
        lines.append(f"Profile: {', '.join(traits[:2])}.")
    lines.append(f"{p.work_style.title()} / {p.team_preference} teams. {p.location}.")
    lines.append(f"Values: {', '.join(p.core_values)}.")
    return " ".join(lines)


def _hiring_template_6(p: PersonSchema) -> str:
    """Ambition-led template."""
    lines = [f"Driven by: {p.ambitions}."]
    lines.append(f"Currently a {p.role_level} {p.career.lower()} with {p.experience_years} years in {p.industry}.")
    lines.append(f"Core skills: {', '.join(p.skills)}. {p.education}.")
    traits = _big5_hiring_traits(p)
    if traits:
        lines.append(f"Professional strengths: {', '.join(traits)}.")
    lines.append(f"Work preferences: {p.work_style}, {p.team_preference} teams, {p.communication_style} communication.")
    lines.append(f"Based in {p.location}. Values: {', '.join(p.core_values)}.")
    return " ".join(lines)


def _hiring_template_7(p: PersonSchema) -> str:
    """Team-fit template."""
    mbti = MBTI_DESCRIPTORS.get(p.myers_briggs, "")
    lines = [f"I'm a {p.communication_style} communicator, {mbti} ({p.myers_briggs})."]
    traits = _big5_hiring_traits(p)
    if traits:
        lines.append(f"In the workplace: {', '.join(traits)}.")
    lines.append(f"I do my best work in {p.work_style} environments with {p.team_preference} teams.")
    lines.append(f"Professional background: {p.role_level} {p.career.lower()}, {p.experience_years} years in {p.industry}.")
    lines.append(f"Key skills: {', '.join(p.skills)}. {p.education}.")
    lines.append(f"What drives me: {', '.join(p.core_values)}.")
    return " ".join(lines)


HIRING_TEMPLATES = [
    _hiring_template_0, _hiring_template_1, _hiring_template_2, _hiring_template_3,
    _hiring_template_4, _hiring_template_5, _hiring_template_6, _hiring_template_7,
]


# ── Public rendering API ─────────────────────────────────────────────────────

def render_dating_profile(p: PersonSchema, template_idx: int | None = None) -> str:
    """Render a person as a dating profile using one of 8 templates."""
    if template_idx is None:
        template_idx = p.person_id % len(DATING_TEMPLATES)
    return DATING_TEMPLATES[template_idx](p)


def render_hiring_resume(p: PersonSchema, template_idx: int | None = None) -> str:
    """Render the same person as a hiring profile using one of 8 templates."""
    if template_idx is None:
        # Use a different template offset than dating to avoid structural correlation
        template_idx = (p.person_id + 3) % len(HIRING_TEMPLATES)
    return HIRING_TEMPLATES[template_idx](p)


def render_people(
    people: Sequence[PersonSchema],
) -> tuple[list[str], list[str]]:
    """Render all people as both dating profiles and hiring resumes.

    Returns (dating_texts, hiring_texts) aligned by index.
    """
    dating = [render_dating_profile(p) for p in people]
    hiring = [render_hiring_resume(p) for p in people]
    return dating, hiring


# ── Hard negative generation (attribute flip) ─────────────────────────────────

FLIP_TYPES = ["religion", "kids", "smoking", "politics", "relationship_style", "lifestyle"]


def generate_hard_negative(
    person: PersonSchema,
    flip_type: str,
    rng: random.Random,
) -> PersonSchema:
    """Clone a person and flip one key attribute to create a hard negative."""
    clone = copy.deepcopy(person)
    clone.person_id = -1  # sentinel

    if flip_type == "religion":
        if person.religion_intensity >= 3:
            clone.religion = "Non-religious"
            clone.religion_intensity = 1
        else:
            clone.religion = rng.choice(["Baptist Christian", "Muslim", "Hindu"])
            clone.religion_intensity = 5

    elif flip_type == "kids":
        if person.kids_wanted in ("1-2", "3+"):
            clone.kids_wanted = "none"
        else:
            clone.kids_wanted = "3+"

    elif flip_type == "smoking":
        clone.smoking = "regular smoker" if person.smoking == "non-smoker" else "non-smoker"

    elif flip_type == "politics":
        opposites = {
            "Liberal": "Conservative", "Conservative": "Liberal",
            "Libertarian": "Liberal", "Moderate": rng.choice(["Liberal", "Conservative"]),
            "Apolitical": rng.choice(["Liberal", "Conservative"]),
        }
        clone.politics = opposites.get(person.politics, "Liberal")
        clone.politics_intensity = 5

    elif flip_type == "relationship_style":
        if person.relationship_style == "monogamous":
            clone.relationship_style = "polyamorous"
        else:
            clone.relationship_style = "monogamous"

    elif flip_type == "lifestyle":
        clone.ambitions = rng.choice([
            "build a billion-dollar company, work 80-hour weeks",
            "travel solo, live in a different country every year",
            "focus entirely on career, no family commitments",
        ])
        clone.interests = _sample(
            [i for i in INTERESTS if i not in person.interests],
            rng.randint(3, 5), rng
        )

    return clone
