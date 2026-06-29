"""
Deterministic scoring logic - pure Python, no LLM involvement.

Extracted from analyze.py. This module owns the hiring policy (weights,
thresholds) and the functions that turn validated category scores into
overall_score / hiring_recommendation / interview_readiness. Keeping this
separate from prompt-building and LLM-calling means a policy change (e.g.
adjusting CATEGORY_WEIGHTS) never requires touching anything that talks
to an LLM provider, and vice versa.
"""

import warnings

from schemas import EvaluationCategories, HiringRecommendation, InterviewReadiness

# ---------------------------------------------------------------------------
# Weights and thresholds — adjust to match actual hiring policy
# ---------------------------------------------------------------------------
CATEGORY_WEIGHTS: dict[str, float] = {
    "technical_knowledge":      0.25,
    "conceptual_understanding": 0.20,
    "communication_skills":     0.15,
    "problem_solving_ability":  0.15,
    "confidence_level":         0.10,
    "completeness_of_answer":   0.15,
}
assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9

CONFIDENCE_SCORE_MAP: dict[str, int] = {"High": 90, "Medium": 65, "Low": 35}

HIRING_THRESHOLDS: list[tuple[int, str]] = [
    (85, "Strong Hire"),
    (70, "Hire"),
    (50, "Borderline"),
    (0,  "Reject"),
]

READINESS_THRESHOLDS: list[tuple[int, str]] = [
    (85, "Highly Ready"),
    (70, "Ready"),
    (50, "Needs Practice"),
    (0,  "Not Ready"),
]


def compute_overall_score(cats: EvaluationCategories) -> int:
    numeric = {
        "technical_knowledge":      cats.technical_knowledge,
        "conceptual_understanding": cats.conceptual_understanding,
        "communication_skills":     cats.communication_skills,
        "problem_solving_ability":  cats.problem_solving_ability,
        "confidence_level":         CONFIDENCE_SCORE_MAP[cats.confidence_level],
        "completeness_of_answer":   cats.completeness_of_answer,
    }
    return round(sum(numeric[k] * w for k, w in CATEGORY_WEIGHTS.items()))


def compute_hiring_recommendation(score: int) -> HiringRecommendation:
    for threshold, label in HIRING_THRESHOLDS:
        if score >= threshold:
            return label
    return "Reject"


def compute_interview_readiness(score: int) -> InterviewReadiness:
    for threshold, label in READINESS_THRESHOLDS:
        if score >= threshold:
            return label
    return "Not Ready"


def warn_if_degenerate(cats: EvaluationCategories) -> None:
    """Warns if all five numeric categories scored identically (likely lazy/undifferentiated LLM output)."""
    values = [
        cats.technical_knowledge, cats.conceptual_understanding,
        cats.communication_skills, cats.problem_solving_ability,
        cats.completeness_of_answer,
    ]
    if len(set(values)) == 1:
        warnings.warn(
            f"All five numeric categories scored identically ({values[0]}) — "
            "possible undifferentiated scoring. Review this sample manually.",
            stacklevel=2,
        )
