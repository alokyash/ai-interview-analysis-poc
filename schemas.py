"""
Output schema for the analysis pipeline.

Extracted from analyze.py so the data contract (what an evaluation looks
like) is decoupled from how it's produced (which LLM provider, which
prompt). Nothing in this file knows about Groq, OpenAI, or any other
provider - it only defines and validates the shape of the result.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ConfidenceLevel = Literal["High", "Medium", "Low"]
HiringRecommendation = Literal["Strong Hire", "Hire", "Borderline", "Reject"]
InterviewReadiness = Literal["Highly Ready", "Ready", "Needs Practice", "Not Ready"]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
class EvaluationCategories(BaseModel):
    technical_knowledge: int = Field(ge=0, le=100)
    conceptual_understanding: int = Field(ge=0, le=100)
    communication_skills: int = Field(ge=0, le=100)
    problem_solving_ability: int = Field(ge=0, le=100)
    confidence_level: ConfidenceLevel
    completeness_of_answer: int = Field(ge=0, le=100)


class InterviewAnalysis(BaseModel):
    job_role: str
    inferred_question: str
    question_inference_confidence: ConfidenceLevel
    ideal_answer: str
    candidate_summary: str
    comparison_summary: str
    detected_topics: list[str] = Field(default_factory=list)

    evaluation_categories: EvaluationCategories

    # These four are always overwritten in Python — LLM values are ignored.
    overall_score: int = Field(ge=0, le=100, default=0)
    hiring_recommendation: HiringRecommendation = "Reject"
    interview_readiness: InterviewReadiness = "Not Ready"
    candidate_confidence: ConfidenceLevel = "Low"

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    suggested_learning_topics: list[str] = Field(default_factory=list)

    final_interviewer_summary: str
    detailed_feedback: str
