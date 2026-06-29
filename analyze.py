"""
Analysis module — orchestrates prompt building, LLM invocation (via an
injectable LLMClient adapter), response validation, multi-sample
aggregation, and deterministic scoring.

Architecture
------------
This module has ZERO knowledge of which LLM provider is actually used.
It depends only on the LLMClient abstraction (llm_clients.LLMClient).
The concrete provider (Groq today; OpenAI/Claude/Gemini tomorrow) is
selected by whichever LLMClient instance is passed to analyze_transcript()
- or, if none is passed, by llm_clients.get_default_llm_client(), which
returns a GroqClient and so preserves the original hardcoded default
behavior exactly, with zero changes required by existing callers (e.g.
main.py).

To add a new provider:
    1. Create llm_clients/<provider>_client.py implementing LLMClient.
    2. Pass an instance of it: analyze_transcript(..., client=NewClient()).
No changes to this file, prompt_builder.py, scoring.py, or schemas.py
are required.

Responsibilities that used to all live in this single file have been
split out:
    - Pydantic output schema        -> schemas.py
    - Prompt construction           -> prompt_builder.py
    - Scoring / weights / thresholds -> scoring.py
    - LLM provider SDK + HTTP calls -> llm_clients/*  (Adapter pattern)
This file now only orchestrates those pieces.

Scoring reliability (unchanged from the original design)
----------------------------------------------------------
overall_score, hiring_recommendation, interview_readiness, and
candidate_confidence are all computed in Python from validated category
scores — the LLM's own placeholder values for these fields are always
overwritten. This makes the four derived fields 100% deterministic: the
same category scores always produce the same outputs regardless of LLM
variance.
"""

import json
import statistics
from typing import Optional

from pydantic import ValidationError

from llm_clients import LLMClient, get_default_llm_client
from prompt_builder import build_system_prompt, USER_TRIGGER
from schemas import EvaluationCategories, InterviewAnalysis
from scoring import (
    compute_overall_score,
    compute_hiring_recommendation,
    compute_interview_readiness,
    warn_if_degenerate,
)


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


# ---------------------------------------------------------------------------
# Single-sample fetch with retry
# ---------------------------------------------------------------------------

def _get_one_sample(
    client: LLMClient,
    model: str,
    job_role: str,
    transcript_text: str,
    temperature: float,
    seed: int,
    max_retries: int,
) -> InterviewAnalysis:
    system_prompt = build_system_prompt(job_role, transcript_text)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        # On retry: append the exact validation error to the system prompt
        # so the model sees context + error + instructions in one place.
        prompt = (
            system_prompt if attempt == 0
            else (
                build_system_prompt(job_role, transcript_text)
                + f"\n\nYour previous response was invalid: {last_error}. "
                "Return ONLY a corrected JSON object matching the required schema."
            )
        )

        raw = client.complete(
            system_prompt=prompt,
            user_message=USER_TRIGGER,
            model=model,
            temperature=temperature,
            seed=seed,
        )

        try:
            analysis = InterviewAnalysis(**json.loads(_strip_fences(raw)))
            warn_if_degenerate(analysis.evaluation_categories)
            return analysis
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e

    raise RuntimeError(
        f"Failed after {max_retries} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Multi-sample aggregation
# ---------------------------------------------------------------------------

def _aggregate(samples: list[InterviewAnalysis]) -> InterviewAnalysis:
    if len(samples) == 1:
        return samples[0]

    cats = [s.evaluation_categories for s in samples]

    merged_categories = EvaluationCategories(
        technical_knowledge=      round(statistics.median(c.technical_knowledge      for c in cats)),
        conceptual_understanding= round(statistics.median(c.conceptual_understanding for c in cats)),
        communication_skills=     round(statistics.median(c.communication_skills     for c in cats)),
        problem_solving_ability=  round(statistics.median(c.problem_solving_ability  for c in cats)),
        confidence_level=         statistics.mode([c.confidence_level for c in cats]),
        completeness_of_answer=   round(statistics.median(c.completeness_of_answer   for c in cats)),
    )

    merged = samples[0].model_copy(deep=True)
    merged.evaluation_categories = merged_categories
    merged.question_inference_confidence = statistics.mode(
        [s.question_inference_confidence for s in samples]
    )

    # Union of detected_topics across samples (de-duplicated, order preserved).
    seen: set[str] = set()
    topics: list[str] = []
    for s in samples:
        for t in s.detected_topics:
            key = t.strip().lower()
            if key and key not in seen:
                seen.add(key)
                topics.append(t.strip())
    merged.detected_topics = topics

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_transcript(
    transcript_text: str,
    job_role: str,
    model: str = "llama-3.3-70b-versatile",
    max_retries: int = 2,
    num_samples: int = 1,
    temperature: float = 0.0,
    seed: int = 42,
    client: Optional[LLMClient] = None,
) -> dict:
    """
    Transcribes a candidate's answer for a given job role and returns a
    structured evaluation report as a plain dict (JSON-serialisable).

    num_samples=1   Fast, good for development.
    num_samples=3   Slower but more consistent; recommended for production.
                    Numeric category scores are aggregated via median, so a
                    single outlier sample cannot skew the final result.

    client: Optional LLMClient adapter (see llm_clients/base.py). If
            omitted, defaults to llm_clients.get_default_llm_client(),
            which returns a GroqClient — identical to the original
            hardcoded behavior, so every existing caller (e.g. main.py)
            keeps working with zero changes.

            Pass any other LLMClient implementation (OpenAIClient,
            a future ClaudeClient, GeminiClient, ...) to switch providers
            for this call only, without touching this function or any
            other file:

                analyze_transcript(..., model="gpt-4o", client=OpenAIClient())
    """
    llm_client = client or get_default_llm_client()

    try:
        samples = [
            _get_one_sample(
                llm_client, model, job_role, transcript_text,
                temperature=temperature,
                seed=seed + i,
                max_retries=max_retries,
            )
            for i in range(num_samples)
        ]
    except RuntimeError as e:
        return {"error": str(e)}

    result = _aggregate(samples)
    result.overall_score         = compute_overall_score(result.evaluation_categories)
    result.hiring_recommendation = compute_hiring_recommendation(result.overall_score)
    result.interview_readiness   = compute_interview_readiness(result.overall_score)
    result.candidate_confidence  = result.evaluation_categories.confidence_level
    return result.model_dump()


# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# ---------------------------------------------------------------------------
# CATEGORY_WEIGHTS, CONFIDENCE_SCORE_MAP, HIRING_THRESHOLDS, READINESS_THRESHOLDS,
# and the three Literal type aliases used to be defined directly in this file.
# They now live in scoring.py / schemas.py, but are re-exported here so any
# existing code (or tests) doing `from analyze import CATEGORY_WEIGHTS`, etc.
# keeps working unchanged. (compute_overall_score, compute_hiring_recommendation,
# compute_interview_readiness, build_system_prompt, EvaluationCategories, and
# InterviewAnalysis are already available as analyze.<name> via the imports
# above, for the same reason.)
from scoring import (  # noqa: E402,F401
    CATEGORY_WEIGHTS,
    CONFIDENCE_SCORE_MAP,
    HIRING_THRESHOLDS,
    READINESS_THRESHOLDS,
)
from schemas import (  # noqa: E402,F401
    ConfidenceLevel,
    HiringRecommendation,
    InterviewReadiness,
)
from prompt_builder import _INSTRUCTION_BLOCK  # noqa: E402,F401

# Old private name, kept as an alias in case any external code referenced it.
_USER_TRIGGER = USER_TRIGGER
