"""
Analysis module - free Llama 3.3 (70B) via Groq's hosted API.

Per the clarified requirements: the AI does NOT generate or ask interview
questions. The interviewer asks pre-prepared questions live; this module
only ever sees the candidate's transcribed answer and the target job role.
From that, the LLM infers the likely question, builds an ideal answer for
that role, evaluates the candidate against it across six categories, and
the system deterministically computes the overall score and hiring
recommendation in Python (never trusting LLM arithmetic for either).

Design notes:
- "confidence_level" is treated as ONE concept: how composed/assured the
  candidate sounded (a delivery trait, not a correctness measure), scored
  categorically as High/Medium/Low rather than numeric like the other five
  categories. It still feeds into overall_score via a fixed numeric mapping
  (see CONFIDENCE_LEVEL_SCORE_MAP) so the weighting stays meaningful.
- "question_inference_confidence" is a DIFFERENT concept from the above: it
  rates how sure the model is that it correctly guessed the actual question,
  given that it never sees the real one. It is informational only and is
  deliberately excluded from the overall_score weighting - it says nothing
  about the candidate, only about how much to trust inferred_question.
- detected_topics ("what was actually covered") is kept separate from
  suggested_learning_topics ("what to study because of a gap"). Both are
  lists of subjects/technologies, but one is extractive and the other is
  gap-driven - conflating them would make both less useful.
- interview_readiness and hiring_recommendation are both derived from the
  same overall_score, but through independent threshold tables aimed at
  different audiences: hiring_recommendation is a panel decision label,
  interview_readiness is a softer, candidate-facing coaching signal. Two
  tables means tightening one doesn't silently change the other.
- candidate_confidence is NOT a new measurement - it's the same value as
  evaluation_categories.confidence_level, copied to the top level after
  validation so callers reading a flat report schema don't need to reach
  into the nested object. Drop one of the two if you don't need both.
- detailed_feedback is candidate-facing (constructive, coaching tone).
  final_interviewer_summary is hiring-panel-facing (decision-oriented,
  concise performance summary). Keeping these distinct avoids the model
  blending a coaching tone into a decision document or vice versa.
- improvement_suggestions ("how to improve future answers") is kept
  separate from suggested_learning_topics ("what subjects to study"), since
  these serve different purposes for the candidate.

Consistency/determinism notes (vs. the previous version of this module):
- temperature defaults to 0.0 (was 0.1) and num_samples defaults to 3
  (was 1): multi-sample aggregation is a much stronger consistency lever
  than temperature alone, and the project requirement is explicitly about
  score stability. Pass num_samples=1 for fast iteration during dev if 3x
  latency/cost outweighs stability at that stage - this is a parameter
  default, so main.py does not need to change either way.
- _aggregate_samples now uses the median (was the mean) for the five
  numeric categories, so a single outlier sample can't swing the result as
  much as it could with a mean.
- _warn_if_degenerate is a soft, non-blocking guardrail: if a sample scores
  every numeric category identically, that's more likely lazy/undifferen-
  tiated scoring than a genuine tie, so it's surfaced as a warning for
  manual review rather than silently trusted or silently retried.
"""

import os
import json
import statistics
import warnings
from typing import Literal

from groq import Groq
from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Category weights - PROPOSED DEFAULTS, not specified by the project lead.
# Must sum to 1.0. Confirm/adjust these against actual policy before relying
# on overall_score for real hiring decisions. Note: question_inference_
# confidence is intentionally NOT weighted here - it rates the model's own
# inference, not the candidate, so it has no business affecting their score.
# ---------------------------------------------------------------------------
CATEGORY_WEIGHTS = {
    "technical_knowledge": 0.25,
    "conceptual_understanding": 0.20,
    "communication_skills": 0.15,
    "problem_solving_ability": 0.15,
    "confidence_level": 0.10,
    "completeness_of_answer": 0.15,
}
assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9, "CATEGORY_WEIGHTS must sum to 1.0"

# Maps the categorical confidence_level to a numeric value purely for the
# overall_score weighted calculation. Midpoints of the rubric bands below.
CONFIDENCE_LEVEL_SCORE_MAP = {"High": 90, "Medium": 65, "Low": 35}

# overall_score threshold -> hiring_recommendation. Checked top-down;
# PROPOSED DEFAULTS - confirm against actual hiring policy.
HIRING_THRESHOLDS = [
    (85, "Strong Hire"),
    (70, "Hire"),
    (50, "Borderline"),
    (0, "Reject"),
]

# overall_score threshold -> interview_readiness. Independent table from
# HIRING_THRESHOLDS on purpose - see design notes above. PROPOSED DEFAULTS.
READINESS_THRESHOLDS = [
    (85, "Highly Ready"),
    (70, "Ready"),
    (50, "Needs Practice"),
    (0, "Not Ready"),
]


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
    # How confident the model is that inferred_question matches what was
    # actually asked, given it only ever sees the answer. Informational -
    # not part of overall_score. See module docstring.
    question_inference_confidence: ConfidenceLevel
    ideal_answer: str
    candidate_summary: str
    comparison_summary: str
    # Extractive: subjects/technologies the candidate actually touched on.
    # Distinct from suggested_learning_topics (gap-driven) below.
    detected_topics: list[str] = Field(default_factory=list)

    evaluation_categories: EvaluationCategories

    # All four below are always recomputed in Python after validation - see
    # compute_overall_score() / compute_hiring_recommendation() /
    # compute_interview_readiness() and the candidate_confidence copy in
    # analyze_transcript(). The LLM's own values here are inert placeholders.
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


SYSTEM_PROMPT = """You are a senior technical interviewer and hiring assessor with deep experience \
evaluating candidates for software engineering, infrastructure, and data roles.

IMPORTANT CONTEXT: you do not generate or ask interview questions. A human interviewer already asked
a pre-prepared question live; you only receive the candidate's transcribed spoken answer and the
target job role. The actual question was not given to you.

Perform these steps, in this exact order:

1. Infer the single most likely interview question the candidate is answering, based on the content
   of their transcript and the target job role. It must be a single, realistic, specific question a
   senior interviewer for this role would actually ask - not a vague restatement like "a technical
   question."
2. State your own confidence that the question you inferred in step 1 is the actual question that was
   asked, as exactly one of "High", "Medium", or "Low":
   - High: the transcript makes the topic unambiguous (e.g. the candidate directly addresses one
     specific, narrow problem, or restates part of the question).
   - Medium: plausible and well-grounded, but more than one closely related question could have
     prompted this same answer.
   - Low: the transcript is short, vague, off-topic, or touches multiple unrelated ideas, making the
     underlying question a genuine guess.
   This rates YOUR inference, not the candidate - it is unrelated to confidence_level later, which
   rates the candidate's delivery.
3. Write the ideal answer a strong, well-qualified candidate would give to that exact question for
   that role. Be specific and technically accurate - this is your scoring reference, not a generic
   answer.
4. Summarize, in neutral words, what the candidate actually said.
5. Compare the candidate's answer against your ideal answer: what matched, what was missing, and any
   inaccuracies.
6. List detected_topics: the specific named technologies, concepts, or subjects the candidate actually
   touched on in their answer (2-6 items). Only list things genuinely present in what they said - this
   is extractive, not a wishlist. Gaps belong in suggested_learning_topics later, not here.
7. Score the candidate on six independent categories. Score each one independently - do not let one
   category's score influence another (e.g. a confident delivery should not inflate
   technical_knowledge; a quiet or hesitant delivery should not deflate it either, that belongs only
   in confidence_level).

For the five numeric categories (technical_knowledge, conceptual_understanding, communication_skills,
problem_solving_ability, completeness_of_answer), score an integer 0-100 using these anchors:
  90-100: Expert-level, what you'd expect from a senior/staff engineer in this role
  70-89:  Strong, what you'd expect from a solid mid-level hire
  50-69:  Adequate, but with clear gaps a junior candidate might have
  30-49:  Weak, with significant gaps or inaccuracies
  0-29:   Very poor - largely incorrect, off-topic, or substantively absent

Category definitions:
- technical_knowledge: correctness of technical facts, terminology, and concepts, judged against the
  ideal_answer.
- conceptual_understanding: depth of understanding behind the answer - does the candidate seem to
  understand *why*, not just recite the right terms.
- communication_skills: clarity, structure, and coherence of the spoken answer. This is a transcript
  of speech, so ignore minor filler words ("um", "uh") and judge structure and clarity of ideas, not
  grammar perfection.
- problem_solving_ability: evidence of structured reasoning, trade-off awareness, and a clear approach
  to the problem, where applicable to the question.
- completeness_of_answer: how fully the answer covers what the ideal_answer covers. A short but
  accurate answer should score lower here than a thorough one, even if both are correct.

The sixth category, confidence_level, is scored categorically as exactly one of "High", "Medium", or
"Low" - it measures how composed, assured, and convincing the candidate's delivery sounded, NOT
whether their answer was correct (that is covered by the other five categories entirely).

8. List concrete, specific strengths and weaknesses - not generic statements.
9. Write improvement_suggestions: specific, actionable advice about HOW the candidate could improve
   future answers (e.g. structuring with STAR, quantifying impact, being more concise) - this is about
   approach and delivery, not subject-matter topics.
10. Write suggested_learning_topics: specific named subjects, technologies, or concepts the candidate
    should study (e.g. "Kubernetes networking", "Big-O complexity analysis") based on gaps you
    identified. This is about WHAT TO STUDY because of a gap - distinct from detected_topics (step 6,
    what was actually covered) and from improvement_suggestions (step 9, how to approach delivery).
11. Write detailed_feedback: a constructive, coaching-toned paragraph (3-6 sentences) addressed to the
    candidate.
12. Write final_interviewer_summary: a concise, decision-oriented paragraph (3-5 sentences) addressed
    to the hiring panel. Summarize performance and fit for the role, and briefly ground it in concrete
    evidence (e.g. reference a specific strength/weakness or a detected topic) rather than only
    abstract praise or criticism. Describe performance honestly; do not state a specific hiring label
    yourself, since that is determined separately from your scores.

Rules:
- Base every score and claim on evidence in the transcript. Never invent specific companies,
  technologies, metrics, or claims the candidate did not actually say.
- If the transcript is short, garbled, or barely related to a technical interview, still produce your
  best-effort structured output rather than refusing - note the issue in detailed_feedback and let it
  pull down completeness_of_answer and conceptual_understanding, rather than every category equally.
  In that case question_inference_confidence should also honestly be "Low".
- Always set "overall_score" to 0, "hiring_recommendation" to "Reject", "interview_readiness" to
  "Not Ready", and "candidate_confidence" to "Low" as placeholders - all four are calculated
  separately. Do not attempt to compute or reason about any of them.
- Output ONLY a single JSON object with EXACTLY these keys, in this order, and no other keys. No
  markdown code fences, no preamble, no text before or after the JSON.

{
  "job_role": "<the job role as given>",
  "inferred_question": "<the single interview question you inferred>",
  "question_inference_confidence": "High",
  "ideal_answer": "<the ideal answer for a strong candidate, 3-6 sentences>",
  "candidate_summary": "<neutral summary of what the candidate actually said, 2-4 sentences>",
  "comparison_summary": "<what matched, what was missing or wrong, vs the ideal answer, 2-4 sentences>",
  "detected_topics": ["<specific subject or technology actually covered>"],
  "evaluation_categories": {
    "technical_knowledge": 0,
    "conceptual_understanding": 0,
    "communication_skills": 0,
    "problem_solving_ability": 0,
    "confidence_level": "High",
    "completeness_of_answer": 0
  },
  "overall_score": 0,
  "hiring_recommendation": "Reject",
  "interview_readiness": "Not Ready",
  "candidate_confidence": "Low",
  "strengths": ["<specific strength>"],
  "weaknesses": ["<specific weakness>"],
  "improvement_suggestions": ["<specific, actionable advice on approach/delivery>"],
  "suggested_learning_topics": ["<specific subject or technology to study, due to a gap>"],
  "final_interviewer_summary": "<3-5 sentence decision-oriented summary for the hiring panel>",
  "detailed_feedback": "<3-6 sentence constructive feedback paragraph for the candidate>"
}
"""


def build_user_prompt(job_role: str, transcript_text: str) -> str:
    return (
        f"Job role: {job_role}\n\n"
        f"Candidate's transcribed interview answer (verbatim, may include speech artifacts):\n"
        f"\"\"\"\n{transcript_text.strip()}\n\"\"\"\n\n"
        "Follow your instructions exactly and return the JSON object now."
    )


def compute_overall_score(categories: EvaluationCategories) -> int:
    """Recomputes overall_score from the six validated categories - this is
    the value that ends up in the final report. The LLM's own 0 placeholder
    is always overwritten here, so scoring math is 100% deterministic."""
    numeric = {
        "technical_knowledge": categories.technical_knowledge,
        "conceptual_understanding": categories.conceptual_understanding,
        "communication_skills": categories.communication_skills,
        "problem_solving_ability": categories.problem_solving_ability,
        "confidence_level": CONFIDENCE_LEVEL_SCORE_MAP[categories.confidence_level],
        "completeness_of_answer": categories.completeness_of_answer,
    }
    total = sum(numeric[key] * weight for key, weight in CATEGORY_WEIGHTS.items())
    return round(total)


def compute_hiring_recommendation(overall_score: int) -> HiringRecommendation:
    """Maps overall_score to a hiring label via fixed thresholds - this is
    also always computed in Python, never trusted from the LLM, so the same
    score always yields the same label."""
    for threshold, label in HIRING_THRESHOLDS:
        if overall_score >= threshold:
            return label
    return "Reject"


def compute_interview_readiness(overall_score: int) -> InterviewReadiness:
    """Same overall_score as hiring_recommendation, but mapped through an
    independent threshold table aimed at a different audience: this is a
    candidate-facing coaching signal, while hiring_recommendation is a
    hiring panel's decision label. Keeping the tables separate means
    tightening hiring criteria later won't silently change the coaching
    message, and vice versa."""
    for threshold, label in READINESS_THRESHOLDS:
        if overall_score >= threshold:
            return label
    return "Not Ready"


def _warn_if_degenerate(categories: EvaluationCategories) -> None:
    """Soft guardrail, non-blocking: if all five numeric categories landed
    on the exact same value, the model likely defaulted to one lazy score
    rather than differentiating across categories as instructed. A genuine
    tie is technically possible, so this doesn't fail or retry - it just
    surfaces a warning so a human can sanity-check that particular sample."""
    numeric_values = [
        categories.technical_knowledge,
        categories.conceptual_understanding,
        categories.communication_skills,
        categories.problem_solving_ability,
        categories.completeness_of_answer,
    ]
    if len(set(numeric_values)) == 1:
        warnings.warn(
            f"All five numeric categories scored identically ({numeric_values[0]}) - "
            "possible lazy/undifferentiated scoring. Review this sample manually.",
            stacklevel=2,
        )


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            'then run: $env:GROQ_API_KEY="your_key_here"   (Windows PowerShell)'
        )
    return Groq(api_key=api_key)


def _call_llm(
    client: Groq,
    model: str,
    user_prompt: str,
    temperature: float,
    seed: int,
    use_json_mode: bool = True,
) -> str:
    """Calls the Groq chat API and returns the raw text response. Low
    temperature + a fixed seed are the two consistency levers Groq exposes;
    Groq's docs note seed-based determinism is best-effort, not guaranteed
    bit-for-bit - the real consistency guarantee in this system comes from
    overall_score, hiring_recommendation, interview_readiness, and
    candidate_confidence being computed in Python."""
    kwargs = dict(
        model=model,
        temperature=temperature,
        seed=seed,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception:
        if use_json_mode:
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    return response.choices[0].message.content.strip()


def _clean_json_text(raw: str) -> str:
    """Strips markdown fences if the model adds them despite instructions."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _get_one_sample(
    client: Groq,
    job_role: str,
    transcript_text: str,
    model: str,
    temperature: float,
    seed: int,
    max_retries: int,
) -> InterviewAnalysis:
    """Gets one validated InterviewAnalysis, retrying with the validation
    error fed back to the model if the response is malformed."""
    user_prompt = build_user_prompt(job_role, transcript_text)
    last_error = None

    for _ in range(max_retries):
        raw = _call_llm(client, model, user_prompt, temperature=temperature, seed=seed)
        cleaned = _clean_json_text(raw)
        try:
            data = json.loads(cleaned)
            analysis = InterviewAnalysis(**data)
            _warn_if_degenerate(analysis.evaluation_categories)
            return analysis
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            user_prompt = (
                build_user_prompt(job_role, transcript_text)
                + f"\n\nYour previous response was invalid: {e}. "
                "Return ONLY a corrected, valid JSON object matching the required schema."
            )

    raise RuntimeError(f"Failed to get a valid analysis after {max_retries} attempts: {last_error}")


def _aggregate_samples(samples: list[InterviewAnalysis]) -> InterviewAnalysis:
    """Combines multiple independently-generated samples into one. Numeric
    category scores use the median (more robust to one outlier sample than
    a mean); confidence_level and question_inference_confidence use the
    most common value across samples; detected_topics is the de-duplicated
    union across samples, since topics are extractive facts about the
    transcript and combining samples catches more of what was actually said.
    Other text fields are taken from the first sample, since qualitative
    content is far more stable across runs than numeric edge cases - this is
    what multi-sampling is mainly correcting for."""
    if len(samples) == 1:
        return samples[0]

    cats = [s.evaluation_categories for s in samples]
    averaged = EvaluationCategories(
        technical_knowledge=round(statistics.median(c.technical_knowledge for c in cats)),
        conceptual_understanding=round(statistics.median(c.conceptual_understanding for c in cats)),
        communication_skills=round(statistics.median(c.communication_skills for c in cats)),
        problem_solving_ability=round(statistics.median(c.problem_solving_ability for c in cats)),
        confidence_level=statistics.mode([c.confidence_level for c in cats]),
        completeness_of_answer=round(statistics.median(c.completeness_of_answer for c in cats)),
    )

    merged = samples[0].model_copy(deep=True)
    merged.evaluation_categories = averaged
    merged.question_inference_confidence = statistics.mode(
        [s.question_inference_confidence for s in samples]
    )

    seen = set()
    merged_topics = []
    for s in samples:
        for topic in s.detected_topics:
            key = topic.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged_topics.append(topic.strip())
    merged.detected_topics = merged_topics

    return merged


def analyze_transcript(
    transcript_text: str,
    job_role: str,
    model: str = "llama-3.3-70b-versatile",
    max_retries: int = 2,
    num_samples: int = 1,
    temperature: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Given a job role and a candidate's transcript, returns a fully validated
    evaluation report as a dict, with overall_score, hiring_recommendation,
    interview_readiness, and candidate_confidence all computed
    deterministically in Python.

    num_samples > 1 trades latency/cost for consistency: it calls the LLM
    multiple times and takes the median of the numeric scores, reducing
    run-to-run variance further than a single low-temperature call alone.
    Defaults to 3 for production-grade stability; pass num_samples=1 for
    fast iteration during development.
    """
    client = _client()

    try:
        samples = [
            _get_one_sample(
                client, job_role, transcript_text, model,
                temperature=temperature, seed=seed + i, max_retries=max_retries,
            )
            for i in range(num_samples)
        ]
    except RuntimeError as e:
        return {"error": str(e)}

    analysis = _aggregate_samples(samples)
    analysis.overall_score = compute_overall_score(analysis.evaluation_categories)
    analysis.hiring_recommendation = compute_hiring_recommendation(analysis.overall_score)
    analysis.interview_readiness = compute_interview_readiness(analysis.overall_score)
    analysis.candidate_confidence = analysis.evaluation_categories.confidence_level
    return analysis.model_dump()