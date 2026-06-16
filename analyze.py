"""
Analysis module - free Llama 3.3 (70B) via Groq's hosted API.

Why Groq instead of running Llama 3.3 locally:
  Running a 70B model locally needs serious GPU memory. Groq hosts it for you
  and offers a free tier with fast inference, so for a POC this is the
  practical "free LLM" path - no payment, no local GPU required.

Setup:
  1. Create a free account at https://console.groq.com
  2. Generate an API key at https://console.groq.com/keys
  3. export GROQ_API_KEY=your_key_here   (or set it in a .env file you load yourself)
"""

import os
import json
from groq import Groq

SYSTEM_PROMPT = """You are an expert technical interview analyst.
You will be given a candidate's spoken interview answer (transcribed from audio),
and optionally the question asked and a reference/expected answer.

Analyze it and return ONLY a JSON object with exactly this structure.
No markdown fences, no extra commentary, just the raw JSON object.

{
  "summary": "2-3 sentence summary of what the candidate said",
  "star": {
    "situation": "string or null if not present in the answer",
    "task": "string or null",
    "action": "string or null",
    "result": "string or null"
  },
  "strengths": ["short point", "short point"],
  "concerns": ["short point", "short point"],
  "relevance_score": 0,
  "relevance_reasoning": "1-2 sentences explaining the relevance_score (0-100). Score against the expected answer if provided, otherwise judge general coherence and depth.",
  "filler_word_estimate": "low | medium | high"
}
"""


def _client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "then run: export GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=api_key)


def analyze_transcript(
    transcript_text: str,
    question: str | None = None,
    expected_answer: str | None = None,
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """
    Sends the transcript (plus optional question/expected answer) to Llama 3.3
    and returns a structured dict: summary, STAR breakdown, strengths,
    concerns, and a relevance score.
    """
    client = _client()

    parts = []
    if question:
        parts.append(f"Interview question: {question}")
    if expected_answer:
        parts.append(f"Expected/reference answer: {expected_answer}")
    parts.append(f"Candidate's transcribed answer:\n{transcript_text}")
    user_prompt = "\n\n".join(parts)

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    # Strip accidental markdown fences if the model adds them anyway.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Could not parse LLM output as JSON", "raw_output": raw}
