"""
Factory for the default LLMClient implementation.

This is the ONE place that decides which concrete adapter is used when a
caller doesn't explicitly pass a `client=` to analyze_transcript(). Today
that's Groq, preserving the original hardcoded behavior exactly. To
change the system-wide default provider, change ONLY this function — no
other file needs to know or care.
"""

from .base import LLMClient
from .groq_client import GroqClient


def get_default_llm_client() -> LLMClient:
    """Returns the default LLMClient adapter (currently: Groq)."""
    return GroqClient()
