"""
Groq adapter — implements LLMClient against Groq's chat-completions API.

This is the ONLY file in the project that imports the `groq` SDK. The
client construction logic (api key lookup, error message) and the
call/fallback logic are carried over unchanged from the original
analyze.py (_get_client / _call_llm) — only the location and the shape
(a class implementing LLMClient) have changed, to satisfy DIP and the
Adapter pattern.
"""

import os
from typing import Optional

from groq import Groq

from .base import LLMClient


class GroqClient(LLMClient):
    """Adapter that implements LLMClient against Groq's chat-completions API."""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n"
                "Get a free key at https://console.groq.com/keys\n"
                'Windows PowerShell: $env:GROQ_API_KEY="gsk_..."'
            )
        self._client = Groq(api_key=api_key)

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: str,
        temperature: float,
        seed: int,
        json_mode: bool = True,
    ) -> str:
        """
        Calls the Groq chat API. Tries strict JSON mode first for cleaner
        parsing; falls back to plain completion if the API rejects that
        param. (Identical behavior to the original _call_llm.)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        kwargs = dict(model=model, temperature=temperature, seed=seed, messages=messages)

        if json_mode:
            try:
                return (
                    self._client.chat.completions.create(
                        **kwargs, response_format={"type": "json_object"}
                    )
                    .choices[0]
                    .message.content.strip()
                )
            except Exception:
                # JSON mode not supported on this model/tier — fall back.
                pass

        return self._client.chat.completions.create(**kwargs).choices[0].message.content.strip()
