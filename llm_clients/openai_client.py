"""
OPTIONAL adapter — demonstrates that a new LLM provider can be plugged in
without touching analyze.py, prompt_builder.py, scoring.py, or schemas.py.

NOT wired into the default factory (llm_clients/factory.py still returns
GroqClient by default, so existing behavior is unchanged). Use it
explicitly when you want OpenAI instead of Groq:

    from llm_clients.openai_client import OpenAIClient

    analysis = analyze_transcript(
        transcript_text=transcript_text,
        job_role=job_role,
        model="gpt-4o",                 # OpenAI model id, not Groq's
        client=OpenAIClient(),
    )

Requires the `openai` package, which is NOT in requirements.txt by
default (the system still defaults to Groq, so this isn't a forced
dependency):

    pip install openai

Claude (Anthropic) and Gemini (Google) adapters follow the exact same
shape: implement LLMClient.complete() against that provider's SDK,
translating `system_prompt` / `user_message` into whatever message
format that SDK expects, and returning the raw text content.
"""

import os
from typing import Optional

from .base import LLMClient


class OpenAIClient(LLMClient):
    """Adapter that implements LLMClient against OpenAI's chat-completions API."""

    def __init__(self, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package is required to use OpenAIClient. "
                "Install it with: pip install openai"
            ) from e

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set.\n"
                "Get a key at https://platform.openai.com/api-keys"
            )
        self._client = OpenAI(api_key=api_key)

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
                # JSON mode not supported on this model — fall back.
                pass

        return self._client.chat.completions.create(**kwargs).choices[0].message.content.strip()
