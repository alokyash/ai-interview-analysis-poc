"""
LLMClient abstraction (Adapter Pattern).

Every concrete provider adapter (GroqClient, OpenAIClient, ClaudeClient,
GeminiClient, ...) implements this single interface. The rest of the
application (analyze.py) depends ONLY on this abstraction - never on a
specific provider's SDK. That is what makes the system model-agnostic
and is the concrete fix for the Dependency Inversion Principle violation
identified in the prior code review (analyze.py used to `import Groq`
directly).
"""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Uniform contract for any chat-completion-style LLM provider."""

    @abstractmethod
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
        Sends one system+user message pair to the underlying LLM and
        returns the raw text content of the response (not yet parsed
        or validated - that remains the caller's responsibility).

        Parameters
        ----------
        system_prompt : the full system prompt (context + instructions).
        user_message  : the user-turn message.
        model         : provider-specific model identifier (e.g.
                        "llama-3.3-70b-versatile" for Groq, "gpt-4o" for
                        OpenAI). Each adapter interprets this in whatever
                        way its own SDK expects.
        temperature   : sampling temperature.
        seed          : sampling seed (for reproducibility, where the
                        provider supports it).
        json_mode     : if True, the adapter should request strict JSON
                        output if the provider supports it, and fall back
                        gracefully (not raise) if it doesn't.

        Returns
        -------
        The raw text content of the model's reply.
        """
        raise NotImplementedError
