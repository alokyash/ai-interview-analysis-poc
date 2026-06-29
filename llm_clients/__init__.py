"""
llm_clients package — provider adapters implementing the LLMClient
abstraction (Adapter Pattern).

Import LLMClient to type-hint against "any provider". Import a concrete
adapter (GroqClient, ...) only at the point where you actually construct
one (e.g. in factory.py, or wherever a caller wants to force a specific
provider).

To add a new provider (OpenAI, Claude, Gemini, ...):
    1. Create llm_clients/<provider>_client.py with a class that inherits
       from LLMClient and implements `complete(...)`.
    2. Use it explicitly: `analyze_transcript(..., client=NewClient())`,
       or make it the system-wide default by editing factory.py.
No other file in the project needs to change.
"""

from .base import LLMClient
from .factory import get_default_llm_client
from .groq_client import GroqClient

__all__ = ["LLMClient", "GroqClient", "get_default_llm_client"]
