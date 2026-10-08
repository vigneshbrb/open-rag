"""Generation model selection."""

import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class ProviderError(RuntimeError):
    """Model provider configuration or request failed."""


def make_model(provider: str):
    if provider == "ollama":
        return ChatOllama(model="phi4-mini", base_url="http://127.0.0.1:11434", temperature=0)
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY is required for --provider openai")
        return ChatOpenAI(model="gpt-4o-mini", api_key=key, temperature=0)
    raise ProviderError(f"Unknown provider: {provider}")
