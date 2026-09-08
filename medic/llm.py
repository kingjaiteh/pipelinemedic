"""Chat model factory. Provider and model come from config, never from code."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from medic.config import LLM, LLMSettings


class MissingApiKey(RuntimeError):
    pass


def make_chat_model(settings: LLMSettings = LLM, *, temperature: float = 0.0) -> BaseChatModel:
    """Return a tool-capable chat model for the configured provider.

    Gemini is the default. The key is read from GEMINI_API_KEY and passed to
    the client explicitly, so langchain-google-genai's own GOOGLE_API_KEY
    lookup never comes into play.
    """
    provider = settings.provider.lower()

    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise MissingApiKey(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.model, google_api_key=key, temperature=temperature
        )

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingApiKey("ANTHROPIC_API_KEY is not set.")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError("Install the extra: uv sync --extra anthropic") from exc
        return ChatAnthropic(model=settings.model, api_key=key, temperature=temperature)

    raise ValueError(f"unknown MEDIC_LLM_PROVIDER {settings.provider!r}; use gemini or anthropic")
