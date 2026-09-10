"""Chat model factory. Provider and model come from config, never from code."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from medic.config import LLM, LLMSettings


class MissingApiKey(RuntimeError):
    pass


# Rate limits and "high demand" overloads are both worth waiting out.
RATE_LIMIT_MARKERS = (
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "rate limit",
    "quota",
    "503",
    "unavailable",
    "high demand",
    "overloaded",
)
DAILY_QUOTA_MARKERS = ("perday", "per_day", "per day", "requestsperday")


class DailyQuotaExhausted(RuntimeError):
    """The per-model daily request quota is gone; retrying today cannot help."""


def is_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def is_daily_quota(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return is_rate_limit(exc) and any(marker in text for marker in DAILY_QUOTA_MARKERS)


def invoke_with_backoff(
    runnable: Runnable,
    messages: Any,
    *,
    config: dict | None = None,
    attempts: int = 5,
    base_delay_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] | None = None,
) -> AIMessage:
    """Invoke, and on a rate-limit error wait 15, 30, 60, 120 s before retrying.

    The key is a shared free-tier key, so hammering it only makes the next
    call fail too. Anything that is not a rate limit is raised at once.
    """
    for attempt in range(1, attempts + 1):
        try:
            return runnable.invoke(messages, config=config)
        except Exception as exc:
            if is_daily_quota(exc):
                raise DailyQuotaExhausted(
                    f"daily request quota exhausted for {LLM.model}; switch MEDIC_MODEL "
                    f"to a model with a fresh quota or wait for the reset. ({exc})"
                ) from exc
            if not is_rate_limit(exc) or attempt == attempts:
                raise
            delay = base_delay_s * 2 ** (attempt - 1)
            if log:
                log(
                    f"rate limited ({type(exc).__name__}); waiting {delay:.0f}s before retry {attempt + 1}/{attempts}"
                )
            sleep(delay)
    raise AssertionError("unreachable")


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

        # Gemini 3.x models use fixed sampling and warn on every call that
        # temperature is ignored. Nothing here depends on it, so drop it there.
        kwargs = {} if settings.model.startswith("gemini-3") else {"temperature": temperature}
        return ChatGoogleGenerativeAI(model=settings.model, google_api_key=key, **kwargs)

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
