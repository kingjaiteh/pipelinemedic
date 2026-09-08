"""Langfuse tracing. Every function here is a no-op when the keys are absent.

Langfuse SDK 4.x is OpenTelemetry based: `Langfuse(...)` registers a client,
`start_as_current_span` opens a trace, and the LangChain `CallbackHandler`
attaches every model and tool call inside that span to it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "https://cloud.langfuse.com"


def _keys() -> tuple[str, str] | None:
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    return (public, secret) if public and secret else None


def base_url() -> str:
    # LANGFUSE_HOST is the pre-4.x name; accept both.
    return (
        os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or DEFAULT_BASE_URL
    )


def tracing_enabled() -> bool:
    return _keys() is not None


def _client():
    from langfuse import Langfuse

    public, secret = _keys()  # type: ignore[misc]
    return Langfuse(public_key=public, secret_key=secret, base_url=base_url())


def make_callbacks() -> list[Any]:
    """LangChain callbacks to pass on every invoke. Empty list without keys."""
    if not tracing_enabled():
        return []
    _client()
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler(public_key=_keys()[0])]  # type: ignore[index]


@dataclass
class TraceHandle:
    trace_id: str | None = None

    @property
    def url(self) -> str | None:
        if not self.trace_id:
            return None
        return f"{base_url().rstrip('/')}/trace/{self.trace_id}"


@contextmanager
def trace(name: str, **attributes: Any) -> Iterator[TraceHandle]:
    """Open one Langfuse trace around a whole run. Yields a handle with the id."""
    handle = TraceHandle()
    if not tracing_enabled():
        yield handle
        return
    client = _client()
    with client.start_as_current_span(name=name, input=attributes or None) as span:
        handle.trace_id = getattr(span, "trace_id", None)
        try:
            yield handle
        finally:
            client.flush()


def flush() -> None:
    if tracing_enabled():
        _client().flush()
