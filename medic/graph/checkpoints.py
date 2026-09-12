"""SQLite checkpoints: one file, one thread per incident run.

The saver stores the graph state after every node, which is what lets
`medic fix` stop at human review and `medic resume` pick the thread up from
another process. Thread state is read here without compiling a graph, so
listing and resuming do not need tools or a model first.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from medic.config import CHECKPOINT_DB
from medic.graph import state as state_module

# Every Pydantic class the state holds, so the serializer restores them
# without the "unregistered type" warning that precedes a future refusal.
ALLOWED_TYPES = [
    ("medic.graph.state", name)
    for name, obj in vars(state_module).items()
    if isinstance(obj, type)
    and issubclass(obj, state_module.BaseModel)
    and obj.__module__ == state_module.__name__
]


def new_thread_id(key: str) -> str:
    return f"{key}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


def thread_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


@contextmanager
def open_saver(path: Path = CHECKPOINT_DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES))
    finally:
        conn.close()


@dataclass
class ThreadInfo:
    thread_id: str
    status: str | None
    run_id: str | None
    scenario: int | None
    updated_at: str | None
    values: dict[str, Any]

    @property
    def waiting(self) -> bool:
        return self.status == "awaiting_review"


def _info(thread_id: str, tup: Any) -> ThreadInfo:
    values = tup.checkpoint.get("channel_values", {})
    incident = values.get("incident")
    return ThreadInfo(
        thread_id=thread_id,
        status=values.get("status"),
        run_id=getattr(incident, "run_id", None),
        scenario=getattr(incident, "scenario", None),
        updated_at=tup.checkpoint.get("ts"),
        values=values,
    )


def thread_state(saver: SqliteSaver, thread_id: str) -> ThreadInfo | None:
    """Latest checkpoint of one thread, or None if the thread is unknown."""
    tup = saver.get_tuple(thread_config(thread_id))
    return None if tup is None else _info(thread_id, tup)


def list_threads(saver: SqliteSaver) -> list[ThreadInfo]:
    """One entry per thread, newest first.

    The saver holds its lock while `list` is being consumed, so nothing else
    on the saver may be called inside the loop.
    """
    latest: dict[str, Any] = {}
    for tup in saver.list(None):
        tid = tup.config["configurable"]["thread_id"]
        if tid not in latest or (tup.checkpoint.get("ts") or "") > (
            latest[tid].checkpoint.get("ts") or ""
        ):
            latest[tid] = tup
    out = [_info(tid, tup) for tid, tup in latest.items()]
    out.sort(key=lambda t: t.updated_at or "", reverse=True)
    return out
