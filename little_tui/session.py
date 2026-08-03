"""Append-only JSONL session log.

Every event (messages, tool executions, usage) is one JSON line. The raw LLM
messages are stored verbatim so a conversation can be *replayed* after a crash
or restart: ``replay_messages()`` returns exactly what was sent to the API.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


def default_session_dir() -> Path:
    override = _env("LITTLE_TUI_SESSION_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "share" / "little-tui" / "sessions"


def _env(name: str) -> str | None:
    import os

    return os.environ.get(name)


@dataclass
class Session:
    """A JSONL log of one agent conversation."""

    path: Path
    _fh: TextIO | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, session_dir: Path | None = None) -> "Session":
        directory = session_dir or default_session_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return cls(path=directory / f"{stamp}.jsonl")

    @classmethod
    def open(cls, path: Path) -> "Session":
        return cls(path=path)

    def _writer(self) -> TextIO:
        if self._fh is None or self._fh.closed:
            self._fh = self.path.open("a", encoding="utf-8")
        return self._fh

    def log(self, kind: str, **data: Any) -> None:
        record: dict[str, Any] = {"ts": time.time(), "kind": kind}
        record.update(data)
        writer = self._writer()
        with self._lock:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.flush()

    def log_message(self, message: dict[str, Any]) -> None:
        self.log("message", message=message)

    def log_tool(self, tool_call: Any, result: Any, duration_ms: float) -> None:
        self.log(
            "tool",
            name=tool_call.name,
            id=tool_call.id,
            arguments=tool_call.arguments,
            result=result,
            duration_ms=round(duration_ms),
        )

    def log_usage(self, usage: Any) -> None:
        self.log(
            "usage",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            cost=usage.cost,
        )

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def replay_messages(self) -> list[dict[str, Any]]:
        """Rebuild the exact message list previously sent to the API."""
        messages: list[dict[str, Any]] = []
        if not self.path.is_file():
            return messages
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if record.get("kind") == "message" and isinstance(record.get("message"), dict):
                    messages.append(record["message"])
        return messages

    def summary(self) -> dict[str, Any]:
        """Tally usage/cost/events for this session (best effort)."""
        events = {"messages": 0, "tools": 0, "usage": 0}
        cost = 0.0
        tokens = 0
        if not self.path.is_file():
            return {"events": events, "cost": cost, "tokens": tokens}
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = record.get("kind")
                if kind == "message":
                    events["messages"] += 1
                elif kind == "tool":
                    events["tools"] += 1
                elif kind == "usage":
                    events["usage"] += 1
                    cost += float(record.get("cost") or 0.0)
                    tokens += int(record.get("prompt_tokens") or 0)
                    tokens += int(record.get("completion_tokens") or 0)
        return {"events": events, "cost": round(cost, 4), "tokens": tokens}
