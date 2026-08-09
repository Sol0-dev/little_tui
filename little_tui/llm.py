"""OpenAI-compatible chat client: streaming SSE, tool-call accumulation, retries.

Only ``requests`` is used. The response schema is OpenAI-compatible, which both
OpenRouter and Groq normalize to, so a single client serves every provider.
Streaming works for all models, and the final SSE chunk carries ``usage`` (with
cost on OpenRouter) even when token deltas were streamed.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .config import Config

DEFAULT_MAX_ATTEMPTS = 3
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

DeltaCallback = Callable[[str], None]


class LLMError(Exception):
    """An API error with OpenRouter's typed ``error_type`` when available."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        error_type: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.error_type = error_type
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE_STATUSES


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: Usage
    model: str | None = None


def _error_from_response(resp: requests.Response) -> LLMError:
    """Parse the provider error envelope from a non-streaming failure."""
    message = f"HTTP {resp.status_code}"
    error_type = None
    retry_after: float | None = None
    try:
        body = resp.json()
        err = body.get("error") or {}
        message = str(err.get("message") or message)
        metadata = err.get("metadata") or {}
        error_type = metadata.get("error_type") or err.get("error_type")
        if isinstance(metadata, dict):
            error_type = metadata.get("error_type")
    except ValueError:
        message = resp.text[:300] or message
    header = resp.headers.get("Retry-After")
    if header and header.isdigit():
        retry_after = float(header)
    return LLMError(
        message, code=resp.status_code, error_type=error_type, retry_after=retry_after
    )


class ChatClient:
    """Minimal streaming client for OpenAI-compatible chat completions.

    Endpoint and headers are read from ``config`` on every request, so the
    provider (and API key) can be switched at runtime without rebuilding.
    """

    def __init__(self, config: Config, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> None:
        self.config = config
        self.max_attempts = max_attempts
        self.session = requests.Session()
        self._abort = threading.Event()

    def abort(self) -> None:
        """Signal an in-flight stream to stop (e.g. user pressed Ctrl+C)."""
        self._abort.set()

    def clear_abort(self) -> None:
        self._abort.clear()

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> ChatResult:
        """Send a chat request, streaming deltas to ``on_delta``.

        Returns the fully assembled response: assistant text, any tool calls the
        model requested, and token/cost usage. Raises ``LLMError`` on failure.
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        last_error: LLMError | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._stream(payload, on_delta)
            except LLMError as exc:
                last_error = exc
                if not exc.retryable or attempt == self.max_attempts:
                    raise
                delay = exc.retry_after or float(2**attempt)
                time.sleep(delay)

        raise last_error  # unreachable; satisfies type checker

    def _stream(self, payload: dict[str, Any], on_delta: DeltaCallback | None) -> ChatResult:
        resp = self.session.post(
            f"{self.config.api_base}/chat/completions",
            headers=self.config.headers(),
            json=payload,
            stream=True,
            timeout=(15, 600),
        )
        if resp.status_code != 200:
            try:
                raise _error_from_response(resp)
            finally:
                resp.close()

        content_parts: list[str] = []
        call_deltas: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage = Usage()
        model: str | None = None

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if self._abort.is_set():
                    raise LLMError("request aborted", code=499)
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith(":"):
                    continue  # SSE keep-alive comment
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("error"):
                    raise LLMError(
                        chunk["error"].get("message", "mid-stream error"),
                        code=chunk["error"].get("code"),
                    )
                if chunk.get("model"):
                    model = chunk["model"]

                if usage_fields := chunk.get("usage"):
                    usage.prompt_tokens = usage_fields.get("prompt_tokens", 0)
                    usage.completion_tokens = usage_fields.get("completion_tokens", 0)
                    usage.cost = float(usage_fields.get("cost") or 0.0)
                    details = usage_fields.get("prompt_tokens_details") or {}
                    usage.cached_tokens = details.get("cached_tokens", 0)

                for choice in chunk.get("choices", []):
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        content_parts.append(text)
                        if on_delta:
                            on_delta(text)
                    for tc in delta.get("tool_calls") or []:
                        index = int(tc.get("index", 0))
                        slot = call_deltas.setdefault(
                            index,
                            {"id": None, "name": None, "arguments": ""},
                        )
                        fn = tc.get("function") or {}
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
        finally:
            resp.close()

        tool_calls: list[ToolCall] = []
        for index in sorted(call_deltas):
            slot = call_deltas[index]
            name = slot["name"]
            if not name:
                continue
            arguments: dict[str, Any] = {}
            if slot["arguments"]:
                try:
                    parsed = json.loads(slot["arguments"])
                    if isinstance(parsed, dict):
                        arguments = parsed
                except json.JSONDecodeError:
                    arguments = {"_raw": slot["arguments"]}
            tool_calls.append(
                ToolCall(id=slot["id"] or f"call_{index}", name=name, arguments=arguments)
            )

        return ChatResult(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model,
        )


# Backwards-compatible alias for code that imported the old name.
OpenRouterClient = ChatClient
