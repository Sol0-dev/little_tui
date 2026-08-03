"""The automated agent loop.

``Agent.run`` is a ReAct loop built on ``OpenRouterClient.chat``:

    model -> (tool calls? -> execute in parallel -> feed results back) -> done

It is bounded by explicit stop conditions (steps and cost) so a run can never
run away, and every step is streamed to a callback and logged to a ``Session``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .llm import ChatResult, LLMError, OpenRouterClient, Usage
from .session import Session
from .tools import Tool, ToolError, tools_specs

MAX_PARALLEL_TOOLS = 4

EventCallback = Callable[[str], None]
ApprovalFn = Callable[[Tool, dict[str, Any]], bool]
ToolCallback = Callable[[Tool, dict[str, Any], Any, float], None]


def _default_approval(tool: Tool, _args: dict[str, Any]) -> bool:
    return not tool.dangerous


@dataclass
class AgentResult:
    content: str
    steps: int
    cost: float
    tokens: int
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


class Agent:
    """Owns the conversation and drives tool execution until a stop condition."""

    def __init__(
        self,
        config: Config,
        client: OpenRouterClient,
        tools: list[Tool],
        *,
        session: Session | None = None,
        on_delta: EventCallback | None = None,
        on_tool: ToolCallback | None = None,
        approve: ApprovalFn = _default_approval,
    ) -> None:
        self.config = config
        self.client = client
        self.tools = {tool.name: tool for tool in tools}
        self.session = session
        self.on_delta = on_delta
        self.on_tool = on_tool
        self.approve = approve
        self.workspace = config.workspace_path()

    # -- public API -------------------------------------------------------

    def run(self, user_input: str, history: list[dict[str, Any]] | None = None) -> AgentResult:
        """Run the loop for a new user turn, optionally continuing *history*."""
        self.client.clear_abort()
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.config.system_prompt}]
        if history:
            messages.extend(history)

        user_msg: dict[str, Any] = {"role": "user", "content": user_input}
        messages.append(user_msg)
        self._log_message(user_msg)

        total_usage = Usage()
        steps = 0
        content = ""
        stop_reason = "complete"
        last_finish: str | None = None
        specs = tools_specs(list(self.tools.values()))

        while steps < self.config.max_steps:
            if total_usage.cost >= self.config.max_cost:
                stop_reason = "max_cost"
                break

            try:
                result = self.client.chat(messages, tools=specs, on_delta=self.on_delta)
            except LLMError as exc:
                stop_reason = f"error: {exc}"
                if steps == 0:
                    raise
                break

            last_finish = result.finish_reason

            self._record_usage(result.usage, total_usage)
            assistant_message: dict[str, Any] = {"role": "assistant", "content": result.content}
            if result.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": _dump(call.arguments)},
                    }
                    for call in result.tool_calls
                ]
            messages.append(assistant_message)
            self._log_message(assistant_message)
            content = result.content

            if not result.tool_calls:
                break

            steps += 1
            self._execute_tool_calls(result, messages)

            if total_usage.cost >= self.config.max_cost:
                stop_reason = "max_cost"

        if steps >= self.config.max_steps and stop_reason == "complete":
            stop_reason = "max_steps"
        if stop_reason == "complete" and last_finish == "length":
            stop_reason = "length"

        return AgentResult(
            content=content,
            steps=steps,
            cost=round(total_usage.cost, 6),
            tokens=total_usage.total_tokens,
            stop_reason=stop_reason,
            usage=total_usage,
        )

    # -- internals ---------------------------------------------------------

    def _execute_tool_calls(self, result: ChatResult, messages: list[dict[str, Any]]) -> None:
        def execute(call: Any) -> dict[str, Any]:
            started = time.monotonic()
            tool = self.tools.get(call.name)
            if tool is None:
                return _tool_response(
                    call.id, None, f"error: unknown tool '{call.name}'", started
                )
            if not self.approve(tool, call.arguments):
                return _tool_response(
                    call.id, None, f"permission denied: tool '{call.name}' was not approved", started
                )
            try:
                outcome = tool.handler(call.arguments)
            except ToolError as exc:
                outcome = {"error": str(exc)}
            except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
                outcome = {"error": f"{type(exc).__name__}: {exc}"}
            duration = (time.monotonic() - started) * 1000
            if self.session is not None:
                self.session.log_tool(call, outcome, duration)
            if self.on_tool is not None:
                self.on_tool(tool, call.arguments, outcome, duration)
            return _tool_response(call.id, call.name, outcome, started)

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TOOLS) as pool:
            responses = list(pool.map(execute, result.tool_calls))

        for message in responses:
            messages.append(message)
            self._log_message(message)

    def _record_usage(self, usage: Usage, total: Usage) -> None:
        total.prompt_tokens += usage.prompt_tokens
        total.completion_tokens += usage.completion_tokens
        total.cached_tokens += usage.cached_tokens
        total.cost += usage.cost
        if self.session is not None:
            self.session.log_usage(usage)

    def _log_message(self, message: dict[str, Any]) -> None:
        if self.session is not None:
            self.session.log_message(message)


def _tool_response(
    call_id: str, name: str | None, outcome: Any, started: float
) -> dict[str, Any]:
    content = outcome if isinstance(outcome, str) else _dump(outcome)
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }


def _dump(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)
