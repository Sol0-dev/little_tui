"""The automated agent loop.

``Agent.run`` is a ReAct loop built on ``ChatClient.chat``:

    model -> (tool calls? -> execute in parallel -> feed results back) -> done

It is bounded by explicit stop conditions (steps and cost) so a run can never
run away, and every step is streamed to a callback and logged to a ``Session``.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .llm import ChatResult, ChatClient, LLMError, Usage
from .session import Session
from .tools import Tool, ToolError, tools_specs

MAX_PARALLEL_TOOLS = 4

EventCallback = Callable[[str], None]
ApprovalFn = Callable[[Tool, dict[str, Any]], bool]
ToolCallback = Callable[[Tool, dict[str, Any], Any, float], None]

_MSG_OVERHEAD_TOKENS = 4
_TOOL_CALL_OVERHEAD_TOKENS = 4


def _message_tokens(message: dict[str, Any]) -> int:
    """Rough prompt-token estimate for one stored message.

    Good enough to bound request size; exact tokenization is provider-side.
    Counts content plus any tool_calls arguments the assistant message carries.
    """
    content = message.get("content") or ""
    if isinstance(content, list):
        text = " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    else:
        text = str(content)
    tokens = _MSG_OVERHEAD_TOKENS + len(text) // 4
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or ""
        tokens += _TOOL_CALL_OVERHEAD_TOKENS + len(str(arguments)) // 4
    return tokens


def trim_history(
    messages: list[dict[str, Any]], max_tokens: int | None
) -> list[dict[str, Any]]:
    """Keep the newest turns that fit within a prompt-token budget.

    Long conversations otherwise send the entire history on every request,
    which blows through provider rate limits (Groq reserves ``max_tokens``
    against its tokens-per-minute budget) and grows unbounded. Messages are
    dropped whole "rounds" at a time — a round starts at a user message and
    ends before the next one — so an assistant ``tool_calls`` message always
    stays paired with its tool responses. The newest round is always kept.
    """
    if max_tokens is None or max_tokens <= 0:
        return []
    if sum(_message_tokens(m) for m in messages) <= max_tokens:
        return messages

    rounds: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user" and current:
            rounds.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        rounds.append(current)

    kept: list[dict[str, Any]] = []
    used = 0
    for round_ in reversed(rounds):
        size = sum(_message_tokens(m) for m in round_)
        if kept and used + size > max_tokens:
            break
        kept = round_ + kept
        used += size
    return kept

_PATH_LIKE = re.compile(
    r"(?:^|[\s(\"'`])[\w./~@-]+\.(?:py|js|jsx|ts|tsx|json|jsonl|md|txt|toml|ya?ml|"
    r"sh|bash|css|html|go|rs|c|cc|cpp|h|hpp|java|kt|rb|sql|lock|cfg|ini|gitignore)\b",
    re.IGNORECASE,
)

_WORKSPACE_WORDS = re.compile(
    r"\b(?:file|files|folder|folders|directory|directories|repo|repos|repository|"
    r"workspace|script|source|config|readme|package|module|function|class|"
    r"database|server|client|endpoint)\b",
    re.IGNORECASE,
)

_QUESTION_WORDS = re.compile(
    r"\b(?:what|why|how|which|who|when|where|explain|define|meaning|difference|"
    r"example|tell me|can you|do you|does|is|are|should)\b",
    re.IGNORECASE,
)

_GREETINGS = re.compile(
    r"^(?:hi|hey|hello|yo|thanks|thank you|ok|okay|morning|evening|bye|goodbye)\b",
    re.IGNORECASE,
)


def prompt_needs_tools(text: str) -> bool:
    """Decide whether a prompt plausibly needs the tool loop.

    Conservative by design: tools are only skipped for clearly conversational
    prompts — a greeting or a short question with no file/workspace/action
    signals. When in doubt, tools are sent; the model simply won't call them
    when the answer needs nothing from the workspace.
    """
    t = " ".join(text.split())
    if len(t) < 4:
        return False
    if _PATH_LIKE.search(t) or _WORKSPACE_WORDS.search(t):
        return True
    if len(t) <= 120 and _QUESTION_WORDS.search(t):
        return False
    if len(t) <= 40 and _GREETINGS.match(t):
        return False
    return True


def _default_approval(tool: Tool, _args: dict[str, Any]) -> bool:
    return not tool.dangerous


def approval_for(config: Config) -> ApprovalFn:
    """Build the approval policy from ``config``.

    ``allow_all`` (``--yolo``) auto-approves every tool call — both edits and
    ``shell`` — without prompting. Otherwise only non-dangerous tools pass, plus
    the ``shell`` tool when ``allow_shell`` is set. The REPL layers an
    interactive prompt on top of this for the remaining dangerous cases.
    """

    def approve(tool: Tool, _args: dict[str, Any]) -> bool:
        return config.allow_all or (not tool.dangerous) or config.allow_shell

    return approve


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
        client: ChatClient,
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

    def run(
        self,
        user_input: str,
        history: list[dict[str, Any]] | None = None,
        *,
        use_tools: bool | None = None,
    ) -> AgentResult:
        """Run the loop for a new user turn, optionally continuing *history*.

        ``use_tools`` forces the tool loop on/off; when ``None`` (default) the
        prompt decides: clearly conversational prompts are sent with no tool
        definitions at all, so the model cannot call tools. Prompts that
        reference files, the workspace, or actions keep the full tool loop.
        """
        self.client.clear_abort()
        if use_tools is None:
            use_tools = prompt_needs_tools(user_input)
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.config.system_prompt}]
        if history:
            messages.extend(trim_history(history, self.config.max_history_tokens))

        user_msg: dict[str, Any] = {"role": "user", "content": user_input}
        messages.append(user_msg)
        self._log_message(user_msg)

        total_usage = Usage()
        steps = 0
        content = ""
        stop_reason = "complete"
        last_finish: str | None = None
        specs = tools_specs(list(self.tools.values())) if use_tools else None

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

            if not result.tool_calls or not use_tools:
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
