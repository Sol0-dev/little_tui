from pathlib import Path

from little_tui.agent import Agent, AgentResult
from little_tui.config import Config
from little_tui.llm import ChatResult, Usage
from little_tui.tools import Tool, build_tools


class FakeClient:
    """Scripted OpenRouter client; no network involved."""

    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = responses
        self.calls: list[list[dict]] = []
        self.aborted = False

    def clear_abort(self) -> None:
        self.aborted = False

    def chat(self, messages, tools=None, on_delta=None):
        self.calls.append(messages)
        return self.responses.pop(0)


def _config(tmp_path: Path) -> Config:
    return Config(api_key="test", workspace=str(tmp_path))


def _usage(cost: float) -> Usage:
    return Usage(prompt_tokens=10, completion_tokens=5, cached_tokens=0, cost=cost)


def _tool_call(name: str, **args):
    from little_tui.llm import ToolCall

    return ToolCall(id=f"call_{name}", name=name, arguments=args)


def test_simple_answer_no_tools(tmp_path: Path) -> None:
    tools = build_tools(tmp_path)
    client = FakeClient([ChatResult(content="hello", tool_calls=[], finish_reason="stop", usage=_usage(0.01))])
    agent = Agent(_config(tmp_path), client, tools)
    result = agent.run("say hi")
    assert isinstance(result, AgentResult)
    assert result.content == "hello"
    assert result.stop_reason == "complete"
    assert result.steps == 0
    assert result.cost == 0.01


def test_tool_call_executed_and_fed_back(tmp_path: Path) -> None:
    tools = build_tools(tmp_path)
    client = FakeClient(
        [
            ChatResult(
                content="",
                tool_calls=[_tool_call("write_file", path="out.txt", content="42")],
                finish_reason="tool_calls",
                usage=_usage(0.02),
            ),
            ChatResult(content="wrote it", tool_calls=[], finish_reason="stop", usage=_usage(0.03)),
        ]
    )
    agent = Agent(_config(tmp_path), client, tools)
    result = agent.run("create out.txt with 42")
    assert result.content == "wrote it"
    assert result.steps == 1
    assert (tmp_path / "out.txt").read_text() == "42"

    second_call = client.calls[1]
    roles = [m["role"] for m in second_call]
    assert "tool" in roles
    tool_msg = next(m for m in second_call if m["role"] == "tool")
    assert '"created": true' in tool_msg["content"]
    assert '"path": "out.txt"' in tool_msg["content"]


def test_unknown_tool_reported_as_error(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ChatResult(content="", tool_calls=[_tool_call("nope")], finish_reason="tool_calls", usage=_usage(0.01)),
            ChatResult(content="done", tool_calls=[], finish_reason="stop", usage=_usage(0.01)),
        ]
    )
    agent = Agent(_config(tmp_path), client, build_tools(tmp_path))
    result = agent.run("x")
    tool_msg = next(m for m in client.calls[1] if m["role"] == "tool")
    assert "unknown tool" in tool_msg["content"]


def test_denied_dangerous_tool(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ChatResult(content="", tool_calls=[_tool_call("shell", command="rm -rf /")], finish_reason="tool_calls", usage=_usage(0.01)),
            ChatResult(content="ok", tool_calls=[], finish_reason="stop", usage=_usage(0.01)),
        ]
    )
    agent = Agent(_config(tmp_path), client, build_tools(tmp_path))  # default approval denies shell
    result = agent.run("x")
    assert result.content == "ok"
    tool_msg = next(m for m in client.calls[1] if m["role"] == "tool")
    assert "permission denied" in tool_msg["content"]


def test_max_steps_bounds_loop(tmp_path: Path) -> None:
    tools = build_tools(tmp_path)
    responses = [
        ChatResult(content="", tool_calls=[_tool_call("read_file", path="f.txt")], finish_reason="tool_calls", usage=_usage(0.01))
        for _ in range(10)
    ]
    client = FakeClient(responses)
    config = _config(tmp_path)
    config.max_steps = 3
    agent = Agent(config, client, tools)
    result = agent.run("x")
    assert result.stop_reason == "max_steps"
    assert result.steps == 3


def test_max_cost_stops_loop(tmp_path: Path) -> None:
    tools = build_tools(tmp_path)
    config = _config(tmp_path)
    config.max_cost = 0.05
    client = FakeClient(
        [
            ChatResult(content="", tool_calls=[_tool_call("read_file", path="f.txt")], finish_reason="tool_calls", usage=_usage(0.03)),
            ChatResult(content="", tool_calls=[_tool_call("read_file", path="f.txt")], finish_reason="tool_calls", usage=_usage(0.03)),
            ChatResult(content="final", tool_calls=[], finish_reason="stop", usage=_usage(0.03)),
        ]
    )
    agent = Agent(config, client, tools)
    result = agent.run("x")
    assert result.stop_reason == "max_cost"
    assert result.cost >= config.max_cost


def test_output_truncation_is_surfaced(tmp_path: Path) -> None:
    client = FakeClient(
        [
            ChatResult(
                content="this got cut off mid-sentence",
                tool_calls=[],
                finish_reason="length",
                usage=_usage(0.01),
            )
        ]
    )
    agent = Agent(_config(tmp_path), client, build_tools(tmp_path))
    result = agent.run("write a long essay")
    assert result.stop_reason == "length"
    assert result.content.endswith("mid-sentence")
