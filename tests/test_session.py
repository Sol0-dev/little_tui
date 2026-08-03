from pathlib import Path

from little_tui.session import Session


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5
    cached_tokens = 0
    cost = 0.5


def test_replay_messages_roundtrip(tmp_path: Path) -> None:
    session = Session.create(tmp_path)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    for msg in messages:
        session.log_message(msg)
    session.log_usage(_FakeUsage())
    session.close()

    reopened = Session.open(session.path)
    assert reopened.replay_messages() == messages
    summary = reopened.summary()
    assert summary["events"]["messages"] == 2
    assert summary["cost"] == 0.5
    assert summary["tokens"] == 15


def test_tool_logging(tmp_path: Path) -> None:
    session = Session.create(tmp_path)

    class _Call:
        name = "write_file"
        id = "call_1"
        arguments = {"path": "a.txt"}

    session.log_tool(_Call(), {"ok": True}, 12.5)
    session.close()
    summary = Session.open(session.path).summary()
    assert summary["events"]["tools"] == 1
    assert summary["cost"] == 0.0


def test_summary_missing_file(tmp_path: Path) -> None:
    session = Session.open(tmp_path / "nope.jsonl")
    assert session.replay_messages() == []
    assert session.summary()["cost"] == 0.0
