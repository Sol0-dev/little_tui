"""Live smoke test: one turn, every tool, on a free OpenRouter model ($0).

Usage: OPENROUTER_API_KEY=... python scripts/smoke_tools.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from little_tui.agent import Agent
from little_tui.config import Config, load_config
from little_tui.llm import OpenRouterClient
from little_tui.session import Session
from little_tui.tools import build_tools

MODEL = os.environ.get("SMOKE_MODEL", "openai/gpt-oss-20b:free")

WORKSPACE = Path("/data/data/com.termux/files/home/.cache/opencode/tmp/smoke-ws")
shutil.rmtree(WORKSPACE, ignore_errors=True)
WORKSPACE.mkdir(parents=True)
(WORKSPACE / "data.txt").write_text("alpha line one\nbeta line two\nalpha line three\n")

PROMPT = """Exercise every available tool, each EXACTLY once, in one go. Use these
specific calls:
1. current_datetime - no args
2. list_dir - list the workspace
3. glob - pattern "**/*"
4. grep - pattern "alpha" (case sensitive)
5. read_file - data.txt
6. write_file - create new.txt with content "smoke test"
7. edit_file - in new.txt replace "smoke" with "edited"
8. shell - run "pwd" (harmless)

Then, in a short final answer, report which of the 8 succeeded and which failed.
Do not call any tool twice."""


def main() -> int:
    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        print(f"config error: {exc}")
        return 2
    config.model = MODEL
    config.workspace = str(WORKSPACE)
    config.max_tokens = 1500
    config.max_steps = 10
    config.allow_shell = True

    session = Session.create()
    client = OpenRouterClient(config)
    called: list[tuple[str, dict, object, float]] = []

    def on_tool(tool, args, outcome, ms):
        called.append((tool.name, args, outcome, ms))
        print(f"  [{tool.name}] args={json.dumps(args, default=str)}")

    agent = Agent(
        config,
        client,
        build_tools(config.workspace_path()),
        session=session,
        on_tool=on_tool,
        approve=lambda tool, _args: (not tool.dangerous) or config.allow_shell,
    )

    print(f"model: {MODEL}")
    result = agent.run(PROMPT)
    print(f"\nfinal answer: {result.content}\n")

    summary = session.summary()
    expected = {"current_datetime", "list_dir", "glob", "grep",
                "read_file", "write_file", "edit_file", "shell"}
    got = {name for name, _, _, _ in called}
    print("=== results ===")
    for name in sorted(expected):
        status = "OK" if name in got else "MISSING"
        print(f"  {name:<16} {status}")
    print(f"tools called: {len(called)}, expected: {len(expected)}")
    print(f"all tools exercised: {expected == got}")
    print(f"cost: ${result.cost:.6f}, tokens: {result.tokens}, steps: {result.steps}")
    print(f"stop: {result.stop_reason}")
    print(f"new.txt exists: {(WORKSPACE / 'new.txt').is_file()}")
    if (WORKSPACE / "new.txt").is_file():
        print(f"new.txt contents: {WORKSPACE / 'new.txt'!s}")
        print("  ->", (WORKSPACE / "new.txt").read_text())
    return 0 if expected == got and result.stop_reason == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
