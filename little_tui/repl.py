"""Interactive REPL: streaming output, tool display, slash commands.

Uses only built-in ``readline``/``input()`` so it works on any terminal,
including Termux. Streamed deltas print as they arrive; tool calls are shown
dimmed; every turn reports its step count and cost.
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path
from typing import Any

try:
    import readline
except ImportError:  # not available on some platforms (e.g. Termux without libreadline)
    readline = None

from .agent import Agent, AgentResult
from .config import Config
from .llm import LLMError, OpenRouterClient
from .session import Session
from .tools import Tool, build_tools

HISTORY_FILE = Path.home() / ".local" / "share" / "little-tui" / "history"

HELP_TEXT = """\
little-tui slash commands:
  /help             show this help
  /model [slug]     show or switch the model (e.g. /model ~openai/gpt-latest)
  /new              start a fresh conversation
  /cost             show session cost and token usage
  /session          show the session log path
  /tools            list available tools
  /quit             exit

Tips:
  - Ctrl+C interrupts an in-progress response.
  - Shell commands require approval (or --allow-shell).
  - Prompt with no slash text runs a normal agent turn.
"""


class Renderer:
    """ANSI-aware output. Falls back to plain text when ``color`` is off."""

    _C = {"dim": "\x1b[2m", "cyan": "\x1b[36m", "green": "\x1b[32m",
          "red": "\x1b[31m", "yellow": "\x1b[33m", "bold": "\x1b[1m", "reset": "\x1b[0m"}

    def __init__(self, color: bool = True) -> None:
        self.color = color and sys.stdout.isatty()

    def _paint(self, text: str, style: str) -> str:
        if not self.color:
            return text
        return f"{self._C[style]}{text}{self._C['reset']}"

    def banner(self, model: str) -> None:
        print(self._paint(f"little-tui — model: {model}", "bold"))
        print(self._paint("type /help for commands, Ctrl+C to interrupt", "dim"))

    def prompt(self) -> str:
        if not self.color:
            return "❯ "
        if readline is not None:
            # \x01/\x02 tell readline to ignore the ANSI codes when it computes
            # the visible prompt width, so long wrapped input lines stay aligned.
            return "\x01\x1b[36m\x02❯ \x01\x1b[0m\x02"
        return self._paint("❯ ", "cyan")

    def user(self, text: str) -> None:
        print(self._paint("you › ", "cyan") + text)

    def delta(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def end_line(self) -> None:
        sys.stdout.write("\n")

    def tool(self, tool: Tool, args: dict[str, Any]) -> None:
        rendered_args = _compact_args(args)
        print(self._paint(f"  ⚙ {tool.name} {rendered_args}", "dim"))

    def separator(self) -> None:
        print(self._paint("─" * 40, "dim"))

    def meta(self, text: str) -> None:
        print(self._paint(text, "dim"))

    def result_meta(self, result: AgentResult) -> None:
        note = f"done in {result.steps} step{'s' if result.steps != 1 else ''}"
        if result.cost > 0:
            note += f", ${result.cost:.4f}"
        if result.stop_reason not in ("complete",):
            note += f" [{result.stop_reason}]"
        print(self._paint(note, "dim"))

    def error(self, text: str) -> None:
        print(self._paint(f"error: {text}", "red"))

    def info(self, text: str) -> None:
        print(self._paint(text, "green"))


def _compact_args(args: dict[str, Any]) -> str:
    """Render tool args compactly for the tool display line."""
    parts = []
    for key in ("path", "pattern", "command", "old_string"):
        if key in args:
            parts.append(f"{key}={args[key]!r}")
            break
    if not parts:
        for key, value in list(args.items())[:2]:
            parts.append(f"{key}={value!r}")
    return "(" + ", ".join(parts) + ")"


class Repl:
    """One interactive session for a model+workspace; resettable via /new."""

    def __init__(self, config: Config, renderer: Renderer) -> None:
        self.config = config
        self.renderer = renderer
        self.client = OpenRouterClient(config)
        self.session: Session = Session.create()
        self.tools: list[Tool] = build_tools(config.workspace_path())
        self.history: list[dict[str, Any]] = []
        self.agent = self._build_agent()

    def _build_agent(self) -> Agent:
        return Agent(
            self.config,
            self.client,
            self.tools,
            session=self.session,
            on_delta=self.renderer.delta,
            on_tool=lambda tool, args, _out, _ms: self.renderer.tool(tool, args),
            approve=self._approve,
        )

    def _approve(self, tool: Tool, args: dict[str, Any]) -> bool:
        if not tool.dangerous or self.config.allow_shell:
            return True
        try:
            answer = input(f"{self.renderer._paint('allow shell? [y/N] ', 'yellow')}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    # -- main loop ---------------------------------------------------------

    def loop(self) -> None:
        self.renderer.banner(self.config.model)
        self._install_history()
        while True:
            try:
                line = input(self.renderer.prompt())
            except EOFError:
                self._bye()
                return
            except KeyboardInterrupt:
                print()
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self._command(line):
                    self._bye()
                    return
            else:
                self._turn(line)

    def _turn(self, text: str) -> None:
        self.renderer.user(text)
        self.renderer.separator()
        try:
            result = self.agent.run(text, history=self.history)
        except LLMError as exc:
            self.renderer.error(str(exc))
            self.renderer.separator()
            return
        self.renderer.end_line()
        self.renderer.separator()
        self.renderer.result_meta(result)
        self.history = self.session.replay_messages()

    # -- slash commands ----------------------------------------------------

    def _command(self, line: str) -> bool:
        parts = line.split()
        cmd, *rest = parts
        if cmd == "/quit":
            return True
        if cmd == "/help":
            print(HELP_TEXT)
        elif cmd == "/model":
            self._cmd_model(rest)
        elif cmd == "/new":
            self.session.close()
            self.session = Session.create()
            self.history = []
            self.agent = self._build_agent()
            self.renderer.info("new conversation started")
        elif cmd == "/cost":
            self.renderer.meta(f"session: {self.session.summary()}")
        elif cmd == "/session":
            self.renderer.meta(f"log: {self.session.path}")
        elif cmd == "/tools":
            for tool in self.tools:
                mark = " [dangerous]" if tool.dangerous else ""
                print(f"  {tool.name}{mark}")
        else:
            self.renderer.error(f"unknown command {cmd!r}; try /help")
        return False

    def _cmd_model(self, rest: list[str]) -> None:
        if not rest:
            self.renderer.meta(f"current model: {self.config.model}")
            return
        slug = rest[0]
        self.config.model = slug
        self.renderer.info(f"model set to {slug}")

    # -- lifecycle ---------------------------------------------------------

    def _install_history(self) -> None:
        if readline is None:
            return
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            readline.read_history_file(HISTORY_FILE)
        except (OSError, FileNotFoundError):
            pass
        readline.set_history_length(500)
        atexit.register(self._save_history)

    def _save_history(self) -> None:
        if readline is None:
            return
        try:
            readline.write_history_file(HISTORY_FILE)
        except OSError:
            pass

    def _bye(self) -> None:
        self.session.close()
        print()


def run_repl(config: Config) -> None:
    color = not os.environ.get("NO_COLOR")
    Repl(config, Renderer(color=color)).loop()
