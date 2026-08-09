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
from typing import Any, Iterable

try:
    import select
    import termios
    import tty

    _HAVE_TTY = True
except ImportError:  # non-POSIX platforms (e.g. Windows) have no raw-mode termios
    select = None
    termios = None
    tty = None
    _HAVE_TTY = False

try:
    import readline
except ImportError:  # not available on some platforms (e.g. Termux without libreadline)
    readline = None

from .agent import Agent, AgentResult
from .config import Config, ConfigError, PROVIDERS
from .llm import ChatClient, LLMError
from .session import Session, SessionInfo, list_sessions
from .tools import Tool, build_tools

HISTORY_FILE = Path.home() / ".local" / "share" / "little-tui" / "history"

HELP_TEXT = """\
little-tui slash commands:
  /help             show this help
  /model [slug]     show or switch the model (e.g. /model ~openai/gpt-latest)
  /provider [name]  show or switch the API provider (openrouter, groq)
  /sessions [id]    pick a past session with the arrow keys, or resume one by id
  /new              start a fresh conversation
  /cost             show session cost and token usage
  /session          show the session log path
  /tools            list available tools
  /quit             exit

Tips:
  - In /sessions, use ↑/↓ to move and Enter to resume (q or Esc to cancel).
  - Ctrl+C interrupts an in-progress response.
  - Shell commands require approval (or --allow-shell).
  - --yolo auto-approves every tool call (edits and shell).
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

    def warn(self, text: str) -> None:
        print(self._paint(f"warning: {text}", "yellow"))

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


def _terminal_width() -> int:
    """Terminal width in columns, or 0 when unknown.

    Menu lines are truncated to the width so they never wrap; a wrapped line
    would occupy two screen rows and break the cursor-up redraw arithmetic.
    """
    try:
        return os.get_terminal_size(sys.stdout.fileno()).columns
    except (OSError, ValueError):
        return 0


class Repl:
    """One interactive session for a model+workspace.

    Conversation state lives in an append-only ``Session`` log, so it survives
    restarts: ``/new`` starts fresh, ``/sessions <id>`` resumes a past one.
    """

    def __init__(self, config: Config, renderer: Renderer) -> None:
        self.config = config
        self.renderer = renderer
        self.client = ChatClient(config)
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
        if self.config.allow_all or not tool.dangerous or self.config.allow_shell:
            return True
        try:
            answer = input(f"{self.renderer._paint('allow shell? [y/N] ', 'yellow')}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    # -- main loop ---------------------------------------------------------

    def loop(self) -> None:
        self.renderer.banner(self.config.model)
        if self.config.allow_all:
            self.renderer.warn(
                "yolo mode: auto-approving every tool call (edits and shell) — no prompts"
            )
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
        elif cmd == "/provider":
            self._cmd_provider(rest)
        elif cmd == "/sessions":
            self._cmd_sessions(rest)
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

    def _cmd_provider(self, rest: list[str]) -> None:
        if not rest:
            self.renderer.meta(f"current provider: {self.config.provider}")
            return
        name = rest[0].strip().lower()
        if name not in PROVIDERS:
            choices = ", ".join(sorted(PROVIDERS))
            self.renderer.error(f"unknown provider {name!r}; choose from: {choices}")
            return
        self.config.set_provider(name)
        self.renderer.info(
            f"provider set to {self.config.provider}, model: {self.config.model}"
        )

    def _cmd_sessions(self, rest: list[str]) -> None:
        if rest:
            self._resume_session(rest[0])
            return
        infos = list_sessions(self.session.path.parent)
        if not infos:
            self.renderer.meta("no past sessions yet; they are saved automatically")
            return
        if self._tty_ok():
            chosen = self._sessions_picker(infos)
            if chosen is None:
                self.renderer.meta("cancelled")
                return
            self._resume_session(chosen.id)
        else:
            self._list_sessions(infos)

    def _tty_ok(self) -> bool:
        return (
            _HAVE_TTY
            and sys.stdin.isatty()
            and sys.stdout.isatty()
        )

    def _sessions_picker(
        self, infos: list[SessionInfo], keys: Iterable[str] | None = None
    ) -> SessionInfo | None:
        """Arrow-key chooser over ``infos``; Enter resumes, q/Esc cancels.

        ``keys`` overrides the terminal key stream so the navigation logic is
        testable without a TTY. Returns the selected ``SessionInfo`` or ``None``
        on cancel.

        The menu redraws in place using only cursor-up and erase-line escape
        sequences (``\\x1b[{n}A`` / ``\\x1b[K``): Termux does not implement the
        SCO save/restore-cursor codes ``\\x1b[s``/``\\x1b[u``, which made every
        redraw append instead of replace. Every line is emitted with ``\\r\\n``
        except the last (``\\r`` only), so after drawing the cursor sits exactly
        ``total - 1`` rows below the menu start — that invariant holds even when
        the menu reaches the bottom of the screen and scrolls.
        """
        n = len(infos)
        idx = 0
        out = sys.stdout
        key_iter = keys if keys is not None else self._interactive_keys()
        total = n + 1  # header + one line per session
        width = _terminal_width()
        drawn = False

        def draw() -> None:
            nonlocal drawn
            if drawn:
                out.write(f"\x1b[{total - 1}A\r")
            drawn = True
            lines = [
                f"past sessions ({n}) — use ↑/↓ to select, Enter to resume, q/Esc to cancel:"
            ]
            for i, info in enumerate(infos):
                marker = "→" if i == idx else " "
                current = "*" if info.path == self.session.path else " "
                lines.append(
                    f"{marker} {current} {info.id}  {info.created}  "
                    f"messages={info.events['messages']} "
                    f"cost=${info.cost:.4f} tokens={info.tokens}"
                    + (f"  {info.preview}" if info.preview else "")
                )
            for i, line in enumerate(lines):
                text = line[: width - 1] if width else line
                style = "bold" if i == 1 + idx else "dim"
                out.write("\x1b[K" + self.renderer._paint(text, style))
                out.write("\r\n" if i < total - 1 else "\r")
            out.flush()

        draw()
        try:
            for key in key_iter:
                if key == "DOWN":
                    idx = (idx + 1) % n
                    draw()
                elif key == "UP":
                    idx = (idx - 1) % n
                    draw()
                elif key == "ENTER":
                    break
                elif key in ("ESC", "q", "CANCEL"):
                    idx = -1
                    break
        finally:
            out.write(f"\x1b[{total - 1}A\r")  # back to the menu start
            for i in range(total):
                out.write("\x1b[K")
                if i < total - 1:
                    out.write("\r\n")
            out.write(f"\x1b[{total - 1}A\r")  # prompt appears where the menu was
            out.flush()
        return None if idx < 0 else infos[idx]

    def _interactive_keys(self) -> Iterable[str]:
        """Yield terminal keys one at a time while stdin is in raw mode."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                key = self._read_key(fd)
                if key:
                    yield key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _read_key(self, fd: int) -> str:
        """Map one raw byte/escape sequence to a key name ("" = ignore)."""
        try:
            first = os.read(fd, 1)
        except OSError:
            return "CANCEL"
        if first in (b"\r", b"\n"):
            return "ENTER"
        if first == b"\x1b":  # arrow keys arrive as \x1b[A / \x1b[B
            ready, _, _ = select.select([fd], [], [], 0.05)
            if ready:
                rest = os.read(fd, 2)
                if rest == b"[A":
                    return "UP"
                if rest == b"[B":
                    return "DOWN"
            return "ESC"
        if first in (b"q", b"Q"):
            return "q"
        if first in (b"\x03", b"\x04"):  # Ctrl+C / Ctrl+D
            return "CANCEL"
        return ""

    def _list_sessions(self, infos: list[SessionInfo] | None = None) -> None:
        if infos is None:
            infos = list_sessions(self.session.path.parent)
        if not infos:
            self.renderer.meta("no past sessions yet; they are saved automatically")
            return
        self.renderer.meta(f"past sessions ({len(infos)}), * = current:")
        for info in infos:
            line = (
                f"{'*' if info.path == self.session.path else ' '} "
                f"{info.id}  {info.created}  "
                f"messages={info.events['messages']} "
                f"cost=${info.cost:.4f} tokens={info.tokens}"
            )
            if info.preview:
                line += f"  {info.preview}"
            if info.path == self.session.path:
                print(self.renderer._paint(line, "bold"))
            else:
                self.renderer.meta(line)

    def _resume_session(self, session_id: str) -> None:
        candidate = self.session.path.parent / f"{session_id}.jsonl"
        if not candidate.is_file():
            self.renderer.error(f"no session {session_id!r}; run /sessions to list them")
            return
        self.session.close()
        self.session = Session.open(candidate)
        self.history = self.session.replay_messages()
        self.agent = self._build_agent()
        self.renderer.info(
            f"resumed session {session_id} ({len(self.history)} messages); keep typing to continue"
        )

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
