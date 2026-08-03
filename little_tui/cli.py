"""Command-line entry point: interactive REPL or one-shot headless runs.

Examples:
    little-tui                                   # interactive
    little-tui --prompt "list the files"         # one-shot
    echo "summarize README.md" | little-tui      # piped stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import Config, ConfigError, load_config
from .llm import LLMError, OpenRouterClient
from .repl import Renderer, run_repl
from .tools import build_tools


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="little-tui",
        description="A self-hosted coding agent with an automated tool loop, built on OpenRouter.",
    )
    parser.add_argument("--version", action="version", version=f"little-tui {__version__}")
    parser.add_argument("--prompt", help="run a single turn non-interactively and exit")
    parser.add_argument("--model", help="override the model slug")
    parser.add_argument("--max-tokens", type=int, help="max output tokens per model call")
    parser.add_argument("--workspace", help="working directory for tools (default: cwd)")
    parser.add_argument("--allow-shell", action="store_true", help="auto-approve shell commands")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument(
        "--config",
        help="path to a JSON config file (default: ~/.config/little-tui/config.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.config:
        import os

        os.environ["LITTLE_TUI_CONFIG"] = str(Path(args.config).expanduser())

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"little-tui: {exc}", file=sys.stderr)
        return 2

    if args.model:
        config.model = args.model
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    if args.workspace:
        config.workspace = str(Path(args.workspace).expanduser())
    if args.allow_shell:
        config.allow_shell = True

    if args.prompt:
        return _run_headless(config, args.prompt)

    if not sys.stdin.isatty():
        prompt = sys.stdin.read()
        if prompt.strip():
            return _run_headless(config, prompt)

    run_repl(config)
    return 0


def _run_headless(config: Config, prompt: str) -> int:
    renderer = Renderer(color=False)
    client = OpenRouterClient(config)
    agent = Agent(
        config,
        client,
        build_tools(config.workspace_path()),
        approve=lambda tool, _args: (not tool.dangerous) or config.allow_shell,
    )
    try:
        result = agent.run(prompt)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.content:
        print(result.content)
    renderer.result_meta(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
