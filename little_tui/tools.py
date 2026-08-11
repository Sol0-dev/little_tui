"""Tool definitions: filesystem, search, shell, and datetime.

Tools are plain callables plus a JSON-schema description. They are executed by
``agent.py``; this module is deliberately free of LLM imports.

Safety properties (kept on purpose, tested in ``tests/``):
  * every path is resolved against the workspace; ``..`` escapes are rejected
  * file reads cap their line count, greps cap their result count
  * the shell tool runs with ``cwd=workspace``, a timeout, and a stdout cap
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MAX_READ_LINES = 2000
MAX_GREP_RESULTS = 200
MAX_SHELL_OUTPUT = 32_000
DEFAULT_SHELL_TIMEOUT = 30

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".turbo",
    "dist",
    "build",
    ".cache",
}


class ToolError(Exception):
    """A user-facing tool failure that is safe to send back to the model."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    dangerous: bool = False

    def spec(self) -> dict[str, Any]:
        """The OpenAI function-tool JSON the API expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _workspace_path(workspace: Path, rel: str) -> Path:
    """Resolve *rel* against *workspace* and reject escapes outside it."""
    if not rel:
        raise ToolError("path must not be empty")
    candidate = (workspace / rel).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise ToolError(f"path escapes workspace: {rel}") from exc
    return candidate


def _fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KiB"
    return f"{size / 1024**2:.1f} MiB"


def _readable_line(line: str) -> str:
    return line.rstrip("\n").rstrip("\r")


def _read_file(workspace: Path, path: str, offset: int = 0, limit: int | None = None) -> dict[str, Any]:
    target = _workspace_path(workspace, path)
    if not target.is_file():
        raise ToolError(f"not a file or missing: {path}")
    limit = MAX_READ_LINES if limit is None else max(1, min(int(limit), MAX_READ_LINES))
    offset = max(0, int(offset))

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        raise ToolError(f"binary or non-UTF-8 file: {path}") from None

    total = len(lines)
    if offset >= total:
        return {"path": path, "total_lines": total, "lines": [], "truncated": False}
    selected = lines[offset : offset + limit]
    truncated = offset + len(selected) < total
    numbered = [f"{offset + i + 1}:{_readable_line(line)}" for i, line in enumerate(selected)]
    return {
        "path": path,
        "total_lines": total,
        "lines": numbered,
        "offset": offset,
        "truncated": truncated,
        "truncation_hint": (
            f"showing lines {offset + 1}-{offset + len(selected)} of {total}"
            if truncated
            else None
        ),
    }


def _write_file(workspace: Path, path: str, content: str) -> dict[str, Any]:
    target = _workspace_path(workspace, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(content, encoding="utf-8")
    return {
        "path": path,
        "created": not existed,
        "bytes": target.stat().st_size,
    }


def _edit_file(workspace: Path, path: str, old_string: str, new_string: str) -> dict[str, Any]:
    target = _workspace_path(workspace, path)
    if not target.is_file():
        raise ToolError(f"not a file or missing: {path}")
    text = target.read_text(encoding="utf-8")
    count = text.count(old_string)
    if count == 0:
        raise ToolError("old_string not found in file")
    if count > 1:
        raise ToolError(f"old_string matches {count} times; include more surrounding context")
    target.write_text(text.replace(old_string, new_string), encoding="utf-8")
    return {"path": path, "replaced": 1}


def _list_dir(workspace: Path, path: str = ".") -> dict[str, Any]:
    target = _workspace_path(workspace, path)
    if not target.is_dir():
        raise ToolError(f"not a directory: {path}")
    entries = []
    for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            if entry.is_dir():
                entries.append(f"{entry.name}/")
            else:
                entries.append(f"{entry.name} ({_fmt_size(entry.stat().st_size)})")
        except OSError:
            continue
    return {"path": path, "entries": entries, "count": len(entries)}


def _glob(workspace: Path, pattern: str) -> dict[str, Any]:
    if not pattern:
        raise ToolError("pattern must not be empty")
    matches: list[str] = []
    for entry in workspace.rglob(pattern):
        if not entry.is_file():
            continue
        if any(part in IGNORED_DIRS for part in entry.parts):
            continue
        matches.append(str(entry.relative_to(workspace)))
    matches.sort()
    return {"matches": matches, "count": len(matches)}


def _grep(workspace: Path, pattern: str, path: str = ".", include: str | None = None) -> dict[str, Any]:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"invalid regex: {exc}") from exc

    root = _workspace_path(workspace, path)
    if not root.is_dir():
        raise ToolError(f"not a directory: {path}")

    results: list[dict[str, Any]] = []
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        if any(part in IGNORED_DIRS for part in entry.parts):
            continue
        if include and not fnmatch.fnmatch(entry.name, include):
            continue
        rel = str(entry.relative_to(workspace))
        for i, line in enumerate(entry.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if regex.search(line):
                results.append({"file": rel, "line": i, "text": _readable_line(line)[:300]})
                if len(results) >= MAX_GREP_RESULTS:
                    break
        if len(results) >= MAX_GREP_RESULTS:
            break
    return {"results": results, "count": len(results), "truncated": len(results) >= MAX_GREP_RESULTS}


def _shell(workspace: Path, command: str, timeout: int = DEFAULT_SHELL_TIMEOUT) -> dict[str, Any]:
    if not command.strip():
        raise ToolError("command must not be empty")
    timeout = max(1, min(int(timeout), 120))
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"command timed out after {timeout}s") from exc

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    truncated = len(stdout) > MAX_SHELL_OUTPUT
    if truncated:
        stdout = stdout[:MAX_SHELL_OUTPUT] + "\n... [output truncated]"
    return {
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": (stderr or "")[:MAX_SHELL_OUTPUT],
        "truncated": truncated,
    }


def _current_datetime() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "utc_iso8601": now.isoformat(),
        "unix_seconds": int(now.timestamp()),
    }


def build_tools(workspace: Path) -> list[Tool]:
    """Build the standard tool set bound to *workspace*."""
    return [
        Tool(
            name="read_file",
            description=(
                "Read a text file relative to the workspace. Returns numbered lines. "
                "Use offset/limit to page through large files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "offset": {"type": "integer", "description": "0-based starting line"},
                    "limit": {"type": "integer", "description": "Max lines to read (<=2000)"},
                },
                "required": ["path"],
            },
            handler=lambda args: _read_file(workspace, **args),
        ),
        Tool(
            name="write_file",
            description="Write a file relative to the workspace, creating parent directories. Overwrites existing content.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=lambda args: _write_file(workspace, **args),
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact old_string with new_string in a file. "
                "The old_string must match exactly once; use unique surrounding context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            handler=lambda args: _edit_file(workspace, **args),
        ),
        Tool(
            name="list_dir",
            description="List a directory relative to the workspace. Directories are suffixed with '/'.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": [],
            },
            handler=lambda args: _list_dir(workspace, args.get("path", ".")),
        ),
        Tool(
            name="glob",
            description="Find files by glob pattern (e.g. '**/*.py'), skipping build/vendor directories.",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
            handler=lambda args: _glob(workspace, args["pattern"]),
        ),
        Tool(
            name="grep",
            description="Search file contents with a regular expression. Returns up to 200 file:line matches.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": ".", "description": "Directory to search"},
                    "include": {"type": "string", "description": "Optional filename glob filter"},
                },
                "required": ["pattern"],
            },
            handler=lambda args: _grep(
                workspace,
                args["pattern"],
                args.get("path", "."),
                args.get("include"),
            ),
        ),
        Tool(
            name="shell",
            description=(
                "Run a shell command in the workspace. Use for build, test, and git commands. "
                "Output is capped; long-running commands may need a timeout."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 30, "description": "Seconds, 1-120"},
                },
                "required": ["command"],
            },
            handler=lambda args: _shell(workspace, args["command"], int(args.get("timeout", 30))),
            dangerous=True,
        ),
        Tool(
            name="current_datetime",
            description="Get the current UTC date and time.",
            parameters={"type": "object", "properties": {}},
            handler=lambda args: _current_datetime(),
        ),
    ]


def tools_specs(tools: list[Tool]) -> list[dict[str, Any]]:
    return [tool.spec() for tool in tools]
