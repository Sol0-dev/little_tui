"""Layered configuration: defaults < config file < environment.

The config file is JSON at ``$LITTLE_TUI_CONFIG`` or
``~/.config/little-tui/config.json``. Environment variables always win.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "~anthropic/claude-sonnet-latest"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_COST = 2.0
DEFAULT_MAX_TOKENS = 4096

DEFAULT_SYSTEM_PROMPT = (
    "You are little-tui, an autonomous coding agent running in a terminal. "
    "You complete tasks by reasoning step by step and calling tools. "
    "Work inside the workspace directory. When a task is done, give a concise "
    "summary of what you changed and why. Prefer many small, verifiable steps "
    "over one large guess."
)

_ENV_MAP = {
    "model": "LITTLE_TUI_MODEL",
    "workspace": "LITTLE_TUI_WORKSPACE",
    "api_key": "OPENROUTER_API_KEY",
    "api_base": "LITTLE_TUI_API_BASE",
    "http_referer": "LITTLE_TUI_HTTP_REFERER",
    "app_title": "LITTLE_TUI_APP_TITLE",
    "max_steps": "LITTLE_TUI_MAX_STEPS",
    "max_cost": "LITTLE_TUI_MAX_COST",
    "max_tokens": "LITTLE_TUI_MAX_TOKENS",
    "allow_shell": "LITTLE_TUI_ALLOW_SHELL",
    "system_prompt_file": "LITTLE_TUI_SYSTEM_PROMPT_FILE",
}


@dataclass
class Config:
    """Runtime configuration for the agent."""

    api_key: str
    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    workspace: str = field(default_factory=lambda: os.getcwd())
    http_referer: str = "https://github.com/OpenRouterTeam/little-tui"
    app_title: str = "little-tui"
    max_steps: int = DEFAULT_MAX_STEPS
    max_cost: float = DEFAULT_MAX_COST
    max_tokens: int = DEFAULT_MAX_TOKENS
    allow_shell: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers


class ConfigError(Exception):
    """Raised when configuration cannot be loaded."""


def default_config_path() -> Path:
    override = os.environ.get("LITTLE_TUI_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "little-tui" / "config.json"


def _coerce(key: str, value: object) -> object:
    """Type-coerce values from JSON/env into Config field types."""
    if key in ("max_steps",):
        if isinstance(value, bool):
            raise ConfigError(f"{key} must be an integer")
        return int(value)
    if key in ("max_cost",):
        return float(value)
    if key in ("allow_shell",):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    return value


def load_config() -> Config:
    """Load configuration, merging defaults, file, and environment."""
    data: dict[str, object] = {}

    path = default_config_path()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"could not read config file {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"config file {path} must contain a JSON object")
        data.update(raw)

    for key, env_name in _ENV_MAP.items():
        value = os.environ.get(env_name)
        if value is not None and value != "":
            data[key] = value

    system_prompt_file = data.get("system_prompt_file")
    if system_prompt_file is None:
        system_prompt_file = os.environ.get("LITTLE_TUI_SYSTEM_PROMPT_FILE")

    if "api_key" not in data or not data["api_key"]:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. "
            "Get a key at https://openrouter.ai/settings/keys and export it."
        )

    coerced: dict[str, object] = {
        key: _coerce(key, value) for key, value in data.items()
    }

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if system_prompt_file:
        try:
            system_prompt = (
                Path(str(system_prompt_file)).expanduser().read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise ConfigError(f"could not read system prompt file: {exc}") from exc

    return Config(
        api_key=str(coerced["api_key"]),
        model=str(coerced.get("model", DEFAULT_MODEL)),
        api_base=str(coerced.get("api_base", DEFAULT_API_BASE)),
        workspace=str(coerced.get("workspace", os.getcwd())),
        http_referer=str(coerced.get("http_referer", "https://github.com/OpenRouterTeam/little-tui")),
        app_title=str(coerced.get("app_title", "little-tui")),
        max_steps=int(coerced.get("max_steps", DEFAULT_MAX_STEPS)),
        max_cost=float(coerced.get("max_cost", DEFAULT_MAX_COST)),
        max_tokens=int(coerced.get("max_tokens", DEFAULT_MAX_TOKENS)),
        allow_shell=bool(coerced.get("allow_shell", False)),
        system_prompt=system_prompt,
    )
