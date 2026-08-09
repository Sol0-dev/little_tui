"""Layered configuration: defaults < config file < environment.

The config file is JSON at ``$LITTLE_TUI_CONFIG`` or
``~/.config/little-tui/config.json``. Environment variables always win.

Providers are OpenAI-compatible APIs that little-tui can talk to. Each one
defines its own API base, API-key environment variable, default model, and
request headers. ``openrouter`` is the default; ``groq`` is also supported.
A config file's model/API base are bound to its own provider selection, so
selecting another provider via ``LITTLE_TUI_PROVIDER`` falls back to that
provider's defaults instead of reusing stale file values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROVIDER = "openrouter"
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
    "provider": "LITTLE_TUI_PROVIDER",
    "model": "LITTLE_TUI_MODEL",
    "workspace": "LITTLE_TUI_WORKSPACE",
    "api_base": "LITTLE_TUI_API_BASE",
    "http_referer": "LITTLE_TUI_HTTP_REFERER",
    "app_title": "LITTLE_TUI_APP_TITLE",
    "max_steps": "LITTLE_TUI_MAX_STEPS",
    "max_cost": "LITTLE_TUI_MAX_COST",
    "max_tokens": "LITTLE_TUI_MAX_TOKENS",
    "allow_shell": "LITTLE_TUI_ALLOW_SHELL",
    "allow_all": "LITTLE_TUI_ALLOW_ALL",
    "system_prompt_file": "LITTLE_TUI_SYSTEM_PROMPT_FILE",
}


@dataclass(frozen=True)
class ProviderSpec:
    """Static per-provider defaults used when nothing overrides them."""

    name: str
    api_base: str
    api_key_env: str
    key_url: str
    model: str
    http_referer: str = ""
    app_title: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "openrouter": ProviderSpec(
        name="openrouter",
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        key_url="https://openrouter.ai/settings/keys",
        model="nvidia/nemotron-3-super-120b-a12b:free",
        http_referer="https://github.com/OpenRouterTeam/little-tui",
        app_title="little-tui",
    ),
    "groq": ProviderSpec(
        name="groq",
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        key_url="https://console.groq.com/keys",
        model="openai/gpt-oss-120b",
    ),
}


def provider_spec(name: str) -> ProviderSpec:
    try:
        return PROVIDERS[name.strip().lower()]
    except KeyError:
        choices = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"unknown provider {name!r}; choose from: {choices}") from None


@dataclass
class Config:
    """Runtime configuration for the agent."""

    api_key: str
    provider: str = DEFAULT_PROVIDER
    model: str = PROVIDERS[DEFAULT_PROVIDER].model
    api_base: str = PROVIDERS[DEFAULT_PROVIDER].api_base
    workspace: str = field(default_factory=lambda: os.getcwd())
    http_referer: str = PROVIDERS[DEFAULT_PROVIDER].http_referer
    app_title: str = PROVIDERS[DEFAULT_PROVIDER].app_title
    max_steps: int = DEFAULT_MAX_STEPS
    max_cost: float = DEFAULT_MAX_COST
    max_tokens: int = DEFAULT_MAX_TOKENS
    allow_shell: bool = False
    allow_all: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    def set_provider(self, name: str) -> None:
        """Switch providers at runtime, resetting endpoint, headers, and model.

        Switching to the already-active provider is a no-op so explicitly
        configured values (e.g. a model set in the config file) are preserved.
        The API key falls back to the current one when the new provider's
        environment variable is not set, so a key stored in the config file
        keeps working.
        """
        spec = provider_spec(name)
        if spec.name == self.provider:
            return
        self.provider = spec.name
        self.api_base = spec.api_base
        self.http_referer = spec.http_referer
        self.app_title = spec.app_title
        self.model = spec.model
        env_key = os.environ.get(spec.api_key_env)
        if env_key:
            self.api_key = env_key

    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            # OpenRouter attribution headers; other providers ignore them.
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
    if key in ("allow_shell", "allow_all"):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    return value


def load_config() -> Config:
    """Load configuration, merging defaults, file, and environment."""
    file_data: dict[str, object] = {}
    path = default_config_path()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"could not read config file {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"config file {path} must contain a JSON object")
        file_data.update(raw)

    env_data = {
        key: os.environ[name]
        for key, name in _ENV_MAP.items()
        if os.environ.get(name) not in (None, "")
    }

    provider = str(
        env_data.get("provider") or file_data.get("provider") or DEFAULT_PROVIDER
    )
    spec = provider_spec(provider)

    # Model/API base/attribution from the config file are bound to the file's
    # own provider selection. An env provider override means the file was
    # written for another provider, so those fields fall back to the new
    # provider's defaults instead of leaking stale values.
    file_scope = "provider" not in env_data

    def resolve(env_key: str, file_key: str, default: object) -> object:
        if env_key in env_data:
            return env_data[env_key]
        if file_scope and file_key in file_data:
            return file_data[file_key]
        return default

    api_key = os.environ.get(spec.api_key_env) or file_data.get("api_key")
    if not api_key:
        raise ConfigError(
            f"{spec.api_key_env} is not set. Get a key at {spec.key_url} and "
            f"export it, or set 'api_key' in the config file."
        )

    data: dict[str, object] = {
        "provider": spec.name,
        "api_key": api_key,
        "model": resolve("model", "model", spec.model),
        "api_base": resolve("api_base", "api_base", spec.api_base),
        "http_referer": resolve("http_referer", "http_referer", spec.http_referer),
        "app_title": resolve("app_title", "app_title", spec.app_title),
        "workspace": resolve("workspace", "workspace", os.getcwd()),
        "max_steps": resolve("max_steps", "max_steps", DEFAULT_MAX_STEPS),
        "max_cost": resolve("max_cost", "max_cost", DEFAULT_MAX_COST),
        "max_tokens": resolve("max_tokens", "max_tokens", DEFAULT_MAX_TOKENS),
        "allow_shell": resolve("allow_shell", "allow_shell", False),
        "allow_all": resolve("allow_all", "allow_all", False),
    }

    system_prompt_file = env_data.get("system_prompt_file")
    if system_prompt_file is None:
        system_prompt_file = (
            file_data.get("system_prompt_file") if file_scope else None
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
        provider=str(coerced["provider"]),
        model=str(coerced["model"]),
        api_base=str(coerced["api_base"]),
        workspace=str(coerced["workspace"]),
        http_referer=str(coerced["http_referer"]),
        app_title=str(coerced["app_title"]),
        max_steps=int(coerced["max_steps"]),
        max_cost=float(coerced["max_cost"]),
        max_tokens=int(coerced["max_tokens"]),
        allow_shell=bool(coerced["allow_shell"]),
        allow_all=bool(coerced["allow_all"]),
        system_prompt=system_prompt,
    )
