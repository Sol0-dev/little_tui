# little-tui

A self-hosted, terminal-based coding agent built on [OpenRouter](https://openrouter.ai),
with an automated tool loop modeled after the OpenRouter Agent SDK's `callModel` pattern.
Written in pure Python (stdlib + `requests`).

## Install

```sh
pip install -e .[dev]     # inside this directory
little-tui --version
```

Requires an API key:

```sh
export OPENROUTER_API_KEY=sk-or-v1-...
```

## Usage

Interactive REPL:

```sh
little-tui
```

One-shot headless run:

```sh
little-tui --prompt "summarize the README"
echo "list the files" | little-tui        # prompt read from stdin
```

Options: `--model`, `--workspace`, `--allow-shell`, `--no-color`, `--config`.

The agent loops autonomously: model → parallel tool calls → feed results → repeat,
bounded by `max_steps` (50) and `max_cost` ($2.00) per turn. Dangerous tools
(`shell`) are denied by default; approve them per-run with `--allow-shell` or
interactively in the REPL.

### REPL commands

| Command   | Action                                  |
| --------- | --------------------------------------- |
| `/help`   | list commands                           |
| `/model`  | switch model                            |
| `/new`    | start a fresh session                   |
| `/cost`   | show current-turn spend                 |
| `/session`| show session path and event counts      |
| `/tools`  | list available tools                    |
| `/quit`   | exit                                    |

## Configuration

Precedence: defaults < JSON file < environment variables.

- JSON config: `~/.config/little-tui/config.json` (override with `LITTLE_TUI_CONFIG`)
- Env vars: `OPENROUTER_API_KEY`, `LITTLE_TUI_MODEL`, `LITTLE_TUI_WORKSPACE`,
  `LITTLE_TUI_MAX_STEPS`, `LITTLE_TUI_MAX_COST`, `LITTLE_TUI_ALLOW_SHELL`,
  `LITTLE_TUI_SYSTEM_PROMPT_FILE`

## Tools

`read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`,
`shell` (dangerous), `current_datetime`. All file paths are resolved against the
workspace; `..` escapes are rejected. Sessions are append-only JSONL logs under
`~/.local/share/little-tui/sessions/` and can be replayed.

## Tests

```sh
python3 -m pytest -q
```
