# little-tui

![image](https://github.com/Sol0-dev/little_tui/blob/main/image.png)

little-tui is a coding assistant that runs inside your terminal. You describe a
task in plain English and it plans, runs tools, and keeps going until the job is
done. It talks to OpenAI-compatible APIs: [OpenRouter](https://openrouter.ai)
(default) or [Groq](https://groq.com). Everything stays in your terminal and
every conversation is saved so you can pick it up again later.

This guide covers what little-tui is, the tools it has, how to install it on a
new device, where to get an API key, how to set the key up, and how to use it.

## What it can do

- Streams replies as they are generated, right in the terminal
- Reads and writes files, searches your project, and runs shell commands on its own
- Works through multiple steps until a task is finished
- Saves every conversation and resumes it later with `/sessions`
- Switches between providers and models at any time
- Runs on Linux and on Termux (Android)

## Tools

little-tui has these tools:

| Tool | What it does |
| --- | --- |
| `read_file` | Shows a file with numbered lines |
| `write_file` | Creates or overwrites a file |
| `edit_file` | Replaces one exact block of text in a file |
| `list_dir` | Shows the contents of a folder |
| `glob` | Finds files by pattern, for example `**/*.py` |
| `grep` | Searches file contents with a regular expression |
| `shell` | Runs a shell command (asks for your approval first) |
| `current_datetime` | Returns the current UTC date and time |

Safety rules:

- All file tools stay inside the workspace folder. Paths that escape the
  workspace (like `..`) are rejected.
- The `shell` tool asks for approval before running. Use `--allow-shell` to
  auto-approve shell commands, or `--yolo` to auto-approve every tool call.
- Each turn is limited to 50 tool steps and a cost cap, so a run can never
  spiral out of control.

## Requirements

- Python 3.10 or newer (check with `python3 --version`)
- `git`
- An API key from one of the providers below

On Termux (Android), install Python and git first:

```sh
pkg install python git
```

---

## 1. Get an API key

little-tui needs an API key from one provider. Both have a free tier.

### Option A: Groq (free and fast, good place to start)

1. Create a free account at <https://console.groq.com> (sign in with Google or GitHub).
2. Open **API Keys** at <https://console.groq.com/keys>.
3. Click **Create API Key**, give it a name, and copy the value.
4. Keys start with `gsk_`.

### Option B: OpenRouter (default, many models in one place)

1. Create an account at <https://openrouter.ai>.
2. Open **Keys** at <https://openrouter.ai/settings/keys>.
3. Click **Create Key** and copy the value.
4. Keys start with `sk-or-v1-`. Free models need no credit.

Keep your key private. Anyone with it can spend money on your account.

---

## 2. Install

Use the quick installer, or install it manually.

### Quick installer (recommended)

```sh
git clone https://github.com/Sol0-dev/little_tui
cd little_tui
./install.sh
```

What the installer does:

1. Creates a virtual environment (`.venv`) and installs little-tui into it.
2. Links the `little-tui` command into `~/.local/bin` (Linux) or `$PREFIX/bin`
   (Termux).
3. Asks which provider you want and where to paste your API key, then saves
   `~/.config/little-tui/config.json` so you can start right away.
4. Prints the next steps.

Prefer to skip the interactive key prompt (for example, if you will export the
keys yourself)?

```sh
NONINTERACTIVE=1 ./install.sh
```

### Manual install

If you manage your own Python environment:

```sh
git clone https://github.com/Sol0-dev/little_tui
cd little_tui
pip install -e .
little-tui --version
```

On Termux the `little-tui` command goes into `$PREFIX/bin`, which is already on
your PATH. On Linux it usually goes into `~/.local/bin`. Add that folder to your
PATH if the command is not found:

```sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 3. Set up your API key

The installer does this for you. If you prefer to do it yourself, use one of
the two options below. Settings priority: defaults, then config file, then
environment variables.

### Option 1: config file

Create a file at `~/.config/little-tui/config.json` (or wherever
`LITTLE_TUI_CONFIG` points):

```json
{
  "provider": "groq",
  "api_key": "gsk_your_key_here",
  "model": "llama-3.3-70b-versatile"
}
```

### Option 2: environment variables

Add these to `~/.bashrc` or `~/.profile`, then run `source ~/.bashrc`:

```sh
export GROQ_API_KEY=gsk_your_key_here
export OPENROUTER_API_KEY=sk-or-v1-your_key_here
export LITTLE_TUI_PROVIDER=groq
```

You only need the key for the provider you are actually using. little-tui uses
the key of the active provider. If you have both keys, switch with
`--provider openrouter` or `--provider groq` (or `/provider` inside the REPL).

---

## 4. Use it

### Interactive terminal

```sh
little-tui
```

Type a task and press Enter. The assistant streams its reply, shows each tool it
runs, and reports steps and cost when it finishes. Press `Ctrl+C` to stop the
current response.

Example:

```
> list the files in this project
  tool list_dir path='.'
...
done in 3 steps, $0.0001
```

### One-shot

Run a single task and exit:

```sh
little-tui --prompt "summarize the README"
echo "list the files" | little-tui
little-tui --prompt "explain config.py" --workspace ~/myproject
```

### Command-line options

| Option | What it does |
| --- | --- |
| `--provider` | API provider: `openrouter` (default) or `groq` |
| `--model` | Model slug, for example `llama-3.3-70b-versatile` |
| `--prompt` | Run one task non-interactively and exit |
| `--workspace` | Working folder for tools (default: current folder) |
| `--allow-shell` | Auto-approve `shell` commands |
| `--yolo` | Auto-approve every tool call (edits and shell) |
| `--max-tokens` | Max output tokens per model call |
| `--config` | Path to a JSON config file |
| `--no-color` | Turn off colors |

### Slash commands inside the REPL

| Command | What it does |
| --- | --- |
| `/help` | List commands |
| `/model [slug]` | Show or switch the model |
| `/provider [name]` | Switch API provider (`openrouter`, `groq`) |
| `/sessions` | Pick a past conversation with the arrow keys |
| `/sessions <id>` | Resume a past conversation by id |
| `/new` | Start a fresh conversation |
| `/cost` | Show cost and token usage |
| `/session` | Show the session log path |
| `/tools` | List available tools |
| `/quit` | Exit |

---

## 5. Past conversations

Every conversation is saved to a JSONL log in
`~/.local/share/little-tui/sessions/` (for example `20260808-103015.jsonl`).
It survives restarts and crashes. Only the 5 most recent sessions are kept; the
oldest are removed automatically when a new session starts.

- `/sessions` opens a picker. Use the up and down arrow keys to move, Enter to
  resume, and `q` or `Esc` to cancel.
- `/sessions <id>` resumes a conversation directly. New turns keep appending to
  the same log.

The logs are plain text, so you can grep, back up, or share them with any JSONL
tool.

---

## 6. Configuration reference

Precedence: defaults, then JSON config file, then environment variables.

Environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter key |
| `GROQ_API_KEY` | Groq key |
| `LITTLE_TUI_PROVIDER` | Active provider |
| `LITTLE_TUI_MODEL` | Model slug |
| `LITTLE_TUI_WORKSPACE` | Default workspace |
| `LITTLE_TUI_MAX_STEPS` | Max tool steps per turn (default 50) |
| `LITTLE_TUI_MAX_COST` | Max dollars per turn (default 2.0) |
| `LITTLE_TUI_MAX_TOKENS` | Max output tokens (default 4096) |
| `LITTLE_TUI_ALLOW_SHELL` | Auto-approve shell tools |
| `LITTLE_TUI_ALLOW_ALL` | Auto-approve every tool call (same as `--yolo`) |
| `LITTLE_TUI_SYSTEM_PROMPT_FILE` | Path to a custom system prompt |

### Providers

| Provider | Key variable | Default model |
| --- | --- | --- |
| `openrouter` | `OPENROUTER_API_KEY` | `nvidia/nemotron-3-super-120b-a12b:free` |
| `groq` | `GROQ_API_KEY` | `openai/gpt-oss-120b` |

Pick a provider with `--provider`, `LITTLE_TUI_PROVIDER`, the `provider` key in
the config file, or `/provider` in the REPL. Switching also switches the API
endpoint, headers, and default model.

Groq note: the free tier caps `openai/gpt-oss-120b` at 8000 tokens per minute,
which is tight for tool-heavy work. Use `llama-3.3-70b-versatile` instead. The
installer sets this model automatically when you pick Groq.

---

## 7. Troubleshooting

| Problem | Fix |
| --- | --- |
| `OPENROUTER_API_KEY is not set` | Export the key for your active provider (see [section 3](#3-set-up-your-api-key)). |
| `GROQ_API_KEY is not set` | Export `GROQ_API_KEY` or set `api_key` in the config file. |
| `unknown provider 'x'` | Use `openrouter` or `groq`. |
| `Request too large ... TPM` (Groq) | Free-tier limit. Switch to `llama-3.3-70b-versatile` or wait a minute. |
| `little-tui: command not found` | Run `./install.sh` again and make sure the bin folder is on your PATH. |
| `ModuleNotFoundError: little_tui` | Reinstall the package. |
