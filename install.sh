#!/usr/bin/env bash
# little-tui installer + setup wizard.
#
# Builds a virtualenv, installs the package, links the `little-tui` command,
# and (optionally) collects your API key so you can start right away.
#
# Usage:  ./install.sh            (interactive; asks for provider + API key)
#         NONINTERACTIVE=1 ./install.sh   (skip the API-key prompt)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

# Termux puts executables in $PREFIX/bin; standard distros use ~/.local/bin.
if [[ -n "${PREFIX:-}" ]]; then
    BIN_DIR="${BIN_DIR:-$PREFIX/bin}"
else
    BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
fi

VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
CONFIG_DIR="${CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/little-tui}"
CONFIG_FILE="$CONFIG_DIR/config.json"

log()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null 2>&1 \
    || die "$PYTHON not found on PATH. Install Python 3.10+ first (on Termux: pkg install python)."

log "creating virtualenv at $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"

log "installing little-tui"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$VENV_DIR/bin/python" -m pip install --quiet -e "$REPO_DIR"

log "linking command into $BIN_DIR"
mkdir -p "$BIN_DIR"
ln -sf "$VENV_DIR/bin/little-tui" "$BIN_DIR/little-tui"
"$BIN_DIR/little-tui" --version >/dev/null 2>&1 \
    || die "install failed: $BIN_DIR/little-tui is not runnable"

# ---------------------------------------------------------------------------
# API-key setup wizard (skipped when NONINTERACTIVE=1 or a key already exists)
# ---------------------------------------------------------------------------
has_key() {
    [[ -f "$CONFIG_FILE" ]] \
        && grep -q '"api_key"' "$CONFIG_FILE" \
        && ! grep -q '"api_key"[[:space:]]*:[[:space:]]*""' "$CONFIG_FILE"
}

if [[ "${NONINTERACTIVE:-0}" == "1" ]] || has_key; then
    log "keeping existing API key in $CONFIG_FILE"
else
    log "setting up your API key"
    mkdir -p "$CONFIG_DIR"

    echo
    echo "Which provider do you want to use?"
    echo "  1) openrouter  - many models in one place (default)"
    echo "  2) groq        - fast, free tier available"
    read -r -p "Choice [1/2]: " choice || true

    case "$choice" in
        2|groq)
            provider="groq"
            model="llama-3.3-70b-versatile"
            key_url="https://console.groq.com/keys"
            key_hint="gsk_..."
            ;;
        *)
            provider="openrouter"
            model="nvidia/nemotron-3-super-120b-a12b:free"
            key_url="https://openrouter.ai/settings/keys"
            key_hint="sk-or-v1-..."
            ;;
    esac

    echo
    echo "Get a free key at: $key_url"
    read -r -p "Paste your $provider API key (starts with $key_hint): " api_key || true
    api_key="$(printf '%s' "$api_key" | tr -d '[:space:]')"
    [[ -n "$api_key" ]] || die "no API key provided. You can export ${provider^^} key later (see README)."

    cat > "$CONFIG_FILE" <<EOF
{
  "provider": "$provider",
  "api_key": "$api_key",
  "model": "$model",
  "max_tokens": 4096
}
EOF
    log "saved your key to $CONFIG_FILE (plain text; keep the file private)"
fi

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo
    echo "Add $BIN_DIR to your PATH so you can run little-tui anywhere:"
    echo "    echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc"
    echo "    source ~/.bashrc"
fi

cat <<EOF

Installed. Run it:

    little-tui                     # interactive terminal agent
    little-tui --help              # all options

Inside the REPL: /help for commands, /sessions to resume past conversations.
Ctrl+C interrupts a response.

Switch providers / models any time:
    little-tui --provider groq --model llama-3.3-70b-versatile
    export OPENROUTER_API_KEY=sk-or-v1-...    # set a key for the other provider
    export GROQ_API_KEY=gsk_...

Uninstall:
    rm -rf "$VENV_DIR" "$BIN_DIR/little-tui"
EOF
