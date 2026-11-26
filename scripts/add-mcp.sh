#!/usr/bin/env bash
# Interactive wizard for `frak mcp add`: asks for an MCP server and writes it to the frak config.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mcp_config() { uv run --quiet --project "$PROJECT_DIR" frak mcp "$@"; }

ask() {
    local prompt="$1" default="${2:-}" reply
    if [[ -n "$default" ]]; then
        read -r -p "$prompt [$default]: " reply
        echo "${reply:-$default}"
    else
        read -r -p "$prompt: " reply
        echo "$reply"
    fi
}

ask_required() {
    local reply=""
    while [[ -z "$reply" ]]; do reply="$(ask "$@")"; done
    echo "$reply"
}

echo "Add an MCP server to frak. Use \${VAR} in any value to read it from the environment at startup."
echo

name=""
until [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]]; do
    name="$(ask_required "Name (letters, digits, _ and -)")"
done

echo
echo "How does frak reach it?"
echo "  1) HTTP URL (streamable HTTP)"
echo "  2) HTTP URL through llama-server's proxy (needs llama-server --webui-mcp-proxy)"
echo "  3) Local command (stdio)"
transport="$(ask "Choice" 1)"

args=(add "$name")
case "$transport" in
    1|2)
        url=""
        until [[ "$url" =~ ^https?:// ]]; do
            url="$(ask_required "URL, e.g. https://example.com/mcp")"
        done
        args+=(--url "$url")
        [[ "$transport" == 2 ]] && args+=(--proxy)
        echo "Headers, one per line as Name: value (for example Authorization: Bearer \${TOKEN}). Empty line to finish."
        while true; do
            header="$(ask "Header")"
            [[ -z "$header" ]] && break
            if [[ "$header" != *:* ]]; then
                echo "  expected Name: value"
                continue
            fi
            header_name="${header%%:*}"
            header_value="${header#*:}"
            args+=(--header "${header_name}=${header_value# }")
        done
        ;;
    3)
        command="$(ask_required "Command, e.g. npx or uvx")"
        args+=(--command "$command")
        read -r -p "Arguments, separated by spaces: " -a command_args
        for value in "${command_args[@]+"${command_args[@]}"}"; do
            args+=(--arg "$value")
        done
        echo "Environment variables, one per line as KEY=value. Empty line to finish."
        while true; do
            env_line="$(ask "Variable")"
            [[ -z "$env_line" ]] && break
            if [[ "$env_line" != *=* ]]; then
                echo "  expected KEY=value"
                continue
            fi
            args+=(--env "$env_line")
        done
        ;;
    *)
        echo "unknown choice $transport" >&2
        exit 1
        ;;
esac

echo
echo "Where should it be saved?"
echo "  1) global config, ~/.config/frakobla-llm/config.toml (all directories)"
echo "  2) project config, .frakobla-llm/config.toml in $(pwd)"
scope="$(ask "Choice" 1)"
[[ "$scope" == 2 ]] && args+=(--scope project) || args+=(--scope global)

errors="$(mktemp)"
trap 'rm -f "$errors"' EXIT
if ! mcp_config "${args[@]}" 2>"$errors"; then
    cat "$errors" >&2
    if grep -q "already exists" "$errors" && [[ "$(ask "Replace it? y/n" n)" == y ]]; then
        mcp_config "${args[@]}" --force
    else
        exit 1
    fi
fi

echo
if [[ "$(ask "Test the connection now? y/n" y)" == y ]]; then
    mcp_config check "$name" || echo "The server is saved anyway. Fix it in the config, then run frak mcp check $name"
fi
