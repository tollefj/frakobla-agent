# frakobla llm

A minimal agentic CLI for a local llama.cpp server (`llama-server`). It streams replies in your terminal, calls file and shell tools with permission prompts, keeps sessions you can resume and undo, and can use MCP servers.

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/) and a running `llama-server` started with `--jinja` so the chat template supports tool calls. Router mode (several models) works too.

## Install

```sh
make install   # put `frak` on your PATH, installed from GitHub (origin)
make update    # reinstall the latest from GitHub
```

Without installing the tool, run it with `uv run frak`.

Without a clone: `uv tool install git+ssh://git@github.com/tollefj/frakobla-agent.git`. The first interactive run without `~/.config/frakobla-llm/config.toml` asks for the llama-server URL (e.g. `http://192.168.0.10:8080`) and writes it there with permission mode `ask`. `--base-url` or `FRAK_BASE_URL` skip the question.

## Versioning

The version comes from the latest `v*` git tag (`frak --version`). `make versioning` tags HEAD with the next minor version and `make versioning major` with the next major one. Commit first, then `git push origin <tag>` so `make update` picks it up.

## Usage

```sh
frak                         # interactive session
frak -p "explain src/frakobla_llm/loop.py"  # one prompt, headless, exit codes 0 / 1 / 130
frak --continue              # continue the latest session in this directory
frak --resume [number|text]  # pick an earlier session
frak --mode ask|readonly|yolo # permission mode (--yolo is a shorthand)
frak --model <id> --base-url <url>
frak --no-session            # do not write a session file
frak --doctor                # check server, template, tool calls and MCP servers, then exit
```

`FRAK_BASE_URL` and `FRAK_MODEL` override the config. `frak --help` lists the options. `make` lists install, update, versioning and dev tasks.

### In the prompt

| Input | Effect |
| --- | --- |
| `/help` | list commands |
| `/new`, `/clear` | new session with empty context |
| `/resume [number\|text]` | switch to an earlier session |
| `/rename <title>` | name the session |
| `/compact [focus]` | summarize the history to free context |
| `/undo` | revert the last turn's file edits and drop it from history (bash changes are not undone) |
| `/retry` | undo the last turn and send its prompt again |
| `/plan [clear]` | show the plan checklist, or drop it |
| `/summary` | show the compaction summary |
| `/context` | token breakdown: system prompt, summary and plan, tool schemas, messages |
| `/model [number\|text]` | pick a model; unloads the loaded model(s) on the server, then loads the new one (router mode) |
| `/unload` | unload the current model from the server, the next request loads it again |
| `/mode [ask\|readonly\|yolo]` | show or set the permission mode |
| `/tools` | list tools |
| `/tools load <name\|keywords>` | load hidden MCP tools in lazy mode |
| `/perms` | show permission rules and session grants |
| `/mcp [enable\|disable <server>]` | list MCP servers or toggle one (saved to config) |
| `/stats` | token usage and speed |
| `/export [path]` | write the conversation as markdown |
| `/doctor` | run the health checks |
| `/quit` | exit |
| `!cmd` | run a shell command locally |
| `!!cmd` | run it and attach the output to your next message |
| `1`-`3` | then Enter, send that suggested follow-up |
| `@path` | complete a file path |
| Alt-Enter | newline |
| Ctrl-C | cancel the running turn, or clear the input |
| Ctrl-C twice, Ctrl-D | quit |

## Tools

Built in: `read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`, `bash`. With `tools.interact` on (default) the model also gets `update_plan` (checklist above the prompt), `ask_user` (numbered choices) and `suggest_next` (follow-up suggestions). Headless mode only registers `update_plan`.

`AGENTS.md` files are added to the system prompt: `~/.config/frakobla-llm/AGENTS.md`, then every `AGENTS.md` from the git root down to the working directory.

## Permissions

Modes: `ask` (default) asks before write, exec and MCP tools; `readonly` denies them, also when an allow rule matches; `yolo` allows them. Read tools are always allowed inside the working directory.

Regardless of mode:

- sensitive files (`.env`, `*.pem`, `*.key`, `id_rsa*`, ...) are denied, and `grep` skips them
- paths outside the working directory ask
- dangerous bash commands (`sudo`, `rm -rf`, `curl | sh`, force push, ...) ask with a warning, except in yolo
- the same call three times in a row asks, four times is denied

At a prompt you answer `y` or Enter (once), `s` (this session), `a` (always, written to `~/.config/frakobla-llm/config.toml`, so it applies in every project), `n` or a free-text reason to deny. A bash grant covers that exact command. A file or MCP tool grant covers every call of that tool. Enter is not accepted for dangerous commands or paths outside the working directory, those need a typed answer.

Rules per tool, with `kind` `glob` (default), `regex` or `exact`. Glob and regex allow rules do not match bash commands with `;`, `|`, `&`, `$(`, backticks or redirects; only exact rules do.

```toml
[[permissions.rules.allow.bash]]
pattern = "uv run pytest*"

[[permissions.rules.deny.write_file]]
pattern = "uv.lock"
kind = "exact"
```

## Configuration

`~/.config/frakobla-llm/config.toml` is merged with `.frakobla-llm/config.toml` in the working directory (project wins). Unknown keys give a warning.

The `server`, `permissions` and `mcp` sections of a project config are only used after you have trusted the file. `frak` asks once per file and again when the file is changed by something other than `frak` itself. Trusted hashes are in `~/.config/frakobla-llm/trusted.json`. Headless runs never ask and ignore those sections of an untrusted file.

Defaults:

```toml
[server]
base_url = "http://127.0.0.1:8080"
model = ""            # empty picks the loaded model
api_key = ""
timeout = 600

[agent]
max_iterations = 50
parallel_tool_calls = false
thinking = "default"   # default, off, low, medium, high

[permissions]
mode = "ask"

[context]
auto_compact = true
compact_at = 0.75     # fraction of the context size

[tools]
output_max_chars = 12000
bash_timeout = 120
interact = true

[mcp]
connect_timeout = 15
lazy = "auto"         # true, false, or auto (on above 12 MCP tools)

[ui]
show_reasoning = false
```

## Context and sessions

When the context estimate reaches `compact_at` of the server's context size, older messages are summarized into the system prompt and the recent tail is kept. A context overflow from the server triggers one compact and retry.

Sessions are JSONL files under `~/.config/frakobla-llm/sessions/<cwd>/`, created on the first message. Before each `write_file` or `edit_file` the original is snapshotted so `/undo` can restore it. Input history is in `~/.config/frakobla-llm/history`. Config, trust hashes, sessions, history and MCP logs left in `~/.config/llm` by older versions are moved over at startup, existing files are not overwritten.

## MCP servers

```toml
[mcp.servers.github]
url = "https://api.githubcopilot.com/mcp/"
headers = { Authorization = "Bearer ${GITHUB_TOKEN}" }

[mcp.servers.files]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
env = { DEBUG = "0" }
timeout = 60
enabled = true
```

`${VAR}` is read from the environment. `proxy = true` on a URL server routes it through llama-server's `/cors-proxy` (start llama-server with `--webui-mcp-proxy`). Stdio stderr goes to `~/.config/frakobla-llm/mcp/<name>.log`. A server that drops its connection is restarted on the next call, and the model is told to send the call again.

Tools show up as `mcp__<server>__<tool>` and ask for permission by default. In lazy mode they are hidden from the model, which finds and loads them with `mcp_search`.

Managing servers from the shell:

```sh
frak mcp list
frak mcp add github --url https://<host>/mcp --header "Authorization=Bearer ${GITHUB_TOKEN}"
frak mcp add fs --command npx --arg -y --arg <package> --scope project
frak mcp check github          # connect and list tools
frak mcp remove github
```

`./scripts/add-mcp.sh` in a checkout asks the same questions interactively.

## Development

```sh
make test
make lint
```
