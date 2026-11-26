# frakobla llm

errything local :)

install (requires uv) and run:

`uv tool install git+ssh://git@github.com/tollefj/frakobla-agent.git`

The first run asks for your llama-server URL (e.g. `http://192.168.0.10:8080`) and stores it in global settings under `~.config/`.

some neat commands:

```sh
frak                         
frak -p "explain src/frakobla_llm/loop.py"  
frak --continue              
frak --resume [number|text]  
frak --mode ask|readonly|yolo 
frak --model <id> --base-url <url>
frak --doctor  # check server, template, tool calls and MCP servers
```

or just run -h for a full list.

## tooools

`read_file`, `write_file`, `edit_file`, `list_dir`, `glob`, `grep`, `bash`. With `tools.interact` on (default) the model also gets `update_plan` (checklist above the prompt), `ask_user` (numbered choices) and `suggest_next` (follow-up suggestions).

`AGENTS.md` files are added to the system prompt: `~/.config/frakobla-llm/AGENTS.md`, and also from the cwd.

## permissions!

- `ask` is default: asks before write, exec and MCP tools
- `readonly` denies them, also when an allow rule matches
- `yolo` for speeeed and uncertainty.

Regardless of mode:

- sensitive files (`.env`, `*.pem`, `*.key`, `id_rsa*`, ...) are denied, and `grep` skips them
- paths outside the working directory ask
- dangerous bash commands (`sudo`, `rm -rf`, `curl | sh`, force push, ...) ask with a warning, except in yolo ofc
- the same call three times in a row asks, four times is denied

At a prompt you answer `y` or Enter (once), `s` (this session), `a` (always, written to `~/.config/frakobla-llm/config.toml`, so it applies in every project. it was tedious to work with local settings), `n` or a free-text reason to deny. 

example:

```toml
[[permissions.rules.allow.bash]]
pattern = "uv run pytest*"

[[permissions.rules.deny.write_file]]
pattern = "uv.lock"
kind = "exact"
```

## config

`~/.config/frakobla-llm/config.toml` is merged with `.frakobla-llm/config.toml` in the working directory (project wins). Unknown keys give a warning.

The `server`, `permissions` and `mcp` sections of a project config are only used after you have trusted the file. `frak` asks once per file and again when the file is changed by something other than `frak` itself. Trusted hashes are in `~/.config/frakobla-llm/trusted.json`. 

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

## context + sessions

When the context estimate reaches `compact_at` of the server's context size, older messages are summarized into the system prompt and the recent tail is kept.

A context overflow from the server triggers one compact and retry, this is to avoid the crash that occurs in the native llama.cpp-interface.

Sessions are JSONL files under `~/.config/frakobla-llm/sessions/<cwd>/`, created on the first message. Before each `write_file` or `edit_file` the original is snapshotted so `/undo` can restore it. Input history is in `~/.config/frakobla-llm/history`. Config, trust hashes, sessions, history and MCP logs left in `~/.config/llm` by older versions are moved over at startup, existing files are not overwritten.

## MCP

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

