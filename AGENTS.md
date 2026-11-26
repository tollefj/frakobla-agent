Agentic CLI for llama-server. Python 3.13, uv. Code in src/frakobla_llm/, tests in tests/.

- Run: `uv run frak`. Tests: `uv run pytest -q`. Lint: `uv run ruff check src tests`.
- New config keys go in `config.DEFAULTS`.
- History must stay valid: an assistant `tool_calls` message always keeps its tool results.
- `mcp_client.py` must not be renamed to `mcp.py`.
- Minimal comments, self-explanatory code.
