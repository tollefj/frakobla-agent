.DEFAULT_GOAL := help
.PHONY: help install install-tool reinstall-tool clean-install-tool uninstall-tool run continue resume ask doctor test test-fast lint mcp mcp-list mcp-check mcp-remove config sessions clean

UV ?= uv
FRAK := $(UV) run frak
MCP := $(UV) run python -m frakobla_llm.mcp_config
SESSIONS_DIR := $(HOME)/.config/frakobla-llm/sessions

help: ## list targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## install dependencies into .venv
	$(UV) sync

install-tool: ## install the frak command on your PATH (editable, follows this checkout)
	$(UV) tool install --editable . --force

reinstall-tool: ## rebuild the frak command and all its dependencies
	$(UV) tool install --editable . --reinstall --force

clean-install-tool: ## uninstall, clear the uv cache for this package, then install fresh
	-$(UV) tool uninstall frakobla-llm
	$(UV) cache clean frakobla-llm
	$(UV) tool install --editable .

uninstall-tool: ## remove the frak command from your PATH
	$(UV) tool uninstall frakobla-llm

run: ## start an interactive session
	$(FRAK)

continue: ## continue the latest session in this directory
	$(FRAK) --continue

resume: ## pick an earlier session in this directory
	$(FRAK) --resume

ask: ## run one prompt headless: make ask P="explain src/frakobla_llm/loop.py"
	@test -n "$(P)" || (echo 'usage: make ask P="your prompt"' && exit 1)
	$(FRAK) -p "$(P)"

doctor: ## check server, chat template, tool calls and MCP servers
	$(FRAK) --doctor

test: ## run all tests
	$(UV) run pytest -q

test-fast: ## run tests except the ones that start MCP servers
	$(UV) run pytest -q -k "not stdio_server"

lint: ## run ruff
	$(UV) run ruff check src tests

mcp: ## add an MCP server interactively (URL, llama-server proxy or local command)
	@./scripts/add-mcp.sh

mcp-list: ## list configured MCP servers
	@$(MCP) list

mcp-check: ## connect to an MCP server and list its tools: make mcp-check NAME=github
	@test -n "$(NAME)" || (echo "usage: make mcp-check NAME=<server>" && exit 1)
	@$(MCP) check "$(NAME)"

mcp-remove: ## remove an MCP server from the config: make mcp-remove NAME=github
	@test -n "$(NAME)" || (echo "usage: make mcp-remove NAME=<server>" && exit 1)
	@$(MCP) remove "$(NAME)"

config: ## show the config files in use
	@for file in "$(HOME)/.config/frakobla-llm/config.toml" ".frakobla-llm/config.toml"; do \
		if [ -f "$$file" ]; then echo "== $$file"; cat "$$file"; echo; else echo "== $$file (missing)"; fi; \
	done

sessions: ## list saved sessions for this directory
	@ls -1t "$(SESSIONS_DIR)/$$(pwd | sed 's|^/||; s|/|-|g')" 2>/dev/null | grep '\.jsonl$$' || echo "no sessions for $$(pwd)"

clean: ## remove caches
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
