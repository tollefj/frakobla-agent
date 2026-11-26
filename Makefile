.DEFAULT_GOAL := help
.PHONY: help install update uninstall versioning major test lint clean

UV ?= uv
ORIGIN := $(shell git remote get-url origin 2>/dev/null || echo https://github.com/tollefj/frakobla-agent.git)
GIT_SOURCE ?= git+$(subst git@github.com:,ssh://git@github.com/,$(ORIGIN))

help: ## list targets (run frak --help for the CLI itself)
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## install frak on your PATH from GitHub
	$(UV) tool install --force "$(GIT_SOURCE)"

update: ## reinstall frak and its dependencies from GitHub
	$(UV) tool install --reinstall --force "$(GIT_SOURCE)"

uninstall: ## remove frak from your PATH
	$(UV) tool uninstall frakobla-llm

versioning: ## tag the next minor version (make versioning major for a major bump)
	@./scripts/bump-version.sh $(if $(filter major,$(MAKECMDGOALS)),major,minor)

major:
	@:

test: ## run tests
	$(UV) run pytest -q

lint: ## run ruff
	$(UV) run ruff check src tests

clean: ## remove caches
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} +
