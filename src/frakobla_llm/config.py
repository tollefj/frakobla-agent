from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import tomlkit
from tomlkit.items import InlineTable

CONFIG_DIR = Path.home() / ".config" / "frakobla-llm"
GLOBAL_CONFIG = CONFIG_DIR / "config.toml"
PROJECT_CONFIG = Path(".frakobla-llm") / "config.toml"
TRUSTED_FILE = CONFIG_DIR / "trusted.json"
LEGACY_DATA_DIR = Path.home() / ".config" / "llm"
LEGACY_DATA_NAMES = ("config.toml", "trusted.json", "AGENTS.md", "history", "sessions", "mcp")
TRUST_GATED_SECTIONS = ("server", "permissions", "mcp")

TrustPrompt = Callable[[Path, list[str]], bool]

DEFAULTS: dict = {
    "server": {
        "base_url": "http://127.0.0.1:8080",
        "model": "",
        "api_key": "",
        "timeout": 600,
    },
    "agent": {"max_iterations": 50, "parallel_tool_calls": False, "thinking": "default"},
    "permissions": {"mode": "ask", "rules": {"allow": {}, "ask": {}, "deny": {}}},
    "context": {"auto_compact": True, "compact_at": 0.75},
    "tools": {"output_max_chars": 12000, "bash_timeout": 120, "interact": True},
    "mcp": {"connect_timeout": 15, "lazy": "auto", "servers": {}},
    "ui": {"show_reasoning": False, "spinner": "dots", "status_bar": True, "gpu_watts": 400},
}

FREE_FORM_SECTIONS = {
    ("permissions", "rules", "allow"),
    ("permissions", "rules", "ask"),
    ("permissions", "rules", "deny"),
    ("mcp", "servers"),
}


class ConfigError(Exception):
    pass


@dataclass
class Config:
    values: dict
    sources: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, section: str) -> dict:
        return self.values[section]


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def unknown_keys(values: dict, schema: dict, path: tuple[str, ...] = ()) -> list[str]:
    if path in FREE_FORM_SECTIONS:
        return []
    unknown = []
    for key, value in values.items():
        here = (*path, key)
        if key not in schema:
            unknown.append(".".join(here))
        elif isinstance(value, dict) and isinstance(schema[key], dict):
            unknown.extend(unknown_keys(value, schema[key], here))
    return unknown


def load_config(cwd: Path, global_path: Path = GLOBAL_CONFIG, trust: TrustPrompt | None = None) -> Config:
    config = Config(values=copy.deepcopy(DEFAULTS))
    project_path = cwd / PROJECT_CONFIG
    for path in (global_path, project_path):
        if not path.is_file():
            continue
        try:
            loaded = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path}: {error}") from error
        if path == project_path:
            loaded = _without_untrusted_sections(path, loaded, trust, config.warnings)
        config.warnings += [f"{path}: unknown key {key}" for key in unknown_keys(loaded, DEFAULTS)]
        config.values = deep_merge(config.values, loaded)
        config.sources.append(path)
    return config


def is_trusted(path: Path) -> bool:
    return path.is_file() and _trusted_digests().get(str(path.resolve())) == _digest(path)


def remember_trust(path: Path) -> None:
    digests = _trusted_digests()
    digests[str(path.resolve())] = _digest(path)
    TRUSTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRUSTED_FILE.write_text(json.dumps(digests, indent=2) + "\n", encoding="utf-8")


def migrate_legacy_data(legacy: Path = LEGACY_DATA_DIR, target: Path | None = None) -> int:
    target = target or CONFIG_DIR
    moved = 0
    for name in LEGACY_DATA_NAMES:
        source = legacy / name
        files = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
        for file in files:
            destination = target / file.relative_to(legacy)
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(file, destination)
            moved += 1
    return moved


def append_rule(path: Path, tier: str, tool: str, pattern: str, kind: str) -> Path:
    keeps_trust = _only_written_by_us(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"\n[[permissions.rules.{tier}.{tool}]]\n"
        f"pattern = {json.dumps(pattern)}\n"
        f"kind = {json.dumps(kind)}\n"
    )
    with path.open("a", encoding="utf-8") as file:
        file.write(entry)
    if keeps_trust:
        remember_trust(path)
    return path


def config_path(cwd: Path, scope: str, global_path: Path = GLOBAL_CONFIG) -> Path:
    return cwd / PROJECT_CONFIG if scope == "project" else global_path


def find_server_file(cwd: Path, name: str, global_path: Path = GLOBAL_CONFIG) -> Path | None:
    for path in (cwd / PROJECT_CONFIG, global_path):
        if name in _servers_table(_read_document(path)):
            return path
    return None


def set_server_enabled(cwd: Path, name: str, enabled: bool, global_path: Path = GLOBAL_CONFIG) -> Path | None:
    path = find_server_file(cwd, name, global_path)
    if path is None:
        return None
    document = _read_document(path)
    _servers_table(document)[name]["enabled"] = enabled
    _write_document(path, document)
    return path


def write_server(path: Path, name: str, entry: dict) -> None:
    document = _read_document(path)
    if "mcp" not in document:
        document["mcp"] = tomlkit.table()
    if "servers" not in document["mcp"]:
        document["mcp"]["servers"] = tomlkit.table(is_super_table=True)
    table = tomlkit.table()
    for key, value in entry.items():
        table[key] = _inline(value) if isinstance(value, dict) else value
    document["mcp"]["servers"][name] = table
    _write_document(path, document)


def remove_server(path: Path, name: str) -> bool:
    document = _read_document(path)
    servers = _servers_table(document)
    if name not in servers:
        return False
    del servers[name]
    _write_document(path, document)
    return True


def _inline(values: dict) -> InlineTable:
    table = tomlkit.inline_table()
    table.update(values)
    return table


def _read_document(path: Path) -> tomlkit.TOMLDocument:
    if not path.is_file():
        return tomlkit.document()
    return tomlkit.parse(path.read_text(encoding="utf-8"))


def _servers_table(document: tomlkit.TOMLDocument) -> dict:
    return document.get("mcp", {}).get("servers", {})


def _write_document(path: Path, document: tomlkit.TOMLDocument) -> None:
    keeps_trust = _only_written_by_us(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(document), encoding="utf-8")
    if keeps_trust:
        remember_trust(path)


def _without_untrusted_sections(path: Path, loaded: dict, trust: TrustPrompt | None, warnings: list[str]) -> dict:
    gated = [section for section in TRUST_GATED_SECTIONS if section in loaded]
    if not gated or is_trusted(path):
        return loaded
    if trust is not None and trust(path, gated):
        remember_trust(path)
        return loaded
    warnings.append(f"{path}: not trusted, ignoring {', '.join(gated)}")
    return {section: values for section, values in loaded.items() if section not in gated}


def _only_written_by_us(path: Path) -> bool:
    return not path.exists() or is_trusted(path)


def _trusted_digests() -> dict[str, str]:
    try:
        digests = json.loads(TRUSTED_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return digests if isinstance(digests, dict) else {}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
