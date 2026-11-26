from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import httpx

from frakobla_llm.config import (
    ConfigError,
    config_path,
    find_server_file,
    load_config,
    remove_server,
    write_server,
)
from frakobla_llm.mcp_client import McpManager, parse_servers

NAME = re.compile(r"^[A-Za-z0-9_-]+$")
EXIT_OK, EXIT_FAILED = 0, 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m frakobla_llm.mcp_config", description="Manage MCP servers in the frak config")
    commands = parser.add_subparsers(dest="action", required=True)

    commands.add_parser("list", help="list configured servers")

    add = commands.add_parser("add", help="add or replace a server")
    add.add_argument("name")
    target = add.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="streamable HTTP endpoint")
    target.add_argument("--command", help="executable for a stdio server")
    add.add_argument("--arg", action="append", default=[], help="stdio argument, repeatable")
    add.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="stdio environment, repeatable")
    add.add_argument("--header", action="append", default=[], metavar="NAME=VALUE", help="HTTP header, repeatable")
    add.add_argument("--proxy", action="store_true", help="connect through llama-server's /cors-proxy")
    add.add_argument("--timeout", type=float, help="seconds per tool call")
    add.add_argument("--disabled", action="store_true", help="add it switched off")
    add.add_argument("--scope", choices=["global", "project"], default="global")
    add.add_argument("--force", action="store_true", help="replace an existing server with the same name")

    check = commands.add_parser("check", help="connect to a server and list its tools")
    check.add_argument("name")

    remove = commands.add_parser("remove", help="remove a server from the config that defines it")
    remove.add_argument("name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = Path.cwd()
    try:
        return {"list": list_servers, "add": add_server, "check": check_server, "remove": remove_entry}[args.action](
            args, cwd
        )
    except (ConfigError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_FAILED


def add_server(args: argparse.Namespace, cwd: Path) -> int:
    if not NAME.match(args.name):
        raise ValueError("name may only contain letters, digits, _ and -")
    existing = find_server_file(cwd, args.name)
    if existing and not args.force:
        raise ValueError(f"{args.name} already exists in {existing}, pass --force to replace it")
    if args.proxy and not args.url:
        raise ValueError("--proxy needs --url")
    path = config_path(cwd, args.scope)
    write_server(path, args.name, server_entry(args))
    print(f"saved {args.name} to {path}")
    return EXIT_OK


def server_entry(args: argparse.Namespace) -> dict:
    entry: dict = {}
    if args.url:
        entry["url"] = args.url
        if args.proxy:
            entry["proxy"] = True
        if args.header:
            entry["headers"] = pairs(args.header)
    else:
        entry["command"] = args.command
        if args.arg:
            entry["args"] = args.arg
        if args.env:
            entry["env"] = pairs(args.env)
    if args.timeout:
        entry["timeout"] = args.timeout
    if args.disabled:
        entry["enabled"] = False
    return entry


def pairs(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, rest = value.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"expected KEY=VALUE, got {value!r}")
        result[key.strip()] = rest
    return result


def list_servers(_: argparse.Namespace, cwd: Path) -> int:
    config = load_config(cwd)
    configs, warnings = parse_servers(config["mcp"]["servers"])
    for warning in [*config.warnings, *warnings]:
        print(f"warning: {warning}", file=sys.stderr)
    if not configs:
        print("no MCP servers configured, add one with: make mcp")
        return EXIT_OK
    for config in configs:
        state = "on " if config.enabled else "off"
        target = config.url or " ".join([config.command, *config.args])
        print(f"{state}  {config.name:<16} {config.transport:<22} {target}  [{find_server_file(cwd, config.name)}]")
    return EXIT_OK


def check_server(args: argparse.Namespace, cwd: Path) -> int:
    config = load_config(cwd)
    base_url = os.environ.get("FRAK_BASE_URL") or config["server"]["base_url"]
    configs, _ = parse_servers(config["mcp"]["servers"])
    server_config = next((entry for entry in configs if entry.name == args.name), None)
    if server_config is None:
        hint = "".join(f" ({warning})" for warning in config.warnings if "not trusted" in warning)
        raise ValueError(f"no MCP server named {args.name}{hint}")
    if server_config.proxy and not proxy_available(base_url):
        print(f"warning: {base_url} does not report cors_proxy_enabled, start llama-server with --webui-mcp-proxy")
    manager = McpManager(
        [server_config],
        cwd,
        config["mcp"]["connect_timeout"],
        proxy_base=base_url,
        proxy_api_key=config["server"]["api_key"],
    )
    try:
        manager.connect([args.name])
        server = manager.servers[args.name]
        if not server.connected:
            print(f"FAIL  {args.name}: {server.error}")
            if server_config.proxy:
                print("      through the proxy, llama-server must be allowed to reach the target address")
            return EXIT_FAILED
        print(f"ok    {args.name} ({server_config.transport}), {len(server.tools)} tools")
        for tool in manager.tools(server):
            print(f"      {tool.name}  {tool.description}")
        return EXIT_OK
    finally:
        manager.close()


def remove_entry(args: argparse.Namespace, cwd: Path) -> int:
    path = find_server_file(cwd, args.name)
    if path is None or not remove_server(path, args.name):
        raise ValueError(f"no MCP server named {args.name}")
    print(f"removed {args.name} from {path}")
    return EXIT_OK


def proxy_available(base_url: str) -> bool:
    try:
        return bool(httpx.get(f"{base_url.rstrip('/')}/props", timeout=5).json().get("cors_proxy_enabled"))
    except (httpx.HTTPError, ValueError):
        return False


if __name__ == "__main__":
    sys.exit(main())
