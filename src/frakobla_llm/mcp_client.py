from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO
from urllib.parse import quote

import httpx2
from mcp import Client, MCPError, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from frakobla_llm.config import CONFIG_DIR
from frakobla_llm.tools.registry import Registry, Risk, Tool, ToolError

LOG_DIR = CONFIG_DIR / "mcp"
MAX_DESCRIPTION_CHARS = 300
DEFAULT_CALL_TIMEOUT = 60
SHUTDOWN_SECONDS = 3
HTTP_TIMEOUT_SECONDS = 30
HTTP_READ_TIMEOUT_SECONDS = 300
VARIABLE = re.compile(r"\$\{(\w+)\}")
UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


@dataclass(frozen=True)
class ServerConfig:
    name: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_CALL_TIMEOUT
    enabled: bool = True
    proxy: bool = False

    @property
    def transport(self) -> str:
        if not self.url:
            return "stdio"
        return "http via llama-server" if self.proxy else "http"


@dataclass
class McpServer:
    config: ServerConfig
    tools: list[mcp_types.Tool] = field(default_factory=list)
    error: str = ""
    client: Client | None = None
    stop: asyncio.Event | None = None
    task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self.client is not None

    @property
    def group(self) -> str:
        return f"mcp:{self.config.name}"


def substitute(value: str, environ: Mapping[str, str], missing: set[str]) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in environ:
            missing.add(name)
        return environ.get(name, "")

    return VARIABLE.sub(replace, value)


def parse_servers(raw: dict, environ: Mapping[str, str] = os.environ) -> tuple[list[ServerConfig], list[str]]:
    configs, warnings = [], []
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            warnings.append(f"mcp server {name}: expected a table")
            continue
        missing: set[str] = set()
        sub = lambda value, missing=missing: substitute(str(value), environ, missing)  # noqa: E731
        try:
            config = ServerConfig(
                name=name,
                command=sub(entry.get("command", "")),
                args=tuple(sub(arg) for arg in entry.get("args", [])),
                env={key: sub(value) for key, value in (entry.get("env") or {}).items()},
                url=sub(entry.get("url", "")),
                headers={key: sub(value) for key, value in (entry.get("headers") or {}).items()},
                timeout=float(entry.get("timeout", DEFAULT_CALL_TIMEOUT)),
                enabled=bool(entry.get("enabled", True)),
                proxy=bool(entry.get("proxy", False)),
            )
        except (TypeError, ValueError, AttributeError) as error:
            warnings.append(f"mcp server {name}: invalid entry ({error})")
            continue
        if not config.command and not config.url:
            warnings.append(f"mcp server {name}: needs command or url")
            continue
        if config.proxy and not config.url:
            warnings.append(f"mcp server {name}: proxy only applies to url servers")
        if missing:
            warnings.append(f"mcp server {name}: unset variables {', '.join(sorted(missing))}")
        configs.append(config)
    return configs, warnings


PROXY_HEADER_PREFIX = "x-llama-server-proxy-header-"
PROXY_KEEP_HEADERS = {"host", "content-length", "content-type", "connection", "transfer-encoding", "accept-encoding"}


def proxied_url(base_url: str, url: str) -> str:
    return f"{base_url.rstrip('/')}/cors-proxy?url={quote(url, safe='')}"


def proxy_headers(headers: Mapping[str, str], proxy_api_key: str = "") -> dict[str, str]:
    result = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in PROXY_KEEP_HEADERS:
            result[lowered] = value
        if lowered not in PROXY_KEEP_HEADERS - {"content-type"}:
            result[PROXY_HEADER_PREFIX + lowered] = value
    if proxy_api_key:
        result["authorization"] = f"Bearer {proxy_api_key}"
    return result


def tool_name(server: str, tool: str) -> str:
    return f"mcp__{UNSAFE_NAME_CHARS.sub('_', server)}__{UNSAFE_NAME_CHARS.sub('_', tool)}"


def shorten(text: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def format_result(result: mcp_types.CallToolResult) -> str:
    parts = [_content_text(item) for item in result.content]
    text = "\n".join(part for part in parts if part)
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    if result.is_error:
        raise ToolError(text or "MCP tool reported an error")
    return text or "(no output)"


def _content_text(item: object) -> str:
    if isinstance(item, mcp_types.TextContent):
        return item.text
    if isinstance(item, mcp_types.ImageContent | mcp_types.AudioContent):
        return f"[{item.type} {item.mime_type}, not shown]"
    if isinstance(item, mcp_types.EmbeddedResource):
        return getattr(item.resource, "text", None) or f"[resource {item.resource.uri}]"
    if isinstance(item, mcp_types.ResourceLink):
        return f"[resource link {item.uri}]"
    return str(item)


def _connection_lost(error: BaseException) -> bool:
    return isinstance(error, MCPError) and error.code == mcp_types.CONNECTION_CLOSED


def _http_client(headers: Mapping[str, str]) -> httpx2.AsyncClient:
    timeout = httpx2.Timeout(HTTP_TIMEOUT_SECONDS, read=HTTP_READ_TIMEOUT_SECONDS)
    return httpx2.AsyncClient(headers=dict(headers), timeout=timeout)


def _describe(error: BaseException) -> str:
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return str(error) or type(error).__name__


class McpManager:
    def __init__(
        self,
        configs: list[ServerConfig],
        cwd: Path,
        connect_timeout: float,
        log_dir: Path = LOG_DIR,
        proxy_base: str = "",
        proxy_api_key: str = "",
    ) -> None:
        self.servers = {config.name: McpServer(config) for config in configs}
        self.cwd = cwd
        self.proxy_base = proxy_base
        self.proxy_api_key = proxy_api_key
        self.connect_timeout = connect_timeout
        self.log_dir = log_dir
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp", daemon=True)
        self._tasks: list[asyncio.Task] = []
        self._logs: list[IO[str]] = []

    def start(self) -> None:
        self.connect([server.config.name for server in self.servers.values() if server.config.enabled])

    def connect(self, names: list[str]) -> None:
        if not names:
            return
        if not self._thread.is_alive():
            self._thread.start()
        servers = [self.servers[name] for name in names]
        asyncio.run_coroutine_threadsafe(self._connect_all(servers), self._loop).result()

    def tools(self, server: McpServer) -> list[Tool]:
        return [self._tool(server, tool) for tool in server.tools]

    def call(self, server: McpServer, name: str, args: dict) -> str:
        client = server.client
        if client is None:
            raise ToolError(f"MCP server {server.config.name} is not connected")
        future = asyncio.run_coroutine_threadsafe(
            client.call_tool(name, args, read_timeout_seconds=server.config.timeout), self._loop
        )
        try:
            result = future.result(timeout=server.config.timeout + 5)
        except KeyboardInterrupt:
            future.cancel()
            raise
        except TimeoutError as error:
            future.cancel()
            raise ToolError(f"MCP call timed out after {server.config.timeout:.0f}s") from error
        except Exception as error:
            if _connection_lost(error):
                raise ToolError(self._restart_text(server)) from error
            raise ToolError(f"MCP call failed: {_describe(error)}") from error
        return format_result(result)

    def _restart_text(self, server: McpServer) -> str:
        name = server.config.name
        asyncio.run_coroutine_threadsafe(self._restart(server), self._loop).result()
        if server.connected:
            return f"MCP server {name} lost its connection and was restarted. The call may not have run, send it again."
        return f"MCP server {name} lost its connection and could not be restarted: {server.error}"

    async def _restart(self, server: McpServer) -> None:
        if server.stop is not None:
            server.stop.set()
        if server.task is not None:
            _, pending = await asyncio.wait([server.task], timeout=SHUTDOWN_SECONDS)
            for task in pending:
                task.cancel()
        server.client = None
        await self._connect(server)

    def close(self) -> None:
        if self._thread.is_alive():
            for server in self.servers.values():
                if server.stop is not None:
                    self._loop.call_soon_threadsafe(server.stop.set)
            done = asyncio.run_coroutine_threadsafe(self._wait_tasks(), self._loop)
            try:
                done.result(timeout=SHUTDOWN_SECONDS + 1)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1)
        for log in self._logs:
            log.close()

    async def _wait_tasks(self) -> None:
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=SHUTDOWN_SECONDS)

    async def _connect_all(self, servers: list[McpServer]) -> None:
        await asyncio.gather(*(self._connect(server) for server in servers))

    async def _connect(self, server: McpServer) -> None:
        if server.connected:
            return
        server.error = ""
        server.stop = asyncio.Event()
        ready = asyncio.Event()
        task = asyncio.create_task(self._serve(server, ready))
        server.task = task
        self._tasks.append(task)
        try:
            await asyncio.wait_for(ready.wait(), self.connect_timeout)
        except TimeoutError:
            server.error = f"no response within {self.connect_timeout:.0f}s"
            task.cancel()

    async def _serve(self, server: McpServer, ready: asyncio.Event) -> None:
        try:
            async with Client(self._transport(server.config)) as client:
                server.tools = await _list_tools(client)
                server.client = client
                ready.set()
                assert server.stop is not None
                await server.stop.wait()
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            if server.task is asyncio.current_task():
                server.error = _describe(error)
        finally:
            if server.task is asyncio.current_task():
                server.client = None
            ready.set()

    def _transport(self, config: ServerConfig):
        if config.url and config.proxy:
            return streamable_http_client(proxied_url(self.proxy_base, config.url), http_client=self._proxy_http(config))
        if config.url:
            http = _http_client(config.headers) if config.headers else None
            return streamable_http_client(config.url, http_client=http)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = (self.log_dir / f"{UNSAFE_NAME_CHARS.sub('_', config.name)}.log").open("a", encoding="utf-8")
        self._logs.append(log)
        params = StdioServerParameters(
            command=config.command, args=list(config.args), env=dict(config.env) or None, cwd=self.cwd
        )
        return stdio_client(params, errlog=log)

    def _proxy_http(self, config: ServerConfig):
        http = _http_client(config.headers)

        async def rewrite(request) -> None:
            rewritten = proxy_headers(request.headers, self.proxy_api_key)
            request.headers.clear()
            request.headers.update(rewritten)

        http.event_hooks["request"].append(rewrite)
        return http

    def _tool(self, server: McpServer, tool: mcp_types.Tool) -> Tool:
        return Tool(
            name=tool_name(server.config.name, tool.name),
            description=shorten(tool.description or tool.title or tool.name),
            parameters=tool.input_schema or {"type": "object", "properties": {}},
            risk=Risk.MCP,
            run=lambda args: self.call(server, tool.name, args),
            group=server.group,
        )


async def _list_tools(client: Client) -> list[mcp_types.Tool]:
    tools, cursor = [], None
    while True:
        page = await client.list_tools(cursor=cursor)
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tools


LAZY_AUTO_ABOVE_TOOLS = 12
SEARCH_DEFAULT_LIMIT = 5
SEARCH_MAX_LIMIT = 10
WORD = re.compile(r"[a-z0-9]+")


def lazy_enabled(setting: bool | str, tool_count: int) -> bool:
    if setting == "auto":
        return tool_count > LAZY_AUTO_ABOVE_TOOLS
    return bool(setting) and tool_count > 0


def search_score(tool: Tool, words: list[str]) -> int:
    name = tool.name.lower()
    text = f"{tool.description} {' '.join(tool.parameters.get('properties', {}))}".lower()
    return sum(3 * (word in name) + (word in text) for word in words)


def mcp_search_tool(registry: Registry) -> Tool:
    def run(args: dict) -> str:
        words = WORD.findall(str(args.get("query", "")).lower())
        limit = min(max(int(args.get("limit") or SEARCH_DEFAULT_LIMIT), 1), SEARCH_MAX_LIMIT)
        candidates = [tool for tool in registry if tool.risk is Risk.MCP]
        if not candidates:
            return "No MCP tools are available."
        scored = sorted(
            ((search_score(tool, words), tool) for tool in candidates), key=lambda pair: pair[0], reverse=True
        )
        matches = [tool for score, tool in scored[:limit] if score > 0]
        if not matches:
            servers = sorted({tool.group.removeprefix("mcp:") for tool in candidates})
            return f"No MCP tools matched. Servers: {', '.join(servers)}. Try other words or a server name."
        registry.show([tool.name for tool in matches])
        lines = [f"- {tool.name}: {tool.description}" for tool in matches]
        return "These tools are now loaded and can be called directly:\n" + "\n".join(lines)

    return Tool(
        name="mcp_search",
        description="Find and load tools from connected MCP servers by keywords. Loaded tools can then be called directly.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "keywords, for example a server name or an action"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        risk=Risk.READ,
        run=run,
    )
