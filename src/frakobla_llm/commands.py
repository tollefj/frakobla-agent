from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from prompt_toolkit import prompt as plain_prompt
from rich.console import Console
from rich.table import Table
from rich.text import Text

from frakobla_llm.client import THINKING_LEVELS, LlamaClient, LlamaError, ModelInfo, ServerInfo, swap_model
from frakobla_llm.config import set_server_enabled
from frakobla_llm.context import truncate_middle, with_summary
from frakobla_llm.loop import Agent, HistoryError
from frakobla_llm.mcp_client import McpManager, McpServer
from frakobla_llm.permissions import Mode, Permissions
from frakobla_llm.session import Session, SessionInfo
from frakobla_llm.tools.interact import Interaction
from frakobla_llm.tools.registry import object_schema
from frakobla_llm.ui.render import plan_text
from frakobla_llm.ui.status import format_tokens

EXPORT_RESULT_CHARS = 2000
DOCTOR_PROMPT = "Call the ping tool with message set to hello. Do not reply with text."


@dataclass
class App:
    console: Console
    client: LlamaClient
    agent: Agent
    permissions: Permissions
    server: ServerInfo
    cwd: Path
    sessions_enabled: bool
    interaction: Interaction
    mcp: McpManager
    config_warnings: list[str]
    running: bool = True

    def switch_session(self, session: Session) -> None:
        self.agent.reset(session)
        self.interaction.reset()


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    run: Callable[[App, str], None]


def run_turn(app: App, text: str) -> None:
    try:
        app.agent.run_turn(text)
    except KeyboardInterrupt:
        app.console.print("[dim]cancelled[/dim]")
    except LlamaError as error:
        app.console.print(Text(str(error), style="red"))


def dispatch(app: App, line: str) -> None:
    name, _, argument = line.partition(" ")
    command = COMMANDS.get(name)
    if command is None:
        app.console.print(f"[red]unknown command {name}[/red], try /help")
        return
    try:
        command.run(app, argument.strip())
    except (KeyboardInterrupt, EOFError):
        app.console.print("[dim]cancelled[/dim]")


def pick_session(console: Console, sessions: list[SessionInfo], query: str = "") -> SessionInfo | None:
    if not sessions:
        console.print("[dim]no earlier sessions in this directory[/dim]")
        return None
    if query.isdigit() and 1 <= int(query) <= len(sessions):
        return sessions[int(query) - 1]
    if query:
        matches = [info for info in sessions if query.lower() in f"{info.title} {info.path.stem}".lower()]
        if len(matches) == 1:
            return matches[0]
        sessions = matches or sessions
    table = Table.grid(padding=(0, 2))
    for number, info in enumerate(sessions, 1):
        table.add_row(
            f"[cyan]{number}[/cyan]", info.modified.strftime("%Y-%m-%d %H:%M"), f"{info.turns} turns", info.title
        )
    console.print(table)
    reply = plain_prompt("session number (empty cancels) > ").strip()
    if reply.isdigit() and 1 <= int(reply) <= len(sessions):
        return sessions[int(reply) - 1]
    return None


def _help(app: App, _: str) -> None:
    table = Table.grid(padding=(0, 2))
    for command in COMMANDS.values():
        table.add_row(f"[cyan]{command.name}[/cyan]", command.help)
    table.add_row("[cyan]!cmd[/cyan]", "run a shell command locally")
    table.add_row("[cyan]!!cmd[/cyan]", "run it and attach the output to your next message")
    table.add_row("[cyan]1-3[/cyan]", "then Enter sends that suggested follow-up")
    table.add_row("[cyan]Alt-Enter[/cyan]", "newline in the input")
    table.add_row("[cyan]Ctrl-C twice[/cyan]", "quit, as does Ctrl-D")
    table.add_row("[cyan]@path[/cyan]", "complete a file path")
    app.console.print(table)


def _new(app: App, _: str) -> None:
    session = Session.create(app.cwd) if app.sessions_enabled else Session.disabled()
    app.switch_session(session)
    app.console.print(f"[dim]new session {session.name}[/dim]")


def _clear(app: App, argument: str) -> None:
    app.console.clear()
    app.console.line(2)
    _new(app, argument)


def _resume(app: App, query: str) -> None:
    if not app.sessions_enabled:
        app.console.print("[red]sessions are disabled (--no-session)[/red]")
        return
    current = app.agent.session.path
    sessions = [info for info in Session.list(app.cwd) if info.path != current]
    chosen = pick_session(app.console, sessions, query)
    if chosen is None:
        return
    session = Session.open(chosen.path)
    app.switch_session(session)
    app.agent.restore(session.entries())
    app.console.print(Text(f"resumed {session.name} ({len(app.agent.messages)} messages)", style="dim"))
    if last := _last_user_text(app.agent.messages):
        app.console.print(Text(f"last prompt: {last}", style="dim"), overflow="ellipsis", no_wrap=True)


def _rename(app: App, title: str) -> None:
    if not title:
        app.console.print(f"session name: {app.agent.session.name}", markup=False)
        return
    if app.agent.session.path is None:
        app.console.print("[red]sessions are disabled (--no-session)[/red]")
        return
    app.agent.session.rename(title)
    app.console.print(Text(f"session renamed to {title}", style="dim"))


def _compact(app: App, instructions: str) -> None:
    before = app.agent.estimated_tokens()
    app.console.print("[dim]compacting...[/dim]")
    try:
        count = app.agent.compact(instructions)
    except (HistoryError, LlamaError) as error:
        app.console.print(f"[red]compaction failed: {error}[/red]")
        return
    after = app.agent.estimated_tokens()
    app.console.print(
        f"[dim]summarized {count} messages, context about {format_tokens(before)} -> {format_tokens(after)} tokens[/dim]"
    )


def _undo(app: App, _: str) -> None:
    try:
        result = app.agent.undo()
    except HistoryError as error:
        app.console.print(f"[red]{error}[/red]")
        return
    for path in result.restored:
        app.console.print(f"restored {path}", markup=False)
    app.console.print(
        f"[dim]removed the last turn ({result.messages_removed} messages). "
        "Changes made through bash are not undone.[/dim]"
    )


def _retry(app: App, _: str) -> None:
    turns = app.agent.turns
    if not turns:
        app.console.print("[red]nothing to retry in the current history[/red]")
        return
    text = str(app.agent.messages[turns[-1].start].get("content") or "")
    _undo(app, "")
    app.console.print(Text(f"retrying: {' '.join(text.split())}", style="dim"), overflow="ellipsis", no_wrap=True)
    run_turn(app, text)


def _plan(app: App, action: str) -> None:
    if action == "clear":
        app.interaction.plan = []
        app.agent.plan = ""
        app.console.print("[dim]plan cleared[/dim]")
        return
    if action:
        app.console.print("[red]usage: /plan, /plan clear[/red]")
        return
    if not app.interaction.plan:
        app.console.print("[dim]no plan[/dim]")
        return
    app.console.print(plan_text(app.interaction.plan))


def _summary(app: App, _: str) -> None:
    if not app.agent.summary:
        app.console.print("[dim]history has not been compacted[/dim]")
        return
    app.console.print(app.agent.summary, markup=False)


def _context(app: App, _: str) -> None:
    agent = app.agent
    model = agent.model
    parts = [
        ("system prompt", agent.system_prompt),
        ("summary and plan", with_summary("", agent.summary, agent.plan).strip()),
        ("tool schemas", json.dumps(agent.registry.schemas())),
        ("messages", json.dumps(agent.messages, ensure_ascii=False) if agent.messages else ""),
    ]
    counts = [(label, app.client.count_tokens(model, text) if text else 0) for label, text in parts]
    total = sum(count for _, count in counts)
    size = app.server.context_size
    table = Table.grid(padding=(0, 2))
    for label, count in counts:
        table.add_row(label, f"{format_tokens(count):>7}", f"[dim]{_share(count, size)}[/dim]")
    table.add_row("[bold]total[/bold]", f"[bold]{format_tokens(total):>7}[/bold]", f"[dim]{_share(total, size)}[/dim]")
    app.console.print(table)
    app.console.print(
        f"[dim]{len(agent.messages)} messages, {len(agent.registry.schemas())} tool schemas, "
        f"context size {format_tokens(size) if size else 'unknown'}[/dim]"
    )


def _share(count: int, size: int) -> str:
    return f"{count / size:.1%}" if size else ""


def _unload(app: App, _: str) -> None:
    model = app.agent.model
    try:
        with app.console.status(f"unloading {model}..."):
            app.client.unload_model(model)
            app.client.wait_for_status(model, "unloaded")
    except LlamaError as error:
        app.console.print(f"[red]{error}[/red]")
        return
    except KeyboardInterrupt:
        app.console.print("[yellow]stopped waiting, the server may still be unloading[/yellow]")
        return
    app.console.print(f"[dim]unloaded {model}, the next request loads it again[/dim]")


def _model(app: App, query: str) -> None:
    try:
        chosen = pick_model(app.console, app.client.models(), app.agent.model, query)
        if chosen is None:
            return
        with app.console.status("switching model...") as status:
            swap_model(app.client, chosen.id, lambda step: status.update(f"{step}..."))
        app.agent.model = chosen.id
        app.agent.session.append("model_change", model=chosen.id)
        app.server = app.client.server_info(chosen.id)
        app.agent.context_size = app.server.context_size
        app.console.print(f"[dim]model set to {chosen.id}[/dim]")
    except LlamaError as error:
        app.console.print(f"[red]{error}[/red]")
    except KeyboardInterrupt:
        app.console.print("[yellow]model switch interrupted, the server may still be loading[/yellow]")


def pick_model(console: Console, models: list[ModelInfo], current: str, query: str = "") -> ModelInfo | None:
    if not models:
        console.print("[dim]server reports no models[/dim]")
        return None
    exact = next((model for model in models if model.id == query), None)
    if exact or (query.isdigit() and 1 <= int(query) <= len(models)):
        return exact or models[int(query) - 1]
    if query:
        matches = [model for model in models if query.lower() in model.id.lower()]
        if len(matches) == 1:
            return matches[0]
        models = matches or models
    table = Table.grid(padding=(0, 2))
    for number, model in enumerate(models, 1):
        marker = "*" if model.id == current else " "
        table.add_row(f"[cyan]{number}[/cyan]", marker, model.id, f"[dim]{model.status}[/dim]")
    console.print(table)
    reply = plain_prompt("model number (empty cancels) > ").strip()
    if reply.isdigit() and 1 <= int(reply) <= len(models):
        return models[int(reply) - 1]
    return None


def _mode(app: App, name: str) -> None:
    if name:
        try:
            app.permissions.mode = Mode(name)
        except ValueError:
            app.console.print(f"[red]unknown mode {name}[/red], expected ask, readonly or yolo")
            return
    app.console.print(f"[dim]permission mode: {app.permissions.mode}[/dim]")


def _think(app: App, level: str) -> None:
    if level:
        if level not in THINKING_LEVELS:
            app.console.print(f"[red]unknown level {level}[/red], expected {', '.join(THINKING_LEVELS)}")
            return
        app.agent.thinking = level
    app.console.print(f"[dim]thinking: {app.agent.thinking}[/dim]")


def _tools(app: App, argument: str) -> None:
    action, _, query = argument.partition(" ")
    if action == "load":
        _load_tools(app, query.strip())
        return
    if action:
        app.console.print("[red]usage: /tools, /tools load <name|keywords>[/red]")
        return
    table = Table.grid(padding=(0, 2))
    registry = app.agent.registry
    for tool in registry:
        state = "[dim](not loaded)[/dim]" if tool.name in registry.hidden else ""
        table.add_row(f"[cyan]{tool.name}[/cyan]", tool.risk, f"{tool.description} {state}".strip())
    app.console.print(table)


def _load_tools(app: App, query: str) -> None:
    registry = app.agent.registry
    if not query:
        app.console.print("[red]usage: /tools load <name|keywords>[/red]")
        return
    if registry.get(query) and query in registry.hidden:
        registry.show([query])
        app.console.print(Text(f"loaded {query}", style="dim"))
        return
    search = registry.get("mcp_search")
    if search is None:
        app.console.print("[dim]lazy MCP mode is off, all tools are already loaded[/dim]")
        return
    app.console.print(search.run({"query": query}), markup=False)


def _perms(app: App, _: str) -> None:
    permissions = app.permissions
    app.console.print(f"mode: {permissions.mode}")
    for tier, by_tool in permissions.rules.items():
        for tool, rules in by_tool.items():
            for rule in rules:
                app.console.print(f"{tier}  {tool}  {rule.kind}  {rule.pattern}", markup=False)
    for tool, rules in permissions.grants.items():
        for rule in rules:
            app.console.print(f"session grant  {tool}  {rule.kind}  {rule.pattern}", markup=False)


def _mcp(app: App, argument: str) -> None:
    action, _, name = argument.partition(" ")
    if action in ("enable", "disable"):
        _toggle_mcp(app, name.strip(), enabled=action == "enable")
        return
    if action:
        app.console.print("[red]usage: /mcp, /mcp enable <server>, /mcp disable <server>[/red]")
        return
    if not app.mcp.servers:
        app.console.print("[dim]no MCP servers configured, add [mcp.servers.<name>] to the config[/dim]")
        return
    table = Table.grid(padding=(0, 2))
    for server in app.mcp.servers.values():
        table.add_row(
            f"[cyan]{server.config.name}[/cyan]",
            server.config.transport,
            _mcp_status(app, server),
            _mcp_tool_count(app, server),
            f"~{format_tokens(_mcp_tokens(app, server))} tokens" if server.tools else "",
        )
    app.console.print(table)


def _toggle_mcp(app: App, name: str, enabled: bool) -> None:
    server = app.mcp.servers.get(name)
    if server is None:
        app.console.print(f"[red]unknown MCP server {name!r}[/red]")
        return
    registry = app.agent.registry
    if enabled and not server.connected:
        app.console.print(f"[dim]connecting {name}...[/dim]")
        app.mcp.connect([name])
        if not server.connected:
            app.console.print(Text(f"{name}: {server.error}", style="red"))
            return
        tools = app.mcp.tools(server)
        registry.add(tools)
        if registry.get("mcp_search"):
            registry.hide([tool.name for tool in tools])
    registry.set_group_enabled(server.group, enabled)
    server.config = replace(server.config, enabled=enabled)
    state = "enabled" if enabled else "disabled"
    saved = set_server_enabled(app.cwd, name, enabled)
    where = f"saved to {saved}" if saved else "for this session only, server not found in a config file"
    app.console.print(Text(f"{name} {state}, {where}", style="dim"))


def _mcp_status(app: App, server: McpServer) -> str:
    if server.error:
        return f"[red]failed: {server.error}[/red]"
    if not server.connected:
        return "[yellow]disabled[/yellow]" if not server.config.enabled else "[dim]not connected[/dim]"
    if server.group in app.agent.registry.disabled_groups:
        return "[yellow]disabled[/yellow]"
    return "[green]connected[/green]"


def _mcp_tool_count(app: App, server: McpServer) -> str:
    count = f"{len(server.tools)} tools"
    if not app.agent.registry.get("mcp_search"):
        return count
    names = {tool.name for tool in app.agent.registry.group(server.group)}
    return f"{count}, {len(names - app.agent.registry.hidden)} loaded"


def _mcp_tokens(app: App, server: McpServer) -> int:
    schemas = [tool.schema() for tool in app.mcp.tools(server)]
    return app.client.count_tokens(app.agent.model, json.dumps(schemas))


def _export(app: App, target: str) -> None:
    agent = app.agent
    if not agent.messages and not agent.summary:
        app.console.print("[dim]nothing to export[/dim]")
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(target).expanduser() if target else app.cwd / f"frak-export-{stamp}.md"
    path = path if path.is_absolute() else app.cwd / path
    path.write_text(export_markdown(agent.session.name, agent.model, agent.summary, agent.messages), encoding="utf-8")
    app.console.print(Text(f"exported to {path}", style="dim"))


def export_markdown(title: str, model: str, summary: str, messages: list[dict]) -> str:
    parts = [f"# {title}", f"Model: {model}"]
    if summary:
        parts.append(f"## Summary of compacted history\n\n{summary}")
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role == "tool":
            parts.append(f"Result:\n\n```\n{truncate_middle(content, EXPORT_RESULT_CHARS)}\n```")
            continue
        if content:
            parts.append(f"## {str(role).capitalize()}\n\n{content}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(f"Tool call `{function.get('name')}`:\n\n```json\n{function.get('arguments')}\n```")
    return "\n\n".join(parts) + "\n"


def _doctor(app: App, _: str) -> None:
    run_doctor(app)


def run_doctor(app: App) -> bool:
    report = Doctor(app.console)
    try:
        models = app.client.models()
        report.ok(f"server {app.client.base_url} reachable, {len(models)} models")
    except LlamaError as error:
        report.fail(str(error))
        return False
    model = next((entry for entry in models if entry.id == app.agent.model), None)
    if model is None:
        report.fail(f"model {app.agent.model} is not listed by the server")
    else:
        report.ok(f"model {model.id} ({'loaded' if model.loaded else 'not loaded yet'})")
    try:
        server = app.client.server_info(app.agent.model)
        size = format_tokens(server.context_size) if server.context_size else "unknown"
        report.ok(f"context size {size}")
        if server.supports_tools is False:
            report.fail("chat template has no tool support, start llama-server with --jinja")
        elif server.supports_tools is None:
            report.warn("could not read the chat template")
        else:
            report.ok("chat template supports tools")
    except LlamaError as error:
        report.fail(f"/props: {error}")
    _doctor_tool_call(app, report)
    for server in app.mcp.servers.values():
        if server.error:
            report.fail(f"mcp {server.config.name}: {server.error}")
        elif server.connected:
            report.ok(f"mcp {server.config.name}: {len(server.tools)} tools")
        else:
            report.warn(f"mcp {server.config.name}: not connected")
    for warning in app.config_warnings:
        report.warn(warning)
    return report.failures == 0


def _doctor_tool_call(app: App, report: Doctor) -> None:
    ping = {
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Send a ping message.",
            "parameters": object_schema(["message"], message={"type": "string"}),
        },
    }
    started = time.monotonic()
    try:
        completion = app.client.stream_chat(
            app.agent.model, [{"role": "user", "content": DOCTOR_PROMPT}], [ping], lambda _: None
        )
    except LlamaError as error:
        report.fail(f"tool-call round trip: {error}")
        return
    seconds = time.monotonic() - started
    calls = [call for call in completion.tool_calls if call.name == "ping"]
    if not calls:
        report.fail(f"tool-call round trip: the model did not call the tool ({seconds:.1f}s)")
        return
    try:
        args = json.loads(calls[0].arguments or "{}")
    except json.JSONDecodeError:
        report.fail(f"tool-call round trip: invalid JSON arguments {calls[0].arguments!r}")
        return
    if not isinstance(args, dict) or "message" not in args:
        report.warn(f"tool-call round trip: unexpected arguments {calls[0].arguments}")
        return
    report.ok(f"tool-call round trip in {seconds:.1f}s")


class Doctor:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.failures = 0

    def ok(self, text: str) -> None:
        self._line("ok  ", "green", text)

    def warn(self, text: str) -> None:
        self._line("warn", "yellow", text)

    def fail(self, text: str) -> None:
        self.failures += 1
        self._line("FAIL", "red", text)

    def _line(self, label: str, style: str, text: str) -> None:
        self.console.print(Text.assemble((label, style), "  ", text))


def _stats(app: App, _: str) -> None:
    completion = app.agent.last_completion
    usage, timings = completion.usage, completion.timings
    app.console.print(f"messages: {len(app.agent.messages)}")
    app.console.print(
        f"context: about {format_tokens(app.agent.estimated_tokens())} of {format_tokens(app.server.context_size)}"
    )
    if app.agent.summary:
        app.console.print("history has been compacted")
    app.console.print(
        f"last request: {usage.get('prompt_tokens', 0)} prompt, {usage.get('completion_tokens', 0)} completion tokens"
    )
    if timings:
        app.console.print(
            f"speed: {timings.get('prompt_per_second', 0):.0f} tok/s prompt, "
            f"{timings.get('predicted_per_second', 0):.0f} tok/s generation"
        )
    app.console.print(f"session: {app.agent.session.path or 'disabled'}", markup=False)


def _quit(app: App, _: str) -> None:
    app.running = False


def _last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return " ".join(str(message.get("content") or "").split())
    return ""


COMMANDS = {
    command.name: command
    for command in [
        Command("/help", "show commands", _help),
        Command("/new", "start a new session with empty context", _new),
        Command("/clear", "clear the screen and start a new session", _clear),
        Command("/resume", "pick an earlier session in this directory, or /resume <number|text>", _resume),
        Command("/rename", "name the current session", _rename),
        Command("/compact", "summarize the history to free context, optional focus instructions", _compact),
        Command("/undo", "revert the last turn's file edits and drop it from history", _undo),
        Command("/retry", "undo the last turn and send its prompt again", _retry),
        Command("/plan", "show the plan checklist, /plan clear drops it", _plan),
        Command("/summary", "show the compaction summary", _summary),
        Command("/context", "token breakdown of the context", _context),
        Command("/model", "pick a model, unloads the current one and loads the new one, or /model <number|text>", _model),
        Command("/mode", "show or set permission mode: ask, readonly, yolo", _mode),
        Command("/think", "show or set thinking level: " + ", ".join(THINKING_LEVELS), _think),
        Command("/unload", "unload the current model from the server", _unload),
        Command("/tools", "list tools, /tools load <name|keywords> loads lazy MCP tools", _tools),
        Command("/perms", "show permission rules and session grants", _perms),
        Command("/mcp", "list MCP servers, /mcp enable|disable <server> (saved to the config)", _mcp),
        Command("/stats", "token usage and speed", _stats),
        Command("/export", "write the conversation to a markdown file, optional path", _export),
        Command("/doctor", "check server, template, tool calls and MCP servers", _doctor),
        Command("/quit", "exit", _quit),
    ]
}
