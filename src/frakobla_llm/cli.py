from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from prompt_toolkit import prompt as plain_prompt
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from frakobla_llm import commands
from frakobla_llm.client import THINKING_LEVELS, LlamaClient, LlamaError, resolve_model
from frakobla_llm.commands import App, pick_session
from frakobla_llm.config import GLOBAL_CONFIG, LEGACY_DATA_DIR, Config, ConfigError, append_rule, load_config, migrate_legacy_data
from frakobla_llm.context import build_system_prompt, discover_agents_files
from frakobla_llm.loop import Agent
from frakobla_llm.mcp_client import McpManager, lazy_enabled, mcp_search_tool, parse_servers
from frakobla_llm.permissions import Permissions, Rule
from frakobla_llm.session import Session
from frakobla_llm.tools.fs import fs_tools
from frakobla_llm.tools.interact import Interaction, interact_tools
from frakobla_llm.tools.registry import Registry, Risk
from frakobla_llm.tools.shell import MAX_TIMEOUT_SECONDS, bash_tool, run_command
from frakobla_llm.ui.prompt import InputPrompt
from frakobla_llm.ui.render import PlainUI, Renderer
from frakobla_llm.ui.status import Snapshot, format_tokens, status_text, top_bar
from frakobla_llm.ui.statusbar import StatusBar

EXIT_OK, EXIT_ERROR, EXIT_INTERRUPTED = 0, 1, 130
QUIT_PRESS_WINDOW_SECONDS = 2.0
INTERACT_HINT = (
    "Track multi-step tasks with update_plan. Use ask_user when you are blocked. "
    "When natural follow-ups exist, end with suggest_next."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frak", description="frakobla llm: minimal agentic CLI for a local llama.cpp server")
    parser.add_argument("-p", "--prompt", help="run one prompt without the interactive UI")
    parser.add_argument("--model", help="model id, defaults to the loaded model")
    parser.add_argument("--thinking", choices=THINKING_LEVELS, help="reasoning level sent to the model")
    parser.add_argument("--base-url", help="llama-server URL")
    parser.add_argument("--mode", choices=["ask", "readonly", "yolo"], help="permission mode")
    parser.add_argument("--yolo", action="store_true", help="shorthand for --mode yolo")
    parser.add_argument("-c", "--continue", dest="resume_latest", action="store_true", help="continue the latest session in this directory")
    parser.add_argument("-r", "--resume", nargs="?", const="", metavar="QUERY", help="pick an earlier session, optionally by number or text")
    parser.add_argument("--no-session", action="store_true", help="do not write a session file")
    parser.add_argument("--doctor", action="store_true", help="check server, template, tool calls and MCP servers, then exit")
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> None:
    server = config["server"]
    server["base_url"] = args.base_url or os.environ.get("FRAK_BASE_URL") or server["base_url"]
    server["model"] = args.model or os.environ.get("FRAK_MODEL") or server["model"]
    if args.yolo or args.mode:
        config["permissions"]["mode"] = "yolo" if args.yolo else args.mode


def open_session(args: argparse.Namespace, cwd: Path, console: Console) -> Session:
    if args.no_session or args.doctor:
        return Session.disabled()
    if args.resume is not None and (chosen := pick_session(console, Session.list(cwd), args.resume)):
        return Session.open(chosen.path)
    if args.resume_latest and (latest := Session.latest(cwd)):
        return latest
    return Session.create(cwd)


def main() -> int:
    args = build_parser().parse_args()
    errors = Console(stderr=True)
    cwd = Path.cwd()
    headless = args.prompt is not None
    try:
        if moved := migrate_legacy_data():
            errors.print(Text(f"moved {moved} files from {LEGACY_DATA_DIR} to the config directory", style="dim"))
        config = load_config(cwd, trust=None if headless else ask_trust)
        apply_overrides(config, args)
        client = LlamaClient(config["server"]["base_url"], config["server"]["api_key"], config["server"]["timeout"])
        model = resolve_model(client, config["server"]["model"])
        server = client.server_info(model)
        console = Console()
        ui: PlainUI | Renderer = PlainUI()
        prompter = None
        if not headless:
            ui = renderer = Renderer(console, config["ui"]["show_reasoning"], config["ui"]["spinner"])
            prompter = renderer.ask_permission
        permissions = Permissions(
            config["permissions"],
            cwd,
            prompter=prompter,
            persist=_persist_rule,
        )
    except (ConfigError, LlamaError, ValueError, OSError) as error:
        errors.print(Text(str(error), style="red"))
        return EXIT_ERROR

    agents_files = discover_agents_files(cwd)
    session = open_session(args, cwd, console)
    interaction = Interaction()
    registry = Registry([*fs_tools(cwd), bash_tool(cwd, config["tools"]["bash_timeout"])])
    system_prompt = build_system_prompt(cwd, agents_files)
    if config["tools"]["interact"]:
        registry.add(interact_tools(interaction, ui, interactive=not headless))
        system_prompt += "\n" + (INTERACT_HINT if not headless else "Track multi-step tasks with update_plan.")

    server_configs, mcp_warnings = parse_servers(config["mcp"]["servers"])
    config.warnings += mcp_warnings
    mcp = McpManager(
        server_configs,
        cwd,
        config["mcp"]["connect_timeout"],
        proxy_base=config["server"]["base_url"],
        proxy_api_key=config["server"]["api_key"],
    )
    try:
        if not headless and server_configs:
            console.print("[dim]connecting MCP servers...[/dim]")
        mcp.start()
        for mcp_server in mcp.servers.values():
            if mcp_server.connected:
                registry.add(mcp.tools(mcp_server))
        system_prompt += enable_lazy_mcp(registry, mcp, config["mcp"]["lazy"])

        agent = Agent(
            client=client,
            model=model,
            registry=registry,
            permissions=permissions,
            session=session,
            ui=ui,
            system_prompt=system_prompt,
            max_iterations=config["agent"]["max_iterations"],
            output_max_chars=config["tools"]["output_max_chars"],
            parallel_tool_calls=config["agent"]["parallel_tool_calls"],
            thinking=args.thinking or config["agent"]["thinking"],
            context_size=server.context_size,
            compact_at=config["context"]["compact_at"],
            auto_compact=config["context"]["auto_compact"],
            interaction=interaction,
        )
        agent.restore(session.entries())

        if headless:
            return run_headless(agent, args.prompt, errors)
        app = App(
            console,
            client,
            agent,
            permissions,
            server,
            cwd,
            sessions_enabled=not args.no_session,
            interaction=interaction,
            mcp=mcp,
            config_warnings=config.warnings,
        )
        if args.doctor:
            return EXIT_OK if commands.run_doctor(app) else EXIT_ERROR
        print_startup_panel(app, config, agents_files)
        return run_interactive(app, config)
    finally:
        mcp.close()


def ask_trust(path: Path, sections: list[str]) -> bool:
    console = Console(stderr=True)
    console.print(Text(f"{path} sets {', '.join(sections)}.", style="yellow"))
    console.print("It can start MCP commands, change the permission rules and point at another server.")
    try:
        reply = plain_prompt("trust this project config? [y/N] > ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return reply in ("y", "yes")


def run_headless(agent: Agent, prompt: str, errors: Console) -> int:
    try:
        agent.run_turn(prompt)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except LlamaError as error:
        errors.print(Text(str(error), style="red"))
        return EXIT_ERROR
    return EXIT_OK


def run_interactive(app: App, config: Config) -> int:
    bar = make_status_bar(app, config) if config["ui"]["status_bar"] and app.console.is_terminal else None

    def toolbar() -> str:
        agent = app.agent
        if bar:
            return f"{app.permissions.mode} | {agent.session.name}"
        return status_text(
            agent.model,
            agent.estimated_tokens(),
            agent.last_completion,
            app.server.context_size,
            app.permissions.mode,
            agent.session.name,
        )

    prompt = InputPrompt(app.cwd, list(commands.COMMANDS), toolbar, app.interaction)
    if bar is None:
        return read_loop(app, prompt)
    bar.install()
    try:
        return read_loop(app, prompt, bar)
    finally:
        bar.remove()


def make_status_bar(app: App, config: Config) -> StatusBar:
    def render(width: int) -> list[Text]:
        agent = app.agent
        snapshot = Snapshot(
            model=agent.model,
            thinking=agent.thinking,
            used=agent.estimated_tokens(),
            context_size=app.server.context_size,
            compact_at=config["context"]["compact_at"],
            usage=agent.stats.current(),
            completion=agent.last_completion,
            gpu_watts=config["ui"]["gpu_watts"],
        )
        return top_bar(snapshot, width)

    bar = StatusBar(app.console, render)
    if isinstance(app.agent.ui, Renderer):
        app.agent.ui.on_tick = lambda force: bar.refresh(force)
    return bar


def read_loop(app: App, prompt: InputPrompt, bar: StatusBar | None = None) -> int:
    attachments: list[str] = []
    last_interrupt = float("-inf")
    while app.running:
        if bar:
            bar.refresh(force=True)
        try:
            text = prompt.read().strip()
        except KeyboardInterrupt:
            now = time.monotonic()
            if now - last_interrupt < QUIT_PRESS_WINDOW_SECONDS:
                break
            last_interrupt = now
            app.console.print("[dim]press Ctrl-C again to quit[/dim]")
            continue
        except EOFError:
            break
        if not text:
            continue
        if text.startswith("/"):
            commands.dispatch(app, text)
            continue
        if text.startswith("!"):
            if attachment := run_local_command(app, text):
                attachments.append(attachment)
            continue
        commands.run_turn(app, "\n\n".join([*attachments, text]))
        attachments.clear()
    return EXIT_OK


def run_local_command(app: App, text: str) -> str:
    share = text.startswith("!!")
    command = text.removeprefix("!!" if share else "!").strip()
    if not command:
        return ""
    try:
        output = run_command(command, app.cwd, MAX_TIMEOUT_SECONDS)
    except KeyboardInterrupt:
        app.console.print("[dim]cancelled[/dim]")
        return ""
    app.console.print(Text(output, style="dim"))
    if not share:
        return ""
    app.console.print("[dim]output will be attached to your next message[/dim]")
    return f"I ran `{command}` in the terminal:\n```\n{output}\n```"


def print_startup_panel(app: App, config: Config, agents_files: list[Path]) -> None:
    agent, server = app.agent, app.server
    tool_support = {True: "yes", False: "no", None: "unknown"}[server.supports_tools]
    rows = [
        ("server", app.client.base_url),
        ("model", agent.model),
        ("context", format_tokens(server.context_size) if server.context_size else "unknown"),
        ("tool calls", tool_support),
        ("mode", app.permissions.mode),
        ("thinking", agent.thinking),
        ("tools", ", ".join(tool.name for tool in agent.registry if not tool.group)),
        *_mcp_rows(app),
        ("config", ", ".join(map(str, config.sources)) or "defaults"),
        ("AGENTS.md", ", ".join(map(str, agents_files)) or "none"),
        ("session", agent.session.name + (f" ({len(agent.messages)} messages)" if agent.messages else "")),
    ]
    body = Text()
    for label, value in rows:
        body.append(f"{label:<11}", style="dim")
        body.append(f"{value}\n")
    warnings = list(config.warnings)
    if server.supports_tools is False:
        warnings.append("chat template has no tool support, start llama-server with --jinja")
    for warning in warnings:
        body.append(f"warning: {warning}\n", style="yellow")
    body.append("/help for commands, Ctrl-C cancels a turn, Ctrl-C twice or Ctrl-D quits", style="dim")
    app.console.print(Panel(body, title="frakobla llm", border_style="blue", expand=False))


def enable_lazy_mcp(registry: Registry, mcp: McpManager, setting: bool | str) -> str:
    mcp_tools = [tool.name for tool in registry if tool.risk is Risk.MCP]
    if not lazy_enabled(setting, len(mcp_tools)):
        return ""
    registry.hide(mcp_tools)
    registry.add([mcp_search_tool(registry)])
    servers = ", ".join(name for name, server in mcp.servers.items() if server.connected)
    return f"\nMore tools from MCP servers ({servers}) are not listed. Find and load them with mcp_search."


def _mcp_rows(app: App) -> list[tuple[str, str]]:
    rows = []
    for server in app.mcp.servers.values():
        if server.connected:
            state = f"{len(server.tools)} tools" + (", lazy" if app.agent.registry.get("mcp_search") else "")
        elif server.error:
            state = f"failed: {server.error}"
        else:
            state = "disabled"
        rows.append(("mcp", f"{server.config.name} ({state})"))
    return rows


def _persist_rule(tool: str, rule: Rule) -> None:
    append_rule(GLOBAL_CONFIG, "allow", tool, rule.pattern, rule.kind)


if __name__ == "__main__":
    sys.exit(main())
