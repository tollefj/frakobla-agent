import argparse
import os
from datetime import datetime
from pathlib import Path

import pytest
from rich.console import Console

from frakobla_llm import cli, commands
from frakobla_llm.commands import Command, export_markdown, pick_session
from frakobla_llm.config import Config
from frakobla_llm.permissions import Choice
from frakobla_llm.session import Session, SessionInfo
from frakobla_llm.tools.registry import Registry, Risk, Tool
from frakobla_llm.tools.shell import run_command
from frakobla_llm.ui.prompt import PERMISSION_KEYS, chosen_suggestion


class AppPlaceholder:
    def __init__(self) -> None:
        self.console = Console(record=True, width=120)


class McpPlaceholder:
    def __init__(self) -> None:
        self.servers: dict = {}


def info(title: str) -> SessionInfo:
    return SessionInfo(Path(f"<dir>/{title}.jsonl"), title, 1, datetime.now())


def test_interrupted_command_does_not_escape_dispatch(monkeypatch):
    def interrupted(app, argument):
        raise KeyboardInterrupt

    monkeypatch.setitem(commands.COMMANDS, "/<slow>", Command("/<slow>", "<help>", interrupted))
    app = AppPlaceholder()

    commands.dispatch(app, "/<slow> <argument>")
    commands.dispatch(app, "/<unknown>")

    output = app.console.export_text()
    assert "cancelled" in output and "unknown command" in output


def test_pick_session_by_number_and_unique_text():
    sessions = [info("<first>"), info("<second>")]
    console = Console(record=True)

    assert pick_session(console, sessions, "2") is sessions[1]
    assert pick_session(console, sessions, "FIRST") is sessions[0]
    assert pick_session(console, [], "") is None


def test_export_keeps_tool_calls_next_to_their_results():
    messages = [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "<tool>", "arguments": "{}"}}]},
        {"role": "tool", "content": "<result>"},
    ]

    exported = export_markdown("<title>", "<model>", "<summary>", messages)

    assert exported.index("<summary>") < exported.index("<prompt>") < exported.index("`<tool>`") < exported.index("<result>")


def test_cli_overrides_beat_config(monkeypatch):
    monkeypatch.delenv("FRAK_BASE_URL", raising=False)
    monkeypatch.setenv("FRAK_MODEL", "<model from variable>")
    config = Config({"server": {"base_url": "<config url>", "model": "<config model>"}, "permissions": {"mode": "ask"}})
    args = argparse.Namespace(base_url=None, model=None, yolo=True, mode="readonly")

    cli.apply_overrides(config, args)

    assert config["server"] == {"base_url": "<config url>", "model": "<model from variable>"}
    assert config["permissions"]["mode"] == "yolo"


def test_latest_session_is_the_most_recently_used_one_for_this_directory(tmp_path):
    here, elsewhere = tmp_path / "a" / "b-c", tmp_path / "a" / "b" / "c"
    older, newer, foreign = (Session.create(cwd, tmp_path) for cwd in (here, here, elsewhere))
    for session in (older, newer, foreign):
        session.append("message", message={"role": "user", "content": "<prompt>"})
    os.utime(older.path, (2_000_000_000, 2_000_000_000))
    os.utime(foreign.path, (2_100_000_000, 2_100_000_000))

    assert older.path.parent == foreign.path.parent
    assert Session.latest(here, tmp_path).path == older.path
    assert [found.path for found in Session.list(elsewhere, tmp_path)] == [foreign.path]


def test_bash_output_is_capped_with_head_and_tail(tmp_path):
    command = "printf 'A%.0s' $(seq 1 3000); printf 'B%.0s' $(seq 1 3000)"

    output = run_command(command, tmp_path, 10, max_output_bytes=1000)

    assert output.startswith("A" * 500) and "B" * 500 in output
    assert "5000 bytes of output dropped" in output and output.endswith("[exit code 0]")


def test_suggestions_are_chosen_by_number_only():
    suggestions = ["<first>", "<second>"]

    assert chosen_suggestion(" 2 ", suggestions) == "<second>"
    assert chosen_suggestion("2 things", suggestions) == ""
    assert chosen_suggestion("3", suggestions) == ""


def test_permission_prompt_accepts_full_words():
    assert PERMISSION_KEYS["always"] is Choice.ALWAYS
    assert PERMISSION_KEYS["session"] is Choice.SESSION
    assert PERMISSION_KEYS["deny"] is Choice.DENY


def test_lazy_mcp_hides_tools_behind_search():
    tools = [
        Tool(f"mcp__<server>__<tool {n}>", "<description>", {}, Risk.MCP, lambda args: "", group="mcp:<server>")
        for n in range(2)
    ]
    registry = Registry(tools)

    assert cli.enable_lazy_mcp(registry, McpPlaceholder(), False) == ""
    assert "mcp_search" in cli.enable_lazy_mcp(registry, McpPlaceholder(), True)
    assert registry.names() == ["mcp_search"]


@pytest.mark.parametrize("newline", ["\x1b[13;2u", "\x1b[27;2;13~", "\x1b\r", "\n"])
def test_shift_enter_and_fallbacks_insert_a_newline(newline):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from frakobla_llm.ui.prompt import newline_bindings

    with create_pipe_input() as pipe:
        pipe.send_text(f"first{newline}second\r")
        session = PromptSession(key_bindings=newline_bindings(), input=pipe, output=DummyOutput())
        assert session.prompt() == "first\nsecond"
