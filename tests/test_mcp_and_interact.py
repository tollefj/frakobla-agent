import sys
from pathlib import Path

import pytest

from frakobla_llm.config import DEFAULTS, load_config, set_server_enabled, unknown_keys
from frakobla_llm.mcp_client import McpManager, ServerConfig, lazy_enabled, mcp_search_tool, parse_servers, shorten, tool_name
from frakobla_llm.permissions import Decision, Permissions
from frakobla_llm.tools.interact import Interaction, interact_tools
from frakobla_llm.tools.registry import Registry, Risk, Tool, ToolError, object_schema
from frakobla_llm.ui.render import PlainUI

ECHO_SERVER = Path(__file__).parent / "fixtures" / "echo_server.py"


def test_server_config_substitutes_variables_and_reports_missing():
    raw = {
        "one": {"command": "${BIN}", "args": ["--token", "${TOKEN}"], "env": {"KEY": "${TOKEN}"}},
        "two": {"url": "http://${HOST}/mcp", "enabled": False},
        "broken": {"args": []},
    }
    configs, warnings = parse_servers(raw, {"BIN": "<bin>", "TOKEN": "<token>"})
    one, two = configs
    assert (one.command, one.args, dict(one.env)) == ("<bin>", ("--token", "<token>"), {"KEY": "<token>"})
    assert (two.url, two.enabled, two.transport) == ("http:///mcp", False, "http")
    assert any("HOST" in warning for warning in warnings)
    assert any("broken" in warning for warning in warnings)


def test_mcp_servers_section_accepts_any_server_name():
    assert unknown_keys({"mcp": {"servers": {"<name>": {"command": "<cmd>"}}}}, DEFAULTS) == []


def test_tool_names_and_descriptions_are_sanitized():
    assert tool_name("my server", "do.thing") == "mcp__my_server__do_thing"
    assert len(shorten("word " * 200)) <= 300


def test_stdio_server_round_trip_and_permission_default(tmp_path):
    manager = McpManager(
        [ServerConfig("echo", command=sys.executable, args=(str(ECHO_SERVER),)), ServerConfig("missing", command="/nonexistent/bin")],
        tmp_path,
        connect_timeout=20,
        log_dir=tmp_path / "logs",
    )
    try:
        manager.start()
        echo, missing = manager.servers["echo"], manager.servers["missing"]
        assert echo.connected and not missing.connected and missing.error
        tools = {tool.name: tool for tool in manager.tools(echo)}
        assert tools["mcp__echo__echo"].run({"text": "<text>"}) == "<text>"
        with pytest.raises(ToolError):
            tools["mcp__echo__fail"].run({})

        permissions = Permissions({"mode": "ask", "rules": {}}, tmp_path)
        assert permissions.evaluate(tools["mcp__echo__echo"], {"text": "<text>"}).decision is Decision.ASK

        registry = Registry(list(tools.values()))
        registry.set_group_enabled(echo.group, False)
        assert registry.names() == [] and registry.get("mcp__echo__echo") is None
        registry.set_group_enabled(echo.group, True)
        assert len(registry.names()) == 3
    finally:
        manager.close()


class AnsweringUI(PlainUI):
    def ask_user(self, question: str, choices: list[str]) -> str:
        return choices[-1]


def test_dead_stdio_server_is_restarted(tmp_path):
    manager = McpManager(
        [ServerConfig("echo", command=sys.executable, args=(str(ECHO_SERVER),))],
        tmp_path,
        connect_timeout=20,
        log_dir=tmp_path / "logs",
    )
    try:
        manager.start()
        tools = {tool.name: tool for tool in manager.tools(manager.servers["echo"])}
        with pytest.raises(ToolError, match="was restarted"):
            tools["mcp__echo__crash"].run({})
        assert tools["mcp__echo__echo"].run({"text": "<text>"}) == "<text>"
    finally:
        manager.close()


def test_interact_tools():
    state = Interaction()
    tools = {tool.name: tool for tool in interact_tools(state, AnsweringUI(), interactive=True)}

    assert tools["ask_user"].run({"question": "<q>", "choices": ["<a>", "<b>"]}) == "User answered: <b>"

    result = tools["update_plan"].run({"steps": [{"step": "<one>", "status": "done"}, {"step": "<two>", "status": "pending"}]})
    assert result == "Plan updated, 1/2 done."
    assert state.plan_open()
    with pytest.raises(ToolError):
        tools["update_plan"].run({"steps": [{"step": "<one>", "status": "<bad>"}]})

    tools["suggest_next"].run({"suggestions": ["<a>", " ", "<b>", "<c>", "<d>"]})
    assert state.suggestions == ["<a>", "<b>", "<c>"]
    assert tools["suggest_next"].ends_turn


def test_headless_only_gets_the_plan_tool():
    assert [tool.name for tool in interact_tools(Interaction(), PlainUI(), interactive=False)] == ["update_plan"]


def mcp_placeholder(name: str, description: str, group: str = "mcp:<server>") -> Tool:
    return Tool(name, description, object_schema([]), Risk.MCP, run=lambda args: "", group=group)


def test_lazy_mode_setting():
    assert lazy_enabled("auto", 13) and not lazy_enabled("auto", 12)
    assert lazy_enabled(True, 1) and not lazy_enabled(True, 0)
    assert not lazy_enabled(False, 100)


def test_mcp_search_loads_matching_hidden_tools():
    issues = mcp_placeholder("mcp__tracker__create_issue", "Create an issue in the tracker")
    comment = mcp_placeholder("mcp__tracker__add_comment", "Add a comment to an issue")
    weather = mcp_placeholder("mcp__weather__forecast", "Weather forecast for a city", "mcp:weather")
    registry = Registry([issues, comment, weather])
    registry.hide([issues.name, comment.name, weather.name])
    search = mcp_search_tool(registry)
    registry.add([search])

    assert registry.names() == ["mcp_search"]
    assert registry.get(issues.name) is issues

    result = search.run({"query": "create issue"})
    assert result.index(issues.name) < result.index(comment.name)
    assert weather.name not in result
    assert set(registry.names()) == {"mcp_search", issues.name, comment.name}

    assert "Servers:" in search.run({"query": "<nothing>"})

    registry.set_group_enabled("mcp:weather", False)
    assert weather.name not in search.run({"query": "weather forecast"})


def test_toggle_is_written_to_the_config_that_defines_the_server(tmp_path):
    global_config = tmp_path / "global.toml"
    global_config.write_text('# my servers\n[mcp.servers.shared]\ncommand = "<cmd>"  # keep me\n')
    project_config = tmp_path / ".frakobla-llm" / "config.toml"
    project_config.parent.mkdir()
    project_config.write_text('[mcp.servers.local]\ncommand = "<cmd>"\nenabled = true\n')

    assert set_server_enabled(tmp_path, "shared", False, global_config) == global_config
    assert set_server_enabled(tmp_path, "local", False, global_config) == project_config
    assert set_server_enabled(tmp_path, "<missing>", False, global_config) is None

    assert "# keep me" in global_config.read_text() and "# my servers" in global_config.read_text()
    servers = load_config(tmp_path, global_config, trust=lambda path, sections: True)["mcp"]["servers"]
    assert servers["shared"]["enabled"] is False and servers["local"]["enabled"] is False


def test_invalid_server_entry_is_a_warning():
    configs, warnings = parse_servers({"broken": {"command": "<cmd>", "timeout": "<not a number>"}})

    assert configs == []
    assert "broken" in warnings[0]
