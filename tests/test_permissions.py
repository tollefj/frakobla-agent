from pathlib import Path

import pytest

from frakobla_llm.permissions import Answer, Choice, Decision, Permissions, Rule
from frakobla_llm.tools.fs import fs_tools
from frakobla_llm.tools.shell import bash_tool


def settings(mode: str = "ask", **rules: dict) -> dict:
    return {"mode": mode, "rules": {"allow": {}, "ask": {}, "deny": {}, **rules}}


@pytest.fixture
def tools(tmp_path: Path) -> dict:
    return {tool.name: tool for tool in [*fs_tools(tmp_path), bash_tool(tmp_path, 5)]}


def decide(permissions: Permissions, tool, **args) -> Decision:
    return permissions.evaluate(tool, args).decision


def test_mode_defaults(tools, tmp_path):
    ask = Permissions(settings("ask"), tmp_path)
    readonly = Permissions(settings("readonly"), tmp_path)
    yolo = Permissions(settings("yolo"), tmp_path)

    assert decide(ask, tools["read_file"], path="a.txt") is Decision.ALLOW
    assert decide(ask, tools["bash"], command="ls") is Decision.ASK
    assert decide(readonly, tools["write_file"], path="a.txt", content="") is Decision.DENY
    assert decide(yolo, tools["bash"], command="ls") is Decision.ALLOW


def test_deny_rule_beats_allow_rule_and_yolo(tools, tmp_path):
    rules = {
        "allow": {"bash": [{"pattern": "git *"}]},
        "deny": {"bash": [{"pattern": "git push*"}]},
    }
    permissions = Permissions(settings("yolo", **rules), tmp_path)

    assert decide(permissions, tools["bash"], command="git status") is Decision.ALLOW
    assert decide(permissions, tools["bash"], command="git push origin") is Decision.DENY


def test_ask_rule_beats_allow_rule(tools, tmp_path):
    rules = {
        "allow": {"bash": [{"pattern": "git *"}]},
        "ask": {"bash": [{"pattern": "^git commit", "kind": "regex"}]},
    }
    permissions = Permissions(settings(**rules), tmp_path)
    assert decide(permissions, tools["bash"], command="git commit -m x") is Decision.ASK


def test_allow_glob_does_not_cover_chained_commands(tools, tmp_path):
    permissions = Permissions(settings(allow={"bash": [{"pattern": "git status*"}]}), tmp_path)

    assert decide(permissions, tools["bash"], command="git status -s") is Decision.ALLOW
    assert decide(permissions, tools["bash"], command="git status; touch x") is Decision.ASK
    assert decide(permissions, tools["bash"], command="git status $(touch x)") is Decision.ASK


def test_sensitive_files_are_denied_even_in_yolo(tools, tmp_path):
    permissions = Permissions(settings("yolo"), tmp_path)

    assert decide(permissions, tools["read_file"], path=".env") is Decision.DENY
    assert decide(permissions, tools["read_file"], path="certs/server.pem") is Decision.DENY
    assert decide(permissions, tools["read_file"], path=".env.example") is Decision.ALLOW


def test_paths_outside_cwd_escalate_to_ask(tools, tmp_path):
    permissions = Permissions(settings(), tmp_path)

    assert decide(permissions, tools["read_file"], path="../other.txt") is Decision.ASK
    assert decide(permissions, tools["grep"], pattern="x", path="/etc") is Decision.ASK
    assert decide(permissions, tools["grep"], pattern="x") is Decision.ALLOW


def test_dangerous_command_asks_with_warning_despite_allow_rule(tools, tmp_path):
    permissions = Permissions(settings(allow={"bash": [{"pattern": "*"}]}), tmp_path)
    verdict = permissions.evaluate(tools["bash"], {"command": "sudo make install"})
    assert verdict.decision is Decision.ASK
    assert "root" in verdict.warning


def test_doom_loop_asks_on_third_and_denies_on_fourth_identical_call(tools, tmp_path):
    asked = []

    def prompter(request):
        asked.append(request.reason)
        return Answer(Choice.ONCE)

    permissions = Permissions(settings("yolo"), tmp_path, prompter)
    args = {"command": "ls"}
    decisions = [permissions.authorize(tools["bash"], args).decision for _ in range(4)]

    assert decisions == [Decision.ALLOW, Decision.ALLOW, Decision.ALLOW, Decision.DENY]
    assert asked == ["third identical call in a row"]
    assert permissions.authorize(tools["bash"], {"command": "pwd"}).decision is Decision.ALLOW


def test_session_grant_covers_exact_command_only(tools, tmp_path):
    answers = iter([Answer(Choice.SESSION), Answer(Choice.DENY, "<reason>")])
    permissions = Permissions(settings(), tmp_path, lambda request: next(answers))

    assert permissions.authorize(tools["bash"], {"command": "make test"}).decision is Decision.ALLOW
    assert decide(permissions, tools["bash"], command="make test") is Decision.ALLOW
    denied = permissions.authorize(tools["bash"], {"command": "make test && make install"})
    assert denied.decision is Decision.DENY
    assert "<reason>" in denied.reason


def test_always_persists_the_grant(tools, tmp_path):
    persisted = []
    permissions = Permissions(
        settings(),
        tmp_path,
        lambda request: Answer(Choice.ALWAYS),
        lambda tool, rule: persisted.append((tool, rule)),
    )
    permissions.authorize(tools["edit_file"], {"path": "a.py", "old_string": "a", "new_string": "b"})

    assert persisted == [("edit_file", Rule("*", "glob"))]
    assert decide(permissions, tools["edit_file"], path="b.py", old_string="a", new_string="b") is Decision.ALLOW
    assert decide(permissions, tools["edit_file"], path="../b.py", old_string="a", new_string="b") is Decision.ASK


def test_ask_without_prompter_is_denied(tools, tmp_path):
    permissions = Permissions(settings(), tmp_path)
    assert permissions.authorize(tools["bash"], {"command": "ls"}).decision is Decision.DENY


def test_invalid_rule_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Permissions(settings(allow={"bash": [{"pattern": "x", "kind": "<unknown>"}]}), tmp_path)


def test_readonly_ignores_allow_rules_and_ask_rules(tools, tmp_path):
    allow = {"write_file": [{"pattern": "*"}], "bash": [{"pattern": "ls", "kind": "exact"}]}
    readonly = Permissions(settings("readonly", allow=allow, ask={"bash": [{"pattern": "git *"}]}), tmp_path)

    assert decide(readonly, tools["write_file"], path="a.txt", content="") is Decision.DENY
    assert decide(readonly, tools["bash"], command="ls") is Decision.DENY
    assert decide(readonly, tools["bash"], command="git status") is Decision.DENY
    assert decide(readonly, tools["bash"], command="sudo ls") is Decision.DENY
    assert decide(readonly, tools["read_file"], path="a.txt") is Decision.ALLOW


def test_malformed_rules_are_value_errors(tmp_path):
    for rules in (
        {"bash": [{"pattern": "(", "kind": "regex"}]},
        {"bash": [{"kind": "glob"}]},
        {"bash": {"pattern": "ls"}},
    ):
        with pytest.raises(ValueError):
            Permissions(settings(allow=rules), tmp_path)


def test_risky_requests_need_an_explicit_answer(tools, tmp_path):
    requests = []

    def prompter(request):
        requests.append(request)
        return Answer(Choice.ONCE)

    permissions = Permissions(settings(), tmp_path, prompter=prompter)
    permissions.authorize(tools["write_file"], {"path": "a.txt", "content": ""})
    permissions.authorize(tools["write_file"], {"path": "/outside/a.txt", "content": ""})
    permissions.authorize(tools["bash"], {"command": "sudo ls"})

    assert [request.needs_explicit_answer for request in requests] == [False, True, True]


def test_permission_prompt_escapes_control_characters_in_commands(tools):
    import io

    from rich.console import Console

    from frakobla_llm.permissions import PermissionRequest
    from frakobla_llm.ui.render import _permission_body

    command = "echo safe\x1b[2K\rrm -rf ~"
    request = PermissionRequest(tools["bash"], {"command": command}, command, "<reason>", "", Rule(command, "exact"))
    out = io.StringIO()
    Console(file=out, color_system=None, width=200).print(_permission_body(request))

    assert "\x1b" not in out.getvalue()
    assert "\\x1b[2K\\rrm -rf ~" in out.getvalue()
