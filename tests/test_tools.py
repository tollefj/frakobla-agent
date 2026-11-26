from pathlib import Path

import pytest

from frakobla_llm.config import DEFAULTS, append_rule, deep_merge, load_config, unknown_keys
from frakobla_llm.context import truncate_middle
from frakobla_llm.tools.fs import FileTools
from frakobla_llm.tools.registry import ToolError
from frakobla_llm.tools.shell import bash_tool
from frakobla_llm.ui.render import split_complete_blocks


@pytest.fixture
def files(tmp_path: Path) -> FileTools:
    return FileTools(tmp_path)


def edit(files: FileTools, old: str, new: str, **extra) -> str:
    return files.edit_file({"path": "f.txt", "old_string": old, "new_string": new, **extra})


def test_edit_replaces_a_unique_match(files, tmp_path):
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    edit(files, "two", "three")
    assert (tmp_path / "f.txt").read_text() == "one\nthree\n"


def test_edit_rejects_ambiguous_missing_and_empty_matches(files, tmp_path):
    (tmp_path / "f.txt").write_text("a a")
    with pytest.raises(ToolError, match="2 times"):
        edit(files, "a", "b")
    with pytest.raises(ToolError, match="not found"):
        edit(files, "z", "b")
    with pytest.raises(ToolError, match="empty"):
        edit(files, "", "b")
    with pytest.raises(ToolError, match="identical"):
        edit(files, "a", "a")
    assert (tmp_path / "f.txt").read_text() == "a a"


def test_edit_replace_all_and_crlf_preserved(files, tmp_path):
    (tmp_path / "f.txt").write_bytes(b"a\r\na\r\n")
    edit(files, "a", "b", replace_all=True)
    assert (tmp_path / "f.txt").read_bytes() == b"b\r\nb\r\n"


def test_edit_missing_file(files):
    with pytest.raises(ToolError, match="does not exist"):
        edit(files, "a", "b")


def test_read_file_paginates_with_line_numbers(files, tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\n")
    output = files.read_file({"path": "f.txt", "offset": 2, "limit": 1})
    assert output.startswith("2\tb")
    assert "offset=3" in output


def test_glob_and_grep_skip_ignored_directories(files, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src" / "a.py").write_text("needle\n")
    (tmp_path / "node_modules" / "b.py").write_text("needle\n")

    assert files.glob({"pattern": "**/*.py"}) == "src/a.py"
    assert files.glob({"pattern": "*.py"}) == "src/a.py"
    assert files.grep({"pattern": "need.e"}) == "src/a.py:1: needle"
    assert files.grep({"pattern": "needle", "include": "*.md"}) == "No matches."


def test_bash_reports_exit_code_and_timeout(tmp_path):
    bash = bash_tool(tmp_path, 5)
    assert bash.run({"command": "echo out; echo err >&2; exit 3"}) == "out\nerr\n[exit code 3]"
    assert "timed out after 1s" in bash.run({"command": "sleep 5", "timeout": 1})


def test_truncate_middle_keeps_head_and_tail():
    text = "a" * 50 + "b" * 50
    result = truncate_middle(text, 20)
    assert result.startswith("a" * 10) and result.endswith("b" * 10)
    assert "80 characters truncated" in result
    assert truncate_middle("short", 20) == "short"


def test_markdown_blocks_split_outside_code_fences():
    assert split_complete_blocks("one\n\ntwo") == ("one\n\n", "two")
    assert split_complete_blocks("```\na\n\nb") == ("", "```\na\n\nb")
    assert split_complete_blocks("```\na\n\nb\n```\n\nrest") == ("```\na\n\nb\n```\n\n", "rest")


def test_config_merge_and_unknown_keys(tmp_path):
    assert deep_merge({"a": {"x": 1, "y": [1]}}, {"a": {"y": [2]}}) == {"a": {"x": 1, "y": [2]}}
    loaded = {"server": {"<typo>": 1}, "permissions": {"rules": {"allow": {"bash": []}}}}
    assert unknown_keys(loaded, DEFAULTS) == ["server.<typo>"]


def test_persisted_rule_round_trips_through_the_project_config(tmp_path):
    append_rule(tmp_path / ".frakobla-llm" / "config.toml", "allow", "bash", 'echo "hi"', "exact")
    config = load_config(tmp_path, global_path=tmp_path / "absent.toml")
    assert config["permissions"]["rules"]["allow"]["bash"] == [{"pattern": 'echo "hi"', "kind": "exact"}]
    assert config.warnings == []


def test_grep_skips_sensitive_files_and_glob_survives_dangling_links(files, tmp_path):
    (tmp_path / "server.key").write_text("needle\n")
    (tmp_path / "notes.txt").write_text("needle\n")
    (tmp_path / "dangling").symlink_to(tmp_path / "<missing>")

    assert files.grep({"pattern": "needle"}) == "notes.txt:1: needle"
    assert "notes.txt" in files.glob({"pattern": "*"})


def test_project_config_needs_trust_for_gated_sections(tmp_path):
    project = tmp_path / ".frakobla-llm" / "config.toml"
    project.parent.mkdir()
    project.write_text('[server]\nbase_url = "<url>"\n[ui]\nshow_reasoning = true\n')
    absent = tmp_path / "absent.toml"

    untrusted = load_config(tmp_path, absent)
    assert untrusted["server"]["base_url"] == DEFAULTS["server"]["base_url"]
    assert untrusted["ui"]["show_reasoning"] is True
    assert "not trusted" in untrusted.warnings[0]

    assert load_config(tmp_path, absent, trust=lambda path, sections: True)["server"]["base_url"] == "<url>"
    assert load_config(tmp_path, absent)["server"]["base_url"] == "<url>"

    append_rule(project, "allow", "bash", "ls", "exact")
    assert load_config(tmp_path, absent)["server"]["base_url"] == "<url>"

    project.write_text('[server]\nbase_url = "<other url>"\n')
    assert load_config(tmp_path, absent)["server"]["base_url"] == DEFAULTS["server"]["base_url"]


def test_legacy_data_is_moved_without_overwriting(tmp_path):
    from frakobla_llm.config import migrate_legacy_data

    legacy, target = tmp_path / "legacy", tmp_path / "target"
    (legacy / "sessions" / "<dir>").mkdir(parents=True)
    (legacy / "sessions" / "<dir>" / "a.jsonl").write_text("<old session>")
    (legacy / "history").write_text("<old history>")
    target.mkdir()
    (target / "history").write_text("<new history>")

    assert migrate_legacy_data(legacy, target) == 1
    assert (target / "sessions" / "<dir>" / "a.jsonl").read_text() == "<old session>"
    assert (target / "history").read_text() == "<new history>"
    assert (legacy / "history").exists()
