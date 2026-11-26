from pathlib import Path

import pytest

from frakobla_llm.client import CompletionBuilder, ContextOverflowError, LlamaClient, is_context_overflow
from frakobla_llm.context import anchor_user_index, find_cut, kept_messages, transcript, with_summary
from frakobla_llm.loop import Agent, HistoryError, Turn
from frakobla_llm.permissions import Permissions
from frakobla_llm.session import Session
from frakobla_llm.tools.registry import Registry
from frakobla_llm.ui.render import PlainUI


def user(text: str = "<user>") -> dict:
    return {"role": "user", "content": text}


def assistant(text: str = "<assistant>", calls: int = 0) -> dict:
    message: dict = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {"id": f"<id_{n}>", "type": "function", "function": {"name": "<tool>", "arguments": "{}"}}
            for n in range(calls)
        ]
    return message


def tool(text: str = "<result>") -> dict:
    return {"role": "tool", "tool_call_id": "<id_0>", "content": text}


def make_agent(tmp_path: Path, session: Session | None = None) -> Agent:
    permissions = Permissions({"mode": "ask", "rules": {}}, tmp_path)
    return Agent(
        client=LlamaClient("http://127.0.0.1:9"),
        model="<model>",
        registry=Registry([]),
        permissions=permissions,
        session=session or Session.disabled(),
        ui=PlainUI(),
        system_prompt="<system>",
        max_iterations=5,
        output_max_chars=1000,
    )


def test_cut_never_separates_tool_calls_from_results():
    messages = [user(), assistant(calls=2), tool(), tool(), assistant(calls=1), tool(), assistant(), user(), assistant()]
    for keep in range(0, 200, 5):
        cut = find_cut(messages, keep)
        assert messages[cut]["role"] in ("user", "assistant")
        assert messages[cut - 1]["role"] != "assistant" or "tool_calls" not in messages[cut - 1]


def test_cut_prefers_the_earliest_boundary_that_fits():
    messages = [user("a" * 50), assistant("b" * 50), user("c" * 30), assistant("d" * 30), user("e" * 10), assistant("f" * 10)]
    assert find_cut(messages, 1000) == 2
    assert find_cut(messages, 25) == 4
    assert find_cut(messages, 5) == 4


def test_cut_inside_a_single_turn_keeps_the_user_request():
    messages = [user("<task>"), assistant(calls=1), tool("x" * 100), assistant(calls=1), tool()]
    cut = find_cut(messages, 50)
    assert cut == 3
    anchor = anchor_user_index(messages, cut)
    assert anchor == 0
    assert kept_messages(messages, cut, anchor) == [user("<task>"), assistant(calls=1), tool()]


def test_nothing_to_cut_in_a_lone_user_message():
    assert find_cut([user()], 0) == 0


def test_transcript_truncates_tool_results_and_shows_calls():
    text = transcript([user("<q>"), assistant("", calls=1), tool("r" * 5000)], 100_000)
    assert "[tool call] <tool> {}" in text
    assert "characters truncated" in text


def test_summary_goes_into_the_system_prompt():
    assert with_summary("<system>", "") == "<system>"
    assert "<summary>" in with_summary("<system>", "<summary>")


def test_live_compaction_and_replay_agree(tmp_path):
    session = Session(tmp_path / "s.jsonl", tmp_path)
    live = make_agent(tmp_path, session)
    for message in [user("<one>"), assistant(), user("<two>"), assistant(calls=1), tool(), assistant()]:
        if message["role"] == "user":
            live.turns.append(Turn(len(live.messages)))
        live._append(message)
    live._apply_compaction("<summary>", 2, None)
    session.append("compaction", summary="<summary>", cut=2, anchor=None)
    live._append(user("<three>"))

    replayed = make_agent(tmp_path)
    replayed.restore(Session.open(session.path).entries())

    assert replayed.messages == [*live.messages[:-1], user("<three>")]
    assert replayed.summary == "<summary>"
    assert [turn.start for turn in replayed.turns] == [0, 4]


def test_compaction_with_anchor_rebases_the_current_turn(tmp_path):
    agent = make_agent(tmp_path)
    agent.messages = [user("<old>"), assistant(), user("<task>"), assistant(calls=1), tool(), assistant(calls=1), tool()]
    agent.turns = [Turn(0), Turn(2)]
    agent._apply_compaction("<summary>", 5, anchor_user_index(agent.messages, 5))
    assert agent.messages == [user("<task>"), assistant(calls=1), tool()]
    assert [turn.start for turn in agent.turns] == [0]


def test_undo_restores_files_and_drops_the_turn(tmp_path):
    session = Session(tmp_path / "s.jsonl", tmp_path)
    agent = make_agent(tmp_path, session)
    edited, created = tmp_path / "edited.txt", tmp_path / "created.txt"
    edited.write_text("before")

    agent.turns.append(Turn(0))
    agent._append(user("<edit>"))
    agent.turns[-1].snapshots += [session.snapshot(edited), session.snapshot(created)]
    edited.write_text("after")
    created.write_text("new")
    agent._append(assistant())

    result = agent.undo()

    assert edited.read_text() == "before"
    assert not created.exists()
    assert result.messages_removed == 2
    assert agent.messages == []
    with pytest.raises(HistoryError):
        agent.undo()

    replayed = make_agent(tmp_path)
    replayed.restore(Session.open(session.path).entries())
    assert replayed.messages == [] and replayed.turns == []


def test_sessions_are_written_lazily_and_listed_with_titles(tmp_path):
    root = tmp_path / "sessions"
    empty = Session.create(tmp_path, root)
    assert not empty.path.exists()

    session = Session.create(tmp_path, root)
    session.append("message", message=user("<first prompt>\nmore"))
    assert [info.title for info in Session.list(tmp_path, root)] == ["<first prompt>"]

    session.rename("<name>")
    infos = Session.list(tmp_path, root)
    assert [(info.title, info.turns) for info in infos] == [("<name>", 1)]
    assert Session.open(session.path).name == "<name>"


def test_context_overflow_is_recognized():
    assert is_context_overflow({"type": "exceed_context_size_error", "message": "<text>"})
    assert is_context_overflow({"message": "the request exceeds the available context size"})
    assert not is_context_overflow({"message": "<other error>"})
    with pytest.raises(ContextOverflowError):
        CompletionBuilder().add({"error": {"type": "exceed_context_size_error", "message": "<text>"}})


def test_open_plan_survives_compaction_and_replay(tmp_path):
    from frakobla_llm.tools.interact import PlanItem

    session = Session(tmp_path / "s.jsonl", tmp_path)
    live = make_agent(tmp_path, session)
    live.interaction.plan = [PlanItem("<step_a>", "done"), PlanItem("<step_b>", "in_progress")]
    for message in [user(), assistant(), user(), assistant()]:
        live._append(message)
    live._summarize = lambda older, instructions: "<summary>"
    live.compact(everything=False)

    system = live.request_messages()[0]["content"]
    assert "- [in_progress] <step_b>" in system
    assert "- [done] <step_a>" in system

    replayed = make_agent(tmp_path)
    replayed.restore(Session.open(session.path).entries())
    assert replayed.request_messages()[0]["content"] == system


def test_finished_plan_is_left_out():
    from frakobla_llm.context import plan_text
    from frakobla_llm.tools.interact import PlanItem

    assert plan_text([PlanItem("<step>", "done")]) == ""
    assert "Plan" not in with_summary("<system>", "<summary>", "")


def test_failing_permission_prompt_still_records_a_tool_result(tmp_path):
    from frakobla_llm.client import Completion, ToolCall
    from frakobla_llm.tools.fs import fs_tools

    class OneWriteClient(LlamaClient):
        def stream_chat(self, *args, **kwargs):
            return Completion(tool_calls=[ToolCall("<id>", "write_file", '{"path": "a.txt", "content": 5}')])

    def broken_prompter(request):
        raise RuntimeError("<prompt failure>")

    agent = make_agent(tmp_path)
    agent.client = OneWriteClient("http://127.0.0.1:9")
    agent.registry = Registry(fs_tools(tmp_path))
    agent.permissions = Permissions({"mode": "ask", "rules": {}}, tmp_path, prompter=broken_prompter)
    agent.max_iterations = 1

    agent.run_turn("<prompt>")

    assert [message["role"] for message in agent.messages] == ["user", "assistant", "tool"]
    assert "<prompt failure>" in agent.messages[-1]["content"]


def test_damaged_session_lines_are_skipped(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"type": "session"}\n{"type": "message", "message": {"role": "user", "content": "<user>"}}\n{"type": "mess')

    assert [entry["type"] for entry in Session(path).entries()] == ["session", "message"]


def test_moved_snapshot_backups_are_found_next_to_the_session(tmp_path):
    session = Session(tmp_path / "s.jsonl", tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("<original>")
    taken = session.snapshot(target)

    stored = session.stored_snapshot(taken.path, "/old/location/" + Path(taken.backup).name)

    assert stored.backup == taken.backup


def test_interrupted_auto_compaction_leaves_a_placeholder_reply(tmp_path):
    class InterruptedClient(LlamaClient):
        def stream_chat(self, *args, **kwargs):
            raise KeyboardInterrupt

    agent = make_agent(tmp_path)
    agent.client = InterruptedClient("http://127.0.0.1:9")
    agent.context_size = 10
    agent.messages = [user("x" * 100), assistant("y" * 100)]

    with pytest.raises(KeyboardInterrupt):
        agent.run_turn("<prompt>")

    assert [message["role"] for message in agent.messages][-2:] == ["user", "assistant"]
