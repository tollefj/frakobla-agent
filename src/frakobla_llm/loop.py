from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from frakobla_llm.client import Completion, ContextOverflowError, Delta, LlamaClient, LlamaError, ToolCall, estimate_tokens
from frakobla_llm.context import (
    anchor_user_index,
    find_cut,
    kept_messages,
    message_chars,
    plan_text,
    summary_request,
    truncate_middle,
    with_summary,
)
from frakobla_llm.permissions import Decision, Permissions
from frakobla_llm.session import Session, Snapshot
from frakobla_llm.stats import Stats
from frakobla_llm.tools.fs import resolve_path
from frakobla_llm.tools.interact import Interaction
from frakobla_llm.tools.registry import Registry, Risk, Tool, ToolError

KEEP_FRACTION = 0.25
SUMMARY_INPUT_FRACTION = 0.6
CHARS_PER_TOKEN = 3
DEFAULT_SUMMARY_CHARS = 100_000


class AgentUI(Protocol):
    def waiting(self) -> None: ...
    def reasoning(self, text: str) -> None: ...
    def content(self, text: str) -> None: ...
    def stream_end(self) -> None: ...
    def tool_call(self, tool: Tool, args: dict) -> None: ...
    def tool_result(self, result: str, ok: bool) -> None: ...
    def notice(self, text: str) -> None: ...


class HistoryError(Exception):
    pass


@dataclass
class Turn:
    start: int
    snapshots: list[Snapshot] = field(default_factory=list)


@dataclass(frozen=True)
class UndoResult:
    restored: list[str]
    messages_removed: int


class Agent:
    def __init__(
        self,
        client: LlamaClient,
        model: str,
        registry: Registry,
        permissions: Permissions,
        session: Session,
        ui: AgentUI,
        system_prompt: str,
        max_iterations: int,
        output_max_chars: int,
        parallel_tool_calls: bool = False,
        thinking: str = "default",
        context_size: int = 0,
        compact_at: float = 0.75,
        auto_compact: bool = True,
        interaction: Interaction | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.registry = registry
        self.permissions = permissions
        self.session = session
        self.ui = ui
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.output_max_chars = output_max_chars
        self.parallel_tool_calls = parallel_tool_calls
        self.thinking = thinking
        self.context_size = context_size
        self.compact_at = compact_at
        self.auto_compact = auto_compact
        self.interaction = interaction or Interaction()
        self.messages: list[dict] = []
        self.summary = ""
        self.plan = ""
        self.turns: list[Turn] = []
        self.last_completion = Completion()
        self.stats = Stats()
        self.context_tokens = 0
        self._unsent_chars = 0

    def run_turn(self, user_text: str) -> None:
        self.turns.append(Turn(len(self.messages)))
        self._append({"role": "user", "content": user_text})
        for _ in range(self.max_iterations):
            if self._needs_compaction():
                self._compact_safely("context is nearly full")
            completion = self._complete()
            self._append(_assistant_message(completion))
            if not completion.tool_calls:
                return
            if self._run_tool_calls(completion.tool_calls):
                return
        self.ui.notice(f"stopped after {self.max_iterations} iterations")

    def reset(self, session: Session) -> None:
        self.messages = []
        self.summary = ""
        self.plan = ""
        self.turns = []
        self.session = session
        self.last_completion = Completion()
        self.context_tokens = 0
        self._unsent_chars = 0

    def request_messages(self) -> list[dict]:
        return [{"role": "system", "content": with_summary(self.system_prompt, self.summary, self.plan)}, *self.messages]

    def estimated_tokens(self) -> int:
        if self.context_tokens:
            return self.context_tokens + self._unsent_chars // CHARS_PER_TOKEN
        return estimate_tokens(json.dumps(self.request_messages(), ensure_ascii=False))

    def compact(self, instructions: str = "", everything: bool = True) -> int:
        if everything:
            cut = len(self.messages)
        else:
            keep_chars = int(self.context_size * KEEP_FRACTION * CHARS_PER_TOKEN) if self.context_size else 0
            cut = find_cut(self.messages, keep_chars)
        if cut == 0:
            raise HistoryError("nothing to compact")
        anchor = anchor_user_index(self.messages, cut)
        summary = self._summarize(self.messages[:cut], instructions)
        plan = plan_text(self.interaction.plan)
        self._apply_compaction(summary, cut, anchor, plan)
        self.session.append("compaction", summary=summary, cut=cut, anchor=anchor, plan=plan)
        self.context_tokens = 0
        self._unsent_chars = 0
        return cut

    def undo(self) -> UndoResult:
        if not self.turns:
            raise HistoryError("nothing to undo in the current history")
        turn = self.turns[-1]
        restored = []
        for snapshot in reversed(turn.snapshots):
            snapshot.restore()
            restored.append(snapshot.path)
        removed = self._drop_last_turn()
        self.session.append("undo")
        return UndoResult(restored, removed)

    def restore(self, entries: Iterable[dict]) -> None:
        for entry in entries:
            kind = entry["type"]
            if kind == "message":
                if entry["message"].get("role") == "user":
                    self.turns.append(Turn(len(self.messages)))
                self.messages.append(entry["message"])
            elif kind == "compaction":
                self._apply_compaction(entry["summary"], entry["cut"], entry.get("anchor"), entry.get("plan", ""))
            elif kind == "snapshot" and self.turns:
                self.turns[-1].snapshots.append(self.session.stored_snapshot(entry["path"], entry.get("backup")))
            elif kind == "undo" and self.turns:
                self._drop_last_turn()

    def _drop_last_turn(self) -> int:
        turn = self.turns.pop()
        removed = len(self.messages) - turn.start
        del self.messages[turn.start :]
        return removed

    def _apply_compaction(self, summary: str, cut: int, anchor: int | None, plan: str = "") -> None:
        self.messages = kept_messages(self.messages, cut, anchor)
        self.summary = summary
        self.plan = plan
        shift = 1 if anchor is not None else 0
        rebased = []
        for turn in self.turns:
            if turn.start >= cut:
                turn.start = turn.start - cut + shift
            elif anchor is not None and turn.start == anchor:
                turn.start = 0
            else:
                continue
            rebased.append(turn)
        self.turns = rebased

    def _summarize(self, older: list[dict], instructions: str) -> str:
        budget = (
            int(self.context_size * SUMMARY_INPUT_FRACTION * CHARS_PER_TOKEN)
            if self.context_size
            else DEFAULT_SUMMARY_CHARS
        )
        self.ui.waiting()
        self.stats.begin()
        completion = None
        try:
            completion = self.client.stream_chat(
                self.model, summary_request(self.summary, older, budget, instructions),
                [],
                lambda delta: self.stats.token(delta.kind),
                thinking=self.thinking,
            )
        finally:
            self.stats.end(completion)
            self.ui.stream_end()
        summary = completion.content.strip()
        if not summary:
            raise HistoryError("the model returned an empty summary")
        return summary

    def _needs_compaction(self) -> bool:
        if not (self.auto_compact and self.context_size):
            return False
        return self.estimated_tokens() >= self.compact_at * self.context_size

    def _compact_safely(self, reason: str) -> bool:
        self.ui.notice(f"{reason}, compacting history")
        try:
            self.compact(everything=False)
        except (HistoryError, LlamaError) as error:
            self.ui.notice(f"compaction failed: {error}")
            return False
        except KeyboardInterrupt:
            self._placeholder("interrupted by user")
            raise
        return True

    def _complete(self) -> Completion:
        try:
            completion = self._stream()
        except ContextOverflowError:
            if not self._compact_safely("context overflow"):
                self._placeholder("context overflow")
                raise LlamaError(self._overflow_text()) from None
            try:
                completion = self._stream()
            except ContextOverflowError:
                self._placeholder("context overflow")
                raise LlamaError(self._overflow_text()) from None
            except (KeyboardInterrupt, LlamaError) as error:
                self._placeholder(_failure_reason(error))
                raise
        except (KeyboardInterrupt, LlamaError) as error:
            self._placeholder(_failure_reason(error))
            raise
        self.last_completion = completion
        self._record_usage(completion)
        if completion.finish_reason == "length":
            self.ui.notice("the model hit its output limit, the reply may be cut off")
        return completion

    def _stream(self) -> Completion:
        self.ui.waiting()
        self.stats.begin()
        completion = None
        try:
            completion = self.client.stream_chat(
                self.model,
                self.request_messages(),
                self.registry.schemas(),
                self._on_delta,
                self.parallel_tool_calls,
                self.thinking,
            )
            return completion
        finally:
            self.stats.end(completion)
            self.ui.stream_end()

    def _record_usage(self, completion: Completion) -> None:
        usage = completion.usage
        if usage.get("prompt_tokens"):
            self.context_tokens = usage["prompt_tokens"] + usage.get("completion_tokens", 0)
        else:
            text = json.dumps(self.request_messages(), ensure_ascii=False) + completion.content
            self.context_tokens = self.client.count_tokens(self.model, text)
        self._unsent_chars = 0

    def _overflow_text(self) -> str:
        size = f" of {self.context_size} tokens" if self.context_size else ""
        return (
            f"context overflow: the conversation does not fit in the model's context{size}, even after compacting. "
            "Use /new to start over, or restart llama-server with a larger --ctx-size."
        )

    def _placeholder(self, reason: str) -> None:
        self._append({"role": "assistant", "content": f"[no response: {reason}]"})

    def _on_delta(self, delta: Delta) -> None:
        self.stats.token(delta.kind)
        if delta.kind == "reasoning":
            self.ui.reasoning(delta.text)
        else:
            self.ui.content(delta.text)

    def _run_tool_calls(self, calls: list[ToolCall]) -> bool:
        for position, call in enumerate(calls):
            try:
                result = self._execute(call)
            except KeyboardInterrupt:
                for cancelled in calls[position:]:
                    self._append_tool_result(cancelled, "Cancelled by user.")
                raise
            except Exception as error:
                result = self._fail(f"Error: {type(error).__name__}: {error}")
            self._append_tool_result(call, result)
        tools = [self.registry.get(call.name) for call in calls]
        return all(tool is not None and tool.ends_turn for tool in tools)

    def _execute(self, call: ToolCall) -> str:
        tool = self.registry.get(call.name)
        if tool is None:
            return self._fail(f"Error: unknown tool {call.name!r}. Available: {', '.join(self.registry.names())}")
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as error:
            return self._fail(f"Error: arguments for {call.name} are not valid JSON ({error}). Send the call again.")
        if not isinstance(args, dict):
            return self._fail(f"Error: arguments for {call.name} must be a JSON object.")
        if missing := tool.missing_arguments(args):
            return self._fail(f"Error: missing required arguments for {call.name}: {', '.join(missing)}")

        self.ui.tool_call(tool, args)
        verdict = self.permissions.authorize(tool, args)
        self.session.append(
            "permission", tool=tool.name, args=args, decision=verdict.decision, reason=verdict.reason
        )
        if verdict.decision is Decision.DENY:
            return self._fail(f"Permission denied: {verdict.reason}. Do not retry the same call.")
        self._snapshot_before_write(tool, args)
        try:
            result = tool.run(args)
        except ToolError as error:
            return self._fail(f"Error: {error}")
        result = truncate_middle(result, self.output_max_chars)
        self.ui.tool_result(result, ok=True)
        return result

    def _snapshot_before_write(self, tool: Tool, args: dict) -> None:
        if tool.risk is not Risk.WRITE or not tool.subject_is_path or not self.turns:
            return
        path = resolve_path(self.permissions.cwd, str(args.get(tool.subject_arg) or "."))
        turn = self.turns[-1]
        if any(snapshot.path == str(path) for snapshot in turn.snapshots):
            return
        turn.snapshots.append(self.session.snapshot(path))

    def _fail(self, message: str) -> str:
        self.ui.tool_result(message, ok=False)
        return message

    def _append_tool_result(self, call: ToolCall, result: str) -> None:
        self._append({"role": "tool", "tool_call_id": call.id, "content": result})

    def _append(self, message: dict) -> None:
        self.messages.append(message)
        if message["role"] != "assistant":
            self._unsent_chars += message_chars([message])
        self.session.append("message", message=message)


def _failure_reason(error: BaseException) -> str:
    return "interrupted by user" if isinstance(error, KeyboardInterrupt) else "server error"


def _assistant_message(completion: Completion) -> dict:
    message: dict = {"role": "assistant", "content": completion.content}
    if completion.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": _valid_json_or_empty(call.arguments)},
            }
            for call in completion.tool_calls
        ]
    return message


def _valid_json_or_empty(arguments: str) -> str:
    try:
        json.loads(arguments)
    except json.JSONDecodeError:
        return "{}"
    return arguments
