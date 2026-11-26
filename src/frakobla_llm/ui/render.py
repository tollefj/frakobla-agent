from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from frakobla_llm.permissions import Answer, PermissionRequest
from frakobla_llm.tools.interact import PlanItem
from frakobla_llm.tools.registry import Tool
from frakobla_llm.ui.prompt import PLAN_MARKS, read_choice, read_permission_answer

RESULT_PREVIEW_LINES = 6
REASONING_TAIL_LINES = 6
ARG_PREVIEW_CHARS = 70
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def strip_control(text: str) -> str:
    return CONTROL_CHARS.sub("", text)


def split_complete_blocks(text: str) -> tuple[str, str]:
    in_fence = False
    cut = position = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        position += len(line)
        if not in_fence and not stripped and line.endswith("\n"):
            cut = position
    return text[:cut], text[cut:]


def summarize_args(args: dict) -> str:
    parts = []
    for key, value in args.items():
        shown = value if isinstance(value, str) else json.dumps(value)
        shown = shown.replace("\n", "\\n")
        if len(shown) > ARG_PREVIEW_CHARS:
            shown = shown[:ARG_PREVIEW_CHARS] + "..."
        parts.append(f"{key}={shown}")
    return strip_control("  ".join(parts))


class Renderer:
    def __init__(self, console: Console, show_reasoning: bool, spinner: str = "dots") -> None:
        self.console = console
        self.show_reasoning = show_reasoning
        self.spinner = spinner
        self._live: Live | None = None
        self._pending_markdown = ""
        self._reasoning = ""
        self._reasoning_tokens = 0
        self._reasoning_started = 0.0
        self.on_tick: Callable[[bool], None] = lambda force: None

    def waiting(self) -> None:
        self.on_tick(True)
        self._show(Spinner(self.spinner, "waiting for model...", style="dim", speed=0.8))

    def reasoning(self, text: str) -> None:
        if not self._reasoning:
            self._reasoning_started = time.monotonic()
        self._reasoning += strip_control(text)
        self._reasoning_tokens += 1
        self.on_tick(False)
        tail = "\n".join(self._reasoning.splitlines()[-REASONING_TAIL_LINES:])
        label = f"thinking... {self._reasoning_tokens} tokens"
        self._show(Group(Spinner(self.spinner, label, style="dim", speed=0.8), Text(tail, style="dim italic")))

    def content(self, text: str) -> None:
        self._finish_reasoning()
        self.on_tick(False)
        self._pending_markdown += strip_control(text)
        complete, self._pending_markdown = split_complete_blocks(self._pending_markdown)
        if complete.strip():
            self.console.print(Markdown(complete))
        self._show(Markdown(self._pending_markdown))

    def stream_end(self) -> None:
        self._finish_reasoning()
        self._hide()
        if self._pending_markdown.strip():
            self.console.print(Markdown(self._pending_markdown))
        self._pending_markdown = ""
        self.on_tick(True)

    def tool_call(self, tool: Tool, args: dict) -> None:
        line = Text.assemble((tool.name, "bold cyan"), "  ", (summarize_args(args), "dim"))
        self.console.print(line, overflow="ellipsis", no_wrap=True)

    def tool_result(self, result: str, ok: bool) -> None:
        lines = strip_control(result).splitlines() or [""]
        shown = lines[:RESULT_PREVIEW_LINES]
        if len(lines) > len(shown):
            shown.append(f"... +{len(lines) - len(shown)} lines")
        style = "dim" if ok else "red"
        for line in shown:
            self.console.print(Text("  " + line, style=style), overflow="ellipsis", no_wrap=True)
        self.on_tick(True)

    def notice(self, text: str) -> None:
        self.console.print(Text(text, style="yellow"))

    def ask_permission(self, request: PermissionRequest) -> Answer:
        self._hide()
        self.console.print(Panel(_permission_body(request), title="permission", border_style="yellow"))
        return read_permission_answer(explicit=request.needs_explicit_answer)

    def ask_user(self, question: str, choices: list[str]) -> str:
        self._hide()
        body = Text(strip_control(question))
        for number, choice in enumerate(choices, 1):
            body.append(f"\n  {number} ", style="bold cyan")
            body.append(strip_control(choice))
        self.console.print(Panel(body, title="question", border_style="magenta"))
        return read_choice(choices)

    def show_plan(self, plan: list[PlanItem]) -> None:
        self._hide()
        self.console.print(plan_text(plan))

    def _finish_reasoning(self) -> None:
        if not self._reasoning:
            return
        self._hide()
        if self.show_reasoning:
            self.console.print(Text(self._reasoning.strip(), style="dim italic"))
        else:
            seconds = time.monotonic() - self._reasoning_started
            self.console.print(
                Text(f"thought for {seconds:.1f}s ({self._reasoning_tokens} tokens)", style="dim italic")
            )
        self._reasoning = ""
        self._reasoning_tokens = 0

    def _show(self, renderable: RenderableType) -> None:
        if self._live is None:
            self._live = Live(
                renderable,
                console=self.console,
                transient=True,
                refresh_per_second=20,
                vertical_overflow="ellipsis",
            )
            self._live.start()
        else:
            self._live.update(renderable)

    def _hide(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


class PlainUI:
    def __init__(self) -> None:
        self._line_open = False

    def waiting(self) -> None:
        pass

    def reasoning(self, text: str) -> None:
        pass

    def content(self, text: str) -> None:
        text = strip_control(text)
        sys.stdout.write(text)
        sys.stdout.flush()
        self._line_open = not text.endswith("\n")

    def stream_end(self) -> None:
        if self._line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._line_open = False

    def tool_call(self, tool: Tool, args: dict) -> None:
        print(f"[{tool.name}] {summarize_args(args)}", file=sys.stderr)

    def tool_result(self, result: str, ok: bool) -> None:
        if not ok:
            print(f"  {strip_control(result)}", file=sys.stderr)

    def notice(self, text: str) -> None:
        print(text, file=sys.stderr)

    def ask_user(self, question: str, choices: list[str]) -> str:
        return "the user is not available, decide yourself"

    def show_plan(self, plan: list[PlanItem]) -> None:
        print(plan_text(plan).plain, file=sys.stderr)


def plan_text(plan: list[PlanItem]) -> Text:
    text = Text()
    for item in plan:
        mark, style = PLAN_MARKS[item.status]
        text.append(f"  {mark}{strip_control(item.step)}\n", style=style.replace("ansi", ""))
    text.rstrip()
    return text


def _command_detail(command: str) -> RenderableType:
    if CONTROL_CHARS.search(command):
        return Text(f"{command!r}\n(contains control characters, shown escaped)", style="bold red")
    return Syntax(command, "bash", theme="ansi_dark", word_wrap=True)


def _permission_body(request: PermissionRequest) -> RenderableType:
    tool = request.tool
    if tool.preview:
        detail: RenderableType = Syntax(strip_control(tool.preview(request.args)), "diff", theme="ansi_dark", word_wrap=True)
    elif tool.subject_arg and not tool.subject_is_path:
        detail = _command_detail(request.subject)
    else:
        detail = Text(summarize_args(request.args))
    lines: list[RenderableType] = [
        Text.assemble((tool.name, "bold cyan"), f"  ({request.reason})"),
        detail,
    ]
    if request.warning:
        lines.append(Text(f"warning: {request.warning}", style="bold red"))
    if request.grant.kind == "exact":
        scope = "this exact command"
    elif tool.subject_is_path:
        scope = f"all {tool.name} calls inside the working directory"
    else:
        scope = f"all {tool.name} calls"
    lines.append(Text(f"session/always grants: {scope} (always applies to every project)", style="dim"))
    return Group(*lines)
