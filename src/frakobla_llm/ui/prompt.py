from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit import prompt as plain_prompt
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from frakobla_llm.config import CONFIG_DIR
from frakobla_llm.permissions import Answer, Choice
from frakobla_llm.tools.interact import Interaction, PlanItem

HISTORY_FILE = CONFIG_DIR / "history"
PERMISSION_KEYS = {
    "": Choice.ONCE,
    "y": Choice.ONCE,
    "yes": Choice.ONCE,
    "once": Choice.ONCE,
    "s": Choice.SESSION,
    "session": Choice.SESSION,
    "a": Choice.ALWAYS,
    "always": Choice.ALWAYS,
    "n": Choice.DENY,
    "no": Choice.DENY,
    "deny": Choice.DENY,
}
PERMISSION_PROMPT = "[Y] once  [s] session  [a] always  [n] deny  (or type a reason to deny) > "
EXPLICIT_PERMISSION_PROMPT = "[y] once  [s] session  [a] always  [n] deny  (or type a reason to deny) > "


def read_permission_answer(explicit: bool = False) -> Answer:
    while True:
        try:
            reply = plain_prompt(EXPLICIT_PERMISSION_PROMPT if explicit else PERMISSION_PROMPT).strip()
        except EOFError:
            return Answer(Choice.DENY, "no input")
        if explicit and not reply:
            continue
        if reply.lower() in PERMISSION_KEYS:
            return Answer(PERMISSION_KEYS[reply.lower()])
        return Answer(Choice.DENY, reply)


def read_choice(choices: list[str], prompt: str = "answer (number or text) > ") -> str:
    while True:
        reply = plain_prompt(prompt).strip()
        if reply.isdigit() and 1 <= int(reply) <= len(choices):
            return choices[int(reply) - 1]
        if reply:
            return reply


PLAN_MARKS = {"done": ("[x] ", "ansigreen"), "in_progress": ("[>] ", "bold"), "pending": ("[ ] ", "")}


def plan_fragments(plan: list[PlanItem]) -> list[tuple[str, str]]:
    fragments = []
    for item in plan:
        mark, style = PLAN_MARKS[item.status]
        fragments.append((style, f"  {mark}{item.step}\n"))
    return fragments


def prompt_message(interaction: Interaction) -> FormattedText:
    fragments: list[tuple[str, str]] = []
    if interaction.plan_open():
        fragments += plan_fragments(interaction.plan)
    if interaction.suggestions:
        for number, text in enumerate(interaction.suggestions, 1):
            fragments += [("bold ansicyan", f"  {number} "), ("", f"{text}\n")]
    fragments.append(("", "> "))
    return FormattedText(fragments)


class InputCompleter(Completer):
    def __init__(self, cwd: Path, command_names: list[str]) -> None:
        self.cwd = cwd
        self.command_names = command_names

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            for name in self.command_names:
                if name.startswith(text):
                    yield Completion(name, start_position=-len(text))
            return
        word = text.split()[-1] if text and not text[-1].isspace() else ""
        if word.startswith("@"):
            yield from self._path_completions(word[1:])

    def _path_completions(self, prefix: str) -> Iterable[Completion]:
        directory, _, partial = prefix.rpartition("/")
        base = self.cwd / directory
        if not base.is_dir():
            return
        for entry in sorted(base.iterdir()):
            if not entry.name.startswith(partial) or (entry.name.startswith(".") and not partial):
                continue
            suffix = "/" if entry.is_dir() else ""
            path = f"{directory}/{entry.name}" if directory else entry.name
            yield Completion(path + suffix, start_position=-len(prefix))


class InputPrompt:
    def __init__(
        self, cwd: Path, command_names: list[str], toolbar: Callable[[], str], interaction: Interaction
    ) -> None:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.interaction = interaction
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        self._session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            completer=InputCompleter(cwd, command_names),
            complete_while_typing=True,
            key_bindings=bindings,
            bottom_toolbar=toolbar,
        )

    def read(self) -> str:
        suggestions = self.interaction.suggestions
        try:
            text = self._session.prompt(lambda: prompt_message(self.interaction))
        finally:
            self.interaction.suggestions = []
        return chosen_suggestion(text, suggestions) or text


def chosen_suggestion(text: str, suggestions: list[str]) -> str:
    reply = text.strip()
    if reply.isdigit() and 1 <= int(reply) <= len(suggestions):
        return suggestions[int(reply) - 1]
    return ""
