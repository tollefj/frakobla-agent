from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from frakobla_llm.tools.fs import is_sensitive, resolve_path
from frakobla_llm.tools.registry import Risk, Tool

ASK_AFTER_REPEATS = 3
DENY_AFTER_REPEATS = 4

DANGEROUS_COMMANDS = [
    (r"\bsudo\b", "runs as root"),
    (r"\brm\s+-\w*[rf]", "recursive or forced delete"),
    (r"\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b", "pipes a download into a shell"),
    (r"\bdd\s+if=", "raw disk write"),
    (r"\bmkfs", "formats a filesystem"),
    (r">\s*/dev/(sd|disk|nvme)", "writes to a block device"),
    (r"\bgit\s+push\b.*(--force|\s-f\b)", "force push"),
    (r"\bchmod\s+-R\s+777\b", "world-writable recursive chmod"),
]
SHELL_CHAINING = re.compile(r"[;&|`\n<>]|\$\(")


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class Mode(StrEnum):
    ASK = "ask"
    READONLY = "readonly"
    YOLO = "yolo"


class Choice(StrEnum):
    ONCE = "once"
    SESSION = "session"
    ALWAYS = "always"
    DENY = "deny"


@dataclass(frozen=True)
class Rule:
    pattern: str
    kind: str = "glob"

    def __post_init__(self) -> None:
        if self.kind not in ("glob", "regex", "exact"):
            raise ValueError(f"unknown rule kind {self.kind!r}, expected glob, regex or exact")
        if self.kind == "regex":
            try:
                re.compile(self.pattern)
            except re.error as error:
                raise ValueError(f"invalid regex rule {self.pattern!r}: {error}") from error

    def matches(self, subject: str) -> bool:
        if self.kind == "exact":
            return subject == self.pattern
        if self.kind == "regex":
            return re.search(self.pattern, subject) is not None
        return fnmatch.fnmatchcase(subject, self.pattern)


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str = ""
    warning: str = ""


@dataclass(frozen=True)
class PermissionRequest:
    tool: Tool
    args: dict
    subject: str
    reason: str
    warning: str
    grant: Rule

    @property
    def needs_explicit_answer(self) -> bool:
        outside_cwd = self.tool.subject_is_path and Path(self.subject).is_absolute()
        return bool(self.warning) or outside_cwd


@dataclass(frozen=True)
class Answer:
    choice: Choice
    reason: str = ""


Prompter = Callable[[PermissionRequest], Answer]
Persist = Callable[[str, Rule], None]


class Permissions:
    def __init__(
        self,
        settings: dict,
        cwd: Path,
        prompter: Prompter | None = None,
        persist: Persist | None = None,
    ) -> None:
        self.mode = Mode(settings["mode"])
        self.cwd = cwd
        self.prompter = prompter
        self.persist = persist
        self.rules = {
            tier: _parse_rules(settings["rules"].get(tier, {}))
            for tier in (Decision.ALLOW, Decision.ASK, Decision.DENY)
        }
        self.grants: dict[str, list[Rule]] = {}
        self._last_call = ""
        self._repeats = 0

    def authorize(self, tool: Tool, args: dict) -> Verdict:
        verdict = self.evaluate(tool, args, self._count_repeat(tool, args))
        if verdict.decision is not Decision.ASK:
            return verdict
        if self.prompter is None:
            return Verdict(Decision.DENY, f"needs approval ({verdict.reason}) and no prompt is available")
        subject = self.subject(tool, args)
        grant = _grant_rule(tool, subject)
        answer = self.prompter(
            PermissionRequest(tool, args, subject, verdict.reason, verdict.warning, grant)
        )
        if answer.choice is Choice.DENY:
            reason = f"denied by user: {answer.reason}" if answer.reason else "denied by user"
            return Verdict(Decision.DENY, reason)
        if answer.choice in (Choice.SESSION, Choice.ALWAYS):
            self.grants.setdefault(tool.name, []).append(grant)
        if answer.choice is Choice.ALWAYS and self.persist:
            self.persist(tool.name, grant)
        return Verdict(Decision.ALLOW, f"approved by user ({answer.choice})")

    def evaluate(self, tool: Tool, args: dict, repeats: int = 1) -> Verdict:
        subject = self.subject(tool, args)
        if tool.subject_is_path and is_sensitive(subject):
            return Verdict(Decision.DENY, "sensitive file")
        if self._matches(Decision.DENY, tool, subject):
            return Verdict(Decision.DENY, "matches a deny rule")
        if repeats >= DENY_AFTER_REPEATS:
            return Verdict(Decision.DENY, f"identical call repeated {repeats} times in a row")
        if repeats >= ASK_AFTER_REPEATS:
            return Verdict(Decision.ASK, "third identical call in a row")
        verdict = self._evaluate_rules(tool, subject)
        if verdict.decision is Decision.ASK and self.mode is Mode.YOLO:
            return Verdict(Decision.ALLOW, "yolo mode", verdict.warning)
        return verdict

    def subject(self, tool: Tool, args: dict) -> str:
        if not tool.subject_arg:
            return json.dumps(args, sort_keys=True)
        if not tool.subject_is_path:
            return str(args.get(tool.subject_arg, ""))
        path = resolve_path(self.cwd, str(args.get(tool.subject_arg) or "."))
        if path.is_relative_to(self.cwd):
            return path.relative_to(self.cwd).as_posix()
        return str(path)

    def _evaluate_rules(self, tool: Tool, subject: str) -> Verdict:
        if self.mode is Mode.READONLY and tool.risk is not Risk.READ:
            return Verdict(Decision.DENY, "readonly mode")
        if self._matches(Decision.ASK, tool, subject):
            return Verdict(Decision.ASK, "matches an ask rule")
        if tool.subject_is_path and Path(subject).is_absolute():
            return Verdict(Decision.ASK, "path is outside the working directory")
        warning = _danger_warning(subject) if tool.risk is Risk.EXEC else ""
        if warning:
            return Verdict(Decision.ASK, "dangerous command", warning)
        if self._is_allowed(tool, subject):
            return Verdict(Decision.ALLOW, "matches an allow rule")
        if tool.risk is Risk.READ:
            return Verdict(Decision.ALLOW, "read-only tool")
        return Verdict(Decision.ASK, f"{tool.risk} tool")

    def _is_allowed(self, tool: Tool, subject: str) -> bool:
        rules = [*self.rules[Decision.ALLOW].get(tool.name, []), *self.grants.get(tool.name, [])]
        chained = tool.risk is Risk.EXEC and SHELL_CHAINING.search(subject) is not None
        return any(
            rule.matches(subject) for rule in rules if rule.kind == "exact" or not chained
        )

    def _matches(self, tier: Decision, tool: Tool, subject: str) -> bool:
        return any(rule.matches(subject) for rule in self.rules[tier].get(tool.name, []))

    def _count_repeat(self, tool: Tool, args: dict) -> int:
        call = f"{tool.name}:{json.dumps(args, sort_keys=True)}"
        self._repeats = self._repeats + 1 if call == self._last_call else 1
        self._last_call = call
        return self._repeats


def _parse_rules(by_tool: dict) -> dict[str, list[Rule]]:
    return {tool: [_parse_rule(tool, entry) for entry in _as_list(tool, entries)] for tool, entries in by_tool.items()}


def _as_list(tool: str, entries: object) -> list:
    if not isinstance(entries, list):
        raise ValueError(f"permission rules for {tool} must be a list of tables")
    return entries


def _parse_rule(tool: str, entry: object) -> Rule:
    if not isinstance(entry, dict) or not isinstance(entry.get("pattern"), str):
        raise ValueError(f"permission rule for {tool} needs a pattern string, got {entry!r}")
    return Rule(entry["pattern"], entry.get("kind", "glob"))


def _grant_rule(tool: Tool, subject: str) -> Rule:
    if tool.subject_is_path or tool.risk is Risk.MCP:
        return Rule("*", "glob")
    return Rule(subject, "exact")


def _danger_warning(command: str) -> str:
    reasons = [reason for pattern, reason in DANGEROUS_COMMANDS if re.search(pattern, command)]
    return ", ".join(reasons)
