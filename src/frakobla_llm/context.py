from __future__ import annotations

import platform
from datetime import date
from pathlib import Path

from frakobla_llm.config import CONFIG_DIR
from frakobla_llm.tools.interact import PlanItem

GLOBAL_AGENTS_FILE = CONFIG_DIR / "AGENTS.md"

BASE_PROMPT = """\
You are a coding agent working in a terminal on the user's machine.
Working directory: {cwd}
Platform: {platform}. Date: {today}.

Inspect files with the tools instead of guessing. Prefer glob, grep and read_file over bash for looking around. Read a file before editing it. edit_file needs old_string to match the file exactly and uniquely. Relative paths resolve against the working directory. Some tool calls need user approval. If a call is denied, do not repeat it, choose another approach or ask the user. Keep answers short."""


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    cut = len(text) - 2 * half
    return f"{text[:half]}\n[... {cut} characters truncated ...]\n{text[-half:]}"


def discover_agents_files(cwd: Path, global_file: Path = GLOBAL_AGENTS_FILE) -> list[Path]:
    ancestors = [cwd, *cwd.parents]
    git_root_depth = next(
        (depth for depth, directory in enumerate(ancestors) if (directory / ".git").exists()), 0
    )
    directories = ancestors[: git_root_depth + 1]
    candidates = [global_file, *(directory / "AGENTS.md" for directory in reversed(directories))]
    return [path for path in candidates if path.is_file()]


def build_system_prompt(cwd: Path, agents_files: list[Path]) -> str:
    prompt = BASE_PROMPT.format(cwd=cwd, platform=platform.system(), today=date.today().isoformat())
    if agents_files:
        prompt += "\n\nAGENTS.md files hold the project's commands and conventions. Consult and follow them."
    for path in agents_files:
        prompt += f"\n\nInstructions from {path}:\n{path.read_text(encoding='utf-8').strip()}"
    return prompt


SUMMARY_PROMPT = """\
You compress a coding agent's conversation so the agent can continue the work with only your summary. \
Keep the user's goals and requests, decisions made, files read or changed and how, commands run and their \
important results, errors and how they were solved, and what is still left to do. Drop pleasantries, full \
file contents and long outputs. Write plain text, at most 400 words."""

TOOL_RESULT_SUMMARY_CHARS = 1500


def message_chars(messages: list[dict]) -> int:
    return sum(len(str(message.get("content") or "")) + len(str(message.get("tool_calls") or "")) for message in messages)


def is_cut_point(messages: list[dict], index: int) -> bool:
    role = messages[index].get("role")
    if role == "user":
        return True
    return role == "assistant" and messages[index - 1].get("role") in ("tool", "assistant")


def find_cut(messages: list[dict], keep_chars: int) -> int:
    cuts = [index for index in range(1, len(messages)) if is_cut_point(messages, index)]
    fitting = [index for index in cuts if message_chars(messages[index:]) <= keep_chars]
    if fitting:
        return fitting[0]
    return cuts[-1] if cuts else 0


def anchor_user_index(messages: list[dict], cut: int) -> int | None:
    if cut >= len(messages) or messages[cut].get("role") == "user":
        return None
    return next((index for index in range(cut - 1, -1, -1) if messages[index].get("role") == "user"), None)


def kept_messages(messages: list[dict], cut: int, anchor: int | None) -> list[dict]:
    kept = messages[cut:]
    return [messages[anchor], *kept] if anchor is not None else kept


def transcript(messages: list[dict], max_chars: int) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "")
        content = str(message.get("content") or "")
        if role == "tool":
            parts.append(f"[tool result]\n{truncate_middle(content, TOOL_RESULT_SUMMARY_CHARS)}")
            continue
        if content:
            parts.append(f"[{role}]\n{content}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            parts.append(f"[tool call] {function.get('name')} {function.get('arguments')}")
    return truncate_middle("\n\n".join(parts), max_chars)


def summary_request(previous: str, older: list[dict], max_chars: int, instructions: str = "") -> list[dict]:
    body = f"Previous summary:\n{previous or '(none)'}\n\nConversation to fold into the summary:\n{transcript(older, max_chars)}"
    if instructions:
        body += f"\n\nExtra instructions from the user: {instructions}"
    body += "\n\nWrite the updated summary."
    return [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": body}]


def plan_text(plan: list[PlanItem]) -> str:
    if not any(item.status != "done" for item in plan):
        return ""
    return "\n".join(f"- [{item.status}] {item.step}" for item in plan)


def with_summary(system_prompt: str, summary: str, plan: str = "") -> str:
    if not summary:
        return system_prompt
    prompt = f"{system_prompt}\n\nSummary of the earlier conversation, which was compacted:\n{summary}"
    if plan:
        prompt += f"\n\nPlan at the time of compaction:\n{plan}"
    return prompt
