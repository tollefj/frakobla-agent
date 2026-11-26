from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum


class Risk(StrEnum):
    READ = "read"
    WRITE = "write"
    EXEC = "exec"
    NETWORK = "network"
    MCP = "mcp"


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    risk: Risk
    run: Callable[[dict], str]
    subject_arg: str = ""
    subject_is_path: bool = False
    preview: Callable[[dict], str] | None = None
    ends_turn: bool = False
    group: str = ""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def missing_arguments(self, args: dict) -> list[str]:
        return [name for name in self.parameters.get("required", []) if name not in args]


class Registry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self.disabled_groups: set[str] = set()
        self.hidden: set[str] = set()

    def hide(self, names: list[str]) -> None:
        self.hidden.update(names)

    def show(self, names: list[str]) -> None:
        self.hidden.difference_update(names)

    def add(self, tools: list[Tool]) -> None:
        self._tools.update((tool.name, tool) for tool in tools)

    def set_group_enabled(self, group: str, enabled: bool) -> None:
        if enabled:
            self.disabled_groups.discard(group)
        else:
            self.disabled_groups.add(group)

    def group(self, group: str) -> list[Tool]:
        return [tool for tool in self._tools.values() if tool.group == group]

    def get(self, name: str) -> Tool | None:
        tool = self._tools.get(name)
        return tool if tool and tool.group not in self.disabled_groups else None

    def names(self) -> list[str]:
        return [tool.name for tool in self.visible()]

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.visible()]

    def visible(self) -> list[Tool]:
        return [tool for tool in self if tool.name not in self.hidden]

    def __iter__(self) -> Iterator[Tool]:
        return (tool for tool in self._tools.values() if tool.group not in self.disabled_groups)


def object_schema(required: list[str], **properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": required}
