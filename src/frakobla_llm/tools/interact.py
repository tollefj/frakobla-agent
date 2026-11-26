from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from frakobla_llm.tools.registry import Risk, Tool, ToolError, object_schema

PLAN_STATUSES = ("pending", "in_progress", "done")
MAX_SUGGESTIONS = 3
STRING_LIST = {"type": "array", "items": {"type": "string"}}


@dataclass(frozen=True)
class PlanItem:
    step: str
    status: str


@dataclass
class Interaction:
    plan: list[PlanItem] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.plan = []
        self.suggestions = []

    def plan_open(self) -> bool:
        return any(item.status != "done" for item in self.plan)


class InteractUI(Protocol):
    def ask_user(self, question: str, choices: list[str]) -> str: ...
    def show_plan(self, plan: list[PlanItem]) -> None: ...


def interact_tools(state: Interaction, ui: InteractUI, interactive: bool) -> list[Tool]:
    def ask_user(args: dict) -> str:
        choices = [str(choice) for choice in args.get("choices") or []]
        answer = ui.ask_user(str(args["question"]), choices)
        return f"User answered: {answer}"

    def update_plan(args: dict) -> str:
        steps = args["steps"]
        if not isinstance(steps, list):
            raise ToolError("steps must be a list of {step, status} objects")
        plan = []
        for entry in steps:
            if not isinstance(entry, dict) or not entry.get("step"):
                raise ToolError("each step needs a non-empty step text")
            status = entry.get("status", "pending")
            if status not in PLAN_STATUSES:
                raise ToolError(f"status must be one of {', '.join(PLAN_STATUSES)}")
            plan.append(PlanItem(str(entry["step"]), status))
        state.plan = plan
        ui.show_plan(plan)
        done = sum(item.status == "done" for item in plan)
        return f"Plan updated, {done}/{len(plan)} done."

    def suggest_next(args: dict) -> str:
        suggestions = [str(text).strip() for text in args["suggestions"] if str(text).strip()]
        state.suggestions = suggestions[:MAX_SUGGESTIONS]
        return "Suggestions shown to the user."

    plan_tool = Tool(
        name="update_plan",
        description="Keep a checklist for multi-step tasks. Send the full list each time, marking steps done as you go.",
        parameters=object_schema(
            ["steps"],
            steps={
                "type": "array",
                "items": object_schema(
                    ["step", "status"],
                    step={"type": "string"},
                    status={"type": "string", "enum": list(PLAN_STATUSES)},
                ),
            },
        ),
        risk=Risk.READ,
        run=update_plan,
    )
    if not interactive:
        return [plan_tool]
    return [
        Tool(
            name="ask_user",
            description="Ask the user a question when you are blocked or the request is ambiguous. Offer choices when possible.",
            parameters=object_schema(["question"], question={"type": "string"}, choices=STRING_LIST),
            risk=Risk.READ,
            run=ask_user,
        ),
        plan_tool,
        Tool(
            name="suggest_next",
            description="At the end of your answer, optionally offer 2 or 3 short follow-up requests the user may want next. Ends your turn.",
            parameters=object_schema(["suggestions"], suggestions=STRING_LIST),
            risk=Risk.READ,
            run=suggest_next,
            ends_turn=True,
        ),
    ]
