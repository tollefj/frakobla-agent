from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

from frakobla_llm.client import Completion
from frakobla_llm.stats import Usage, energy_kwh

CONTEXT_BAR_CELLS = 12
BAR_STYLE = "on grey15"


@dataclass(frozen=True)
class Snapshot:
    model: str
    thinking: str
    used: int
    context_size: int
    compact_at: float
    usage: Usage
    completion: Completion
    gpu_watts: float


def format_tokens(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def status_text(model: str, used: int, completion: Completion, context_size: int, mode: str, session: str) -> str:
    context = f"ctx {format_tokens(used)}"
    if context_size:
        context += f"/{format_tokens(context_size)} ({used * 100 // context_size}%)"
    parts = [model, context]
    if speed := completion.timings.get("predicted_per_second"):
        parts.append(f"{speed:.0f} tok/s")
    return " | ".join([*parts, mode, session])


def context_bar(used: int, size: int, compact_at: float) -> Text:
    if not size:
        return Text(f"ctx {format_tokens(used)}")
    fraction = min(used / size, 1.0)
    filled = round(fraction * CONTEXT_BAR_CELLS)
    color = "green" if fraction < compact_at * 0.66 else "yellow" if fraction < compact_at else "red"
    return Text.assemble(
        "ctx ",
        ("━" * filled, color),
        ("─" * (CONTEXT_BAR_CELLS - filled), "grey35"),
        f" {fraction * 100:.0f}% {format_tokens(used)}/{format_tokens(size)}",
    )


def top_bar(snapshot: Snapshot, width: int) -> list[Text]:
    usage = snapshot.usage
    model_line = [
        Text(snapshot.model, style="bold cyan"),
        Text(f"think {snapshot.thinking}", style="magenta"),
        context_bar(snapshot.used, snapshot.context_size, snapshot.compact_at),
    ]
    usage_line = [
        Text(
            f"in {format_tokens(usage.prompt_tokens)}  "
            f"out {format_tokens(usage.generated_tokens)}  "
            f"think {format_tokens(usage.reasoning_tokens)}"
        ),
    ]
    if speed := snapshot.completion.timings.get("predicted_per_second"):
        usage_line.append(Text(f"{speed:.0f} tok/s"))
    usage_line.append(Text(f"gpu {usage.inference_seconds:.0f}s"))
    usage_line.append(Text(f"{energy_kwh(usage, snapshot.gpu_watts):.4f} kWh", style="yellow"))
    return [_bar_line(parts, width) for parts in (model_line, usage_line)]


def _bar_line(parts: list[Text], width: int) -> Text:
    line = Text(" ", style=BAR_STYLE)
    line.append_text(Text(" | ", style="grey50").join(parts))
    line.truncate(width, overflow="ellipsis", pad=True)
    return line
