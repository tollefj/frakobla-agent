import io

from rich.console import Console

from frakobla_llm.client import Completion
from frakobla_llm.stats import Stats, Usage, energy_kwh
from frakobla_llm.ui.render import Renderer, reasoning_mode
from frakobla_llm.ui.status import Snapshot, top_bar


def test_stats_prefers_server_timings_over_counted_deltas():
    stats = Stats()
    stats.begin()
    for kind in ["reasoning", "reasoning", "content"]:
        stats.token(kind)
    completion = Completion(
        timings={"prompt_n": 100, "predicted_n": 5, "prompt_ms": 1000, "predicted_ms": 2000},
    )
    stats.end(completion)
    assert stats.total == Usage(prompt_tokens=100, generated_tokens=5, reasoning_tokens=2, inference_seconds=3.0)


def test_stats_accumulate_and_include_live_request():
    stats = Stats()
    stats.begin()
    stats.token("content")
    stats.end(Completion(timings={"prompt_n": 10, "predicted_n": 1, "prompt_ms": 0, "predicted_ms": 0}))
    stats.begin()
    stats.token("content")
    stats.token("content")
    current = stats.current()
    assert current.prompt_tokens == 10
    assert current.generated_tokens == 3


def test_energy_for_one_hour_at_400_watts():
    assert energy_kwh(Usage(inference_seconds=3600), 400) == 0.4


def test_top_bar_shows_model_thinking_context_and_energy():
    snapshot = Snapshot(
        model="qwen",
        thinking="high",
        used=8000,
        context_size=32000,
        compact_at=0.75,
        usage=Usage(prompt_tokens=1500, generated_tokens=200, reasoning_tokens=50, inference_seconds=36),
        completion=Completion(timings={"predicted_per_second": 42.0}),
        gpu_watts=400,
    )
    first, second = (line.plain for line in top_bar(snapshot, 80))
    for part in ["qwen", "think high", "25% 8.0k/32.0k"]:
        assert part in first
    for part in ["in 1.5k  out 200  think 50", "42 tok/s", "gpu 36s", "0.0040 kWh"]:
        assert part in second
    assert len(first) == len(second) == 80


def test_reasoning_summary_counts_tokens():
    output = io.StringIO()
    renderer = Renderer(Console(file=output, width=80), reasoning="summary")
    for piece in ["one", " two", " three"]:
        renderer.reasoning(piece)
    renderer.stream_end()
    assert "(3 tokens)" in output.getvalue()


def test_reasoning_off_prints_nothing():
    output = io.StringIO()
    renderer = Renderer(Console(file=output, width=80), reasoning="off")
    renderer.reasoning("secret")
    renderer.stream_end()
    assert output.getvalue() == ""


def test_reasoning_mode_accepts_legacy_bools():
    assert reasoning_mode(True) == "full"
    assert reasoning_mode(False) == "summary"
    assert reasoning_mode("off") == "off"
