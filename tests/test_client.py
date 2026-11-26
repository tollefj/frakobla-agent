import pytest

from frakobla_llm.client import CompletionBuilder, LlamaClient, LlamaError, thinking_params


def chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def tool_part(index: int, arguments: str, name: str = "", call_id: str = "") -> dict:
    part: dict = {"index": index, "function": {"arguments": arguments}}
    if name:
        part["function"]["name"] = name
    if call_id:
        part["id"] = call_id
    return {"tool_calls": [part]}


def test_tool_call_arguments_are_concatenated_per_index():
    builder = CompletionBuilder()
    builder.add(chunk(tool_part(0, "", name="<tool_a>", call_id="<id_a>")))
    builder.add(chunk(tool_part(0, '{"key":')))
    builder.add(chunk(tool_part(1, '{"other"', name="<tool_b>", call_id="<id_b>")))
    builder.add(chunk(tool_part(0, '"value"}')))
    builder.add(chunk(tool_part(1, ":1}")))
    builder.add(chunk({}, finish_reason="tool_calls"))

    completion = builder.build()

    assert [(call.id, call.name, call.arguments) for call in completion.tool_calls] == [
        ("<id_a>", "<tool_a>", '{"key":"value"}'),
        ("<id_b>", "<tool_b>", '{"other":1}'),
    ]
    assert completion.finish_reason == "tool_calls"


def test_missing_tool_call_id_gets_a_fallback():
    builder = CompletionBuilder()
    builder.add(chunk(tool_part(0, "{}", name="<tool>")))
    assert builder.build().tool_calls[0].id == "call_0"


def test_content_and_reasoning_are_separated():
    builder = CompletionBuilder()
    deltas = builder.add(chunk({"reasoning_content": "<thought>"}))
    deltas += builder.add(chunk({"content": "<answer>"}))

    completion = builder.build()

    assert [(delta.kind, delta.text) for delta in deltas] == [("reasoning", "<thought>"), ("content", "<answer>")]
    assert (completion.reasoning, completion.content) == ("<thought>", "<answer>")


def test_usage_and_timings_come_from_the_final_chunk():
    builder = CompletionBuilder()
    builder.add({"choices": [], "usage": {"prompt_tokens": 3}, "timings": {"predicted_per_second": 1.5}})
    completion = builder.build()
    assert completion.usage == {"prompt_tokens": 3}
    assert completion.timings == {"predicted_per_second": 1.5}


def test_error_chunk_raises():
    with pytest.raises(LlamaError, match="<message>"):
        CompletionBuilder().add({"error": {"message": "<message>"}})


def test_thinking_params():
    assert thinking_params("default") == {}
    assert thinking_params("off")["chat_template_kwargs"]["enable_thinking"] is False
    high = thinking_params("high")
    assert high["reasoning_effort"] == "high"
    assert high["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "high"}


def test_unreachable_server_raises():
    with pytest.raises(LlamaError, match="failed"):
        LlamaClient("http://127.0.0.1:9", timeout=2).check_reachable()
