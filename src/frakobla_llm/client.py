from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

THINKING_LEVELS = ("default", "off", "low", "medium", "high")


class LlamaError(Exception):
    pass


class ContextOverflowError(LlamaError):
    pass


@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class Completion:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Delta:
    kind: str
    text: str


@dataclass(frozen=True)
class ModelInfo:
    id: str
    status: str = "loaded"
    failed: bool = False

    @property
    def loaded(self) -> bool:
        return self.status == "loaded"


@dataclass(frozen=True)
class ServerInfo:
    context_size: int
    supports_tools: bool | None


class CompletionBuilder:
    def __init__(self) -> None:
        self._completion = Completion()
        self._calls: dict[int, ToolCall] = {}

    def add(self, chunk: dict) -> list[Delta]:
        if "error" in chunk:
            raise _error_from(chunk["error"])
        if chunk.get("usage"):
            self._completion.usage = chunk["usage"]
        if chunk.get("timings"):
            self._completion.timings = chunk["timings"]
        deltas = []
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if text := delta.get("reasoning_content"):
                self._completion.reasoning += text
                deltas.append(Delta("reasoning", text))
            if text := delta.get("content"):
                self._completion.content += text
                deltas.append(Delta("content", text))
            for part in delta.get("tool_calls") or []:
                self._add_tool_call_part(part)
            if choice.get("finish_reason"):
                self._completion.finish_reason = choice["finish_reason"]
        return deltas

    def _add_tool_call_part(self, part: dict) -> None:
        index = part.get("index", 0)
        call = self._calls.setdefault(index, ToolCall())
        function = part.get("function") or {}
        if part.get("id"):
            call.id = part["id"]
        if function.get("name"):
            call.name = function["name"]
        call.arguments += function.get("arguments") or ""

    def build(self) -> Completion:
        calls = [self._calls[index] for index in sorted(self._calls)]
        for position, call in enumerate(calls):
            call.id = call.id or f"call_{position}"
        self._completion.tool_calls = calls
        return self._completion


class LlamaClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 600) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5),
        )

    def models(self) -> list[ModelInfo]:
        entries = self._get("/v1/models").get("data") or []
        return [_model_info(entry) for entry in entries]

    def load_model(self, model: str) -> None:
        self._post("/models/load", {"model": model})

    def unload_model(self, model: str) -> None:
        self._post("/models/unload", {"model": model})

    def wait_for_status(self, model: str, status: str, timeout: float = 600, interval: float = 0.5) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = next((entry for entry in self.models() if entry.id == model), None)
            if info is None:
                raise LlamaError(f"model {model} is no longer listed by the server")
            if info.status == status:
                return
            if info.failed and status == "loaded":
                raise LlamaError(f"model {model} failed to load, check the llama-server log")
            time.sleep(interval)
        raise LlamaError(f"timed out after {timeout:.0f}s waiting for {model} to become {status}")

    def server_info(self, model: str) -> ServerInfo:
        props = self._get("/props", model=model)
        settings = props.get("default_generation_settings") or {}
        return ServerInfo(
            context_size=settings.get("n_ctx") or 0,
            supports_tools=_supports_tools(props),
        )

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        on_delta: Callable[[Delta], None],
        parallel_tool_calls: bool = False,
        thinking: str = "default",
    ) -> Completion:
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **thinking_params(thinking),
        }
        if tools:
            body["tools"] = tools
            body["parallel_tool_calls"] = parallel_tool_calls
        builder = CompletionBuilder()
        try:
            with self._http.stream("POST", "/v1/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    response.read()
                    raise _http_error(response)
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    for delta in builder.add(json.loads(data)):
                        on_delta(delta)
        except httpx.HTTPError as error:
            raise LlamaError(self._unreachable_text(error)) from error
        return builder.build()

    def count_tokens(self, model: str, text: str) -> int:
        try:
            response = self._http.post("/tokenize", json={"model": model, "content": text})
            response.raise_for_status()
            return len(response.json()["tokens"])
        except (httpx.HTTPError, ValueError, KeyError):
            return estimate_tokens(text)

    def _post(self, path: str, body: dict) -> dict:
        try:
            response = self._http.post(path, json=body)
        except httpx.HTTPError as error:
            raise LlamaError(self._unreachable_text(error)) from error
        if response.status_code >= 400:
            raise _http_error(response)
        return response.json() if response.content else {}

    def _get(self, path: str, **params: str) -> dict:
        try:
            response = self._http.get(path, params=params)
        except httpx.HTTPError as error:
            raise LlamaError(self._unreachable_text(error)) from error
        if response.status_code >= 400:
            raise _http_error(response)
        return response.json()

    def _unreachable_text(self, error: Exception) -> str:
        return f"llama-server request to {self.base_url} failed: {error or type(error).__name__}"


def resolve_model(client: LlamaClient, configured: str) -> str:
    if configured:
        return configured
    models = client.models()
    if not models:
        raise LlamaError("server reports no models")
    loaded = [model for model in models if model.loaded]
    return (loaded or models)[0].id


def swap_model(client: LlamaClient, target: str, on_step: Callable[[str], None]) -> None:
    for model in client.models():
        if model.id != target and model.status in ("loaded", "loading"):
            on_step(f"unloading {model.id}")
            client.unload_model(model.id)
            client.wait_for_status(model.id, "unloaded")
    current = next((model for model in client.models() if model.id == target), None)
    if current is None:
        raise LlamaError(f"unknown model {target}")
    if current.loaded:
        return
    on_step(f"loading {target}")
    if current.status != "loading":
        client.load_model(target)
    client.wait_for_status(target, "loaded")


def thinking_params(level: str) -> dict:
    if level == "default":
        return {}
    effort = "low" if level == "off" else level
    return {
        "reasoning_effort": effort,
        "chat_template_kwargs": {"enable_thinking": level != "off", "reasoning_effort": effort},
    }


def _model_info(entry: dict) -> ModelInfo:
    status = entry.get("status") or {}
    return ModelInfo(id=entry["id"], status=status.get("value", "loaded"), failed=bool(status.get("failed")))


def _supports_tools(props: dict) -> bool | None:
    caps = props.get("chat_template_caps")
    if caps and "supports_tools" in caps:
        return bool(caps["supports_tools"])
    template = props.get("chat_template")
    if template:
        return "tools" in template
    return None


def estimate_tokens(text: str) -> int:
    return len(text) // 3


def is_context_overflow(error: object) -> bool:
    if isinstance(error, dict) and error.get("type") == "exceed_context_size_error":
        return True
    text = str(error.get("message") if isinstance(error, dict) else error).lower()
    return "context size" in text or "context length" in text or "context window" in text


def _error_text(error: object) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)


def _error_from(error: object, prefix: str = "") -> LlamaError:
    kind = ContextOverflowError if is_context_overflow(error) else LlamaError
    return kind(prefix + _error_text(error))


def _http_error(response: httpx.Response) -> LlamaError:
    prefix = f"llama-server returned HTTP {response.status_code}: "
    try:
        return _error_from(response.json().get("error"), prefix)
    except (ValueError, AttributeError):
        return LlamaError(prefix + response.text[:500])
