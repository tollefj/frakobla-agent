from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import IO

from frakobla_llm.tools.registry import Risk, Tool, object_schema

MAX_TIMEOUT_SECONDS = 600
MAX_OUTPUT_BYTES = 1_000_000
READ_CHUNK_BYTES = 65536
DRAIN_SECONDS = 1


class BoundedOutput:
    def __init__(self, stream: IO[bytes], limit: int) -> None:
        self._half = limit // 2
        self._head = bytearray()
        self._tail = bytearray()
        self._dropped = 0
        self._thread = threading.Thread(target=self._read, args=(stream,), daemon=True)
        self._thread.start()

    def wait(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def text(self) -> str:
        parts = [bytes(self._head)]
        if self._dropped:
            parts.append(f"\n[... {self._dropped} bytes of output dropped ...]\n".encode())
        parts.append(bytes(self._tail))
        return b"".join(parts).decode("utf-8", errors="replace").replace("\r\n", "\n")

    def _read(self, stream: IO[bytes]) -> None:
        while chunk := os.read(stream.fileno(), READ_CHUNK_BYTES):
            room = max(self._half - len(self._head), 0)
            self._head += chunk[:room]
            self._tail += chunk[room:]
            overflow = len(self._tail) - self._half
            if overflow > 0:
                self._dropped += overflow
                del self._tail[:overflow]


def run_command(command: str, cwd: Path, timeout: int, max_output_bytes: int = MAX_OUTPUT_BYTES) -> str:
    process = subprocess.Popen(
        command,
        shell=True,
        executable=shutil.which("bash"),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    output = BoundedOutput(process.stdout, max_output_bytes)
    deadline = time.monotonic() + timeout
    try:
        if not output.wait(timeout):
            raise subprocess.TimeoutExpired(command, timeout)
        process.wait(max(deadline - time.monotonic(), 0))
    except subprocess.TimeoutExpired:
        _kill_group(process)
        output.wait(DRAIN_SECONDS)
        process.wait()
        return f"{output.text().rstrip()}\n[timed out after {timeout}s]".lstrip()
    except KeyboardInterrupt:
        _kill_group(process)
        raise
    return f"{output.text().rstrip()}\n[exit code {process.returncode}]".lstrip()


def bash_tool(cwd: Path, default_timeout: int) -> Tool:
    def run(args: dict) -> str:
        timeout = min(int(args.get("timeout") or default_timeout), MAX_TIMEOUT_SECONDS)
        return run_command(args["command"], cwd, timeout)

    return Tool(
        name="bash",
        description="Run a shell command in the working directory. Returns combined stdout and stderr.",
        parameters=object_schema(
            ["command"],
            command={"type": "string"},
            timeout={"type": "integer", "description": "seconds"},
        ),
        risk=Risk.EXEC,
        run=run,
        subject_arg="command",
    )


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
