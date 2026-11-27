from __future__ import annotations

import io
import shutil
import time
from collections.abc import Callable

from rich.console import Console
from rich.text import Text

REFRESH_SECONDS = 0.25
SAVE, RESTORE = "\x1b7", "\x1b8"
NO_WRAP, WRAP = "\x1b[?7l", "\x1b[?7h"


class StatusBar:
    def __init__(self, console: Console, render: Callable[[int], list[Text]], height: int = 2) -> None:
        self.console = console
        self.render = render
        self.height = height
        self._last = 0.0

    def install(self) -> None:
        self._write("\n" * self.height + f"\x1b[{self.height}A")
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < REFRESH_SECONDS:
            return
        self._last = now
        columns, rows = shutil.get_terminal_size()
        top = rows - self.height
        self.console.height = top
        lines = self.render(columns)[: self.height]
        drawn = "".join(
            f"\x1b[{row};1H\x1b[2K{self._ansi(line, columns)}\x1b[0m" for row, line in enumerate(lines, top + 1)
        )
        self._write(f"{SAVE}\x1b[1;{top}r{RESTORE}{SAVE}{NO_WRAP}{drawn}{WRAP}{RESTORE}")

    def remove(self) -> None:
        self.console.height = None
        rows = shutil.get_terminal_size().lines
        clear = "".join(f"\x1b[{row};1H\x1b[2K" for row in range(rows - self.height + 1, rows + 1))
        self._write(f"{SAVE}\x1b[r{clear}{RESTORE}")

    def _ansi(self, text: Text, width: int) -> str:
        buffer = io.StringIO()
        Console(file=buffer, force_terminal=True, color_system=self.console.color_system, width=width).print(
            text, end="", no_wrap=True, overflow="ellipsis", crop=True
        )
        return buffer.getvalue()

    def _write(self, data: str) -> None:
        with self.console._lock:
            self.console.file.write(data)
            self.console.file.flush()
