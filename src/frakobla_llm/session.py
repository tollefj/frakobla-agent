from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from frakobla_llm.config import CONFIG_DIR

SESSIONS_ROOT = CONFIG_DIR / "sessions"
FORMAT_VERSION = 1
TITLE_CHARS = 60


@dataclass(frozen=True)
class Snapshot:
    path: str
    backup: str | None

    def restore(self) -> None:
        target = Path(self.path)
        if self.backup is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.backup, target)


@dataclass(frozen=True)
class SessionInfo:
    path: Path
    title: str
    turns: int
    modified: datetime


class Session:
    def __init__(self, path: Path | None, cwd: Path | None = None) -> None:
        self.path = path
        self.cwd = cwd
        self.title = ""
        self._snapshot_count = 0
        self._temp_snapshots: Path | None = None

    @classmethod
    def disabled(cls) -> Session:
        return cls(None)

    @classmethod
    def create(cls, cwd: Path, root: Path = SESSIONS_ROOT) -> Session:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        return cls(_directory(cwd, root) / f"{stamp}_{uuid.uuid4().hex[:8]}.jsonl", cwd)

    @classmethod
    def open(cls, path: Path) -> Session:
        session = cls(path)
        for entry in session.entries():
            if entry["type"] == "rename":
                session.title = entry["title"]
            elif entry["type"] == "snapshot":
                session._snapshot_count += 1
        return session

    @classmethod
    def latest(cls, cwd: Path, root: Path = SESSIONS_ROOT) -> Session | None:
        infos = cls.list(cwd, root)
        return cls.open(infos[0].path) if infos else None

    @classmethod
    def list(cls, cwd: Path, root: Path = SESSIONS_ROOT) -> list[SessionInfo]:
        infos = [_info(path) for path in _directory(cwd, root).glob("*.jsonl") if _started_in(path, cwd)]
        return sorted((info for info in infos if info.turns), key=lambda info: info.modified, reverse=True)

    @property
    def name(self) -> str:
        if self.title:
            return self.title
        return self.path.stem if self.path else "no session"

    def rename(self, title: str) -> None:
        self.title = title
        self.append("rename", title=title)

    def append(self, entry_type: str, **data: object) -> None:
        if self.path is None:
            return
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write({"type": "session", "version": FORMAT_VERSION, "cwd": str(self.cwd or "")})
        self._write({"type": entry_type, "time": datetime.now().isoformat(timespec="seconds"), **data})

    def entries(self) -> Iterator[dict]:
        if self.path is None or not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and "type" in entry:
                yield entry

    def stored_snapshot(self, path: str, backup: str | None) -> Snapshot:
        if backup and not Path(backup).exists():
            backup = str(self._snapshot_dir() / Path(backup).name)
        return Snapshot(path, backup)

    def snapshot(self, path: Path) -> Snapshot:
        backup = None
        if path.is_file():
            self._snapshot_count += 1
            target = self._snapshot_dir() / f"{self._snapshot_count:04d}_{path.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            backup = str(target)
        snapshot = Snapshot(str(path), backup)
        self.append("snapshot", path=snapshot.path, backup=snapshot.backup)
        return snapshot

    def _snapshot_dir(self) -> Path:
        if self.path is not None:
            return self.path.with_suffix("") / "snapshots"
        if self._temp_snapshots is None:
            self._temp_snapshots = Path(tempfile.mkdtemp(prefix="frak-snapshots-"))
        return self._temp_snapshots

    def _write(self, entry: dict) -> None:
        assert self.path is not None
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _info(path: Path) -> SessionInfo:
    title, first_prompt, turns = "", "", 0
    for entry in Session(path).entries():
        if entry["type"] == "rename":
            title = entry["title"]
        elif entry["type"] == "message" and entry["message"].get("role") == "user":
            turns += 1
            first_prompt = first_prompt or _first_line(entry["message"].get("content"))
        elif entry["type"] == "undo":
            turns -= 1
    title = title or first_prompt[:TITLE_CHARS] or path.stem
    return SessionInfo(path, title, turns, datetime.fromtimestamp(path.stat().st_mtime))


def _started_in(path: Path, cwd: Path) -> bool:
    header = next(Session(path).entries(), {})
    return header.get("cwd") in (None, "", str(cwd))


def _first_line(content: object) -> str:
    lines = str(content or "").strip().splitlines()
    return lines[0] if lines else ""


def _directory(cwd: Path, root: Path) -> Path:
    return root / str(cwd).strip("/").replace("/", "-")
