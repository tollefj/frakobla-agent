from __future__ import annotations

import difflib
import fnmatch
import glob
import os
import re
from collections.abc import Iterator
from pathlib import Path

from frakobla_llm.tools.registry import Risk, Tool, ToolError, object_schema

SENSITIVE_NAMES = [".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*"]
SENSITIVE_EXCEPTIONS = [".env.example", ".env.sample", ".env.template", "*.pub"]
SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
MAX_RESULTS = 200
MAX_GREP_FILE_BYTES = 2_000_000
MAX_LINE_CHARS = 300
DEFAULT_READ_LINES = 500
MAX_PREVIEW_LINES = 80

STRING = {"type": "string"}
INTEGER = {"type": "integer"}
BOOLEAN = {"type": "boolean"}


def is_sensitive(subject: str) -> bool:
    name = Path(subject).name
    if any(fnmatch.fnmatchcase(name, pattern) for pattern in SENSITIVE_EXCEPTIONS):
        return False
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in SENSITIVE_NAMES)


def resolve_path(cwd: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else cwd / path).resolve()


def fs_tools(cwd: Path) -> list[Tool]:
    files = FileTools(cwd)
    return [
        Tool(
            name="read_file",
            description="Read a text file with line numbers. Use offset and limit for large files.",
            parameters=object_schema(["path"], path=STRING, offset=INTEGER, limit=INTEGER),
            risk=Risk.READ,
            run=files.read_file,
            subject_arg="path",
            subject_is_path=True,
        ),
        Tool(
            name="write_file",
            description="Create or overwrite a file with the given content.",
            parameters=object_schema(["path", "content"], path=STRING, content=STRING),
            risk=Risk.WRITE,
            run=files.write_file,
            subject_arg="path",
            subject_is_path=True,
            preview=files.preview_write,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace old_string with new_string in a file. old_string must match exactly "
                "and be unique unless replace_all is true."
            ),
            parameters=object_schema(
                ["path", "old_string", "new_string"],
                path=STRING,
                old_string=STRING,
                new_string=STRING,
                replace_all=BOOLEAN,
            ),
            risk=Risk.WRITE,
            run=files.edit_file,
            subject_arg="path",
            subject_is_path=True,
            preview=files.preview_edit,
        ),
        Tool(
            name="list_dir",
            description="List the entries of a directory.",
            parameters=object_schema([], path=STRING),
            risk=Risk.READ,
            run=files.list_dir,
            subject_arg="path",
            subject_is_path=True,
        ),
        Tool(
            name="glob",
            description="Find files by glob pattern such as **/*.py, newest first.",
            parameters=object_schema(["pattern"], pattern=STRING, path=STRING),
            risk=Risk.READ,
            run=files.glob,
            subject_arg="path",
            subject_is_path=True,
        ),
        Tool(
            name="grep",
            description="Search file contents with a regular expression. Returns path:line: text.",
            parameters=object_schema(
                ["pattern"],
                pattern=STRING,
                path=STRING,
                include={"type": "string", "description": "glob filter for file names, e.g. *.py"},
                ignore_case=BOOLEAN,
            ),
            risk=Risk.READ,
            run=files.grep,
            subject_arg="path",
            subject_is_path=True,
        ),
    ]


class FileTools:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def read_file(self, args: dict) -> str:
        path = self._resolve(args["path"])
        lines = _read_text(path).splitlines()
        offset = max(int(args.get("offset") or 1), 1)
        limit = max(int(args.get("limit") or DEFAULT_READ_LINES), 1)
        selected = lines[offset - 1 : offset - 1 + limit]
        if not selected:
            return f"[no lines at offset {offset}; file has {len(lines)} lines]"
        body = "\n".join(f"{offset + index}\t{line}" for index, line in enumerate(selected))
        last = offset + len(selected) - 1
        if last < len(lines):
            body += f"\n[lines {offset}-{last} of {len(lines)}; pass offset={last + 1} to continue]"
        return body

    def write_file(self, args: dict) -> str:
        path = self._resolve(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8", newline="")
        return f"Wrote {len(args['content'].splitlines())} lines to {self._display(path)}"

    def edit_file(self, args: dict) -> str:
        path = self._resolve(args["path"])
        original = _read_text(path)
        path.write_text(_apply_edit(original, args), encoding="utf-8", newline="")
        return f"Edited {self._display(path)}"

    def list_dir(self, args: dict) -> str:
        path = self._resolve(args.get("path") or ".")
        if not path.is_dir():
            raise ToolError(f"{path} is not a directory")
        entries = sorted(path.iterdir(), key=lambda entry: (entry.is_file(), entry.name))
        names = [entry.name + ("/" if entry.is_dir() else "") for entry in entries]
        return _capped(names) or "[empty directory]"

    def glob(self, args: dict) -> str:
        base = self._resolve(args.get("path") or ".")
        pattern = args["pattern"]
        matcher = re.compile(glob.translate(pattern, recursive=True, include_hidden=True))
        matches = [
            file
            for file in _walk(base)
            if matcher.match(file.relative_to(base).as_posix() if "/" in pattern else file.name)
        ]
        matches.sort(key=_modified, reverse=True)
        return _capped([self._display(file) for file in matches]) or "No files matched."

    def grep(self, args: dict) -> str:
        base = self._resolve(args.get("path") or ".")
        try:
            regex = re.compile(args["pattern"], re.IGNORECASE if args.get("ignore_case") else 0)
        except re.error as error:
            raise ToolError(f"invalid regular expression: {error}") from error
        include = re.compile(glob.translate(args["include"])) if args.get("include") else None
        hits = []
        for file in _walk(base):
            if include and not include.match(file.name):
                continue
            target = file.resolve()
            if is_sensitive(file.name) or is_sensitive(target.name) or not target.is_relative_to(base):
                continue
            hits.extend(self._grep_file(file, regex))
            if len(hits) > MAX_RESULTS:
                break
        return _capped(hits) or "No matches."

    def preview_edit(self, args: dict) -> str:
        path = self._resolve(args["path"])
        try:
            original = _read_text(path)
            return _diff(original, _apply_edit(original, args), self._display(path))
        except ToolError as error:
            return f"(this call will fail: {error})"

    def preview_write(self, args: dict) -> str:
        path = self._resolve(args["path"])
        original = ""
        if path.is_file():
            try:
                original = _read_text(path)
            except ToolError:
                return f"(overwrites non-text file {path})"
        return _diff(original, args["content"], self._display(path))

    def _grep_file(self, file: Path, regex: re.Pattern) -> list[str]:
        try:
            if file.stat().st_size > MAX_GREP_FILE_BYTES:
                return []
            data = file.read_bytes()
        except OSError:
            return []
        if b"\0" in data[:1024]:
            return []
        name = self._display(file)
        return [
            f"{name}:{number}: {line[:MAX_LINE_CHARS]}"
            for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1)
            if regex.search(line)
        ]

    def _resolve(self, raw: str) -> Path:
        return resolve_path(self.cwd, str(raw))

    def _display(self, path: Path) -> str:
        if path.is_relative_to(self.cwd):
            return path.relative_to(self.cwd).as_posix() or "."
        return str(path)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise ToolError(f"{path} does not exist")
    if not path.is_file():
        raise ToolError(f"{path} is not a file")
    try:
        return path.read_text(encoding="utf-8", newline="")
    except UnicodeDecodeError as error:
        raise ToolError(f"{path} is not a UTF-8 text file") from error


def _apply_edit(text: str, args: dict) -> str:
    old, new = args["old_string"], args["new_string"]
    if not old:
        raise ToolError("old_string is empty; use write_file to create a file")
    if old == new:
        raise ToolError("old_string and new_string are identical")
    count = text.count(old)
    if count == 0:
        raise ToolError("old_string not found; it must match the file exactly, including whitespace")
    if count > 1 and not args.get("replace_all"):
        raise ToolError(
            f"old_string matches {count} times; add surrounding lines to make it unique or set replace_all"
        )
    return text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)


def _diff(before: str, after: str, name: str) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), f"a/{name}", f"b/{name}", lineterm="", n=2
        )
    )
    if len(lines) > MAX_PREVIEW_LINES:
        hidden = len(lines) - MAX_PREVIEW_LINES
        lines = [*lines[:MAX_PREVIEW_LINES], f"... {hidden} more diff lines"]
    return "\n".join(lines) or "(no changes)"


def _walk(base: Path) -> Iterator[Path]:
    if base.is_file():
        yield base
        return
    if not base.is_dir():
        raise ToolError(f"{base} does not exist")
    for directory, subdirs, names in os.walk(base):
        subdirs[:] = sorted(name for name in subdirs if name not in SKIP_DIRS)
        for name in sorted(names):
            yield Path(directory) / name


def _modified(file: Path) -> float:
    try:
        return file.stat().st_mtime
    except OSError:
        return 0.0


def _capped(items: list[str]) -> str:
    if len(items) <= MAX_RESULTS:
        return "\n".join(items)
    return "\n".join([*items[:MAX_RESULTS], f"[showing the first {MAX_RESULTS} results; narrow the search]"])
