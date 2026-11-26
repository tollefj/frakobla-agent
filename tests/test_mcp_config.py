from pathlib import Path

import pytest

from frakobla_llm import mcp_config
from frakobla_llm.config import load_config
from frakobla_llm.mcp_client import PROXY_HEADER_PREFIX, parse_servers, proxied_url, proxy_headers


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def servers(project: Path) -> dict:
    return load_config(project, project / "<no global>")["mcp"]["servers"]


def add(*args: str) -> int:
    return mcp_config.main(["add", *args, "--scope", "project"])


def test_add_writes_project_servers_and_refuses_duplicates(project):
    assert add("web", "--url", "https://<host>/mcp", "--proxy", "--header", "Authorization=Bearer ${TOKEN}") == 0
    assert add("local", "--command", "<cmd>", "--arg", "<a>", "--arg", "<b>", "--env", "KEY=<v>") == 0

    written = servers(project)
    assert written["web"] == {"url": "https://<host>/mcp", "proxy": True, "headers": {"Authorization": "Bearer ${TOKEN}"}}
    assert written["local"] == {"command": "<cmd>", "args": ["<a>", "<b>"], "env": {"KEY": "<v>"}}
    configs, warnings = parse_servers(written, {"TOKEN": "<token>"})
    assert [config.transport for config in configs] == ["http via llama-server", "stdio"] and not warnings

    assert add("local", "--command", "<other>") == 1
    assert add("local", "--command", "<other>", "--force") == 0
    assert servers(project)["local"] == {"command": "<other>"}


def test_add_validates_input(project):
    assert add("bad name", "--command", "<cmd>") == 1
    assert add("ok", "--command", "<cmd>", "--proxy") == 1
    assert add("ok", "--url", "https://<host>", "--header", "no-separator") == 1
    assert not (project / ".frakobla-llm" / "config.toml").exists()


def test_remove(project):
    add("local", "--command", "<cmd>")
    assert mcp_config.main(["remove", "local"]) == 0
    assert servers(project) == {}
    assert mcp_config.main(["remove", "local"]) == 1


def test_proxy_url_and_headers():
    assert (
        proxied_url("http://<llama>:8080/", "https://<host>/mcp?a=1")
        == "http://<llama>:8080/cors-proxy?url=https%3A%2F%2F%3Chost%3E%2Fmcp%3Fa%3D1"
    )
    headers = proxy_headers(
        {
            "Host": "<llama>",
            "Content-Type": "application/json",
            "Mcp-Session-Id": "<id>",
            "Authorization": "Bearer <mcp>",
        },
        proxy_api_key="<llama key>",
    )
    assert headers["host"] == "<llama>" and PROXY_HEADER_PREFIX + "host" not in headers
    assert headers["content-type"] == headers[PROXY_HEADER_PREFIX + "content-type"] == "application/json"
    assert headers[PROXY_HEADER_PREFIX + "mcp-session-id"] == "<id>" and "mcp-session-id" not in headers
    assert headers[PROXY_HEADER_PREFIX + "authorization"] == "Bearer <mcp>"
    assert headers["authorization"] == "Bearer <llama key>"
