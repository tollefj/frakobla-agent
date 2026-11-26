import os
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

server = MCPServer("echo")


@server.tool()
def echo(text: str) -> str:
    """Return the text unchanged."""
    return text


@server.tool()
def fail() -> str:
    """Always fails."""
    raise ValueError("boom")


@server.tool()
def crash() -> str:
    """Kills the server process."""
    os._exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        host, port = sys.argv[2], int(sys.argv[3])
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        server.run(transport="streamable-http", host=host, port=port, transport_security=security)
    else:
        server.run()
