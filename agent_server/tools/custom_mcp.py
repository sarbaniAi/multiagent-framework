"""Custom MCP subagent — runs a local MCP server process via stdio."""

from agents.mcp import MCPServerStdio


def build_custom_mcp(config: dict) -> MCPServerStdio:
    """
    Build a custom MCP server that runs as a subprocess (stdio transport).

    Required config keys:
        command:      List of command + args (e.g. ["python", "my_server.py"])
        description:  When to use this tool

    Optional config keys:
        env:          Dict of environment variables for the subprocess
        cwd:          Working directory for the subprocess
    """
    command = config["command"]
    description = config["description"]
    env = config.get("env", {})
    cwd = config.get("cwd")

    return MCPServerStdio(
        params={
            "command": command[0],
            "args": command[1:] if len(command) > 1 else [],
            "env": env or None,
            "cwd": cwd,
        },
        name=description,
    )
