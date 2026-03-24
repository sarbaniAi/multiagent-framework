"""External MCP subagent — connects to an external MCP server via HTTP."""

from databricks_openai.agents import McpServer


def build_external_mcp(config: dict) -> McpServer:
    """
    Build an external MCP server connection.

    Required config keys:
        url:          MCP server URL (e.g. https://my-service.com/mcp)
        description:  When to use this tool

    Optional config keys:
        headers:      Dict of HTTP headers (e.g. {"Authorization": "Bearer ${TOKEN}"})
    """
    url = config["url"]
    description = config["description"]
    headers = config.get("headers", {})

    return McpServer(
        url=url,
        name=description,
        headers=headers,
    )
