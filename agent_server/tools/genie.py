"""Genie MCP subagent — queries a Databricks Genie space for structured data."""

from databricks_openai.agents import McpServer
from agent_server.utils import build_mcp_url


def build_genie_mcp(config: dict) -> McpServer:
    """
    Build a Genie MCP server from config.

    Required config keys:
        space_id:    Genie space UUID
        description: Used as the MCP server name (tells the LLM when to use it)
    """
    space_id = config["space_id"]
    description = config["description"]

    return McpServer(
        url=build_mcp_url(f"/api/2.0/mcp/genie/{space_id}"),
        name=description,
    )
