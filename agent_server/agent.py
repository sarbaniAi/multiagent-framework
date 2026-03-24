"""
Generic Multi-Agent Orchestrator Framework.

Reads agents.yaml to build an orchestrator agent with dynamically
configured subagents. No Python changes needed — all configuration
is in YAML.

Supported subagent types:
  - genie:          Databricks Genie space (structured data via MCP)
  - vector_search:  Databricks Vector Search index (RAG)
  - uc_function:    Unity Catalog SQL function (governed rules)
  - external_mcp:   External MCP server (HTTP)
  - custom_mcp:     Custom MCP server (subprocess / stdio)

Built from: databricks/app-templates (agent-openai-agents-sdk-multiagent)
"""

import litellm
import logging
import os
from contextlib import AsyncExitStack
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.config import load_config
from agent_server.tools.genie import build_genie_mcp
from agent_server.tools.vector_search import build_vector_search_tool
from agent_server.tools.uc_function import build_uc_function_tool
from agent_server.tools.external_mcp import build_external_mcp
from agent_server.tools.custom_mcp import build_custom_mcp
from agent_server.utils import get_session_id, process_agent_stream_events

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])
mlflow.openai.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Load config + build tools at import time
# ---------------------------------------------------------------------------

_config = load_config()
_orch_config = _config["orchestrator"]
_subagents = _config["subagents"]

# Separate subagents into MCP servers (need async context) and function tools
_mcp_builders = []   # List of (name, builder_func, config) — built per-request
_function_tools = [] # List of function_tool objects — built once

for sa in _subagents:
    sa_type = sa["type"]
    sa_name = sa["name"]

    if sa_type == "genie":
        _mcp_builders.append((sa_name, build_genie_mcp, sa))
        print(f"  [Config] Registered MCP subagent: {sa_name} (genie)")

    elif sa_type == "vector_search":
        tool = build_vector_search_tool(sa)
        _function_tools.append(tool)
        print(f"  [Config] Registered tool subagent: {sa_name} (vector_search)")

    elif sa_type == "uc_function":
        tool = build_uc_function_tool(sa)
        _function_tools.append(tool)
        print(f"  [Config] Registered tool subagent: {sa_name} (uc_function)")

    elif sa_type == "external_mcp":
        _mcp_builders.append((sa_name, build_external_mcp, sa))
        print(f"  [Config] Registered MCP subagent: {sa_name} (external_mcp)")

    elif sa_type == "custom_mcp":
        _mcp_builders.append((sa_name, build_custom_mcp, sa))
        print(f"  [Config] Registered MCP subagent: {sa_name} (custom_mcp)")

print(f"  [Config] Total: {len(_function_tools)} tools + {len(_mcp_builders)} MCP servers")

# ---------------------------------------------------------------------------
# Build orchestrator instructions
# ---------------------------------------------------------------------------


def _build_instructions() -> str:
    """
    Build orchestrator instructions from config.

    If the user provided custom instructions, append auto-generated
    tool descriptions. If not, generate everything automatically.
    """
    base = _orch_config.get("instructions", "")
    company = os.environ.get("COMPANY_NAME", _orch_config.get("name", "Assistant"))

    # Auto-generate tool routing section
    tool_section = "\n\nAVAILABLE TOOLS:\n"
    for i, sa in enumerate(_subagents, 1):
        tool_section += f"\n{i}. **{sa['name']}** ({sa['type']})\n"
        tool_section += f"   {sa['description']}\n"

    if base:
        return base + tool_section
    else:
        return (
            f"You are {company}, an AI assistant.\n"
            f"Route the user's request to the most appropriate tool based on the descriptions below.\n"
            f"If a query requires multiple tools, call them sequentially and synthesize results.\n"
            f"If you cannot answer confidently, say so.\n"
            + tool_section
        )


INSTRUCTIONS = _build_instructions()

# ---------------------------------------------------------------------------
# Orchestrator agent factory
# ---------------------------------------------------------------------------


def create_orchestrator(mcp_servers: list) -> Agent:
    return Agent(
        name=_orch_config.get("name", "Orchestrator"),
        instructions=INSTRUCTIONS,
        model=_orch_config["model"],
        mcp_servers=mcp_servers,
        tools=_function_tools,
    )


# ---------------------------------------------------------------------------
# MLflow Responses API handlers
# ---------------------------------------------------------------------------


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    async with AsyncExitStack() as stack:
        # Build MCP servers (each needs async context management)
        mcp_servers = []
        for name, builder, config in _mcp_builders:
            server = builder(config)
            entered = await stack.enter_async_context(server)
            mcp_servers.append(entered)

        agent = create_orchestrator(mcp_servers)
        messages = [i.model_dump() for i in request.input]
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream_handler(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    async with AsyncExitStack() as stack:
        mcp_servers = []
        for name, builder, config in _mcp_builders:
            server = builder(config)
            entered = await stack.enter_async_context(server)
            mcp_servers.append(entered)

        agent = create_orchestrator(mcp_servers)
        messages = [i.model_dump() for i in request.input]
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
