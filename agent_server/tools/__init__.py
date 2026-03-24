"""
Tool builders for each subagent type.

Each module exports a build_* function that takes a subagent config dict
and returns either:
  - A function_tool (for vector_search, uc_function)
  - A McpServer (for genie, external_mcp, custom_mcp)
"""
