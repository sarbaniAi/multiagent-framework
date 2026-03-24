"""UC Function subagent — calls a Unity Catalog SQL function."""

import os

from agents import function_tool
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Disposition, StatementState


_ws = None
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")


def _get_ws() -> WorkspaceClient:
    global _ws
    if _ws is None:
        _ws = WorkspaceClient()
    return _ws


async def _exec_sql(sql: str) -> str:
    """Execute SQL and return the first cell value."""
    resp = _get_ws().statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        wait_timeout="30s",
        disposition=Disposition.INLINE,
    )
    if resp.status.state != StatementState.SUCCEEDED:
        return f"Error: {resp.status.error.message if resp.status.error else 'Query failed'}"
    if resp.result and resp.result.data_array:
        # Return all columns of first row as key=value pairs
        cols = [c.name for c in resp.manifest.schema.columns]
        row = resp.result.data_array[0]
        if len(cols) == 1:
            return str(row[0])
        return ", ".join(f"{c}={v}" for c, v in zip(cols, row))
    return "No result returned."


def build_uc_function_tool(config: dict):
    """
    Build a function_tool that calls a UC SQL function.

    Required config keys:
        function:     Fully qualified function name (catalog.schema.fn_name)
        description:  When to use this tool

    Optional config keys:
        parameters:   List of {name, type} dicts describing function params.
                      If omitted, the tool accepts a single 'query' string
                      and the LLM must format the SQL call itself.

    Example config:
        function: "catalog.schema.fn_check_limit"
        parameters:
          - name: product_id
            type: string
          - name: age
            type: integer
    """
    func_name = config["function"]
    description = config["description"]
    params = config.get("parameters", [])
    tool_name = f"call_{config['name']}"

    if params:
        # Build a tool with typed parameters
        # Generate a function signature dynamically
        param_names = [p["name"] for p in params]
        param_types = [p.get("type", "string") for p in params]

        async def _call(**kwargs) -> str:
            args = []
            for p in params:
                val = kwargs.get(p["name"])
                if val is None:
                    args.append("NULL")
                elif p.get("type") in ("integer", "int", "float", "double", "boolean"):
                    args.append(str(val).lower() if p.get("type") == "boolean" else str(val))
                else:
                    args.append(f"'{str(val)}'")
            sql = f"SELECT {func_name}({', '.join(args)})"
            result = await _exec_sql(sql)
            return f"{func_name}({', '.join(str(kwargs.get(p['name'],'')) for p in params)}) = {result}"

        # Build docstring with parameter info
        param_doc = "\n".join(f"    {p['name']} ({p.get('type','string')})" for p in params)
        _call.__name__ = tool_name
        _call.__doc__ = f"{description}\n\nParameters:\n{param_doc}"

        # For the OpenAI Agents SDK, we need proper function signature.
        # function_tool with **kwargs won't have schema. Build explicit params.
        # Use a wrapper that maps positional string args.
        async def _call_with_args(arguments: str) -> str:
            """Parse arguments string and call the function."""
            # The LLM will pass arguments as described in the docstring
            parts = [a.strip() for a in arguments.split(",")]
            kwargs = {}
            for i, p in enumerate(params):
                if i < len(parts):
                    val = parts[i].strip().strip("'\"")
                    if p.get("type") in ("integer", "int"):
                        try:
                            val = int(val)
                        except ValueError:
                            pass
                    elif p.get("type") in ("float", "double"):
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    elif p.get("type") == "boolean":
                        val = val.lower() in ("true", "1", "yes")
                    kwargs[p["name"]] = val
            return await _call(**kwargs)

        _call_with_args.__name__ = tool_name
        _call_with_args.__doc__ = (
            f"{description}\n\n"
            f"Pass arguments as a comma-separated string in this order:\n"
            f"{', '.join(p['name'] + ' (' + p.get('type','string') + ')' for p in params)}\n\n"
            f"Example: \"{', '.join('value' + str(i+1) for i in range(len(params)))}\""
        )
        return function_tool(_call_with_args)

    else:
        # No parameters defined — accept a free-form query
        async def _call_freeform(query: str) -> str:
            sql = f"SELECT {func_name}({query})"
            result = await _exec_sql(sql)
            return f"{func_name}({query}) = {result}"

        _call_freeform.__name__ = tool_name
        _call_freeform.__doc__ = description
        return function_tool(_call_freeform)
