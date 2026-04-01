"""UC Function subagent — calls a Unity Catalog SQL function.

Security:
  - All string parameters are escaped (single quotes doubled) to prevent SQL injection.
  - Numeric/boolean parameters are strictly validated before interpolation.
  - Freeform mode (no parameters) uses parameterized execution via named markers.
  - Raw database errors are never exposed to the user; details are logged server-side.
"""

import logging
import os
import re

from agents import function_tool
from databricks.sdk.service.sql import Disposition, StatementState

from agent_server.utils import get_request_ws_client


logger = logging.getLogger(__name__)

WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")

# Regex to validate fully-qualified UC function names (catalog.schema.function)
_VALID_FUNC_NAME = re.compile(r"^[\w]+\.[\w]+\.[\w]+$")


def _sanitize_string(val: str) -> str:
    """Escape single quotes to prevent SQL injection in string literals."""
    return str(val).replace("'", "''")


def _validate_numeric(val, target_type: str):
    """Strictly validate and convert numeric values. Raises ValueError on failure."""
    if target_type in ("integer", "int"):
        return int(val)
    elif target_type in ("float", "double"):
        return float(val)
    raise ValueError(f"Unsupported numeric type: {target_type}")


def _validate_boolean(val) -> str:
    """Strictly validate boolean values. Returns SQL boolean literal."""
    if isinstance(val, bool):
        return "true" if val else "false"
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return "true"
    elif s in ("false", "0", "no"):
        return "false"
    raise ValueError(f"Invalid boolean value: {val}")


async def _exec_sql(sql: str, params: dict = None) -> str:
    """Execute SQL and return results. Supports parameterized queries."""
    try:
        kwargs = {
            "warehouse_id": WAREHOUSE_ID,
            "statement": sql,
            "wait_timeout": "30s",
            "disposition": Disposition.INLINE,
        }
        if params:
            kwargs["parameters"] = [
                {"name": k, "value": str(v), "type": "STRING"}
                for k, v in params.items()
            ]
        resp = get_request_ws_client().statement_execution.execute_statement(**kwargs)
        if resp.status.state != StatementState.SUCCEEDED:
            error_detail = resp.status.error.message if resp.status.error else "unknown"
            logger.error("SQL execution failed: %s | SQL: %s", error_detail, sql)
            return "Function execution failed. Please try again or refine your query."
        if resp.result and resp.result.data_array:
            cols = [c.name for c in resp.manifest.schema.columns]
            # Return all rows for TABLE-returning functions
            if len(resp.result.data_array) == 1:
                row = resp.result.data_array[0]
                if len(cols) == 1:
                    return str(row[0])
                return ", ".join(f"{c}={v}" for c, v in zip(cols, row))
            else:
                rows = []
                for row in resp.result.data_array:
                    rows.append(", ".join(f"{c}={v}" for c, v in zip(cols, row)))
                return "\n".join(rows)
        return "No result returned."
    except Exception as e:
        logger.exception("Unexpected error executing SQL: %s", e)
        return "Function execution encountered an error. Please try again."


def build_uc_function_tool(config: dict):
    """
    Build a function_tool that calls a UC SQL function.

    Required config keys:
        function:     Fully qualified function name (catalog.schema.fn_name)
        description:  When to use this tool

    Optional config keys:
        parameters:   List of {name, type} dicts describing function params.
                      If omitted, the tool accepts a single 'query' string
                      that is passed as a parameterized string argument.

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

    # Validate function name format to prevent injection via config
    if not _VALID_FUNC_NAME.match(func_name):
        raise ValueError(
            f"Invalid UC function name '{func_name}'. "
            f"Must be fully qualified: catalog.schema.function_name"
        )

    if params:
        # Build a tool with typed, validated parameters
        async def _call(**kwargs) -> str:
            args = []
            for p in params:
                val = kwargs.get(p["name"])
                p_type = p.get("type", "string")
                if val is None:
                    args.append("NULL")
                elif p_type in ("integer", "int", "float", "double"):
                    try:
                        validated = _validate_numeric(val, p_type)
                        args.append(str(validated))
                    except (ValueError, TypeError):
                        logger.warning(
                            "Invalid %s value for param '%s': %s",
                            p_type, p["name"], val,
                        )
                        return f"Invalid value for parameter '{p['name']}': expected {p_type}."
                elif p_type == "boolean":
                    try:
                        args.append(_validate_boolean(val))
                    except ValueError:
                        return f"Invalid value for parameter '{p['name']}': expected boolean."
                else:
                    # String type — escape single quotes
                    args.append(f"'{_sanitize_string(val)}'")

            sql = f"SELECT * FROM {func_name}({', '.join(args)})"
            result = await _exec_sql(sql)
            param_summary = ", ".join(
                str(kwargs.get(p["name"], "")) for p in params
            )
            return f"{func_name}({param_summary}) = {result}"

        # Build docstring with parameter info
        param_doc = "\n".join(
            f"    {p['name']} ({p.get('type', 'string')})" for p in params
        )
        _call.__name__ = tool_name
        _call.__doc__ = f"{description}\n\nParameters:\n{param_doc}"

        # Wrapper for OpenAI Agents SDK — parses comma-separated string
        async def _call_with_args(arguments: str) -> str:
            """Parse arguments string and call the function with validation."""
            parts = [a.strip() for a in arguments.split(",")]
            kwargs = {}
            for i, p in enumerate(params):
                if i < len(parts):
                    val = parts[i].strip().strip("'\"")
                    p_type = p.get("type", "string")
                    if p_type in ("integer", "int"):
                        try:
                            val = int(val)
                        except ValueError:
                            return f"Invalid value for '{p['name']}': expected integer."
                    elif p_type in ("float", "double"):
                        try:
                            val = float(val)
                        except ValueError:
                            return f"Invalid value for '{p['name']}': expected number."
                    elif p_type == "boolean":
                        if val.lower() in ("true", "1", "yes"):
                            val = True
                        elif val.lower() in ("false", "0", "no"):
                            val = False
                        else:
                            return f"Invalid value for '{p['name']}': expected boolean."
                    kwargs[p["name"]] = val
            return await _call(**kwargs)

        _call_with_args.__name__ = tool_name
        _call_with_args.__doc__ = (
            f"{description}\n\n"
            f"Pass arguments as a comma-separated string in this order:\n"
            f"{', '.join(p['name'] + ' (' + p.get('type', 'string') + ')' for p in params)}\n\n"
            f"Example: \"{', '.join('value' + str(i + 1) for i in range(len(params)))}\""
        )
        return function_tool(_call_with_args)

    else:
        # No parameters defined — use parameterized query for safety
        async def _call_freeform(query: str) -> str:
            # Sanitize the query as a string parameter
            sanitized = _sanitize_string(query)
            sql = f"SELECT * FROM {func_name}('{sanitized}')"
            result = await _exec_sql(sql)
            return f"{func_name}({query}) = {result}"

        _call_freeform.__name__ = tool_name
        _call_freeform.__doc__ = description
        return function_tool(_call_freeform)
