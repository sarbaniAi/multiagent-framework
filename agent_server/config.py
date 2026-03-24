"""
Configuration loader for the multi-agent framework.

Reads agents.yaml and validates the structure. Supports environment
variable substitution in values using ${VAR_NAME} syntax.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _resolve_env_vars(value: Any) -> Any:
    """Replace ${VAR_NAME} placeholders with environment variable values."""
    if isinstance(value, str):
        def replacer(match):
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise ValueError(f"Environment variable '{var_name}' is not set (referenced in agents.yaml)")
            return env_val
        return re.sub(r'\$\{([^}]+)\}', replacer, value)
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


VALID_TYPES = {"genie", "vector_search", "uc_function", "external_mcp", "custom_mcp"}


def validate_subagent(agent: dict, index: int) -> None:
    """Validate a single subagent definition."""
    name = agent.get("name")
    if not name:
        raise ValueError(f"Subagent at index {index} is missing 'name'")

    agent_type = agent.get("type")
    if not agent_type:
        raise ValueError(f"Subagent '{name}' is missing 'type'")
    if agent_type not in VALID_TYPES:
        raise ValueError(
            f"Subagent '{name}' has invalid type '{agent_type}'. "
            f"Valid types: {', '.join(sorted(VALID_TYPES))}"
        )

    if not agent.get("description"):
        raise ValueError(f"Subagent '{name}' is missing 'description' (required for LLM routing)")

    # Type-specific validation
    if agent_type == "genie" and not agent.get("space_id"):
        raise ValueError(f"Subagent '{name}' (genie) is missing 'space_id'")

    if agent_type == "vector_search" and not agent.get("index_name"):
        raise ValueError(f"Subagent '{name}' (vector_search) is missing 'index_name'")

    if agent_type == "uc_function" and not agent.get("function"):
        raise ValueError(f"Subagent '{name}' (uc_function) is missing 'function'")

    if agent_type == "external_mcp" and not agent.get("url"):
        raise ValueError(f"Subagent '{name}' (external_mcp) is missing 'url'")

    if agent_type == "custom_mcp" and not agent.get("command"):
        raise ValueError(f"Subagent '{name}' (custom_mcp) is missing 'command'")


def load_config(config_path: str = None) -> dict:
    """
    Load and validate agents.yaml.

    Returns:
        dict with keys: orchestrator, subagents
    """
    if config_path is None:
        config_path = os.environ.get(
            "AGENTS_CONFIG_PATH",
            str(Path(__file__).parent.parent / "agents.yaml"),
        )

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy agents.example.yaml to agents.yaml and configure your subagents."
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Config file is empty: {config_path}")

    # Resolve environment variables
    config = _resolve_env_vars(raw)

    # Validate orchestrator
    orch = config.get("orchestrator", {})
    if not orch.get("model"):
        raise ValueError("orchestrator.model is required in agents.yaml")

    # Validate subagents
    subagents = config.get("subagents", [])
    if not subagents:
        raise ValueError("At least one subagent is required in agents.yaml")

    for i, agent in enumerate(subagents):
        validate_subagent(agent, i)

    # Check for duplicate names
    names = [a["name"] for a in subagents]
    dupes = [n for n in names if names.count(n) > 1]
    if dupes:
        raise ValueError(f"Duplicate subagent names: {set(dupes)}")

    return config
