import contextvars
import logging
from typing import AsyncGenerator, AsyncIterator, Optional
from uuid import uuid4

from agents.result import StreamEvent
from databricks.sdk import WorkspaceClient
from mlflow.genai.agent_server import get_request_headers
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentStreamEvent


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-request user context — set in request handlers, read by tools
# ---------------------------------------------------------------------------

_user_ws_client: contextvars.ContextVar[WorkspaceClient | None] = contextvars.ContextVar(
    "_user_ws_client", default=None
)

# Shared (app service principal) client — singleton
_app_ws_client: WorkspaceClient | None = None


def get_app_workspace_client() -> WorkspaceClient:
    """Get the shared app-level WorkspaceClient (service principal credentials)."""
    global _app_ws_client
    if _app_ws_client is None:
        _app_ws_client = WorkspaceClient()
    return _app_ws_client


def get_user_workspace_client() -> WorkspaceClient:
    """
    Get a per-request WorkspaceClient using the authenticated user's token.

    The Databricks App proxy injects x-forwarded-access-token with the
    user's OAuth token. This enables user-scoped access control.

    Falls back to the app-level client if the header is not present
    (e.g., during local development or evaluation).
    """
    token = get_request_headers().get("x-forwarded-access-token")
    if token:
        return WorkspaceClient(token=token, auth_type="pat")
    logger.debug("No user token found in request headers, falling back to app client")
    return get_app_workspace_client()


def set_request_ws_client() -> None:
    """Set the per-request WorkspaceClient in context. Call at the start of each request."""
    _user_ws_client.set(get_user_workspace_client())


def get_request_ws_client() -> WorkspaceClient:
    """
    Get the WorkspaceClient for the current request.

    Tools should call this instead of creating their own WorkspaceClient.
    Returns the user-scoped client if set, otherwise the app-level client.
    """
    client = _user_ws_client.get()
    if client is not None:
        return client
    return get_app_workspace_client()


def get_session_id(request: ResponsesAgentRequest) -> str | None:
    if request.context and request.context.conversation_id:
        return request.context.conversation_id
    if request.custom_inputs and isinstance(request.custom_inputs, dict):
        return request.custom_inputs.get("session_id")
    return None


def get_databricks_host(workspace_client: WorkspaceClient | None = None) -> Optional[str]:
    workspace_client = workspace_client or get_app_workspace_client()
    try:
        return workspace_client.config.host
    except Exception as e:
        logger.exception("Error getting databricks host from env: %s", e)
        return None


def build_mcp_url(path: str, workspace_client: WorkspaceClient | None = None) -> str:
    if not path.startswith("/"):
        return path
    hostname = get_databricks_host(workspace_client)
    return f"{hostname}{path}"


async def process_agent_stream_events(
    async_stream: AsyncIterator[StreamEvent],
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    curr_item_id = str(uuid4())
    async for event in async_stream:
        if event.type == "raw_response_event":
            event_data = event.data.model_dump()
            if event_data["type"] == "response.output_item.added":
                curr_item_id = str(uuid4())
                event_data["item"]["id"] = curr_item_id
            elif event_data.get("item") is not None and event_data["item"].get("id") is not None:
                event_data["item"]["id"] = curr_item_id
            elif event_data.get("item_id") is not None:
                event_data["item_id"] = curr_item_id
            yield event_data
        elif event.type == "run_item_stream_event" and event.item.type == "tool_call_output_item":
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=event.item.to_input_item(),
            )
