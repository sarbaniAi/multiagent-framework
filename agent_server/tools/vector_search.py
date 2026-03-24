"""Vector Search RAG subagent — searches a Databricks Vector Search index."""

from agents import function_tool
from databricks.sdk import WorkspaceClient


_ws = None


def _get_ws() -> WorkspaceClient:
    global _ws
    if _ws is None:
        _ws = WorkspaceClient()
    return _ws


def build_vector_search_tool(config: dict):
    """
    Build a function_tool that queries a Vector Search index.

    Required config keys:
        index_name:   Full index name (catalog.schema.index)
        description:  When to use this tool

    Optional config keys:
        columns:      List of columns to return (default: ["content", "source"])
        num_results:  Number of results (default: 5)
    """
    index_name = config["index_name"]
    description = config["description"]
    columns = config.get("columns", ["content", "source"])
    num_results = config.get("num_results", 5)
    tool_name = f"search_{config['name']}"

    async def _search(query: str) -> str:
        try:
            result = _get_ws().vector_search_indexes.query_index(
                index_name=index_name,
                query_text=query,
                columns=columns,
                num_results=num_results,
            )
            if not result.result or not result.result.data_array:
                return "No relevant documents found."

            col_names = [c.name for c in result.manifest.columns]
            chunks, sources = [], set()
            for row in result.result.data_array:
                r = dict(zip(col_names, row))
                # Use first column as content, rest as metadata
                content_col = columns[0] if columns else col_names[0]
                chunks.append(r.get(content_col, str(r)))
                for c in columns[1:]:
                    if r.get(c):
                        sources.add(str(r[c]))

            context = "\n\n---\n\n".join(chunks)
            src = ", ".join(sources) if sources else "knowledge base"
            return f"Sources: {src}\n\n{context}"
        except Exception as e:
            return f"Vector search error: {e}"

    _search.__name__ = tool_name
    _search.__doc__ = description
    return function_tool(_search)
