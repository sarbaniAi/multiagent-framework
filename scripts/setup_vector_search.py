#!/usr/bin/env python3
"""
Set up a Vector Search index from PDF documents in a UC Volume.

This script automates the full pipeline from raw PDFs to a queryable
Vector Search index, ready to wire into agents.yaml as a vector_search
subagent.

Pipeline:
  1. List PDFs in a UC Volume
  2. Extract text with PyPDF (no ai_parse_document — works in all regions)
  3. Chunk text with configurable size/overlap
  4. Create a Delta table with change data feed enabled
  5. Create a Vector Search endpoint (if needed)
  6. Create a Delta Sync Vector Search index
  7. Print the agents.yaml snippet to add

Usage:
    uv run setup-vector-search --volume-path /Volumes/cat/schema/pdfs \\
        --catalog my_catalog --schema my_schema --name my_docs

Options:
    --volume-path       UC Volume path containing PDF files (required)
    --catalog           UC catalog for output table and index (required)
    --schema            UC schema for output table and index (required)
    --name              Base name for table and index (required)
    --endpoint-name     VS endpoint name (default: multiagent_vs_endpoint)
    --embedding-model   Embedding model (default: databricks-gte-large-en)
    --chunk-size        Chunk size in characters (default: 1000)
    --chunk-overlap     Chunk overlap in characters (default: 200)
    --warehouse-id      SQL warehouse ID (or set DATABRICKS_WAREHOUSE_ID)
    --profile           Databricks CLI config profile
    -h, --help          Show this help message
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


# ---------------------------------------------------------------------------
# Print helpers (match quickstart.py pattern)
# ---------------------------------------------------------------------------

def print_header(text: str) -> None:
    print(f"\n{'=' * 67}")
    print(text)
    print("=" * 67)


def print_step(text: str) -> None:
    print(f"\n{text}")


def print_success(text: str) -> None:
    print(f"  \u2713 {text}")


def print_error(text: str) -> None:
    print(f"  \u2717 {text}", file=sys.stderr)


def print_warn(text: str) -> None:
    print(f"  ! {text}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Set up a Vector Search index from PDFs in a UC Volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--volume-path", required=True,
        help="UC Volume path, e.g. /Volumes/catalog/schema/my_pdfs",
    )
    p.add_argument("--catalog", required=True, help="UC catalog for output table and index")
    p.add_argument("--schema", required=True, help="UC schema for output table and index")
    p.add_argument(
        "--name", required=True,
        help="Base name — creates {name}_chunks table and {name}_index",
    )
    p.add_argument(
        "--endpoint-name", default="multiagent_vs_endpoint",
        help="Vector Search endpoint name (created if missing, default: multiagent_vs_endpoint)",
    )
    p.add_argument(
        "--embedding-model", default="databricks-gte-large-en",
        help="Embedding model endpoint (default: databricks-gte-large-en)",
    )
    p.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in chars (default: 1000)")
    p.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in chars (default: 200)")
    p.add_argument("--warehouse-id", help="SQL warehouse ID (falls back to DATABRICKS_WAREHOUSE_ID)")
    p.add_argument("--profile", help="Databricks CLI config profile")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_warehouse_id(args) -> str:
    wid = args.warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
    if not wid:
        print_error(
            "No warehouse ID provided. Use --warehouse-id or set DATABRICKS_WAREHOUSE_ID in .env"
        )
        sys.exit(1)
    return wid


def sanitize_sql_string(val: str) -> str:
    """Escape single quotes for safe SQL string interpolation."""
    return str(val).replace("'", "''").replace("\\", "\\\\")


def execute_sql(w, warehouse_id: str, statement: str):
    """Execute SQL via statement execution API. Returns result or raises."""
    from databricks.sdk.service.sql import Disposition, StatementState

    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        wait_timeout="50s",
        disposition=Disposition.INLINE,
    )
    if resp.status.state != StatementState.SUCCEEDED:
        error_msg = resp.status.error.message if resp.status.error else "Unknown SQL error"
        raise RuntimeError(f"SQL failed: {error_msg}")
    return resp


# ---------------------------------------------------------------------------
# Step 1: List PDFs
# ---------------------------------------------------------------------------

def list_pdfs(w, volume_path: str) -> list[str]:
    """List all .pdf files in the UC Volume."""
    try:
        entries = w.files.list_directory_contents(volume_path)
    except Exception as e:
        print_error(f"Cannot list files at '{volume_path}': {e}")
        print("  Verify the UC Volume exists and you have READ access.")
        sys.exit(1)

    pdf_paths = []
    for entry in entries:
        path = entry.path if hasattr(entry, "path") else str(entry)
        if path.lower().endswith(".pdf"):
            # Ensure full path
            if not path.startswith("/"):
                path = f"{volume_path.rstrip('/')}/{path}"
            pdf_paths.append(path)

    return sorted(pdf_paths)


# ---------------------------------------------------------------------------
# Step 2: Extract text from PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(w, file_path: str) -> str:
    """Download a PDF from UC Volume and extract text with PyPDF."""
    from pypdf import PdfReader

    try:
        resp = w.files.download(file_path)
        pdf_bytes = resp.contents.read()
    except Exception as e:
        print_warn(f"Cannot download '{file_path}': {e}")
        return ""

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        print_warn(f"Cannot parse '{file_path}': {e}")
        return ""


# ---------------------------------------------------------------------------
# Step 3: Chunk text
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks with sentence-boundary awareness.

    Tries to break at sentence boundaries ('. ', '\\n') near the chunk boundary
    to avoid splitting mid-sentence.
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
            # Look for a sentence boundary near the end (within last 20% of chunk)
            search_start = start + int(chunk_size * 0.8)
            best_break = -1

            # Prefer paragraph breaks, then sentence boundaries
            for sep in ["\n\n", "\n", ". ", "? ", "! ", "; "]:
                idx = text.rfind(sep, search_start, end)
                if idx > best_break:
                    best_break = idx + len(sep)

            if best_break > search_start:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (end - overlap), but never backward
        start = max(start + 1, end - overlap)

    return chunks


# ---------------------------------------------------------------------------
# Step 4: Create Delta table
# ---------------------------------------------------------------------------

def create_chunks_table(w, warehouse_id: str, catalog: str, schema: str, name: str) -> str:
    """Create the chunks Delta table. Returns fully qualified table name."""
    table_name = f"{catalog}.{schema}.{name}_chunks"

    sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGINT GENERATED ALWAYS AS IDENTITY,
        content STRING NOT NULL,
        source STRING,
        chunk_index INT
    )
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
    execute_sql(w, warehouse_id, sql)
    return table_name


# ---------------------------------------------------------------------------
# Step 5: Insert chunks
# ---------------------------------------------------------------------------

def insert_chunks(w, warehouse_id: str, table_name: str, chunks: list[dict]) -> int:
    """Insert chunk rows into the Delta table in batches."""
    BATCH_SIZE = 500
    total = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        values = []
        for c in batch:
            content = sanitize_sql_string(c["content"])
            source = sanitize_sql_string(c["source"])
            idx = c["chunk_index"]
            values.append(f"('{content}', '{source}', {idx})")

        sql = f"INSERT INTO {table_name} (content, source, chunk_index) VALUES\n" + ",\n".join(values)

        try:
            execute_sql(w, warehouse_id, sql)
            total += len(batch)
        except RuntimeError as e:
            # If batch too large, retry with smaller batches
            if "size" in str(e).lower() or "limit" in str(e).lower():
                print_warn(f"Batch of {len(batch)} too large, retrying in smaller batches...")
                for j in range(0, len(batch), 100):
                    mini = batch[j : j + 100]
                    vals = []
                    for c in mini:
                        content = sanitize_sql_string(c["content"])
                        source = sanitize_sql_string(c["source"])
                        vals.append(f"('{content}', '{source}', {c['chunk_index']})")
                    sql2 = f"INSERT INTO {table_name} (content, source, chunk_index) VALUES\n" + ",\n".join(vals)
                    execute_sql(w, warehouse_id, sql2)
                    total += len(mini)
            else:
                raise

    return total


# ---------------------------------------------------------------------------
# Step 6: Ensure Vector Search endpoint
# ---------------------------------------------------------------------------

def ensure_vs_endpoint(w, endpoint_name: str) -> None:
    """Create VS endpoint if it doesn't exist. Wait for ONLINE."""
    from databricks.sdk.service.vectorsearch import EndpointType

    # Check if it exists
    try:
        ep = w.vector_search_endpoints.get_endpoint(endpoint_name)
        state = ep.endpoint_status.state.value if ep.endpoint_status else "UNKNOWN"
        if state == "ONLINE":
            print_success(f"Endpoint '{endpoint_name}' already exists and is ONLINE")
            return
        print_step(f"  Endpoint '{endpoint_name}' exists (state: {state}), waiting...")
    except Exception:
        print_step(f"  Creating Vector Search endpoint '{endpoint_name}'...")
        try:
            w.vector_search_endpoints.create_endpoint(
                name=endpoint_name,
                endpoint_type=EndpointType.STANDARD,
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                pass  # Race condition — another process created it
            else:
                print_error(f"Cannot create endpoint: {e}")
                print("  You may have hit the endpoint limit. Use --endpoint-name to reuse an existing one.")
                sys.exit(1)

    # Poll until ONLINE (max 20 minutes)
    for attempt in range(40):
        time.sleep(30)
        try:
            ep = w.vector_search_endpoints.get_endpoint(endpoint_name)
            state = ep.endpoint_status.state.value if ep.endpoint_status else "UNKNOWN"
            if state == "ONLINE":
                print_success(f"Endpoint '{endpoint_name}' is ONLINE")
                return
            print(f"    Endpoint status: {state} (attempt {attempt + 1}/40)")
        except Exception:
            pass

    print_error(f"Endpoint '{endpoint_name}' did not come ONLINE within 20 minutes.")
    print("  Check the Databricks UI for details.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 7: Create Vector Search index
# ---------------------------------------------------------------------------

def create_vs_index(
    w, endpoint_name: str, index_name: str, table_name: str, embedding_model: str
) -> None:
    """Create a Delta Sync VS index. Wait for ONLINE."""
    from databricks.sdk.service.vectorsearch import (
        DeltaSyncVectorIndexSpecRequest,
        EmbeddingSourceColumn,
        PipelineType,
        VectorIndexType,
    )

    # Check if it already exists
    try:
        existing = w.vector_search_indexes.get_index(index_name)
        state = existing.status.ready if existing.status else False
        if state:
            print_success(f"Index '{index_name}' already exists and is ONLINE")
            return
        print_step(f"  Index '{index_name}' exists but not ready, waiting...")
    except Exception:
        print_step(f"  Creating Delta Sync index '{index_name}'...")
        try:
            w.vector_search_indexes.create_index(
                name=index_name,
                endpoint_name=endpoint_name,
                primary_key="id",
                index_type=VectorIndexType.DELTA_SYNC,
                delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                    source_table=table_name,
                    pipeline_type=PipelineType.TRIGGERED,
                    embedding_source_columns=[
                        EmbeddingSourceColumn(
                            name="content",
                            embedding_model_endpoint_name=embedding_model,
                        )
                    ],
                ),
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                pass
            else:
                print_error(f"Cannot create index: {e}")
                sys.exit(1)

    # Poll until ONLINE (max 30 minutes)
    for attempt in range(60):
        time.sleep(30)
        try:
            idx = w.vector_search_indexes.get_index(index_name)
            ready = idx.status.ready if idx.status else False
            state_msg = idx.status.message if idx.status else "unknown"

            if ready:
                print_success(f"Index '{index_name}' is ONLINE")
                return

            # Check for failure (use getattr for SDK compatibility)
            detail = getattr(idx.status, "detailed_state", "") or ""
            if "FAILED" in str(detail).upper() or "fail" in str(state_msg).lower():
                print_error(f"Index creation failed: {state_msg}")
                sys.exit(1)

            print(f"    Index status: ready={ready} rows={rows} {state_msg[:80]} (attempt {attempt + 1}/60)")
        except Exception:
            pass

    print_error(f"Index '{index_name}' did not come ONLINE within 30 minutes.")
    print("  Check the Databricks UI for details.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 8: Print agents.yaml snippet
# ---------------------------------------------------------------------------

def print_yaml_snippet(name: str, index_name: str) -> None:
    """Print the agents.yaml snippet for the new vector_search subagent."""
    print("\n  Add this to your agents.yaml under 'subagents:'")
    print("  " + "-" * 55)
    print(f"""
  - name: {name}
    type: vector_search
    index_name: "{index_name}"
    columns: ["content", "source"]
    num_results: 5
    description: >
      Search {name.replace('_', ' ')} documents for relevant information.
      Use for questions about content from the ingested PDF documents.
""")
    print("  " + "-" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    from databricks.sdk import WorkspaceClient

    print_header("PDF -> Vector Search Pipeline")

    # Init workspace client
    ws_kwargs = {}
    if args.profile:
        ws_kwargs["profile"] = args.profile
    w = WorkspaceClient(**ws_kwargs)

    warehouse_id = get_warehouse_id(args)
    table_name = f"{args.catalog}.{args.schema}.{args.name}_chunks"
    index_name = f"{args.catalog}.{args.schema}.{args.name}_index"

    print(f"  Volume:    {args.volume_path}")
    print(f"  Table:     {table_name}")
    print(f"  Index:     {index_name}")
    print(f"  Endpoint:  {args.endpoint_name}")
    print(f"  Embedding: {args.embedding_model}")
    print(f"  Chunks:    {args.chunk_size} chars, {args.chunk_overlap} overlap")

    # Step 1: List PDFs
    print_step("Step 1/7: Scanning for PDFs...")
    pdf_paths = list_pdfs(w, args.volume_path)
    if not pdf_paths:
        print_error(f"No PDF files found in '{args.volume_path}'")
        sys.exit(1)
    print_success(f"Found {len(pdf_paths)} PDF(s)")
    for p in pdf_paths:
        print(f"    - {p.rsplit('/', 1)[-1]}")

    # Step 2+3: Parse and chunk
    print_step("Step 2/7: Extracting and chunking text...")
    all_chunks = []
    files_parsed = 0
    for path in pdf_paths:
        text = extract_text_from_pdf(w, path)
        if not text.strip():
            print_warn(f"Skipping {path.rsplit('/', 1)[-1]} (no extractable text)")
            continue
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        source_name = path.rsplit("/", 1)[-1]
        for i, chunk in enumerate(chunks):
            all_chunks.append({"content": chunk, "source": source_name, "chunk_index": i})
        files_parsed += 1
        print_success(f"{source_name}: {len(chunks)} chunks ({len(text)} chars)")

    if not all_chunks:
        print_error("No text extracted from any PDF. For scanned PDFs, consider OCR preprocessing.")
        sys.exit(1)
    print_success(f"Total: {len(all_chunks)} chunks from {files_parsed} PDFs")

    # Step 4: Create table
    print_step("Step 3/7: Creating chunks table...")
    create_chunks_table(w, warehouse_id, args.catalog, args.schema, args.name)
    print_success(f"Table ready: {table_name}")

    # Step 5: Insert chunks
    print_step("Step 4/7: Inserting chunks...")
    count = insert_chunks(w, warehouse_id, table_name, all_chunks)
    print_success(f"Inserted {count} chunks")

    # Step 6: Ensure VS endpoint
    print_step("Step 5/7: Setting up Vector Search endpoint...")
    ensure_vs_endpoint(w, args.endpoint_name)

    # Step 7: Create VS index
    print_step("Step 6/7: Creating Vector Search index...")
    create_vs_index(w, args.endpoint_name, index_name, table_name, args.embedding_model)

    # Step 8: Output
    print_header("Step 7/7: Setup Complete!")
    print_success(f"Table:    {table_name} ({count} chunks)")
    print_success(f"Index:    {index_name} (ONLINE)")
    print_success(f"Endpoint: {args.endpoint_name}")
    print_yaml_snippet(args.name, index_name)

    print("  Next steps:")
    print("    1. Copy the YAML snippet above into your agents.yaml")
    print("    2. Deploy: databricks bundle deploy && databricks bundle run multiagent_app")
    print("    3. Grant the app's service principal SELECT on the source table")
    print(f"       GRANT SELECT ON TABLE {table_name} TO `<app-sp-client-id>`")
    print()


if __name__ == "__main__":
    main()
