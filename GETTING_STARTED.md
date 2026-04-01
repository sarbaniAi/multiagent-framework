# Getting Started — Build a Multi-Agent Copilot

Build a fully functional multi-agent AI assistant on Databricks in **12 steps, zero Python code**.

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html) installed (`databricks --version`)
- [uv](https://docs.astral.sh/uv/) installed (Python package manager)
- Git installed
- Access to a Databricks workspace with serverless compute or a SQL warehouse

---

## Step 1: Clone the framework

```bash
git clone https://github.com/sarbaniAi/multiagent-framework.git
cd multiagent-framework
```

## Step 2: Authenticate with Databricks

```bash
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

This creates a profile in `~/.databrickscfg`. Note the **profile name** (e.g., `DEFAULT`).

Verify:
```bash
databricks auth profiles
```

## Step 3: Find your SQL Warehouse ID

```bash
databricks warehouses list --profile <your-profile>
```

Copy the **ID** of a running warehouse (e.g., `91dbe14a27ddabad`).

## Step 4: Set up environment

```bash
cp .env.example .env
```

Edit `.env` with your values:
```
DATABRICKS_CONFIG_PROFILE=<your-profile>
DATABRICKS_WAREHOUSE_ID=<your-warehouse-id>
UC_CATALOG=<your-catalog>
UC_SCHEMA=<your-schema>
COMPANY_NAME=My Assistant
```

## Step 5: Discover available resources

```bash
uv run discover-tools --profile <your-profile> --catalog <your-catalog> --schema <your-schema>
```

This lists available **Genie spaces, Vector Search indexes, UC functions, and tables** on your workspace. Note down what you want to use as sub-agents.

## Step 6: (Optional) Create a Vector Search index from PDFs

If you have PDFs in a UC Volume and want a RAG sub-agent:

```bash
uv run setup-vector-search \
  --volume-path /Volumes/<catalog>/<schema>/<volume_name> \
  --catalog <catalog> \
  --schema <schema> \
  --name <index_base_name> \
  --endpoint-name <vs-endpoint-name> \
  --profile <your-profile>
```

This creates:
- Table: `<catalog>.<schema>.<name>_chunks`
- Index: `<catalog>.<schema>.<name>_index`
- Prints a ready-to-paste YAML snippet for Step 7

Uses PyPDF for text extraction (works in all regions, no `ai_parse_document` needed).

## Step 7: Configure your agents

```bash
cp agents.example.yaml agents.yaml
```

Edit `agents.yaml` — this is the **only file you need to author**:

```yaml
orchestrator:
  name: "My Assistant"
  model: "databricks-claude-sonnet-4-5"    # Any Databricks FMAPI model
  instructions: |
    You are a helpful assistant.
    Route questions to the most appropriate tool based on descriptions.
    If a query requires multiple tools, call them sequentially and synthesize.

subagents:

  # Genie — for structured data queries
  - name: data_analyst
    type: genie
    space_id: "<your-genie-space-id>"
    description: >
      Query structured data. Use for revenue, targets, aggregations.

  # Vector Search — for document RAG
  - name: knowledge_base
    type: vector_search
    index_name: "<catalog>.<schema>.<name>_index"
    columns: ["content", "source"]
    num_results: 5
    description: >
      Search policy documents and guidelines.

  # UC Function — for governed business logic
  - name: calculator
    type: uc_function
    function: "<catalog>.<schema>.my_function"
    parameters:
      - name: input_param
        type: string
    description: >
      Run business calculations.
```

### Supported sub-agent types

| Type | What it does | Required config |
|------|-------------|-----------------|
| `genie` | Natural language SQL via Databricks Genie | `space_id` |
| `vector_search` | RAG over documents | `index_name` |
| `uc_function` | Call a Unity Catalog SQL function | `function`, optional `parameters` |
| `external_mcp` | Connect to an external MCP server (HTTP) | `url`, optional `headers` |
| `custom_mcp` | Run a local MCP server (subprocess) | `command` |

## Step 8: Create an MLflow experiment

```bash
databricks experiments create-experiment \
  "/Users/<your-email>/my-assistant" \
  --profile <your-profile>
```

Note the returned **experiment_id**.

## Step 9: Configure deployment

Edit `databricks.yml`:

```yaml
bundle:
  name: multiagent_framework

resources:
  apps:
    multiagent_app:
      name: "my-assistant"
      description: "My multi-agent assistant"
      source_code_path: ./
      config:
        command: ["uvicorn", "agent_server.start_server:app", "--host", "0.0.0.0", "--port", "8000"]
        env:
          - name: MLFLOW_TRACKING_URI
            value: "databricks"
          - name: MLFLOW_REGISTRY_URI
            value: "databricks-uc"
          - name: MLFLOW_EXPERIMENT_ID
            value_from: "experiment"
          - name: DATABRICKS_WAREHOUSE_ID
            value: "<your-warehouse-id>"
          - name: COMPANY_NAME
            value: "My Assistant"
      resources:
        - name: "experiment"
          experiment:
            experiment_id: "<experiment-id>"
            permission: "CAN_MANAGE"

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: https://<your-workspace>.cloud.databricks.com
```

## Step 10: Deploy

```bash
# Validate
DATABRICKS_CONFIG_PROFILE=<your-profile> databricks bundle validate

# Deploy
DATABRICKS_CONFIG_PROFILE=<your-profile> databricks bundle deploy

# Start the app
DATABRICKS_CONFIG_PROFILE=<your-profile> databricks bundle run multiagent_app
```

Note the **App URL** printed at the end.

## Step 11: Grant permissions to the app's service principal

Get the app's SP client ID:
```bash
databricks apps get <your-app-name> --profile <your-profile> -o json | grep service_principal_client_id
```

### UC permissions (via SQL)

Run these in a Databricks SQL editor or notebook:
```sql
GRANT USE CATALOG ON CATALOG <catalog> TO `<sp-client-id>`;
GRANT USE SCHEMA ON SCHEMA <catalog>.<schema> TO `<sp-client-id>`;
GRANT SELECT ON SCHEMA <catalog>.<schema> TO `<sp-client-id>`;
GRANT EXECUTE ON SCHEMA <catalog>.<schema> TO `<sp-client-id>`;
```

### Workspace permissions (via Databricks UI)

| Resource | Where to grant | Permission |
|----------|---------------|------------|
| **Genie space** | Genie → Share → Add SP | Can Run |
| **SQL Warehouse** | SQL Warehouses → Permissions → Add SP | Can Use |
| **VS endpoint** | Vector Search → Endpoints → Permissions → Add SP | Can Manage |

## Step 12: Use your app

Open the **App URL** from Step 10. The built-in Chat UI is ready.

---

## Example: Sales Copilot

With these workspace resources:
- 6 tables in `my_catalog.sales_schema` (distributor_sales, retailer_pos, sku_master, etc.)
- 1 Genie space connected to those tables
- 2 UC functions (calculate_margin_gap, territory_prioritizer)
- PDFs in a UC Volume → Vector Search index via `setup-vector-search`

You get a copilot that can:
- "What are the top 5 districts by revenue?" → Genie
- "What is our shelf share policy?" → Vector Search
- "What's the margin gap in Beverages in Hyderabad?" → UC Function
- "Which territories need immediate attention?" → UC Function
- "Give me a full picture of Pune district" → Multiple tools combined

**Zero Python. Just YAML.**

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| App crashes on deploy | Check `agents.yaml` — are all `${VAR_NAME}` references set as env vars in `databricks.yml`? |
| "Config file not found" | Ensure `agents.yaml` exists (not gitignored) and is in the project root |
| Tool returns errors | Check that the app's SP has permissions on the resource (UC, Genie, VS, warehouse) |
| VS index not working | Verify the index is ONLINE: `databricks vector-search indexes list --profile <profile>` |
| Rate limited | Default: 30 requests/60s. Override with env vars `RATE_LIMIT_MAX_REQUESTS` and `RATE_LIMIT_WINDOW_SECONDS` |
