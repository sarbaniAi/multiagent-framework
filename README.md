# Multi-Agent Framework for Databricks

A generic, industry-agnostic framework for building multi-agent AI assistants on Databricks. Define your agents in a single YAML config file — no Python changes needed.

**Think of it as a self-hosted Agent Bricks alternative** — same ease of configuration, but runs in your own workspace with your own models, full data residency control, and any number of custom agents.

## How It Works

```
agents.yaml (you configure)          Databricks App (auto-deployed)
┌─────────────────────┐              ┌───────────────────────┐
│ orchestrator:       │              │   Orchestrator Agent   │
│   model: claude-4-5 │──  deploy ──►│   (OpenAI Agents SDK)  │
│                     │              │                       │
│ subagents:          │              │   ┌── Genie MCP       │
│   - genie           │              │   ├── Vector Search   │
│   - vector_search   │              │   ├── UC Functions    │
│   - uc_function     │              │   ├── External MCP    │
│   - external_mcp    │              │   └── Custom MCP      │
│   - custom_mcp      │              └───────────────────────┘
└─────────────────────┘
```

## Supported Subagent Types

| Type | Backend | Use Case |
|------|---------|----------|
| `genie` | Databricks Genie space (MCP) | Structured data queries (SQL tables) |
| `vector_search` | Databricks Vector Search | RAG over documents, manuals, policies |
| `uc_function` | Unity Catalog SQL function | Governed business rules, calculations |
| `external_mcp` | Any HTTP MCP server | Third-party APIs, SaaS integrations |
| `custom_mcp` | Local subprocess (stdio) | Custom tools, internal services |

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/sarbaniAi/multiagent-framework.git
cd multiagent-framework

# Auth
databricks auth login --host https://<your-workspace> --profile my-profile

# Environment
cp .env.example .env
# Edit .env: set DATABRICKS_CONFIG_PROFILE, DATABRICKS_WAREHOUSE_ID, etc.

# Agent config
cp agents.example.yaml agents.yaml
# Edit agents.yaml: define your orchestrator + subagents
```

### 2. Edit `agents.yaml`

This is the only file that matters. Define your orchestrator and subagents:

```yaml
orchestrator:
  name: "My Company Assistant"
  model: "databricks-claude-sonnet-4-5"
  instructions: |
    You are a helpful assistant for our organization.
    Route questions to the right tool based on descriptions.

subagents:
  - name: company_data
    type: genie
    space_id: "your-genie-space-id"
    description: "Query structured data about customers and orders"

  - name: policy_docs
    type: vector_search
    index_name: "catalog.schema.docs_index"
    description: "Search policy documents and guidelines"

  - name: check_eligibility
    type: uc_function
    function: "catalog.schema.fn_eligibility"
    parameters:
      - name: customer_id
        type: string
    description: "Check customer eligibility for products"
```

### 3. Configure deployment files

Edit `databricks.yml`:
- `targets.dev.workspace.host` → your workspace URL
- `DATABRICKS_WAREHOUSE_ID` → your warehouse ID

Edit `app.yaml` (same values, for manual deploy fallback).

### 4. Deploy

```bash
# Create MLflow experiment
databricks experiments create-experiment "/Users/<your-email>/multiagent-app" --profile my-profile
# Set the returned ID in databricks.yml (experiment_id) and app.yaml (MLFLOW_EXPERIMENT_ID)

# Deploy
DATABRICKS_CONFIG_PROFILE=my-profile databricks bundle deploy
DATABRICKS_CONFIG_PROFILE=my-profile databricks bundle run multiagent_app
```

### 5. Grant permissions

The app's service principal needs access to your resources. Grant via:
- Unity Catalog: `GRANT USE CATALOG`, `USE SCHEMA`, `SELECT`, `EXECUTE` on functions
- SQL Warehouse: `CAN_USE`
- Genie Space: `CAN_RUN` (via `/api/2.0/permissions/genie/<space_id>`)
- MLflow Experiment: `CAN_MANAGE`
- Vector Search endpoint: `CAN_QUERY` (if applicable)

---

## Example Configs

Pre-built configs for common industries — copy one to `agents.yaml` as a starting point:

| File | Industry | Subagents |
|------|----------|-----------|
| `examples/insurance_underwriting.yaml` | Insurance | Genie + VS + 2 UC functions |
| `examples/banking_customer_service.yaml` | Banking | Genie + VS + 2 UC functions |
| `examples/retail_analytics.yaml` | Retail | Genie + VS + UC function + External MCP |

```bash
cp examples/insurance_underwriting.yaml agents.yaml
# Edit with your resource IDs, then deploy
```

---

## Project Structure

```
multiagent-framework/
├── agents.example.yaml            # Template — copy to agents.yaml
├── agents.yaml                    # YOUR CONFIG (gitignored)
├── .env.example                   # Environment template
├── databricks.yml                 # DAB deployment config
├── app.yaml                       # Manual deploy fallback
│
├── agent_server/
│   ├── agent.py                   # Orchestrator — reads agents.yaml dynamically
│   ├── config.py                  # YAML loader + validator
│   ├── tools/
│   │   ├── genie.py               # Genie MCP builder
│   │   ├── vector_search.py       # Vector Search tool builder
│   │   ├── uc_function.py         # UC Function tool builder
│   │   ├── external_mcp.py        # External MCP builder
│   │   └── custom_mcp.py          # Custom MCP (stdio) builder
│   ├── start_server.py            # FastAPI + chat UI
│   └── utils.py                   # Auth + streaming helpers
│
├── examples/                      # Industry-specific configs
│   ├── insurance_underwriting.yaml
│   ├── banking_customer_service.yaml
│   └── retail_analytics.yaml
│
└── scripts/                       # Local dev (from template)
    ├── start_app.py
    ├── quickstart.py
    └── preflight.py
```

---

## How It Compares to Agent Bricks

| | Agent Bricks | This Framework |
|---|---|---|
| Model hosting | US-hosted (data residency issue) | Your workspace, any region |
| Configuration | UI-driven | YAML file (git-friendly, code-reviewable) |
| Subagent types | Fixed brick types | Genie, VS, UC functions, External MCP, Custom MCP |
| Customization | Limited | Full control — any MCP server, any function |
| Deployment | Databricks-managed | DAB (your infra, your control) |
| Agent framework | Proprietary | OpenAI Agents SDK (open source) |
| Observability | Built-in | MLflow tracing + experiments |

---

## Adding a New Subagent Type

To add a new tool type beyond the 5 built-in types:

1. Create `agent_server/tools/my_type.py` with a `build_my_type(config)` function
2. Add the type to `VALID_TYPES` in `agent_server/config.py`
3. Add the builder import + registration in `agent_server/agent.py`

---

## Built With

| Component | Source |
|-----------|--------|
| Agent framework | [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) |
| App template | [databricks/app-templates](https://github.com/databricks/app-templates) |
| Development toolkit | [databricks-solutions/ai-dev-kit](https://github.com/databricks-solutions/ai-dev-kit) |
| Serving | MLflow ResponsesAgent |
| Deployment | Databricks Asset Bundles (DAB) |
