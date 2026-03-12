# Data Analysis Agent

A lightweight, standalone data analysis agent that converts natural language questions into SQL queries, runs exploratory data analysis, and generates interactive charts — all powered by LLM function calling.

Inspired by [WrenAI](https://github.com/Canner/WrenAI)'s GenBI architecture, rebuilt as a single-file agent with no infrastructure dependencies.

## Features

- **Text-to-SQL** — LLM generates DuckDB SQL from natural language + schema context
- **Agent Loop** — LLM autonomously decides which tools to call via function calling (execute_sql, run_eda, generate_chart)
- **Auto Error Correction** — structured error feedback on SQL failure, LLM self-corrects up to 3 retries
- **Smart Chart Selection** — rule-based column classification (temporal/categorical/quantitative) + scale ratio detection for chart type decisions
- **8 Chart Types** — bar, grouped_bar, stacked_bar, line, multi_line, area, pie, dual_axis (bar+line with independent Y-axes)
- **MDL Semantic Layer** — JSON-defined business concepts (models, dimensions, joins, metrics, derived fields) → annotated DDL for LLM context + DuckDB VIEWs for query execution
- **RAG Retrieval** — SQL pairs few-shot examples via local embedding similarity search (sentence-transformers, no API key needed)
- **Multi-turn Chat** — interactive REPL with conversation memory for follow-up questions
- **Star Schema Support** — automatic preprocessing VIEWs + denormalized fact view generation

## Architecture

```
src/text_to_sql.py          Minimal Text → SQL (single LLM call)
    ↓
src/text_to_sql_viz.py      Pipeline: Text → SQL → Data → Chart (4-step sequential)
    ↓
src/data_agent.py           Agent: LLM-driven tool loop + semantic layer + RAG
    ↑
src/mdl_rag.py              MDL parsing, DDL generation, VIEW generation, embedding search
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your LLM API key
export DEEPSEEK_API_KEY=sk-...

# Run built-in demo (customers + orders)
python run.py

# Single question with CSV files
python run.py "Which product has the highest sales?" data/sales.csv

# With MDL semantic layer + RAG
python run.py "Which store type has the highest sales?" \
  data/train.csv data/stores.csv data/features.csv \
  --mdl examples/walmart_mdl.json \
  --sql-pairs examples/walmart_sql_pairs.json

# Interactive multi-turn chat
python run.py --chat \
  data/train.csv data/stores.csv data/features.csv \
  --mdl examples/walmart_mdl.json \
  --sql-pairs examples/walmart_sql_pairs.json
```

You can also run as a Python module:

```bash
python -m src "Which product has the highest sales?" data/sales.csv
```

## How It Works

### 1. MDL Semantic Layer (`src/mdl_rag.py`)

Define your data model in a JSON file:

```json
{
  "models": [{"name": "orders", "columns": [...], "description": "..."}],
  "dimensions": [{"name": "customers", "columns": [...]}],
  "joins": [{"left": "orders", "right": "customers", "on": "orders.customer_id = customers.id"}],
  "metrics": [{"name": "monthly_revenue", "baseObject": "orders", "measure": [{"expression": "SUM(amount)"}]}],
  "derivedFields": [{"name": "is_vip", "expression": "total_spent > 1000"}]
}
```

The MDL is converted to:
- **Annotated DDL** — SQL comments with business descriptions, fed to the LLM as context
- **DuckDB VIEWs** — preprocessing (COALESCE, CAST) + star schema joins, ready to query

### 2. RAG SQL Examples (`SqlPairsStore`)

Provide example question-SQL pairs in JSON. The agent embeds them locally with `all-MiniLM-L6-v2` (no API key needed) and retrieves the most similar examples as few-shot context for the LLM.

### 3. Agent Loop (`src/data_agent.py`)

The LLM receives the schema + examples + question and autonomously calls tools:

1. `execute_sql(sql)` — run SQL, get results or structured error
2. `run_eda(sql)` — statistics, correlations, missing values
3. `generate_chart(sql, filename)` — Vega-Lite chart as self-contained HTML

### 4. Chart Type Decision (`src/text_to_sql_viz.py`)

Data preprocessing auto-classifies columns and computes scale ratios. The LLM follows decision rules:

| Data Pattern | Chart Type |
|---|---|
| 1 categorical + 1 quantitative | bar |
| 2 categorical + 1 quantitative | grouped_bar |
| 1 temporal + 1 quantitative | line |
| Multiple metrics, similar scale | multi_line |
| 2 metrics, scale ratio > 10x | dual_axis |
| Parts of a whole | pie |

## File Structure

```
data-agent/
├── run.py                     # Convenience entry point
├── requirements.txt
├── .gitignore
├── src/                       # Core Python source
│   ├── __init__.py
│   ├── __main__.py            # python -m src entry point
│   ├── data_agent.py          # Main agent (CLI + agent loop + tools)
│   ├── text_to_sql_viz.py     # Chart generation pipeline + Vega-Lite rendering
│   ├── text_to_sql.py         # Minimal text-to-SQL (standalone)
│   └── mdl_rag.py             # MDL parsing + DDL/VIEW generation + embedding RAG
├── data/                      # User-provided data files (gitignored)
│   ├── train.csv
│   ├── stores.csv
│   └── ...
├── examples/                  # MDL and SQL pairs config examples
│   ├── demo_mdl.json
│   ├── demo_sql_pairs.json
│   ├── walmart_mdl.json
│   └── walmart_sql_pairs.json
└── README.md
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | Yes | — | DeepSeek API key for LLM |
| `TEXT_TO_SQL_MODEL` | No | `deepseek/deepseek-chat` | LiteLLM model string |

Any LiteLLM-compatible model works (OpenAI, Anthropic, local Ollama, etc.) — just set the appropriate API key and model string.

## License

MIT
