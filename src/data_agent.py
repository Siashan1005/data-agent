#!/usr/bin/env python3
"""
Data Analysis Agent — Text → SQL → EDA → Visualization

Evolved from text_to_sql_viz.py (pipeline) into a real Agent:
  - MDL semantic layer: maps business concepts to DB schema (descriptions,
    relationships, metrics, calculated fields)
  - RAG retrieval: SQL pairs embedding similarity search for LLM few-shot examples
  - Intent classification: pre-flight check whether the question is data-answerable
  - LLM autonomously decides which tools to call and how many times via function calling
  - Structured error feedback on SQL failure (error type + suggestion), stops after 3 retries
  - LLM optionally runs EDA and decides which chart to generate

Tools:
  execute_sql(sql)           Execute SQL, return data or structured error
  run_eda(sql)               Exploratory data analysis on SQL results
  generate_chart(sql, file)  Generate Vega-Lite chart, save as HTML

Usage:
  export DEEPSEEK_API_KEY=sk-...
  python data_agent.py                                        # built-in demo (MDL + SQL pairs)
  python data_agent.py "your question"                        # with demo data
  python data_agent.py "your question" data.csv               # custom CSV
  python data_agent.py "your question" data.csv --mdl schema.mdl.json
  python data_agent.py "your question" data.csv --mdl s.json --sql-pairs p.json
"""

import asyncio
import json
import os
import sys
import webbrowser
from pathlib import Path

import duckdb
import pandas as pd

from .text_to_sql_viz import (
    _DEMO_DATAFRAMES,
    _DEMO_SCHEMA,
    generate_chart as _generate_chart_fn,
    preprocess_data,
    render_html,
)
from .mdl_rag import (
    load_mdl,
    mdl_to_ddl,
    mdl_to_views,
    retrieve_relevant_tables,
    filter_mdl_by_tables,
    SqlPairsStore,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas (DeepSeek / OpenAI function calling format)
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute a SQL SELECT query against the database with DuckDB. "
                "Returns the result rows and column names, or an error message. "
                "If the query fails, read the error and fix the SQL before retrying."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A valid DuckDB SQL SELECT statement.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_eda",
            "description": (
                "Run exploratory data analysis on a SQL query result. "
                "Returns shape, column types, missing values, summary statistics, "
                "and numeric correlations. Useful before deciding on visualization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query whose results to analyze.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": (
                "Generate a Vega-Lite chart from a SQL query result and save it "
                "as a self-contained HTML file. The LLM picks the best chart type "
                "based on the data. Call this after you have a working SQL query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query that produces the data to visualize.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output HTML filename, e.g. 'result.html'.",
                    },
                },
                "required": ["sql", "filename"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

# Module-level state set by run_agent() before the loop starts
# Either dataframes (in-memory) or csv_paths (large files) is used, not both.
_dataframes: dict[str, pd.DataFrame] = {}
_csv_paths: dict[str, str] = {}   # table_name → absolute CSV file path
_view_stmts: list[str] = []       # MDL-generated CREATE VIEW statements
_question: str = ""
_model: str = "deepseek/deepseek-chat"

MAX_SQL_RETRIES = 3  # max consecutive SQL failures before aborting the agent


def _make_conn() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with all tables registered."""
    conn = duckdb.connect()
    # In-memory DataFrames
    for name, df in _dataframes.items():
        conn.register(name, df)
    # Large CSV files — DuckDB reads them lazily without loading into RAM
    for name, path in _csv_paths.items():
        conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{path}')")
    # MDL-generated views (preprocessing + star schema joins)
    # Tolerate failures for missing tables (e.g., MDL references features.csv
    # but only train.csv and stores.csv were provided)
    for stmt in _view_stmts:
        try:
            conn.execute(stmt)
        except Exception:
            pass  # skip views that reference unregistered tables
    return conn


def _tool_execute_sql(sql: str) -> dict:
    try:
        conn = _make_conn()
        result_df = conn.execute(sql).df()
        rows = result_df.to_dict(orient="records")
        return {
            "success": True,
            "row_count": len(result_df),
            "columns": list(result_df.columns),
            # Send only first 10 rows back to LLM to save tokens
            "sample_rows": rows[:10],
        }
    except Exception as e:
        error_msg = str(e)
        available_tables = list(_dataframes) + list(_csv_paths)

        # Categorize error for actionable LLM feedback
        em = error_msg.lower()
        if "column" in em and ("does not exist" in em or "not found" in em):
            error_type = "column_not_found"
            suggestion = (
                f"The column name is wrong. Run execute_sql('SELECT * FROM {available_tables[0]} LIMIT 1') "
                f"to inspect actual column names." if available_tables else
                "Check column names with SELECT * FROM <table> LIMIT 1."
            )
        elif ("table" in em or "view" in em) and ("does not exist" in em or "not found" in em):
            error_type = "table_not_found"
            suggestion = f"Available tables are: {available_tables}. Use one of these exact names."
        elif "syntax error" in em or "parser error" in em:
            error_type = "syntax_error"
            suggestion = (
                "SQL syntax is invalid. Use double quotes for identifiers, single quotes for strings. "
                "DuckDB does not support TO_CHAR — use strftime('%Y-%m', col) instead."
            )
        elif "cannot be cast" in em or "conversion" in em or "type mismatch" in em:
            error_type = "type_error"
            suggestion = "Type mismatch. Cast explicitly, e.g. CAST(col AS DOUBLE) or TRY_CAST(col AS INTEGER)."
        else:
            error_type = "other"
            suggestion = "Review the SQL and try an alternative approach."

        return {
            "success": False,
            "error": error_msg,
            "error_type": error_type,
            "suggestion": suggestion,
        }


def _tool_run_eda(sql: str) -> dict:
    exec_result = _tool_execute_sql(sql)
    if not exec_result["success"]:
        return exec_result

    conn = _make_conn()
    df = conn.execute(sql).df()

    eda: dict = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "column_types": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": json.loads(df.describe(include="all").fillna("").to_json()),
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        eda["correlations"] = json.loads(df[numeric_cols].corr().round(3).to_json())

    return {"success": True, "eda": eda}


async def _tool_generate_chart(sql: str, filename: str) -> dict:
    exec_result = _tool_execute_sql(sql)
    if not exec_result["success"]:
        return exec_result

    conn = _make_conn()
    df = conn.execute(sql).df()

    sample_data, sample_column_values = preprocess_data(df)

    chart_result = await _generate_chart_fn(
        question=_question,
        sql=sql,
        sample_data=sample_data,
        sample_column_values=sample_column_values,
        model=_model,
    )

    html_path = render_html(
        _question, sql, chart_result, df.to_dict(orient="records"), filename
    )

    return {
        "success": True,
        "chart_type": chart_result["chart_type"],
        "reasoning": chart_result["reasoning"],
        "html_path": str(Path(html_path).resolve()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification (runs before the agent loop)
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_PROMPT = """\
You are a data analyst assistant. Given a database schema and a user question,
decide whether the question can be answered with SQL queries on the available data.

Database Schema:
{schema}

User Question: {question}

Respond ONLY in JSON:
{{
  "answerable": true or false,
  "reason": "1-2 sentence explanation",
  "tables_needed": ["table1", "table2"]
}}

If answerable is false, briefly state what data would be needed."""


async def classify_intent(
    question: str,
    schema: str,
    model: str,
    api_key: str | None,
) -> dict:
    """
    Pre-flight check: ask the LLM whether the question is answerable
    given the schema, before spinning up the full agent loop.

    Returns dict with keys: answerable (bool), reason (str), tables_needed (list).
    """
    from litellm import acompletion

    prompt = _INTENT_PROMPT.format(schema=schema.strip(), question=question.strip())
    response = await acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        api_key=api_key,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # Fallback: assume answerable if parsing fails
        return {"answerable": True, "reason": "(classification parse error)", "tables_needed": []}


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_SYSTEM_PROMPT = """
You are a data analyst agent. Your goal is to answer the user's question
about their data by writing SQL, analyzing results, and generating charts.

You have three tools:
1. execute_sql  — run a SQL query; if it fails, fix the SQL and retry
2. run_eda      — get statistics and correlations (use when the user asks
                  for analysis or when you need to understand the data)
3. generate_chart — create a Vega-Lite HTML chart (always do this last,
                    once you have a correct SQL query)

Workflow:
  1. Read the DATABASE SCHEMA carefully — comments contain business descriptions,
     column aliases, relationships, calculated fields, and metric definitions.
     Use these to understand the business meaning of each table and column.
  2. If SQL EXAMPLES are provided, reference them for correct syntax patterns.
  3. If a "sales_fact" view exists, PREFER using it — it pre-joins all tables
     with cleaned data (NAs handled, types cast). You can also query raw tables
     (train, stores, features) directly when needed.
  4. Use METRICS definitions to write correct aggregations — they tell you the
     exact expressions for business KPIs (e.g., holiday-weighted sales).
  5. Write SQL to answer the question → execute_sql
  6. If SQL fails → read the error, fix the SQL → execute_sql again
  7. If the question asks for analysis → run_eda
  8. Generate a chart → generate_chart
  9. Summarize findings in plain language

Rules:
  - Database engine is DuckDB. Use strftime('%Y-%m', col) for month formatting.
    Do NOT use TO_CHAR.
  - Always end with generate_chart unless the data is clearly not visualizable.
  - Name the output file based on the question (e.g. 'store_sales.html').
  - Pay attention to column descriptions — they tell you the business meaning
    and data caveats (e.g., "MarkDown1: NA before Nov 2011, use COALESCE").
  - Use STAR SCHEMA JOINS info to write correct JOINs when querying raw tables.
  - DERIVED FIELDS (like Active_Promotion, Total_MarkDown) are available in
    the sales_fact view — use them directly instead of recomputing.
"""


async def run_agent(
    question: str,
    dataframes: dict[str, pd.DataFrame] | None = None,
    schema: str = "",
    csv_paths: dict[str, str] | None = None,
    mdl: dict | None = None,
    sql_pairs_store: SqlPairsStore | None = None,
    model: str = "deepseek/deepseek-chat",
    max_iterations: int = 10,
    open_browser: bool = True,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Run the data analysis agent.

    Supply either:
      dataframes  — dict of table_name → pd.DataFrame  (small/medium data)
      csv_paths   — dict of table_name → CSV file path  (large files, lazy loading)

    Optional semantic layer:
      mdl              — MDL dict (from load_mdl). Overrides schema with annotated DDL.
      sql_pairs_store  — SqlPairsStore instance for few-shot SQL example retrieval.

    Multi-turn:
      conversation_history — pass the 'messages' from a previous run_agent() call
                             to continue the conversation. The new question is appended.

    The LLM decides which tools to call (execute_sql / run_eda / generate_chart)
    and in what order, including automatic SQL error correction.

    Returns dict with 'answer' (final text) and 'messages' (full history).
    """
    global _dataframes, _csv_paths, _view_stmts, _question, _model
    _dataframes = dataframes or {}
    _csv_paths = csv_paths or {}
    _question = question
    _model = model

    from litellm import acompletion

    api_key = os.getenv("DEEPSEEK_API_KEY")

    # ── First turn: build schema context ──
    if conversation_history is None:
        _view_stmts = []

        # ── MDL → annotated DDL + executable views ──
        if mdl:
            relevant_tables = await retrieve_relevant_tables(question, mdl, model=model)
            filtered_mdl = filter_mdl_by_tables(mdl, relevant_tables)
            schema = mdl_to_ddl(filtered_mdl)

            _view_stmts = mdl_to_views(mdl)
            if _view_stmts:
                print(f"[MDL] Created {len(_view_stmts)} DuckDB views "
                      f"(including 'sales_fact' star schema view)")

            n_dims = len(filtered_mdl.get('dimensions', []))
            n_models = len(filtered_mdl.get('models', []))
            n_metrics = len(filtered_mdl.get('metrics', []))
            print(f"[MDL] Semantic layer: {n_models} models, {n_dims} dimensions, "
                  f"{n_metrics} metrics")

        # ── SQL pairs → few-shot examples via embedding retrieval ──
        sql_examples_prompt = ""
        if sql_pairs_store:
            examples = await sql_pairs_store.retrieve(question, top_k=3)
            sql_examples_prompt = sql_pairs_store.format_as_prompt(examples)
            if examples:
                print(f"[RAG] Retrieved {len(examples)} SQL examples "
                      f"(scores: {[e['score'] for e in examples]})")

        # ── Build initial messages ──
        user_content_parts = [f"### Database Schema\n{schema}"]
        if sql_examples_prompt:
            user_content_parts.append(sql_examples_prompt)
        user_content_parts.append(f"### Question\n{question}")

        messages = [
            {"role": "system", "content": _AGENT_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": "\n\n".join(user_content_parts)},
        ]
    else:
        # ── Follow-up turn: append new question to existing history ──
        messages = list(conversation_history)

        # RAG for follow-up question too
        if sql_pairs_store:
            examples = await sql_pairs_store.retrieve(question, top_k=2)
            sql_hint = sql_pairs_store.format_as_prompt(examples)
            if sql_hint:
                question = f"{question}\n\n{sql_hint}"

        messages.append({"role": "user", "content": question})

    print(f"\n{'='*65}")
    print(f"  Agent started: {_question!r}")
    print(f"{'='*65}")

    html_path = None

    for iteration in range(max_iterations):
        print(f"\n[Iteration {iteration + 1}] Calling LLM...")

        response = await acompletion(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            api_key=api_key,
        )

        msg = response.choices[0].message

        # Append assistant message (handle both tool_calls and plain text)
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls → agent is done
        if not msg.tool_calls:
            print(f"\n[Agent Final Answer]\n{msg.content}")
            break

        # Execute each tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            # Pretty-print what the agent is doing
            args_display = {k: v for k, v in fn_args.items() if k != "sql"}
            print(f"\n  → Tool: {fn_name}({args_display})")
            if "sql" in fn_args:
                sql_preview = fn_args["sql"].replace("\n", " ")[:100]
                print(f"     SQL: {sql_preview}...")

            # Dispatch
            if fn_name == "execute_sql":
                tool_result = _tool_execute_sql(**fn_args)
                if tool_result["success"]:
                    print(f"     OK : {tool_result['row_count']} rows, "
                          f"cols={tool_result['columns']}")
                else:
                    print(f"     ERR: {tool_result['error']}")

            elif fn_name == "run_eda":
                tool_result = _tool_run_eda(**fn_args)
                if tool_result["success"]:
                    shape = tool_result["eda"]["shape"]
                    print(f"     OK : EDA done ({shape['rows']}×{shape['columns']})")
                else:
                    print(f"     ERR: {tool_result['error']}")

            elif fn_name == "generate_chart":
                tool_result = await _tool_generate_chart(**fn_args)
                if tool_result["success"]:
                    html_path = tool_result["html_path"]
                    print(f"     OK : {tool_result['chart_type']} chart → {html_path}")
                else:
                    print(f"     ERR: {tool_result['error']}")

            else:
                tool_result = {"error": f"Unknown tool: {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, ensure_ascii=False, default=str),
            })

    if html_path and open_browser:
        webbrowser.open(f"file://{html_path}")

    final_answer = next(
        (m["content"] for m in reversed(messages)
         if m["role"] == "assistant" and m.get("content")),
        "(no final message)",
    )
    return {
        "answer": final_answer,
        "html_path": html_path,
        "iterations": iteration + 1,
        "messages": messages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_QUESTIONS = [
    "How many orders are there for each status? Show me a chart.",
    "Who are the top customers by total spending? Analyze and visualize.",
    "Show the monthly revenue trend. Include EDA before charting.",
]


async def _demo(model: str) -> None:
    # Load MDL semantic layer and SQL pairs for demo
    demo_dir = Path(__file__).parent.parent / "examples"
    mdl_path = demo_dir / "demo_mdl.json"
    pairs_path = demo_dir / "demo_sql_pairs.json"

    mdl = load_mdl(str(mdl_path)) if mdl_path.exists() else None
    sql_pairs_store = None
    if pairs_path.exists():
        sql_pairs_store = SqlPairsStore.from_file(str(pairs_path))
        await sql_pairs_store.build_index()

    if mdl:
        print("[Demo] Using MDL semantic layer + SQL pairs RAG")
    else:
        print("[Demo] MDL/SQL pairs files not found, using raw schema")

    for question in _DEMO_QUESTIONS:
        await run_agent(
            question=question,
            dataframes=_DEMO_DATAFRAMES,
            schema=_DEMO_SCHEMA,
            mdl=mdl,
            sql_pairs_store=sql_pairs_store,
            model=model,
            open_browser=False,
        )
        print()
    # Open last chart
    await run_agent(
        question=_DEMO_QUESTIONS[-1],
        dataframes=_DEMO_DATAFRAMES,
        schema=_DEMO_SCHEMA,
        mdl=mdl,
        sql_pairs_store=sql_pairs_store,
        model=model,
        open_browser=True,
    )


def _infer_schema(table: str, abs_path: str) -> str:
    """Auto-generate DDL for a CSV file using DuckDB's type inference."""
    conn = duckdb.connect()
    conn.execute(f"CREATE VIEW _tmp_{table} AS SELECT * FROM read_csv_auto('{abs_path}')")
    col_info = conn.execute(
        f"SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_name = '_tmp_{table}'"
    ).fetchall()
    cols_ddl = ",\n  ".join(f'"{name}" {dtype}' for name, dtype in col_info)
    return f"CREATE TABLE {table} (\n  {cols_ddl}\n);"


def _setup_data(
    csv_specs: list[str],
    mdl_path: str | None = None,
) -> tuple[dict[str, str], str, dict | None]:
    """
    Parse CSV specs, infer schema, load MDL. Returns (csv_paths, schema, mdl).
    """
    mdl = None
    if mdl_path:
        mdl = load_mdl(mdl_path)
        print(f"[MDL] Loaded semantic layer from {mdl_path}")

    csv_paths: dict[str, str] = {}
    schema = ""

    if csv_specs:
        for spec in csv_specs:
            if "=" in spec:
                table, path = spec.split("=", 1)
            else:
                path = spec
                table = Path(path).stem
            abs_path = str(Path(path).resolve())
            csv_paths[table] = abs_path
            print(f"Table '{table}': {abs_path}")

        schema_parts = []
        for table, abs_path in csv_paths.items():
            ddl = _infer_schema(table, abs_path)
            schema_parts.append(ddl)
            print(f"Inferred schema for '{table}':\n{ddl}\n")

        schema = "\n\n".join(schema_parts)

    return csv_paths, schema, mdl


async def _single(
    question: str,
    csv_specs: list[str],
    model: str,
    mdl_path: str | None = None,
    sql_pairs_path: str | None = None,
) -> None:
    """Single question mode."""
    csv_paths, schema, mdl = _setup_data(csv_specs, mdl_path)

    sql_pairs_store = None
    if sql_pairs_path:
        sql_pairs_store = SqlPairsStore.from_file(sql_pairs_path)
        await sql_pairs_store.build_index()

    if csv_paths:
        await run_agent(
            question=question,
            csv_paths=csv_paths,
            schema=schema,
            mdl=mdl,
            sql_pairs_store=sql_pairs_store,
            model=model,
        )
    else:
        await run_agent(
            question=question,
            dataframes=_DEMO_DATAFRAMES,
            schema=_DEMO_SCHEMA,
            mdl=mdl,
            sql_pairs_store=sql_pairs_store,
            model=model,
        )


async def _chat(
    csv_specs: list[str],
    model: str,
    mdl_path: str | None = None,
    sql_pairs_path: str | None = None,
) -> None:
    """Interactive multi-turn chat mode."""
    csv_paths, schema, mdl = _setup_data(csv_specs, mdl_path)

    sql_pairs_store = None
    if sql_pairs_path:
        sql_pairs_store = SqlPairsStore.from_file(sql_pairs_path)
        await sql_pairs_store.build_index()

    use_csv = bool(csv_paths)
    history: list[dict] | None = None

    print("\n" + "=" * 65)
    print("  Interactive Chat Mode (multi-turn)")
    print("  Type your question, or 'quit'/'exit' to stop.")
    print("  Type 'reset' to clear conversation history.")
    print("=" * 65)

    while True:
        try:
            question = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if question.lower() == "reset":
            history = None
            print("[Chat] Conversation history cleared.")
            continue

        result = await run_agent(
            question=question,
            csv_paths=csv_paths if use_csv else None,
            dataframes=None if use_csv else _DEMO_DATAFRAMES,
            schema=schema if use_csv else _DEMO_SCHEMA,
            mdl=mdl,
            sql_pairs_store=sql_pairs_store,
            model=model,
            open_browser=True,
            conversation_history=history,
        )

        history = result["messages"]
        msg_count = sum(1 for m in history if m["role"] == "user")
        print(f"\n[Chat] Turn {msg_count} complete. "
              f"History: {len(history)} messages.")


def _parse_args(argv: list[str]) -> tuple[str | None, list[str], str | None, str | None, bool]:
    """
    Parse CLI args, extracting --mdl, --sql-pairs, and --chat flags.
    Returns (question, csv_specs, mdl_path, sql_pairs_path, chat_mode).

    In --chat mode there is no question arg — all positional args are csv_specs.
    In normal mode the first positional arg is the question, rest are csv_specs.
    """
    question = None
    csv_specs = []
    mdl_path = None
    sql_pairs_path = None
    chat_mode = False

    # First pass: extract flags
    positional = []
    i = 1  # skip argv[0]
    while i < len(argv):
        arg = argv[i]
        if arg == "--mdl" and i + 1 < len(argv):
            mdl_path = argv[i + 1]
            i += 2
        elif arg == "--sql-pairs" and i + 1 < len(argv):
            sql_pairs_path = argv[i + 1]
            i += 2
        elif arg == "--chat":
            chat_mode = True
            i += 1
        else:
            positional.append(arg)
            i += 1

    # Second pass: assign positional args
    if chat_mode:
        # In chat mode, all positional args are CSV specs (no question)
        csv_specs = positional
    else:
        # In normal mode, first positional is question, rest are CSV specs
        if positional:
            question = positional[0]
            csv_specs = positional[1:]

    return question, csv_specs, mdl_path, sql_pairs_path, chat_mode


def main() -> None:
    """
    Usage:
      python data_agent.py                                          # built-in demo
      python data_agent.py "question"                               # demo data
      python data_agent.py "question" file.csv                      # one CSV
      python data_agent.py "question" a.csv b.csv                   # multiple CSVs
      python data_agent.py "question" a.csv --mdl schema.json       # with MDL
      python data_agent.py "question" a.csv --mdl s.json --sql-pairs p.json
      python data_agent.py --chat a.csv --mdl s.json                # interactive multi-turn
      python data_agent.py --chat a.csv --mdl s.json --sql-pairs p.json
    """
    model = os.getenv("TEXT_TO_SQL_MODEL", "deepseek/deepseek-chat")

    if len(sys.argv) == 1:
        asyncio.run(_demo(model))
    else:
        question, csv_specs, mdl_path, sql_pairs_path, chat_mode = _parse_args(sys.argv)
        if chat_mode:
            asyncio.run(_chat(csv_specs, model, mdl_path, sql_pairs_path))
        else:
            asyncio.run(_single(question, csv_specs, model, mdl_path, sql_pairs_path))


if __name__ == "__main__":
    main()
