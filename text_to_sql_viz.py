#!/usr/bin/env python3
"""
Text → SQL → Data → Visualization pipeline.

Reproduces WrenAI's core ask + chart generation flow as a standalone script.

Pipeline steps
──────────────
  1. Text → SQL          LLM generates SQL from question + DDL schema
                         (prompt from sql_generation.py / utils/sql.py)
  2. SQL → Data          DuckDB executes the SQL against CSV / DataFrame
  3. Data Preprocessing  Sample 15 rows + 5 unique values per column
                         (logic from chart.py :: ChartDataPreprocessor)
  4. Chart Generation    LLM picks chart type + produces Vega-Lite schema
                         (prompt from chart_generation.py / utils/chart.py)
  5. Render              Writes a self-contained HTML file (Vega-Embed)

Requirements
────────────
  pip install litellm duckdb pandas
  export DEEPSEEK_API_KEY=sk-...

Usage
─────
  python text_to_sql_viz.py                         # built-in demo
  python text_to_sql_viz.py "question" data.csv     # your own CSV
  python text_to_sql_viz.py "question" data.csv schema.sql
"""

import asyncio
import json
import os
import sys
import textwrap
import webbrowser
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SQL generation prompts
# Source: wren-ai-service/src/pipelines/generation/sql_generation.py
#         wren-ai-service/src/pipelines/generation/utils/sql.py
# ─────────────────────────────────────────────────────────────────────────────

_SQL_RULES = """
### SQL RULES ###
- ONLY USE SELECT statements.
- ONLY USE the tables and columns mentioned in the database schema.
- DON'T INCLUDE comments in the generated SQL query.
- YOU MUST USE "JOIN" if you choose columns from multiple tables!
- PREFER USING CTEs over subqueries.
- Put double quotes around column and table names, single quotes around strings.
- Aggregate functions are not allowed in WHERE; use HAVING.
- For ranking, use DENSE_RANK() then filter with WHERE.
- The database is DuckDB. Use DuckDB-compatible date functions:
    - Use strftime('%Y-%m', col) instead of TO_CHAR(col, 'YYYY-MM')
    - Use DATE_TRUNC('month', col) for month truncation
    - DON'T USE TO_CHAR function.
"""

SQL_SYSTEM_PROMPT = f"""
You are a helpful assistant that converts natural language queries into ANSI SQL.
Think carefully and generate SQL based on the schema and question.

{_SQL_RULES}

### FINAL ANSWER FORMAT ###
Return only JSON:
{{"sql": "<SQL_QUERY_STRING>"}}
"""

SQL_USER_TEMPLATE = """
### DATABASE SCHEMA ###
{schema}

### QUESTION ###
{question}

Let's think step by step.
"""

_SQL_RESPONSE_FORMAT = {"type": "json_object"}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Chart generation prompts
# Source: wren-ai-service/src/pipelines/generation/chart_generation.py
#         wren-ai-service/src/pipelines/generation/utils/chart.py
# ─────────────────────────────────────────────────────────────────────────────

_CHART_INSTRUCTIONS = """
### INSTRUCTIONS ###
- Supported chart types: bar, line, multi_line, area, pie, grouped_bar, stacked_bar, dual_axis
- Choose the chart type that best answers the user's question given the data.
- If data is empty or unsuitable for visualization, return empty strings.
- For grouped_bar: use xOffset encoding, add "stack": null to y-encoding.
- For pie: use mark {"type":"arc"}, add theta and color encodings.
- For temporal x-axis: set timeUnit (year/yearmonth/yearmonthdate) matching the question.
- Each axis must have a human-readable title in the user's language.
- All fields in encoding must exist in the column names of the data.

### CHART TYPE DECISION RULES (follow in order) ###

STEP 1 — Classify the x-axis column:
  - TEMPORAL: the column is a date, datetime, month, year, or continuous time series
    with many sequential values (>= 8 distinct ordered time points)
  - CATEGORICAL: the column is a string label, type code, status, boolean, or a small
    set of discrete values (< 8 distinct values), even if they happen to be numbers
    (e.g., store IDs 1-5, department IDs, type codes A/B/C)

STEP 2 — Check if dual_axis is needed (BEFORE picking chart type):
  Look at the Column Info for "scale_ratio". If scale_ratio > 10 between two
  quantitative columns, their magnitudes are very different (e.g., total_sales in
  millions vs store_count in tens, or revenue in USD vs growth_rate in %).
  When you have 2 quantitative columns with scale_ratio > 10:
    → Use dual_axis: primary metric as bar (left Y), secondary metric as line (right Y).

STEP 3 — Pick chart type based on classification:
  If x-axis is CATEGORICAL:
    - 1 categorical + 1 quantitative               → bar
    - 2 categorical + 1 quantitative (sub-groups)   → grouped_bar
    - composition / parts of a whole within groups   → stacked_bar
    - proportions summing to 100% (few categories)   → pie
    - 2 quantitative with large scale difference     → dual_axis
  If x-axis is TEMPORAL:
    - 1 time series + 1 quantitative                → line
    - multiple metrics, similar scale over time      → multi_line
    - multiple metrics, different scale over time    → dual_axis
    - emphasize cumulative volume over time          → area

### dual_axis CHART FORMAT ###
Use Vega-Lite "layer" with two layers + resolve for independent Y scales:
{
  "layer": [
    {
      "mark": {"type": "bar", "color": "#6366f1", "opacity": 0.8},
      "encoding": {
        "x": {"field": "<x_field>", "type": "<nominal|temporal>", "title": "<X Title>"},
        "y": {"field": "<primary_metric>", "type": "quantitative", "title": "<Left Y Title>", "axis": {"titleColor": "#6366f1"}}
      }
    },
    {
      "mark": {"type": "line", "color": "#ef4444", "strokeWidth": 2.5, "point": {"color": "#ef4444", "size": 50}},
      "encoding": {
        "x": {"field": "<x_field>", "type": "<nominal|temporal>"},
        "y": {"field": "<secondary_metric>", "type": "quantitative", "title": "<Right Y Title>", "axis": {"titleColor": "#ef4444"}}
      }
    }
  ],
  "resolve": {"scale": {"y": "independent"}}
}
- The PRIMARY metric (larger values) goes on the LEFT Y-axis as bar.
- The SECONDARY metric (smaller values or rate/ratio) goes on the RIGHT Y-axis as line.
- Use contrasting colors and add titleColor to each y-axis so readers know which axis belongs to which metric.

### COMMON MISTAKES TO AVOID ###
- Do NOT use line chart for categorical comparisons (e.g., store types A/B/C, status
  labels, city names). Line implies continuous ordered progression between points.
- Do NOT use line just because there are numbers on the x-axis. Store ID, department
  number, or rank are categorical, not temporal.
- When in doubt between bar and line: if the x-axis values could be reordered without
  losing meaning, use bar. If order matters (time), use line.
- Do NOT use pie when there are more than 6 categories — use bar instead.
- Do NOT put two metrics with very different scales on the same Y-axis (e.g., revenue
  in millions + count in tens). The smaller metric becomes invisible. Use dual_axis.
- Do NOT use multi_line or grouped_bar when one metric is 10x+ larger than another.
  Use dual_axis instead so both metrics are readable.
"""

CHART_SYSTEM_PROMPT = f"""
You are a data analyst expert at visualizing data with Vega-Lite.
Given a question, SQL, sample data and sample column values, generate a Vega-Lite schema.

{_CHART_INSTRUCTIONS}

### OUTPUT FORMAT ###
Return only JSON:
{{
    "reasoning": "<why this chart type fits the data and question>",
    "chart_type": "line" | "multi_line" | "bar" | "pie" | "grouped_bar" | "stacked_bar" | "area" | "dual_axis" | "",
    "chart_schema": <VEGA_LITE_JSON_SCHEMA_WITHOUT_data_AND_$schema_FIELDS>
}}
"""

CHART_USER_TEMPLATE = """
Question: {question}
SQL: {sql}
Sample Data: {sample_data}
Column Info (type: temporal/categorical/quantitative, n_unique, abs_mean/min/max for numbers, _scale_ratios between numeric pairs):
{sample_column_values}
Language: {language}

Steps:
1. Classify each column (temporal/categorical/quantitative)
2. Check _scale_ratios — if any ratio > 10, consider dual_axis
3. Pick chart type following the decision rules
Think step by step.
"""

_CHART_RESPONSE_FORMAT = {"type": "json_object"}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — generate_sql()
# ─────────────────────────────────────────────────────────────────────────────


async def generate_sql(
    question: str,
    schema: str,
    model: str = "deepseek/deepseek-chat",
) -> str:
    """Call LLM to convert question + DDL schema → SQL string."""
    from litellm import acompletion

    api_key = os.getenv("DEEPSEEK_API_KEY")
    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": SQL_USER_TEMPLATE.format(
                schema=schema.strip(), question=question.strip()
            )},
        ],
        response_format=_SQL_RESPONSE_FORMAT,
        api_key=api_key,
    )
    return json.loads(response.choices[0].message.content)["sql"]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — execute_sql()
# ─────────────────────────────────────────────────────────────────────────────


def execute_sql(sql: str, dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Execute SQL via DuckDB against a dict of named DataFrames.
    The SQL can reference any DataFrame by its key as a table name.

    Example:
        execute_sql("SELECT * FROM orders LIMIT 5",
                    {"orders": orders_df, "customers": customers_df})
    """
    conn = duckdb.connect()
    for name, df in dataframes.items():
        conn.register(name, df)
    return conn.execute(sql).df()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — preprocess_data()
# Source: ChartDataPreprocessor in utils/chart.py
# ─────────────────────────────────────────────────────────────────────────────


def _classify_column(series: pd.Series) -> str:
    """Classify a pandas Series as temporal, quantitative, or categorical."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"
    # Check if string column looks like dates
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        sample = series.dropna().head(5)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, format="mixed")
                n_unique = series.nunique()
                if n_unique >= 6:
                    return "temporal"
            except (ValueError, TypeError):
                pass
        return "categorical"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    # Numeric: categorical if few distinct values, else quantitative
    n_unique = series.nunique()
    if n_unique <= 8:
        return "categorical"
    return "quantitative"


def preprocess_data(
    df: pd.DataFrame,
    sample_data_count: int = 15,
    sample_column_size: int = 5,
) -> tuple[list[dict], dict]:
    """
    Prepare data for the chart LLM:
      - sample_data          : up to 15 rows as list-of-dicts
      - sample_column_values : up to 5 unique values per column + type hints
                               + scale info for quantitative columns
    """
    sample_column_values = {}
    for col in df.columns:
        col_type = _classify_column(df[col])
        unique_vals = list(df[col].dropna().unique())[:sample_column_size]
        info: dict = {
            "type": col_type,
            "n_unique": int(df[col].nunique()),
            "sample_values": unique_vals,
        }
        # Add magnitude info for quantitative columns so LLM can detect scale gaps
        if col_type == "quantitative":
            abs_mean = df[col].abs().mean()
            info["abs_mean"] = round(float(abs_mean), 2) if abs_mean == abs_mean else 0
            info["min"] = round(float(df[col].min()), 2)
            info["max"] = round(float(df[col].max()), 2)
        sample_column_values[col] = info

    # Compute pairwise scale ratios between quantitative columns
    quant_cols = [c for c, v in sample_column_values.items() if v["type"] == "quantitative"]
    if len(quant_cols) >= 2:
        scale_ratios = {}
        for i, a in enumerate(quant_cols):
            for b in quant_cols[i + 1:]:
                mean_a = sample_column_values[a]["abs_mean"]
                mean_b = sample_column_values[b]["abs_mean"]
                if mean_a > 0 and mean_b > 0:
                    ratio = max(mean_a, mean_b) / min(mean_a, mean_b)
                    scale_ratios[f"{a} vs {b}"] = round(ratio, 1)
        if scale_ratios:
            sample_column_values["_scale_ratios"] = scale_ratios

    if len(df) > sample_data_count:
        sample_data = df.sample(n=sample_data_count, random_state=42).to_dict(
            orient="records"
        )
    else:
        sample_data = df.to_dict(orient="records")

    return sample_data, sample_column_values


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — generate_chart()
# ─────────────────────────────────────────────────────────────────────────────


async def generate_chart(
    question: str,
    sql: str,
    sample_data: list[dict],
    sample_column_values: dict,
    language: str = "English",
    model: str = "deepseek/deepseek-chat",
) -> dict:
    """
    Call LLM to produce a Vega-Lite chart schema from question + data samples.
    Returns dict with keys: reasoning, chart_type, chart_schema.
    """
    from litellm import acompletion

    api_key = os.getenv("DEEPSEEK_API_KEY")
    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": CHART_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": CHART_USER_TEMPLATE.format(
                question=question,
                sql=sql,
                sample_data=json.dumps(sample_data[:15], ensure_ascii=False, default=str),
                sample_column_values=json.dumps(
                    sample_column_values, ensure_ascii=False, default=str
                ),
                language=language,
            )},
        ],
        response_format=_CHART_RESPONSE_FORMAT,
        api_key=api_key,
    )

    result = json.loads(response.choices[0].message.content)
    chart_schema = result.get("chart_schema", {})

    # Allow chart_schema to come back as a JSON string
    if isinstance(chart_schema, str) and chart_schema:
        chart_schema = json.loads(chart_schema)

    return {
        "reasoning": result.get("reasoning", ""),
        "chart_type": result.get("chart_type", ""),
        "chart_schema": chart_schema,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — build_vega_spec() + render_html()
# ─────────────────────────────────────────────────────────────────────────────


def build_vega_spec(chart_schema: dict, data: list[dict]) -> dict:
    """
    Inject $schema, data, sizing, and a clean visual theme into the Vega-Lite schema.
    Source: ChartGenerationPostProcessor in utils/chart.py
    """
    spec = dict(chart_schema)
    spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
    spec["data"] = {"values": data}
    spec["width"] = "container"
    spec["height"] = 380

    # Ensure dual_axis (layered) charts have independent Y scales
    if "layer" in spec:
        spec.setdefault("resolve", {}).setdefault("scale", {})["y"] = "independent"

    spec["config"] = {
        "background": "#ffffff",
        "font": "Inter, system-ui, sans-serif",
        "title": {
            "fontSize": 16,
            "fontWeight": 600,
            "color": "#111827",
            "offset": 16,
        },
        "axis": {
            "labelFontSize": 12,
            "titleFontSize": 13,
            "titleColor": "#6b7280",
            "labelColor": "#374151",
            "gridColor": "#f3f4f6",
            "domainColor": "#e5e7eb",
            "tickColor": "#e5e7eb",
        },
        "legend": {
            "labelFontSize": 12,
            "titleFontSize": 12,
            "titleColor": "#6b7280",
            "labelColor": "#374151",
        },
        "range": {
            "category": [
                "#6366f1", "#f59e0b", "#10b981", "#ef4444",
                "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6",
            ]
        },
        "bar": {"cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
        "view": {"stroke": "transparent"},
    }
    return spec


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Inter', system-ui, sans-serif;
      background: #f8fafc;
      color: #111827;
      min-height: 100vh;
      padding: 32px 24px;
    }}

    .page {{
      max-width: 1100px;
      margin: 0 auto;
    }}

    /* ── Header ── */
    .header {{
      margin-bottom: 28px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #ede9fe;
      color: #6d28d9;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: .05em;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 999px;
      margin-bottom: 12px;
    }}
    .badge svg {{ width:12px; height:12px; }}
    h1 {{
      font-size: 22px;
      font-weight: 700;
      color: #111827;
      line-height: 1.3;
    }}

    /* ── Cards ── */
    .card {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,.05);
    }}
    .card-title {{
      font-size: 11px;
      font-weight: 600;
      letter-spacing: .07em;
      text-transform: uppercase;
      color: #9ca3af;
      margin-bottom: 14px;
    }}

    /* ── Chart container ── */
    #chart-wrap {{
      width: 100%;
      min-height: 420px;
    }}

    /* ── SQL block ── */
    .sql-block {{
      background: #0f172a;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
      font-size: 13px;
      line-height: 1.7;
      border-radius: 8px;
      padding: 18px 20px;
      overflow-x: auto;
      white-space: pre;
    }}
    /* minimal SQL keyword colouring via CSS */
    .sql-block {{ position: relative; }}

    /* ── Reasoning box ── */
    .reasoning {{
      background: #f0fdf4;
      border-left: 3px solid #10b981;
      border-radius: 0 8px 8px 0;
      padding: 14px 16px;
      font-size: 14px;
      color: #065f46;
      line-height: 1.6;
    }}

    /* ── Two-column meta row ── */
    .meta-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 640px) {{
      .meta-row {{ grid-template-columns: 1fr; }}
    }}
    .meta-item {{ font-size: 13px; color: #374151; }}
    .meta-item strong {{ display: block; color: #6b7280; font-size: 11px;
                         font-weight: 600; text-transform: uppercase;
                         letter-spacing: .06em; margin-bottom: 4px; }}
    .chip {{
      display: inline-block;
      background: #ede9fe;
      color: #5b21b6;
      font-size: 12px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 999px;
    }}

    /* ── Footer ── */
    .footer {{
      text-align: center;
      font-size: 12px;
      color: #9ca3af;
      margin-top: 32px;
    }}
  </style>
</head>
<body>
  <div class="page">

    <div class="header">
      <div class="badge">
        <svg viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1.5a6.5 6.5 0 1 0 0 13A6.5 6.5 0 0 0 8 1.5zM0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8z"/>
          <path d="M6.5 4.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0zM6 7.5a.5.5 0 0 1 .5-.5h1a.5.5 0 0 1 .5.5V11h.5a.5.5 0 0 1 0 1h-2a.5.5 0 0 1 0-1H7V8h-.5a.5.5 0 0 1-.5-.5z"/>
        </svg>
        Data Analysis
      </div>
      <h1>{title}</h1>
    </div>

    <!-- Chart -->
    <div class="card">
      <div class="card-title">Visualization</div>
      <div id="chart-wrap"></div>
    </div>

    <!-- Meta row: question + chart type -->
    <div class="meta-row">
      <div class="card">
        <div class="card-title">Question</div>
        <div class="meta-item">{question}</div>
      </div>
      <div class="card">
        <div class="card-title">Chart Type</div>
        <div class="meta-item"><span class="chip">{chart_type}</span></div>
      </div>
    </div>

    <!-- Reasoning -->
    <div class="card">
      <div class="card-title">Why this chart?</div>
      <div class="reasoning">{reasoning}</div>
    </div>

    <!-- SQL -->
    <div class="card">
      <div class="card-title">Generated SQL</div>
      <div class="sql-block">{sql}</div>
    </div>

    <div class="footer">Generated by Data Agent · Powered by DeepSeek + Vega-Lite</div>

  </div>
  <script>
    vegaEmbed('#chart-wrap', {spec}, {{
      actions: {{ export: true, source: false, compiled: false, editor: false }},
      renderer: 'svg',
      theme: 'none',
    }}).catch(console.error);
  </script>
</body>
</html>
"""


def render_html(
    question: str,
    sql: str,
    chart_result: dict,
    all_data: list[dict],
    output_path: str = "chart_output.html",
) -> str:
    """Write a self-contained HTML file and return its path."""
    spec = build_vega_spec(chart_result["chart_schema"], all_data)
    title = spec.get("title", question[:60])

    html = _HTML_TEMPLATE.format(
        title=title,
        question=question,
        sql=sql.strip(),
        chart_type=chart_result.get("chart_type", "—"),
        reasoning=chart_result.get("reasoning", "—"),
        spec=json.dumps(spec, ensure_ascii=False, indent=2, default=str),
    )

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────


async def run_pipeline(
    question: str,
    dataframes: dict[str, pd.DataFrame],
    schema: str,
    output_path: str = "chart_output.html",
    model: str = "deepseek/deepseek-chat",
    language: str = "English",
    open_browser: bool = True,
) -> dict:
    """
    Run the full Text → SQL → Data → Visualization pipeline.

    Args:
        question    : natural language question
        dataframes  : dict of table_name → DataFrame (used as DuckDB tables)
        schema      : DDL string describing the tables
        output_path : where to write the HTML output
        model       : LiteLLM model string
        language    : chart label language

    Returns:
        dict with sql, dataframe, chart_result, html_path
    """
    print(f"[1/4] Generating SQL for: {question!r}")
    sql = await generate_sql(question, schema, model=model)
    print(f"      SQL: {sql}\n")

    print("[2/4] Executing SQL with DuckDB...")
    df = execute_sql(sql, dataframes)
    print(f"      Rows returned: {len(df)}")
    print(f"      Columns: {list(df.columns)}\n")

    print("[3/4] Preprocessing data for chart LLM...")
    sample_data, sample_column_values = preprocess_data(df)
    print(f"      Sample rows: {len(sample_data)}, "
          f"Columns with unique values: {len(sample_column_values)}\n")

    print("[4/4] Generating Vega-Lite chart schema...")
    chart_result = await generate_chart(
        question=question,
        sql=sql,
        sample_data=sample_data,
        sample_column_values=sample_column_values,
        language=language,
        model=model,
    )
    print(f"      Chart type : {chart_result['chart_type']}")
    print(f"      Reasoning  : {textwrap.shorten(chart_result['reasoning'], 120)}\n")

    all_data = df.to_dict(orient="records")
    html_path = render_html(question, sql, chart_result, all_data, output_path)
    print(f"Chart saved to: {html_path}")

    if open_browser:
        webbrowser.open(f"file://{Path(html_path).resolve()}")

    return {
        "sql": sql,
        "dataframe": df,
        "chart_result": chart_result,
        "html_path": html_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo data
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_CUSTOMERS = pd.DataFrame([
    (1, "Alice",   "alice@example.com",  "New York"),
    (2, "Bob",     "bob@example.com",    "San Francisco"),
    (3, "Carol",   "carol@example.com",  "New York"),
    (4, "Dave",    "dave@example.com",   "Chicago"),
    (5, "Eve",     "eve@example.com",    "San Francisco"),
], columns=["customer_id", "name", "email", "city"])

_DEMO_ORDERS = pd.DataFrame([
    (101, 1, 250.0,  "delivered",  "2024-01-10"),
    (102, 2, 89.5,   "shipped",    "2024-01-15"),
    (103, 1, 430.0,  "delivered",  "2024-02-03"),
    (104, 3, 120.0,  "cancelled",  "2024-02-10"),
    (105, 4, 310.0,  "delivered",  "2024-02-20"),
    (106, 5, 540.0,  "shipped",    "2024-03-01"),
    (107, 2, 75.0,   "delivered",  "2024-03-05"),
    (108, 3, 199.0,  "pending",    "2024-03-10"),
    (109, 1, 620.0,  "delivered",  "2024-03-15"),
    (110, 4, 88.0,   "shipped",    "2024-03-20"),
    (111, 5, 450.0,  "delivered",  "2024-04-01"),
    (112, 2, 330.0,  "delivered",  "2024-04-10"),
], columns=["order_id", "customer_id", "total_amount", "status", "ordered_at"])
_DEMO_ORDERS["ordered_at"] = pd.to_datetime(_DEMO_ORDERS["ordered_at"])

_DEMO_DATAFRAMES = {
    "customers": _DEMO_CUSTOMERS,
    "orders":    _DEMO_ORDERS,
}

_DEMO_SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        VARCHAR,
    email       VARCHAR,
    city        VARCHAR
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER,
    total_amount DOUBLE,
    status       VARCHAR,
    ordered_at   DATE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

DEMO_QUESTIONS = [
    ("How many orders are there for each status?",          "orders_by_status.html"),
    ("What is the total spending per customer?",            "spending_per_customer.html"),
    ("Show monthly total revenue trend over time",          "monthly_revenue.html"),
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


async def _cli_demo(model: str) -> None:
    print("=" * 65)
    print("  Text → SQL → Data → Visualization  (demo mode)")
    print("=" * 65)
    for question, out_file in DEMO_QUESTIONS:
        print(f"\n{'─'*65}")
        await run_pipeline(
            question=question,
            dataframes=_DEMO_DATAFRAMES,
            schema=_DEMO_SCHEMA,
            output_path=out_file,
            model=model,
            open_browser=False,
        )
    # open last chart
    webbrowser.open(f"file://{Path(DEMO_QUESTIONS[-1][1]).resolve()}")


async def _cli_single(
    question: str, csv_path: str, schema_path: Optional[str], model: str
) -> None:
    df = pd.read_csv(csv_path)
    table_name = Path(csv_path).stem
    dataframes = {table_name: df}

    if schema_path:
        schema = Path(schema_path).read_text()
    else:
        # Auto-generate a simple DDL from the DataFrame
        type_map = {
            "int64": "BIGINT", "float64": "DOUBLE", "object": "VARCHAR",
            "bool": "BOOLEAN", "datetime64[ns]": "TIMESTAMP",
        }
        cols = ", ".join(
            f'    "{c}" {type_map.get(str(df[c].dtype), "VARCHAR")}'
            for c in df.columns
        )
        schema = f"CREATE TABLE {table_name} (\n{cols}\n);"

    await run_pipeline(
        question=question,
        dataframes=dataframes,
        schema=schema,
        model=model,
    )


def main() -> None:
    model = os.getenv("TEXT_TO_SQL_MODEL", "deepseek/deepseek-chat")

    if len(sys.argv) == 1:
        asyncio.run(_cli_demo(model))
    elif len(sys.argv) == 2:
        asyncio.run(_cli_single(sys.argv[1], None, None, model))
    elif len(sys.argv) == 3:
        asyncio.run(_cli_single(sys.argv[1], sys.argv[2], None, model))
    elif len(sys.argv) >= 4:
        asyncio.run(_cli_single(sys.argv[1], sys.argv[2], sys.argv[3], model))


if __name__ == "__main__":
    main()
