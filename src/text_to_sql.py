#!/usr/bin/env python3
"""
Minimal runnable Text-to-SQL pipeline.

Draws directly on the prompt templates and SQL rules from:
  wren-ai-service/src/pipelines/generation/sql_generation.py
  wren-ai-service/src/pipelines/generation/utils/sql.py

The full service uses Hamilton DAGs, Haystack components, Qdrant vector search,
and Wren Engine SQL validation.  This script strips all of that out so you can
run a query with nothing but an LLM API key:

  pip install litellm
  export DEEPSEEK_API_KEY=sk-...
  python text_to_sql.py

Pass a question on the command line to query a built-in demo schema:

  python text_to_sql.py "Which customers spent the most last month?"

Or supply your own schema DDL file as a second argument:

  python text_to_sql.py "Count orders by status" my_schema.sql

Set TEXT_TO_SQL_MODEL to override the model (default: deepseek/deepseek-chat):

  TEXT_TO_SQL_MODEL=deepseek/deepseek-reasoner python text_to_sql.py "..."
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Optional

# ---------------------------------------------------------------------------
# Prompt templates
# Sourced verbatim from:
#   wren-ai-service/src/pipelines/generation/utils/sql.py  (system prompt)
#   wren-ai-service/src/pipelines/generation/sql_generation.py  (user prompt)
# ---------------------------------------------------------------------------

# From get_sql_generation_system_prompt() in utils/sql.py
_SQL_RULES = """
### SQL RULES ###
- ONLY USE SELECT statements, NO DELETE, UPDATE OR INSERT etc. statements that might change the data in the database.
- ONLY USE the tables and columns mentioned in the database schema.
- ONLY USE "*" if the user query asks for all the columns of a table.
- ONLY CHOOSE columns belong to the tables mentioned in the database schema.
- DON'T INCLUDE comments in the generated SQL query.
- YOU MUST USE "JOIN" if you choose columns from multiple tables!
- PREFER USING CTEs over subqueries.
- When generating SQL query, always:
    - Put double quotes around column and table names.
    - Put single quotes around string literals.
    - Never quote numeric literals.
    For example: SELECT "customers"."customer_name" FROM "customers" WHERE "customers"."city" = 'Taipei' and "customers"."year" = 1992;
- YOU MUST USE "lower(<table_name>.<column_name>) like lower(<value>)" or "lower(...) = lower(<value>)" for case-insensitive comparison.
- Aggregate functions are not allowed in the WHERE clause. Instead, they belong in the HAVING clause.
- PREFER USING CTEs over subqueries.
- For the ranking problem, use DENSE_RANK() and then filter with WHERE.
"""

SYSTEM_PROMPT = f"""
You are a helpful assistant that converts natural language queries into ANSI SQL queries.

Given user's question, database schema, etc., you should think deeply and carefully
and generate the SQL query based on the given reasoning plan step by step.

### GENERAL RULES ###
1. YOU MUST FOLLOW SQL Rules strictly.
2. YOU MUST ONLY USE SELECT statements.

{_SQL_RULES}

### FINAL ANSWER FORMAT ###
The final answer must be a ANSI SQL query in JSON format:

{{
    "sql": <SQL_QUERY_STRING>
}}
"""

# From sql_generation_user_prompt_template in sql_generation.py
USER_PROMPT_TEMPLATE = """
### DATABASE SCHEMA ###
{schema}

### QUESTION ###
User's Question: {question}

Let's think step by step.
"""

# Structured output schema (from SQL_GENERATION_MODEL_KWARGS in utils/sql.py)
_RESPONSE_FORMAT = {"type": "json_object"}


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------


async def generate_sql(
    question: str,
    schema: str,
    model: str = "deepseek/deepseek-chat",
    extra_context: Optional[str] = None,
) -> dict:
    """
    Convert a natural language question to SQL given a DDL schema string.

    This is the minimal equivalent of the full pipeline:
        DbSchemaRetrieval  →  SQLGeneration  →  SQLGenPostProcessor

    The retrieval step is replaced by passing ``schema`` directly.
    The post-processor (Wren Engine dry-run) is skipped; only the raw SQL
    and prompt metadata are returned.

    Args:
        question:      Natural language question from the user.
        schema:        One or more CREATE TABLE / CREATE VIEW DDL statements.
        model:         LiteLLM model string (e.g. "gpt-4o-mini", "claude-3-5-haiku").
        extra_context: Optional additional context appended to the schema section.

    Returns:
        dict with keys:
            sql      – generated SQL string
            model    – model that produced it
            tokens   – usage dict from the LLM response
    """
    try:
        from litellm import acompletion
    except ImportError as exc:
        raise ImportError(
            "litellm is required.  Install it with: pip install litellm"
        ) from exc

    schema_block = schema
    if extra_context:
        schema_block += f"\n\n{extra_context}"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        schema=schema_block.strip(),
        question=question.strip(),
    )

    api_key = os.getenv("DEEPSEEK_API_KEY")

    response = await acompletion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        response_format=_RESPONSE_FORMAT,
        api_key=api_key,
    )

    content = response.choices[0].message.content
    sql = json.loads(content)["sql"]

    return {
        "sql": sql,
        "model": response.model,
        "tokens": dict(response.usage) if response.usage else {},
    }


# ---------------------------------------------------------------------------
# Demo schema and questions (stand-alone test)
# ---------------------------------------------------------------------------

DEMO_SCHEMA = """
/* {"alias":"customers","description":"Customer master data"} */
CREATE TABLE customers (
    -- {"description":"Unique customer identifier"}
    customer_id INTEGER PRIMARY KEY,
    -- {"description":"Full name of the customer"}
    name VARCHAR,
    -- {"description":"Customer email address"}
    email VARCHAR,
    -- {"description":"City the customer is located in"}
    city VARCHAR,
    -- {"description":"Timestamp when the customer account was created"}
    created_at TIMESTAMP
);

/* {"alias":"orders","description":"Sales orders"} */
CREATE TABLE orders (
    -- {"description":"Unique order identifier"}
    order_id INTEGER PRIMARY KEY,
    -- {"description":"Reference to the customer who placed the order"}
    customer_id INTEGER,
    -- {"description":"Total monetary value of the order"}
    total_amount DOUBLE,
    -- {"description":"Order lifecycle status: pending, shipped, delivered, cancelled"}
    status VARCHAR,
    -- {"description":"Timestamp when the order was placed"}
    ordered_at TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

/* {"alias":"order_items","description":"Line items within an order"} */
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    -- {"description":"Name of the product"}
    product_name VARCHAR,
    quantity INTEGER,
    unit_price DOUBLE,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
"""

DEMO_QUESTIONS = [
    "How many orders does each customer have?",
    "What are the top 5 customers by total spending?",
    "List all delivered orders placed by customers in New York, ordered by date descending.",
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _run_demo(model: str) -> None:
    print(f"Model : {model}")
    print(f"Schema:\n{textwrap.indent(DEMO_SCHEMA.strip(), '  ')}\n")
    print("=" * 70)
    for question in DEMO_QUESTIONS:
        print(f"\nQuestion : {question}")
        result = await generate_sql(question, DEMO_SCHEMA, model=model)
        print(f"SQL      :\n{textwrap.indent(result['sql'], '  ')}")
        print(f"Tokens   : {result['tokens']}")
        print("-" * 70)


async def _run_single(question: str, schema_path: Optional[str], model: str) -> None:
    if schema_path:
        with open(schema_path) as f:
            schema = f.read()
    else:
        schema = DEMO_SCHEMA

    result = await generate_sql(question, schema, model=model)
    print(result["sql"])


def main() -> None:
    model = os.getenv("TEXT_TO_SQL_MODEL", "deepseek/deepseek-chat")

    if len(sys.argv) == 1:
        asyncio.run(_run_demo(model))
    elif len(sys.argv) == 2:
        asyncio.run(_run_single(sys.argv[1], None, model))
    elif len(sys.argv) >= 3:
        asyncio.run(_run_single(sys.argv[1], sys.argv[2], model))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
