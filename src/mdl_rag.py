#!/usr/bin/env python3
"""
Lightweight MDL semantic layer + RAG retrieval for data_agent.py.

Features:
  1. MDL JSON → annotated DDL conversion (business semantics as SQL comments)
     - Star schema support: dimensions, fact tables, joins
     - Derived/calculated fields with expressions
     - Column-level preprocessing (COALESCE, CAST, etc.)
     - Metric definitions with dimensions and measures
  2. SQL pairs in-memory vector search (embedding-based few-shot retrieval)
     - Local embedding via sentence-transformers (no API key needed)
     - Optional LiteLLM API embedding as fallback
  3. Star schema → DuckDB VIEW generation (materialized joins)
  4. No external vector DB required — all in-memory with numpy cosine similarity

Usage:
  from mdl_rag import load_mdl, mdl_to_ddl, mdl_to_views, SqlPairsStore

  mdl = load_mdl("walmart_mdl.json")
  ddl = mdl_to_ddl(mdl)           # For LLM prompt context
  views = mdl_to_views(mdl)       # For DuckDB execution (CREATE VIEW statements)
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Local embedding with sentence-transformers (lazy-loaded singleton)
# ─────────────────────────────────────────────────────────────────────────────

_local_model = None
_local_model_name: str = "all-MiniLM-L6-v2"


def _get_local_model():
    """Lazy-load sentence-transformers model (downloads ~80MB on first use)."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedding] Loading local model '{_local_model_name}'...")
        _local_model = SentenceTransformer(_local_model_name)
        print(f"[Embedding] Model ready (dim={_local_model.get_sentence_embedding_dimension()})")
    return _local_model


def local_embed(texts: list[str]) -> list[np.ndarray]:
    """Embed texts using local sentence-transformers model."""
    model = _get_local_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [embeddings[i] for i in range(len(texts))]

# ─────────────────────────────────────────────────────────────────────────────
# 1. MDL Schema — Extended for Star Schema
# ─────────────────────────────────────────────────────────────────────────────
#
# Full MDL JSON structure:
#
# {
#   "catalog": "walmart",
#   "schema": "public",
#
#   "models": [
#     {
#       "name": "train",
#       "description": "Weekly sales fact table",
#       "source": "train.csv",              ← CSV file or table name
#       "primaryKey": "Store,Dept,Date",     ← composite key OK
#       "columns": [
#         {"name": "Store", "type": "INTEGER", "description": "Store ID"},
#         {"name": "Weekly_Sales", "type": "DOUBLE", "description": "..."},
#         {
#           "name": "Sales_Rank",            ← derived/calculated field
#           "type": "INTEGER",
#           "description": "Store ranking by total sales",
#           "isCalculated": true,
#           "expression": "DENSE_RANK() OVER (ORDER BY SUM(Weekly_Sales) DESC)"
#         }
#       ],
#       "preprocessing": {                   ← column-level data cleaning
#         "Date": "CAST(Date AS DATE)"
#       }
#     }
#   ],
#
#   "dimensions": [                          ← NEW: dimension tables
#     {
#       "name": "dim_store",
#       "description": "Store dimension with type and size",
#       "source": "stores.csv",
#       "columns": [
#         {"name": "Store", "type": "INTEGER", "description": "Store ID", "isPrimaryKey": true},
#         {"name": "Type", "type": "VARCHAR", "description": "Store type: A (large), B (medium), C (small)"},
#         {"name": "Size", "type": "INTEGER", "description": "Store square footage"}
#       ]
#     }
#   ],
#
#   "joins": [                               ← NEW: explicit star schema joins
#     {
#       "name": "train_to_store",
#       "left": "train",
#       "right": "dim_store",
#       "type": "LEFT",
#       "on": "train.Store = dim_store.Store"
#     },
#     {
#       "name": "train_to_features",
#       "left": "train",
#       "right": "features",
#       "type": "LEFT",
#       "on": ["train.Store = features.Store", "train.Date = features.Date"]  ← composite
#     }
#   ],
#
#   "relationships": [...],                  ← kept for backward compat
#
#   "metrics": [
#     {
#       "name": "holiday_weighted_sales",
#       "description": "Holiday-weighted sales (5x weight for holidays)",
#       "baseObject": "train",
#       "dimension": [...],
#       "measure": [
#         {"name": "weighted_sales", "type": "DOUBLE",
#          "expression": "SUM(CASE WHEN IsHoliday THEN Weekly_Sales * 5 ELSE Weekly_Sales END)"}
#       ]
#     }
#   ],
#
#   "derivedFields": [                       ← NEW: cross-table computed columns
#     {
#       "name": "Active_Promotion",
#       "description": "True if any MarkDown is active",
#       "expression": "COALESCE(MarkDown1,0)+COALESCE(MarkDown2,0)+COALESCE(MarkDown3,0)+COALESCE(MarkDown4,0)+COALESCE(MarkDown5,0) > 0",
#       "type": "BOOLEAN",
#       "sourceTable": "features"
#     },
#     {
#       "name": "Total_MarkDown",
#       "expression": "COALESCE(MarkDown1,0)+COALESCE(MarkDown2,0)+COALESCE(MarkDown3,0)+COALESCE(MarkDown4,0)+COALESCE(MarkDown5,0)",
#       "type": "DOUBLE",
#       "sourceTable": "features"
#     }
#   ]
# }


def load_mdl(path: str) -> dict:
    """Load an MDL JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. MDL → Annotated DDL Conversion (for LLM prompt)
# ─────────────────────────────────────────────────────────────────────────────

def _column_to_ddl(col: dict, preprocessing: dict | None = None, indent: str = "  ") -> str:
    """Convert a single MDL column to a DDL line with comment."""
    parts = []

    # Build comment from description / displayName / calculated
    comment_items = {}
    if col.get("description"):
        comment_items["description"] = col["description"]
    if col.get("displayName"):
        comment_items["alias"] = col["displayName"]
    if col.get("isCalculated"):
        comment_items["calculated"] = True
        if col.get("expression"):
            comment_items["expression"] = col["expression"]
    # Show preprocessing if any
    if preprocessing and col["name"] in preprocessing:
        comment_items["preprocessing"] = preprocessing[col["name"]]

    if comment_items:
        parts.append(f"{indent}-- {json.dumps(comment_items, ensure_ascii=False)}")

    # Column definition
    name = col["name"]
    dtype = col.get("type", "VARCHAR")
    pk = " PRIMARY KEY" if col.get("isPrimaryKey") else ""
    parts.append(f'{indent}"{name}" {dtype}{pk}')

    return "\n".join(parts)


def _model_to_ddl(
    model: dict,
    relationships: list[dict],
    joins: list[dict],
    is_dimension: bool = False,
) -> str:
    """Convert a single MDL model/dimension to a CREATE TABLE statement with annotations."""
    lines = []

    # Table-level comment
    table_comment = {}
    if model.get("description"):
        table_comment["description"] = model["description"]
    if is_dimension:
        table_comment["role"] = "dimension"
    elif any(j.get("left") == model["name"] for j in joins):
        table_comment["role"] = "fact"
    if model.get("source"):
        table_comment["source"] = model["source"]
    if table_comment:
        lines.append(f"/* {json.dumps(table_comment, ensure_ascii=False)} */")

    table_name = model["name"]
    lines.append(f'CREATE TABLE "{table_name}" (')

    preprocessing = model.get("preprocessing", {})

    # Columns
    col_lines = []
    for col in model.get("columns", []):
        col_lines.append(_column_to_ddl(col, preprocessing))

    # Foreign keys from relationships (backward compat)
    for rel in relationships:
        models = rel.get("models", [])
        if table_name in models and rel.get("condition"):
            join_type = rel.get("joinType", "")
            condition = rel["condition"]
            comment = f'  -- {{"joinType": "{join_type}", "condition": "{condition}"}}'
            fk_ddl = _relationship_to_fk(table_name, rel)
            if fk_ddl:
                col_lines.append(f"{comment}\n{fk_ddl}")

    lines.append(",\n".join(col_lines))
    lines.append(");")

    return "\n".join(lines)


def _relationship_to_fk(table_name: str, rel: dict) -> Optional[str]:
    """
    Try to extract a FOREIGN KEY clause from a relationship condition.
    FK is only placed on the "many" side of the relationship.
    """
    condition = rel.get("condition", "")
    join_type = rel.get("joinType", "")
    models = rel.get("models", [])

    parts = condition.replace(" ", "").split("=")
    if len(parts) != 2:
        return None

    left_table, left_col = parts[0].split(".", 1) if "." in parts[0] else (None, None)
    right_table, right_col = parts[1].split(".", 1) if "." in parts[1] else (None, None)

    if not all([left_table, left_col, right_table, right_col]):
        return None

    if len(models) == 2:
        if join_type == "ONE_TO_MANY":
            fk_table = models[1]
        elif join_type == "MANY_TO_ONE":
            fk_table = models[0]
        else:
            fk_table = table_name
    else:
        fk_table = table_name

    if table_name != fk_table:
        return None

    if left_table == table_name:
        fk_col, ref_table, ref_col = left_col, right_table, right_col
    elif right_table == table_name:
        fk_col, ref_table, ref_col = right_col, left_table, left_col
    else:
        return None

    return f'  FOREIGN KEY ("{fk_col}") REFERENCES "{ref_table}"("{ref_col}")'


def _join_to_ddl(join: dict) -> str:
    """Convert a star schema join definition to a comment block."""
    on_clause = join.get("on", "")
    if isinstance(on_clause, list):
        on_str = " AND ".join(on_clause)
    else:
        on_str = on_clause
    return (
        f'-- JOIN: {join["left"]} {join.get("type", "LEFT")} JOIN {join["right"]} '
        f'ON {on_str}'
    )


def _metric_to_ddl(metric: dict) -> str:
    """Convert an MDL metric to a commented DDL-like block."""
    lines = []
    desc = metric.get("description", "")
    lines.append(f'/* {{"metric": "{metric["name"]}", "description": "{desc}", '
                 f'"baseObject": "{metric.get("baseObject", "")}"}} */')
    for dim in metric.get("dimension", []):
        expr = dim.get("expression", dim["name"])
        lines.append(f'-- Dimension: {dim["name"]} ({dim.get("type", "")}) = {expr}')
    for msr in metric.get("measure", []):
        expr = msr.get("expression", msr["name"])
        lines.append(f'-- Measure: {msr["name"]} ({msr.get("type", "")}) = {expr}')
    return "\n".join(lines)


def _derived_field_to_ddl(field: dict) -> str:
    """Convert a derived field definition to a comment block."""
    desc = field.get("description", "")
    return (
        f'-- DERIVED FIELD: "{field["name"]}" {field.get("type", "")}'
        f' = {field["expression"]}'
        f'{" -- " + desc if desc else ""}'
        f' [from: {field.get("sourceTable", "?")}]'
    )


def mdl_to_ddl(mdl: dict) -> str:
    """
    Convert a full MDL dict to annotated DDL string.

    The output includes:
      - CREATE TABLE for models and dimensions (with role annotations)
      - Column descriptions, preprocessing, calculated fields
      - Star schema join definitions
      - Derived field definitions
      - Metric definitions with dimensions and measures
    """
    relationships = mdl.get("relationships", [])
    joins = mdl.get("joins", [])
    parts = []

    # Dimension tables
    for dim in mdl.get("dimensions", []):
        parts.append(_model_to_ddl(dim, relationships, joins, is_dimension=True))

    # Fact/model tables
    for model in mdl.get("models", []):
        parts.append(_model_to_ddl(model, relationships, joins))

    # Star schema joins
    if joins:
        join_lines = ["### STAR SCHEMA JOINS ###"]
        for j in joins:
            join_lines.append(_join_to_ddl(j))
        parts.append("\n".join(join_lines))

    # Derived fields
    derived = mdl.get("derivedFields", [])
    if derived:
        df_lines = ["### DERIVED FIELDS ###"]
        for f in derived:
            df_lines.append(_derived_field_to_ddl(f))
        parts.append("\n".join(df_lines))

    # Metrics
    if mdl.get("metrics"):
        parts.append("### METRICS ###")
        for metric in mdl["metrics"]:
            parts.append(_metric_to_ddl(metric))

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MDL → DuckDB VIEWs (for query execution)
# ─────────────────────────────────────────────────────────────────────────────


def mdl_to_views(mdl: dict) -> list[str]:
    """
    Generate DuckDB CREATE VIEW statements from MDL joins and derived fields.

    This creates executable VIEWs so the agent can query a single denormalized
    view instead of manually writing JOINs every time.

    Returns a list of SQL statements to execute.
    """
    stmts = []

    # 1. Preprocessing views for tables that have column preprocessing
    all_tables = list(mdl.get("models", [])) + list(mdl.get("dimensions", []))
    for table in all_tables:
        preprocessing = table.get("preprocessing", {})
        if preprocessing:
            cols = []
            for col in table.get("columns", []):
                name = col["name"]
                if name in preprocessing:
                    cols.append(f'{preprocessing[name]} AS "{name}"')
                else:
                    cols.append(f'"{name}"')
            source = table["name"]
            view_name = f"_clean_{table['name']}"
            stmts.append(
                f'CREATE OR REPLACE VIEW {view_name} AS\n'
                f'SELECT {", ".join(cols)} FROM {source};'
            )

    # 2. Star schema joined view
    joins = mdl.get("joins", [])
    if joins:
        # Find the fact table (the "left" of the first join)
        fact_name = joins[0]["left"]

        # Determine if we use the cleaned version
        fact_tables = [m for m in mdl.get("models", []) if m["name"] == fact_name]
        fact_table = fact_tables[0] if fact_tables else None
        fact_source = f"_clean_{fact_name}" if (fact_table and fact_table.get("preprocessing")) else fact_name

        # Build SELECT columns: fact.*, plus dimension/joined columns (avoid duplicates)
        fact_cols = set()
        if fact_table:
            fact_cols = {c["name"] for c in fact_table.get("columns", [])}

        select_parts = [f'{fact_source}.*']

        # Track which tables are joined and their join keys to avoid duplicates
        join_key_cols = set()
        for j in joins:
            on = j.get("on", "")
            if isinstance(on, list):
                for clause in on:
                    # Parse "right.col" from "left.col = right.col"
                    for part in clause.replace(" ", "").split("="):
                        if "." in part:
                            tbl, col = part.split(".", 1)
                            if tbl == j["right"] or tbl == f'_clean_{j["right"]}':
                                join_key_cols.add((j["right"], col))
            else:
                for part in on.replace(" ", "").split("="):
                    if "." in part:
                        tbl, col = part.split(".", 1)
                        if tbl == j["right"] or tbl == f'_clean_{j["right"]}':
                            join_key_cols.add((j["right"], col))

        # Add non-duplicate columns from joined tables
        all_tables_dict = {t["name"]: t for t in all_tables}
        for j in joins:
            right_name = j["right"]
            right_table = all_tables_dict.get(right_name)
            if right_table:
                right_source = f"_clean_{right_name}" if right_table.get("preprocessing") else right_name
                for col in right_table.get("columns", []):
                    col_name = col["name"]
                    # Skip join keys (already in fact table) and duplicates
                    if (right_name, col_name) in join_key_cols:
                        continue
                    if col_name in fact_cols:
                        # Prefix with table name to avoid ambiguity
                        select_parts.append(f'{right_source}."{col_name}" AS "{right_name}_{col_name}"')
                    else:
                        select_parts.append(f'{right_source}."{col_name}"')

        # Add derived fields (rewrite table references to use cleaned source names)
        derived = mdl.get("derivedFields", [])
        for df in derived:
            expr = df["expression"]
            # Rewrite table references: features. → _clean_features. etc.
            for t in all_tables:
                t_name = t["name"]
                t_source = f"_clean_{t_name}" if t.get("preprocessing") else t_name
                if t_name != t_source:
                    expr = expr.replace(f'{t_name}.', f'{t_source}.')
            select_parts.append(f'({expr}) AS "{df["name"]}"')

        # Build JOIN clauses
        join_clauses = []
        for j in joins:
            right_name = j["right"]
            right_table = all_tables_dict.get(right_name)
            right_source = f"_clean_{right_name}" if (right_table and right_table.get("preprocessing")) else right_name

            on_clause = j.get("on", "")
            if isinstance(on_clause, list):
                # Replace table names with source names
                on_parts = []
                for clause in on_clause:
                    c = clause.replace(fact_name, fact_source).replace(right_name, right_source)
                    on_parts.append(c)
                on_str = " AND ".join(on_parts)
            else:
                on_str = on_clause.replace(fact_name, fact_source).replace(right_name, right_source)

            join_type = j.get("type", "LEFT")
            join_clauses.append(f'{join_type} JOIN {right_source} ON {on_str}')

        view_sql = (
            f'CREATE OR REPLACE VIEW sales_fact AS\n'
            f'SELECT\n  {",\n  ".join(select_parts)}\n'
            f'FROM {fact_source}\n'
            f'{chr(10).join(join_clauses)};'
        )
        stmts.append(view_sql)

    return stmts


# ─────────────────────────────────────────────────────────────────────────────
# 4. SQL Pairs In-Memory Vector Store
# ─────────────────────────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SqlPairsStore:
    """
    In-memory vector store for SQL question-answer pairs.

    Embedding backends (checked in order):
      1. Local sentence-transformers (default, no API key needed)
      2. LiteLLM API embedding (if use_local=False and OPENAI_API_KEY is set)

    Uses cosine similarity for retrieval. No external vector DB needed.
    """

    def __init__(
        self,
        pairs: list[dict],
        model: str = "deepseek/deepseek-chat",
        embedding_model: Optional[str] = None,
        similarity_threshold: float = 0.3,
        use_local: bool = True,
    ):
        self.pairs = pairs
        self.model = model
        self.embedding_model = embedding_model or "text-embedding-3-small"
        self.similarity_threshold = similarity_threshold
        self.use_local = use_local
        self._embeddings: list[np.ndarray] = []
        self._indexed = False

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "SqlPairsStore":
        """Load SQL pairs from a JSON file."""
        pairs = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(pairs=pairs, **kwargs)

    def _embed_local(self, texts: list[str]) -> list[np.ndarray]:
        """Embed using local sentence-transformers (no API key needed)."""
        return local_embed(texts)

    async def _embed_api(self, texts: list[str]) -> list[np.ndarray]:
        """Embed using LiteLLM API (requires OPENAI_API_KEY)."""
        from litellm import aembedding

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        response = await aembedding(
            model=self.embedding_model,
            input=texts,
            api_key=api_key,
        )
        return [np.array(item["embedding"]) for item in response.data]

    async def _embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed texts using the configured backend."""
        if self.use_local:
            return self._embed_local(texts)
        return await self._embed_api(texts)

    async def build_index(self) -> None:
        """Embed all SQL pair questions and build the in-memory index."""
        if not self.pairs:
            self._indexed = True
            return

        questions = [p["question"] for p in self.pairs]
        backend = "local" if self.use_local else f"API ({self.embedding_model})"
        print(f"[SqlPairsStore] Embedding {len(questions)} SQL pairs ({backend})...")
        self._embeddings = await self._embed(questions)
        self._indexed = True
        print(f"[SqlPairsStore] Index built. Dimension: {len(self._embeddings[0])}")

    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve top-K most similar SQL pairs for a query.

        Returns list of {"question": str, "sql": str, "score": float}
        sorted by descending similarity, filtered by threshold.
        """
        if not self._indexed:
            await self.build_index()

        if not self._embeddings:
            return []

        query_embedding = (await self._embed([query]))[0]

        scored = []
        for i, emb in enumerate(self._embeddings):
            score = _cosine_similarity(query_embedding, emb)
            if score >= self.similarity_threshold:
                scored.append({
                    "question": self.pairs[i]["question"],
                    "sql": self.pairs[i]["sql"],
                    "score": round(score, 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def format_as_prompt(self, examples: list[dict]) -> str:
        """Format retrieved SQL pairs as a prompt section."""
        if not examples:
            return ""
        lines = ["### SQL EXAMPLES ###"]
        for ex in examples:
            lines.append(f"Question: {ex['question']}")
            lines.append(f"SQL: {ex['sql']}")
            lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Schema Retrieval Helper (for large MDL with many tables)
# ─────────────────────────────────────────────────────────────────────────────


async def retrieve_relevant_tables(
    question: str,
    mdl: dict,
    model: str = "deepseek/deepseek-chat",
    top_k: int = 10,
    use_local: bool = True,
) -> list[str]:
    """
    Use embedding similarity to find the most relevant tables for a question.

    For small schemas (< top_k tables), returns all tables.
    For large schemas, embeds table descriptions and ranks by similarity.
    Uses local sentence-transformers by default (no API key needed).
    """
    all_items = []

    for m in mdl.get("models", []):
        desc = f"{m['name']}: {m.get('description', '')}. "
        desc += "Columns: " + ", ".join(c["name"] for c in m.get("columns", []))
        all_items.append({"name": m["name"], "type": "model", "description": desc})

    for d in mdl.get("dimensions", []):
        desc = f"{d['name']}: {d.get('description', '')}. "
        desc += "Columns: " + ", ".join(c["name"] for c in d.get("columns", []))
        all_items.append({"name": d["name"], "type": "dimension", "description": desc})

    for m in mdl.get("metrics", []):
        desc = f"{m['name']}: {m.get('description', '')}. "
        desc += "Base: " + m.get("baseObject", "")
        all_items.append({"name": m["name"], "type": "metric", "description": desc})

    # Small schema — return everything
    if len(all_items) <= top_k:
        return [item["name"] for item in all_items]

    # Large schema — use embedding to find relevant tables
    texts = [item["description"] for item in all_items] + [question]

    if use_local:
        embeddings = local_embed(texts)
    else:
        from litellm import aembedding
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        response = await aembedding(
            model="text-embedding-3-small", input=texts, api_key=api_key,
        )
        embeddings = [np.array(item["embedding"]) for item in response.data]

    query_emb = embeddings[-1]
    scored = []
    for i, emb in enumerate(embeddings[:-1]):
        score = _cosine_similarity(query_emb, emb)
        scored.append((all_items[i]["name"], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:top_k]]


def filter_mdl_by_tables(mdl: dict, table_names: list[str]) -> dict:
    """Return a subset of MDL containing only the specified tables."""
    name_set = set(table_names)
    filtered = dict(mdl)
    filtered["models"] = [m for m in mdl.get("models", []) if m["name"] in name_set]
    filtered["dimensions"] = [d for d in mdl.get("dimensions", []) if d["name"] in name_set]
    filtered["relationships"] = [
        r for r in mdl.get("relationships", [])
        if any(t in name_set for t in r.get("models", []))
    ]
    filtered["joins"] = [
        j for j in mdl.get("joins", [])
        if j.get("left") in name_set or j.get("right") in name_set
    ]
    filtered["metrics"] = [
        m for m in mdl.get("metrics", [])
        if m["name"] in name_set or m.get("baseObject") in name_set
    ]
    filtered["derivedFields"] = mdl.get("derivedFields", [])
    return filtered
