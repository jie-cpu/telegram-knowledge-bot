"""Database query tool — simulated database with sample tables.

Demonstrates:
- Structured data querying
- Simulated database with realistic schemas
- SQL-like interface
"""

from __future__ import annotations

import csv
import io
from typing import Any

from agent.tools.base import BaseTool

# ── In-memory sample database tables ──────────────────────────────

_SAMPLE_DB: dict[str, list[dict[str, Any]]] = {
    "products": [
        {"id": 1, "name": "Wireless Mouse", "category": "Electronics", "price": 29.99, "stock": 150},
        {"id": 2, "name": "Mechanical Keyboard", "category": "Electronics", "price": 89.99, "stock": 75},
        {"id": 3, "name": "USB-C Hub", "category": "Electronics", "price": 45.00, "stock": 200},
        {"id": 4, "name": "Standing Desk", "category": "Furniture", "price": 349.00, "stock": 30},
        {"id": 5, "name": "Ergonomic Chair", "category": "Furniture", "price": 499.00, "stock": 20},
        {"id": 6, "name": "Monitor 27inch", "category": "Electronics", "price": 329.00, "stock": 60},
        {"id": 7, "name": "Noise Cancelling Headphones", "category": "Electronics", "price": 249.00, "stock": 45},
        {"id": 8, "name": "Notebook (Pack of 5)", "category": "Stationery", "price": 12.99, "stock": 500},
    ],
    "customers": [
        {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "city": "San Francisco", "tier": "premium"},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "city": "New York", "tier": "standard"},
        {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "city": "Austin", "tier": "standard"},
        {"id": 4, "name": "Diana Lee", "email": "diana@example.com", "city": "Seattle", "tier": "premium"},
        {"id": 5, "name": "Eve Garcia", "email": "eve@example.com", "city": "Los Angeles", "tier": "standard"},
    ],
    "orders": [
        {"id": 101, "customer_id": 1, "total": 119.98, "status": "shipped", "date": "2025-03-15"},
        {"id": 102, "customer_id": 2, "total": 45.00, "status": "delivered", "date": "2025-03-20"},
        {"id": 103, "customer_id": 3, "total": 499.00, "status": "pending", "date": "2025-04-01"},
        {"id": 104, "customer_id": 1, "total": 329.00, "status": "processing", "date": "2025-04-05"},
        {"id": 105, "customer_id": 4, "total": 648.99, "status": "shipped", "date": "2025-04-10"},
        {"id": 106, "customer_id": 5, "total": 29.99, "status": "cancelled", "date": "2025-04-12"},
    ],
}


class DatabaseQueryTool(BaseTool):
    """Simulated database query tool."""

    @property
    def name(self) -> str:
        return "database_query"

    @property
    def description(self) -> str:
        return (
            "Query the company database. Available tables: products, customers, orders. "
            "Use this to retrieve structured data about products, customers, and orders."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The query to run. Supported formats: "
                        "'SELECT * FROM {table}' - list all rows from a table. "
                        "'SELECT {columns} FROM {table} WHERE {condition}' - filtered query. "
                        "'SELECT {columns} FROM {table} ORDER BY {col} [ASC|DESC]' - sorted query. "
                        "'SELECT COUNT(*) FROM {table}' - count rows. "
                        "'SELECT {func}({col}) FROM {table}' - aggregate function (SUM, AVG, MIN, MAX).  "
                        "'DESCRIBE {table}' - show table schema. "
                        "'SHOW TABLES' - list all tables."
                    ),
                },
                "format": {
                    "type": "string",
                    "enum": ["json", "csv"],
                    "description": "Output format (default: json)",
                    "default": "json",
                },
            },
            "required": ["query"],
        }

    async def _run(self, query: str, format: str = "json") -> dict[str, Any]:
        query = query.strip()
        query_upper = query.upper()

        # ── SHOW TABLES ────────────────────────────────────────────
        if query_upper == "SHOW TABLES":
            tables = list(_SAMPLE_DB.keys())
            return {"tables": tables, "count": len(tables), "format": format}

        # ── DESCRIBE TABLE ──────────────────────────────────────────
        if query_upper.startswith("DESCRIBE"):
            table_name = query.split(None, 1)[1].strip().rstrip(";")
            if table_name not in _SAMPLE_DB:
                return {"error": f"Table '{table_name}' not found. Available: {list(_SAMPLE_DB.keys())}"}
            if not _SAMPLE_DB[table_name]:
                return {"table": table_name, "columns": [], "message": "Table is empty"}
            columns = list(_SAMPLE_DB[table_name][0].keys())
            return {"table": table_name, "columns": columns, "row_count": len(_SAMPLE_DB[table_name]), "format": format}

        # ── SELECT queries ──────────────────────────────────────────
        if query_upper.startswith("SELECT"):
            result = self._execute_select(query)
            if format == "csv":
                output = self._to_csv(result["columns"], result["rows"])
                return {"query": query, "data": output, "format": "csv", "row_count": len(result["rows"])}
            return {
                "query": query,
                "columns": result["columns"],
                "rows": result["rows"],
                "row_count": len(result["rows"]),
                "format": "json",
            }

        return {"error": f"Unsupported query syntax: {query}"}

    def _execute_select(self, query: str) -> dict[str, Any]:
        """Parse and execute a simplified SELECT query."""
        # Strip trailing semicolon
        query = query.rstrip(";").strip()
        query_upper = query.upper()
        parts = query.split()

        # ── Extract FROM clause ────────────────────────────────────
        try:
            from_idx = parts.index("FROM")
        except ValueError:
            return {"columns": [], "rows": []}

        table_name = parts[from_idx + 1].strip().rstrip(";")
        if table_name not in _SAMPLE_DB:
            return {"columns": [], "rows": []}

        table = _SAMPLE_DB[table_name]
        all_columns = list(table[0].keys()) if table else []

        # ── Extract columns ─────────────────────────────────────────
        select_clause = " ".join(parts[1:from_idx])
        if select_clause.strip() == "*":
            columns = all_columns
        elif select_clause.upper().startswith("COUNT"):
            return {"columns": ["count"], "rows": [{"count": len(table)}]}
        elif select_clause.upper().startswith(("SUM(", "AVG(", "MIN(", "MAX(")):
            func_match = select_clause.split("(")
            func_name = func_match[0].upper()
            col_name = func_match[1].rstrip(")").strip()
            if col_name not in all_columns:
                return {"columns": [], "rows": []}
            values = [r[col_name] for r in table if isinstance(r[col_name], (int, float))]
            if not values:
                return {"columns": [f"{func_name}({col_name})"], "rows": [{f"{func_name}({col_name})": None}]}
            funcs = {"SUM": sum, "AVG": lambda v: sum(v) / len(v), "MIN": min, "MAX": max}
            result = funcs[func_name](values)
            result = round(result, 2) if isinstance(result, float) else result
            return {"columns": [f"{func_name}({col_name})"], "rows": [{f"{func_name}({col_name})": result}]}
        else:
            columns = [c.strip() for c in select_clause.split(",")]

        # ── WHERE clause ────────────────────────────────────────────
        filtered = list(table)
        if "WHERE" in query_upper:
            where_idx = query_upper.index("WHERE")
            where_clause = query[where_idx + 5:].strip()
            # Handle ORDER BY after WHERE
            if "ORDER BY" in where_clause.upper():
                order_idx = where_clause.upper().index("ORDER BY")
                where_clause = where_clause[:order_idx].strip()
            filtered = self._apply_where(filtered, where_clause)

        # ── ORDER BY clause ─────────────────────────────────────────
        if "ORDER BY" in query_upper:
            order_idx = query_upper.index("ORDER BY")
            order_clause = query[order_idx + 8:].strip().rstrip(";")
            parts_order = order_clause.split()
            order_col = parts_order[0].strip()
            order_dir = "ASC"
            if len(parts_order) > 1 and parts_order[1].upper() in ("ASC", "DESC"):
                order_dir = parts_order[1].upper()
            filtered.sort(
                key=lambda r: r.get(order_col, ""),
                reverse=(order_dir == "DESC"),
            )

        # ── Project columns ─────────────────────────────────────────
        result_rows = []
        for row in filtered:
            projected = {}
            for col in columns:
                if col in row:
                    projected[col] = row[col]
            result_rows.append(projected)

        return {"columns": columns, "rows": result_rows}

    def _apply_where(self, rows: list[dict], condition: str) -> list[dict]:
        """Simple WHERE clause parser. Supports: column OPERATOR value."""
        import re

        # Handle: column = 'value' or column = "value" or column = value
        patterns = [
            (r"(\w+)\s*=\s*'([^']*)'", lambda r, c, v: r.get(c) == v),
            (r'(\w+)\s*=\s*"([^"]*)"', lambda r, c, v: r.get(c) == v),
            (r"(\w+)\s*=\s*(\d+)", lambda r, c, v: r.get(c) == int(v)),
            (r"(\w+)\s*=\s*(\d+\.\d+)", lambda r, c, v: r.get(c) == float(v)),
            (r"(\w+)\s*!=\s*'([^']*)'", lambda r, c, v: r.get(c) != v),
            (r"(\w+)\s*>\s*(\d+)", lambda r, c, v: r.get(c, 0) > int(v)),
            (r"(\w+)\s*<\s*(\d+)", lambda r, c, v: r.get(c, 0) < int(v)),
            (r"(\w+)\s*>=\s*(\d+)", lambda r, c, v: r.get(c, 0) >= int(v)),
            (r"(\w+)\s*<=\s*(\d+)", lambda r, c, v: r.get(c, 0) <= int(v)),
        ]

        for pattern, evaluator in patterns:
            match = re.search(pattern, condition.strip())
            if match:
                col, val = match.group(1), match.group(2)
                if val.replace(".", "").isdigit() and "." in val:
                    val = float(val)
                elif val.isdigit():
                    val = int(val)
                return [row for row in rows if evaluator(row, col, val)]

        return rows

    def _to_csv(self, columns: list[str], rows: list[dict]) -> str:
        """Convert results to CSV format."""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
        return output.getvalue()
