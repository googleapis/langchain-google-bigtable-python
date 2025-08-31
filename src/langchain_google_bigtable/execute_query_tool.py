from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from google.cloud.bigtable.data.execute_query import QueryResultRow
from pydantic import BaseModel, Field
from google.cloud.bigtable.data import BigtableDataClient
from langchain_core.tools import BaseTool
from typing import Type
import base64


def try_convert_bytes_to_str(x: Any) -> Any:
    """
    Try to convert bytes to UTF-8 string, else base64-encode(ascii). otherwise return as-is.
    """
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(x).decode("ascii")
    return str(x)


def unpack_value(x: Any) -> Any:
    """
    Unpack row values, converting bytes to strings where possible.
    """
    if isinstance(x, bytes):
        return try_convert_bytes_to_str(x)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            key = unpack_value(k)
            out[key] = unpack_value(v)
        return out
    return x


def row_to_dict(row: QueryResultRow) -> Dict[str, Any]:
    """Convert a Bigtable QueryResultRow into a dictionary.

    Notes:
    - `execute_query_iterator` returns QueryResultRow objects (see:
      https://cloud.google.com/python/docs/reference/bigtable/latest/google.cloud.bigtable.data.execute_query.values.QueryResultRow).
    - QueryResultRow.fields is a list of (column, value) tuples, not a dict.
    - Each row is indexed by a single row key, with data stored under (family, qualifier) pairs.
      (Docs: https://cloud.google.com/bigtable/docs/overview)
    - This function normalizes that structure into a nested dict with bytes converted.
    """
    fields: Optional[List[Tuple[Optional[str], Any]]] = getattr(row, "fields", None)

    """ We treat `fields` as a list of (name, value) pairs
        The first entry "_key" stores the row key,
        and the other entries hold the column families and their values."""
    out: Dict[str, Any] = {}
    for idx, (name, value) in enumerate(fields):
        out[name] = unpack_value(value)
    return out


class BigtableExecuteQueryInput(BaseModel):
    """Input for BigtableExecuteQueryTool."""

    instance_id: str = Field(..., description="The Bigtable instance ID.")
    query: str = Field(..., description="The SQL query to execute in Bigtable.")


class BigtableExecuteQueryTool(BaseTool):
    """Tool for executing a query within a Google Bigtable instance."""

    name: str = "bigtable_execute_query"
    # TODO: Play around with the description to see if the description is needed?
    description: str = (
        "A tool for executing a SQL query in Google Bigtable. The following are examples of common queries for Bigtable data:" \
        "Retrieve the latest version of all columns for a given row key:  SELECT cf1['col1'], cf1['col2'], cf2['col1'] FROM myTable WHERE _key = 'r1';" \
        "Retrieve all versions of all columns for a given row key: SELECT cf1['col1'], cf1['col2'], cf2['col1'] FROM myTable(with_history => TRUE) WHERE _key = 'r1'"
    )
    args_schema: Type[BaseModel] = BigtableExecuteQueryInput
    _client: BigtableDataClient

    def __init__(self, client: BigtableDataClient, **kwargs: Any):
        """Initialize with a Bigtable Data client."""
        super().__init__(**kwargs)
        self._client = client

    def _run(self, instance_id: str, query: str) -> Any:
        """
        Run execute query with the Bigtable Data client
        """
        try:
            rows = []
            for row in self._client.execute_query(query, instance_id):
                rows.append(row_to_dict(row))
            return rows
        except Exception as e:
            return f"Error: {str(e)}"

