# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional, Tuple, Type

from google.cloud.bigtable.data.execute_query import QueryResultRow
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_google_bigtable.engine import BigtableEngine


def try_convert_bytes_to_str(x: Any) -> Any:
    """
    Try to convert bytes to UTF-8 string, else base64-encode(ascii). Otherwise return as-is.
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
    """
    Convert a Bigtable QueryResultRow into a dictionary.

    Notes:
    - `execute_query_iterator` returns QueryResultRow objects (see:
      https://cloud.google.com/python/docs/reference/bigtable/latest/google.cloud.bigtable.data.execute_query.values.QueryResultRow).
    - QueryResultRow.fields is a list of (column, value) tuples, not a dict.
    - Each row is indexed by a single row key, with data stored under (family, qualifier) pairs.
      (Docs: https://cloud.google.com/bigtable/docs/overview)
    - This function normalizes that structure into a nested dict and converts bytes to strings.
    """

    fields: Optional[List[Tuple[Optional[str], Any]]] = getattr(row, "fields", None)
    """ We treat `fields` as a list of (name, value) pairs
        The first entry "_key" stores the row key,
        and the other entries hold the column families and their values.
    """
    out: Dict[str, Any] = {}
    if fields is None:
        return out

    for idx, (name, value) in enumerate(fields):
        if name is None:
            key = f"column_{idx}"
        else:
            key = name
        out[key] = unpack_value(value)
    return out


class BigtableExecuteQueryInput(BaseModel):
    """
    Input for BigtableExecuteQueryTool.
    """

    instance_id: str = Field(..., description="The Bigtable instance ID.")
    query: str = Field(..., description="The SQL query to execute in Bigtable.")


class BigtableExecuteQueryTool(BaseTool):
    """
    Tool for executing a query within a Google Bigtable instance.
    """

    name: str = "bigtable_execute_query"
    description: str = (
        "A tool for executing a SQL query in Google Bigtable. The following are examples of common queries for Bigtable data:"
        "Retrieve the latest version of some columns for a given row key:  SELECT cf1['col1'], cf1['col2'], cf2['col1'] FROM myTable WHERE _key = 'r1';"
        "Retrieve all versions of some columns for a given row key: SELECT cf1['col1'], cf1['col2'], cf2['col1'] FROM myTable(with_history => TRUE) WHERE _key = 'r1'"
        "Use SELECT * FROM myTable to retrieve all data from a table if you are not sure of what column to get."
        "Make sure to set a LIMIT clause, e.g. SELECT * FROM myTable LIMIT 10, to avoid retrieving too much data at once."
    )
    args_schema: Type[BaseModel] = BigtableExecuteQueryInput

    def __init__(self, engine: BigtableEngine, **kwargs: Any):
        super().__init__(**kwargs)
        self._engine = engine

    async def _execute_query_internal(self, instance_id: str, query: str) -> Any:
        result = await self._engine.async_client.execute_query(query, instance_id)
        rows = []
        async for row in result:
            rows.append(row_to_dict(row))
        return rows

    def _run(self, instance_id: str, query: str) -> Any:
        return self._engine._run_as_sync(
            self._execute_query_internal(instance_id, query)
        )

    async def _arun(self, instance_id: str, query: str) -> Any:
        return await self._engine._run_as_async(
            self._execute_query_internal(instance_id, query)
        )


class PresetBigtableQueryInput(BaseModel):
    """
    Input for PresetBigtableExecuteQueryTool with parameterized query.
    """

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description='Parameters to fill in the query, e.g. {"min_age": 18, "status": "active"}',
    )


class PresetBigtableExecuteQueryTool(BaseTool):
    """
    A tool for executing a preset SQL query in Google Bigtable.
    """

    name: str = "preset_bigtable_execute_query"
    description: str = (
        "A preset tool for executing a fixed or parameterized SQL query in Google Bigtable. "
        "The instance_id and query are set at initialization. "
        "If the query contains parameters, they can be provided at runtime. "
        "This is a placeholder description; the actual description will be set during initialization."
    )

    args_schema: Type[BaseModel] = PresetBigtableQueryInput
    _engine: BigtableEngine
    _instance_id: str
    _query: str

    def __init__(
        self,
        engine: BigtableEngine,
        instance_id: str,
        query: str,
        tool_name: str,
        description: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the tool with a BigtableEngine, instance ID, fixed query, and required tool name.
        Optionally, a description can be provided.
        """
        super().__init__(**kwargs)
        self._engine = engine
        self._instance_id = instance_id
        self._query = query
        self.name = tool_name
        default_description = (
            "A preset tool for executing a fixed or parameterized SQL query in Google Bigtable.\n"
            'If the query contains parameters (e.g., @location), provide them as a dictionary in the input: {"location": "Basel"}.\n'
            'All parameter values should match the type stored in Bigtable (e.g., string for string fields: {"location": "Basel"}).\n\n'
            f"Instance ID: {instance_id}\n"
            f"Query: {query}\n"
        )
        if description is not None:
            self.description = (
                f"{default_description}\n\nUser Description: {description}"
            )
        else:
            self.description = default_description

    def _convert_parameters_to_bytes(
        self, parameters: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert string parameters to bytes, as Bigtable expects byte values. The agent can't provide parameters in bytes.
        """
        if parameters is None:
            return None
        converted = {}
        for k, v in parameters.items():
            if isinstance(v, str):
                converted[k] = v.encode("utf-8")
            else:
                converted[k] = v
        return converted

    async def _execute_query_internal(
        self, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        parameters = self._convert_parameters_to_bytes(parameters)
        result = await self._engine.async_client.execute_query(
            self._query, self._instance_id, parameters=parameters
        )
        rows = []
        async for row in result:
            rows.append(row_to_dict(row))
        return rows

    def _run(self, parameters: Optional[Dict[str, Any]] = None) -> Any:
        return self._engine._run_as_sync(
            self._execute_query_internal(parameters=parameters)
        )

    async def _arun(self, parameters: Optional[Dict[str, Any]] = None) -> Any:
        return await self._engine._run_as_async(
            self._execute_query_internal(parameters=parameters)
        )
