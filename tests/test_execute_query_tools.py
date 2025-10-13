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
import uuid
from typing import Iterator, List, Tuple

import google.api_core.exceptions
import pytest
from google.cloud import bigtable

from langchain_google_bigtable.engine import BigtableEngine
from langchain_google_bigtable.execute_query_tools import (
    BigtableExecuteQueryTool,
    PresetBigtableExecuteQueryTool,
)

TOOL_NAME = "hotel_query_tool"

# TODO: Add tests for asserting that tool-calling happens.


@pytest.fixture
def expected_data() -> List[dict]:
    return [
        {
            "_key": "hotels#1#Basel#Hilton Basel#Luxury",
            "cf": {
                "booked": "False",
                "checkin_date": "2024-04-20",
                "checkout_date": "2024-04-22",
                "id": "1",
                "location": "Basel",
                "name": "Hilton Basel",
                "price_tier": "Luxury",
            },
        },
        {
            "_key": "hotels#2#Zurich#Marriott Zurich#Upscale",
            "cf": {
                "booked": "False",
                "checkin_date": "2024-04-14",
                "checkout_date": "2024-04-21",
                "id": "2",
                "location": "Zurich",
                "name": "Marriott Zurich",
                "price_tier": "Upscale",
            },
        },
        {
            "_key": "hotels#3#Basel#Hyatt Regency Basel#Upper Upscale",
            "cf": {
                "booked": "False",
                "checkin_date": "2024-04-02",
                "checkout_date": "2024-04-20",
                "id": "3",
                "location": "Basel",
                "name": "Hyatt Regency Basel",
                "price_tier": "Upper Upscale",
            },
        },
        {
            "_key": "hotels#4#Lucerne#Radisson Blu Lucerne#Midscale",
            "cf": {
                "booked": "False",
                "checkin_date": "2024-04-05",
                "checkout_date": "2024-04-24",
                "id": "4",
                "location": "Lucerne",
                "name": "Radisson Blu Lucerne",
                "price_tier": "Midscale",
            },
        },
        {
            "_key": "hotels#5#Bern#Best Western Bern#Upper Midscale",
            "cf": {
                "booked": "False",
                "checkin_date": "2024-04-01",
                "checkout_date": "2024-04-23",
                "id": "5",
                "location": "Bern",
                "name": "Best Western Bern",
                "price_tier": "Upper Midscale",
            },
        },
    ]


@pytest.fixture(scope="session")
def managed_table(
    project_id: str, instance_id: str, admin_client: bigtable.Client
) -> Iterator[Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]]]:
    """
    Fixture to create a Bigtable table and insert data.
    """
    instance = admin_client.instance(instance_id)
    table_id = f"test-table-{uuid.uuid4().hex[:8]}"
    table = instance.table(table_id)

    column_families = {
        "cf": bigtable.column_family.MaxVersionsGCRule(1),
    }

    try:
        if not table.exists():
            table.create(column_families=column_families)

            columns = [
                "id",
                "name",
                "location",
                "price_tier",
                "checkin_date",
                "checkout_date",
                "booked",
            ]
            data = [
                [
                    1,
                    "Hilton Basel",
                    "Basel",
                    "Luxury",
                    "2024-04-20",
                    "2024-04-22",
                    False,
                ],
                [
                    2,
                    "Marriott Zurich",
                    "Zurich",
                    "Upscale",
                    "2024-04-14",
                    "2024-04-21",
                    False,
                ],
                [
                    3,
                    "Hyatt Regency Basel",
                    "Basel",
                    "Upper Upscale",
                    "2024-04-02",
                    "2024-04-20",
                    False,
                ],
                [
                    4,
                    "Radisson Blu Lucerne",
                    "Lucerne",
                    "Midscale",
                    "2024-04-05",
                    "2024-04-24",
                    False,
                ],
                [
                    5,
                    "Best Western Bern",
                    "Bern",
                    "Upper Midscale",
                    "2024-04-01",
                    "2024-04-23",
                    False,
                ],
            ]

            mutations = []
            batcher = table.mutations_batcher(max_row_bytes=1024)
            for row in data:
                row_key = f"hotels#{row[0]}#{row[2]}#{row[1]}#{row[3]}"
                mutation = table.direct_row(row_key)
                for col, value in zip(columns, row):
                    mutation.set_cell(
                        "cf", col.encode("utf-8"), str(value).encode("utf-8")
                    )
                mutations.append(mutation)
            for mutation in mutations:
                batcher.mutate(mutation)

            yield instance_id, table_id, column_families

    finally:
        if table.exists():
            table.delete()


def test_execute_query_tool_sync(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    expected_data: List[dict],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test the synchronous ExecuteQueryTool functionality.
    """
    instance_id, table_id, column_families = managed_table
    tool = BigtableExecuteQueryTool(engine=bigtable_engine)
    query = f"SELECT * FROM `{table_id}`"

    input_data = {"instance_id": instance_id, "query": query}
    result = tool.invoke(input=input_data)

    assert result == expected_data


def test_execute_query_tool_error_sync(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test the error handling of BigtableExecuteQueryTool when querying a non-existent table.
    """

    instance_id, _, _ = managed_table

    tool = BigtableExecuteQueryTool(engine=bigtable_engine)
    query = "SELECT * FROM `non_existent_table`"

    input_data = {"instance_id": instance_id, "query": query}
    with pytest.raises(google.api_core.exceptions.InvalidArgument) as excinfo:
        tool.invoke(input=input_data)
    assert "Table not found: non_existent_table" in str(excinfo.value)


@pytest.mark.asyncio
async def test_execute_query_tool_async(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    expected_data: List[dict],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test the async ExecuteQueryTool functionality.
    """
    instance_id, table_id, column_families = managed_table

    tool = BigtableExecuteQueryTool(engine=bigtable_engine)
    query = f"SELECT * FROM `{table_id}`"

    input_data = {"instance_id": instance_id, "query": query}
    result = await tool.ainvoke(input=input_data)

    assert result == expected_data


@pytest.mark.asyncio
async def test_execute_query_tool_error_async(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test the error handling of BigtableExecuteQueryTool (async) when querying a non-existent table.
    """
    instance_id, _, _ = managed_table

    tool = BigtableExecuteQueryTool(engine=bigtable_engine)
    query = "SELECT * FROM `non_existent_table`"

    input_data = {"instance_id": instance_id, "query": query}
    with pytest.raises(google.api_core.exceptions.InvalidArgument) as excinfo:
        await tool.ainvoke(input=input_data)
    assert "Table not found: non_existent_table" in str(excinfo.value)


def test_preset_bigtable_execute_query_tool_sync(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    expected_data: List[dict],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test the synchronous PresetBigtableExecuteQueryTool functionality.
    """
    instance_id, table_id, column_families = managed_table
    query = f"SELECT * FROM `{table_id}`"

    tool = PresetBigtableExecuteQueryTool(
        engine=bigtable_engine,
        instance_id=instance_id,
        query=query,
        tool_name=TOOL_NAME,
    )
    result = tool.invoke(input={})
    assert result == expected_data


def test_preset_bigtable_execute_query_tool_error_sync(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test error handling of PresetBigtableExecuteQueryTool when querying a non-existent table.
    """
    instance_id, _, _ = managed_table
    query = "SELECT * FROM `non_existent_table`"
    tool = PresetBigtableExecuteQueryTool(
        engine=bigtable_engine,
        instance_id=instance_id,
        query=query,
        tool_name=TOOL_NAME,
    )
    with pytest.raises(google.api_core.exceptions.InvalidArgument) as excinfo:
        tool.invoke(input={})
    assert "Table not found: non_existent_table" in str(excinfo.value)


@pytest.mark.asyncio
async def test_preset_bigtable_execute_query_tool_async(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    expected_data: List[dict],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test the async PresetBigtableExecuteQueryTool functionality.
    """
    instance_id, table_id, column_families = managed_table
    query = f"SELECT * FROM `{table_id}`"

    tool = PresetBigtableExecuteQueryTool(
        engine=bigtable_engine,
        instance_id=instance_id,
        query=query,
        tool_name=TOOL_NAME,
    )
    result = await tool.ainvoke(input={})
    assert result == expected_data


@pytest.mark.asyncio
async def test_preset_bigtable_execute_query_tool_error_async(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test async error handling of PresetBigtableExecuteQueryTool when querying a non-existent table.
    """
    instance_id, _, _ = managed_table
    query = "SELECT * FROM `non_existent_table`"
    tool = PresetBigtableExecuteQueryTool(
        engine=bigtable_engine,
        instance_id=instance_id,
        query=query,
        tool_name=TOOL_NAME,
    )
    with pytest.raises(google.api_core.exceptions.InvalidArgument) as excinfo:
        await tool.ainvoke(input={})
    assert "Table not found: non_existent_table" in str(excinfo.value)


def test_preset_bigtable_execute_query_tool_with_parameter_sync(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test PresetBigtableExecuteQueryTool with a parameterized query.
    """
    instance_id, table_id, column_families = managed_table
    query = f"SELECT * FROM `{table_id}` WHERE cf['location'] = @location"
    tool = PresetBigtableExecuteQueryTool(
        engine=bigtable_engine,
        instance_id=instance_id,
        query=query,
        tool_name=TOOL_NAME,
    )
    parameters = {"location": "Basel"}
    result = tool.invoke(input={"parameters": parameters})
    assert isinstance(result, list)
    assert all(row["cf"]["location"] == "Basel" for row in result)


@pytest.mark.asyncio
async def test_preset_bigtable_execute_query_tool_with_parameter_async(
    managed_table: Tuple[str, str, dict[str, bigtable.column_family.MaxVersionsGCRule]],
    bigtable_engine: BigtableEngine,
) -> None:
    """
    Test async PresetBigtableExecuteQueryTool with a parameterized query.
    """
    instance_id, table_id, column_families = managed_table
    query = f"SELECT * FROM `{table_id}` WHERE cf['location'] = @location"
    tool = PresetBigtableExecuteQueryTool(
        engine=bigtable_engine,
        instance_id=instance_id,
        query=query,
        tool_name=TOOL_NAME,
    )
    parameters = {"location": "Basel"}
    result = await tool.ainvoke(input={"parameters": parameters})
    assert isinstance(result, list)
    assert all(row["cf"]["location"] == "Basel" for row in result)
