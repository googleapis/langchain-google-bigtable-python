import functools
import sys
import os
from typing import Generator, Iterator, List
import uuid
import pytest
from google.cloud.bigtable.data import BigtableDataClient, Table
from google.cloud import bigtable
from langchain_google_bigtable.execute_query_tool import BigtableExecuteQueryTool


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v

@pytest.fixture(scope="session")
def project_id() -> str:
    return get_env_var("PROJECT_ID", "GCP Project ID")

@pytest.fixture(scope="session")
def instance_id() -> str:
    return get_env_var("INSTANCE_ID", "Bigtable Instance ID")

@pytest.fixture(scope="session")
def admin_client(project_id: str):
    """
    Fixture to create a Bigtable client.
    """
    client = bigtable.Client(project=project_id, admin=True)
    yield client

@pytest.fixture(scope="session")
def data_client(project_id: str):
    """
    Fixture to create a Bigtable client.
    """
    try:
        client = BigtableDataClient(project=project_id, admin=True)
        yield client
    finally:
        client.close()
    

@pytest.fixture
def expected_data():
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
    project_id: str, instance_id: instance_id, admin_client: admin_client
) -> Iterator[tuple[str, str, List[str]]]:
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
        
            columns = ["id", "name", "location", "price_tier", "checkin_date", "checkout_date", "booked"]
            data = [
                [1, 'Hilton Basel', 'Basel', 'Luxury', '2024-04-20', '2024-04-22', False],
                [2, 'Marriott Zurich', 'Zurich', 'Upscale', '2024-04-14', '2024-04-21', False],
                [3, 'Hyatt Regency Basel', 'Basel', 'Upper Upscale', '2024-04-02', '2024-04-20', False],
                [4, 'Radisson Blu Lucerne', 'Lucerne', 'Midscale', '2024-04-05', '2024-04-24', False],
                [5, 'Best Western Bern', 'Bern', 'Upper Midscale', '2024-04-01', '2024-04-23', False],
            ]

            mutations = []
            batcher = table.mutations_batcher(max_row_bytes=1024)
            for row in data:
                row_key = f"hotels#{row[0]}#{row[2]}#{row[1]}#{row[3]}"
                mutation = table.direct_row(row_key)
                for col, value in zip(columns, row):
                    mutation.set_cell("cf", col.encode("utf-8"), str(value).encode("utf-8"))
                mutations.append(mutation)
            for mutation in mutations:
                batcher.mutate(mutation)

            yield instance_id, table_id, column_families  

    finally:
        if table.exists():
            table.delete()


def test_execute_query_tool(
    managed_table: Iterator[tuple[str, str, List[str]]], expected_data: List[dict], data_client: data_client
):
    """
    Test the synchronous ExecuteQueryTool functionality.
    """
    instance_id, table_id, column_families = managed_table

    tool = BigtableExecuteQueryTool(client=data_client)
    query = f"SELECT * FROM `{table_id}`"
    
    input_data = {"instance_id": instance_id, "query": query}
    result = tool.invoke(input=input_data)

    assert result == expected_data

def test_execute_query_tool_error(managed_table: Iterator[tuple[str, str, List[str]]], project_id: str, data_client: data_client):
    """
    Test the error handling of BigtableExecuteQueryTool when querying a non-existent table.
    """

    instance_id, _, _ = managed_table

    tool = BigtableExecuteQueryTool(client=data_client)
    query = "SELECT * FROM `non_existent_table`" 

    input_data = {"instance_id": instance_id, "query": query}
    result = tool.invoke(input=input_data)

    assert "Error" in result
    assert "Table not found: non_existent_table" in result


