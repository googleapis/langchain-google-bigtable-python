import pytest
from google.cloud import bigtable
from langchain_google_bigtable.get_instances_and_table_schema_tool import BigtableGetInstancesAndTableSchemaTool
from typing import Generator
import uuid


@pytest.fixture(scope="session")
def managed_table(
    project_id: str, instance_id: str, admin_client: bigtable.Client
) -> Generator[str, None, None]:
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

        yield table_id 

    finally:
        if table.exists():
            table.delete()


def test_get_instances_and_table_schema_tool(admin_client, managed_table, instance_id):
    """
    Test the synchronous GetInstancesAndTableSchemaTool functionality.
    """
    tool = BigtableGetInstancesAndTableSchemaTool(client=admin_client)
    result = tool.invoke({})
    table_id = managed_table

    expected = {
        instance_id: {
            table_id: {
                "column_families": ["cf"],
                "possible_columns": [
                    "cf['checkin_date']",
                    "cf['name']",
                    "cf['price_tier']",
                    "cf['id']",
                    "cf['location']",
                    "cf['checkout_date']",
                    "cf['booked']",
                ],
            },
        }
    }
   
    # Assert that the expected result exists.
    assert set(expected.keys()).issubset(set(result.keys()))
    for instance in expected.keys():
        assert set(expected[instance].keys()).issubset(set(result[instance].keys()))
        for table in expected[instance].keys():
            assert sorted(expected[instance][table]) == sorted(result[instance][table])
