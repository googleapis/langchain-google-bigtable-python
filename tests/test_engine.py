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
import os
import uuid
from typing import Any, Iterator, Optional
from unittest.mock import MagicMock

import google.auth
import pytest
import pytest_asyncio
from google.api_core import exceptions
from google.cloud import bigtable
from google.cloud.bigtable.data import (
    BigtableDataClientAsync,
    ReadRowsQuery,
    TableAsync,
    row_filters,
)
from google.cloud.bigtable.data.mutations import (
    DeleteAllFromRow,
    RowMutationEntry,
    SetCell,
)

from langchain_google_bigtable.engine import BigtableEngine

TEST_COLUMN_FAMILY = "cf1"
TEST_COLUMN = "test_col".encode("utf-8")
TEST_ROW_PREFIX = "pytest-engine-test-"


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
def dynamic_table_id(project_id: str, instance_id: str) -> Iterator[str]:
    # Uses the admin client for table creation and deletion
    admin_client = bigtable.Client(project=project_id, admin=True)
    instance = admin_client.instance(instance_id)

    table_id = f"test-suite-{uuid.uuid4().hex[:8]}"
    table = instance.table(table_id)

    column_families = {TEST_COLUMN_FAMILY: bigtable.column_family.MaxVersionsGCRule(1)}

    try:
        table.create(column_families=column_families)
        yield table_id
    except exceptions.Conflict:  # Already exists
        yield table_id
    except Exception as e:
        pytest.fail(f"Failed to create table {table_id}: {e}")
    finally:
        try:
            table.delete()
        except Exception as e:
            raise e


@pytest_asyncio.fixture(scope="session", autouse=True)
async def shutdown_bigtable_engine_loop() -> Any:
    yield
    await BigtableEngine.shutdown_default_loop()
    assert BigtableEngine._default_loop is None
    assert BigtableEngine._default_thread is None


class TestBigtableEngineSyncInitialized:
    """Tests for BigtableEngine.initialize()"""

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, project_id: str) -> Any:
        credentials, _ = google.auth.default()
        engine = BigtableEngine.initialize(
            project_id=project_id, credentials=credentials
        )
        yield engine
        await engine.close()

    @pytest.fixture
    def table_id(self, dynamic_table_id: str) -> str:
        return dynamic_table_id

    @pytest.mark.asyncio
    async def test_engine_initialized(self, engine: BigtableEngine) -> None:
        """Test that the engine is initialized correctly via initialize()."""
        assert engine is not None
        assert engine.async_client is not None
        assert isinstance(engine.async_client, BigtableDataClientAsync)
        assert engine._loop is not None
        assert engine._loop.is_running()

    @pytest.mark.asyncio
    async def test_get_table_success(
        self, engine: BigtableEngine, instance_id: str, table_id: str
    ) -> None:
        """Test get_async_table on an engine from initialize()."""
        table = await engine.get_async_table(instance_id, table_id)
        assert table is not None
        assert isinstance(table, TableAsync)
        assert table.table_id == table_id

    @pytest.mark.asyncio
    async def test_data_operations(
        self, engine: BigtableEngine, instance_id: str, table_id: str
    ) -> None:
        """Test data operations using an engine from initialize()."""
        table = await engine.get_async_table(instance_id, table_id)
        test_row_key = TEST_ROW_PREFIX + uuid.uuid4().hex
        test_value = f"value-{uuid.uuid4().hex}"

        # Set up the mutations to be made for this test
        set_cell_mutation = SetCell(
            family=TEST_COLUMN_FAMILY,
            qualifier=TEST_COLUMN,
            new_value=test_value.encode("utf-8"),
        )
        row_mutation_entry = RowMutationEntry(
            row_key=test_row_key, mutations=[set_cell_mutation]
        )

        # Performs a write operation on the mutations for test_row_key row
        async def perform_write() -> None:
            await table.bulk_mutate_rows([row_mutation_entry])

        # Run the write operation asynchronously using the engine and assert
        await engine._run_as_async(perform_write())

        # Performs a read operation using test_row_key
        async def perform_read():
            query = ReadRowsQuery(
                row_keys=[test_row_key],
                row_filter=row_filters.CellsColumnLimitFilter(1),
            )
            rows = await table.read_rows(query)
            return rows[0] if rows else None

        # Run the read operation asynchronously using the engine and assert
        row = await engine._run_as_async(perform_read())
        assert row is not None, f"Row {test_row_key} not found after write"
        cell = row.get_cells(family=TEST_COLUMN_FAMILY, qualifier=TEST_COLUMN)[0]
        assert cell.value.decode("utf-8") == test_value

        # Run the read operation synchronously using the engine and assert
        sync_row = engine._run_as_sync(perform_read())
        assert sync_row is not None
        sync_cell = sync_row.get_cells(
            family=TEST_COLUMN_FAMILY, qualifier=TEST_COLUMN
        )[0]
        assert sync_cell.value.decode("utf-8") == test_value

        # Prepare delete mutation
        delete_mutation = DeleteAllFromRow()
        delete_entry = RowMutationEntry(test_row_key, [delete_mutation])

        # Performs a delete operation
        async def perform_delete() -> None:
            await table.bulk_mutate_rows([delete_entry])

        # Perform delete operation asynchronously using the engine and assert
        await engine._run_as_async(perform_delete())

        deleted_row = await engine._run_as_async(perform_read())
        assert deleted_row is None

    @pytest.mark.asyncio
    async def test_engine_close_behavior(self, project_id: str) -> None:
        """Test close behavior on an engine from initialize()"""
        credentials, _ = google.auth.default()
        local_engine = BigtableEngine.initialize(
            project_id=project_id, credentials=credentials
        )
        await local_engine.close()
        assert local_engine._client is None
        with pytest.raises(RuntimeError, match="Client not initialized"):
            local_engine.async_client

    def test_constructor_key(self) -> None:
        with pytest.raises(Exception, match="Use factory method 'initialize'"):
            BigtableEngine(object(), MagicMock(), None, None)


class TestBigtableEngineAsyncInitialized:
    """Tests for BigtableEngine.async_initialize()"""

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, project_id: str) -> Any:
        credentials, _ = google.auth.default()
        engine = await BigtableEngine.async_initialize(
            project_id=project_id, credentials=credentials
        )
        yield engine
        await engine.close()

    @pytest.fixture
    def table_id(self, dynamic_table_id: str) -> str:
        return dynamic_table_id

    @pytest.mark.asyncio
    async def test_engine_async_initialized(self, engine: BigtableEngine) -> None:
        """Test that the engine is initialized correctly via async_initialize()."""
        assert engine is not None
        assert engine.async_client is not None
        assert isinstance(engine.async_client, BigtableDataClientAsync)
        assert engine._loop is not None
        assert engine._loop.is_running()
        assert engine._thread is not None
        # Check it uses the shared class-level loop and thread
        assert engine._loop is BigtableEngine._default_loop
        assert engine._thread is BigtableEngine._default_thread

    @pytest.mark.asyncio
    async def test_get_table_success_async(
        self, engine: BigtableEngine, instance_id: str, table_id: str
    ) -> None:
        """Test get_async_table on an engine from async_initialize()."""
        table = await engine.get_async_table(instance_id, table_id)
        assert table is not None
        assert isinstance(table, TableAsync)
        assert table.table_id == table_id

    @pytest.mark.asyncio
    async def test_data_operations_async(
        self, engine: BigtableEngine, instance_id: str, table_id: str
    ) -> None:
        """Test data operations using an engine from async_initialize()."""
        table = await engine.get_async_table(instance_id, table_id)
        test_row_key = TEST_ROW_PREFIX + uuid.uuid4().hex
        test_value = f"value-{uuid.uuid4().hex}"

        # Set up the mutations to be made for this test
        set_cell_mutation = SetCell(
            family=TEST_COLUMN_FAMILY,
            qualifier=TEST_COLUMN,
            new_value=test_value.encode("utf-8"),
        )
        row_mutation_entry = RowMutationEntry(
            row_key=test_row_key, mutations=[set_cell_mutation]
        )

        # Performs a write operation on the mutations for test_row_key row
        async def perform_write() -> None:
            await table.bulk_mutate_rows([row_mutation_entry])

        # Run the write operation asynchronously using the engine and assert
        await engine._run_as_async(perform_write())

        # Performs a read operation using test_row_key
        async def perform_read():
            query = ReadRowsQuery(
                row_keys=[test_row_key],
                row_filter=row_filters.CellsColumnLimitFilter(1),
            )
            rows = await table.read_rows(query)
            return rows[0] if rows else None

        # Run the read operation asynchronously using the engine and assert
        row = await engine._run_as_async(perform_read())
        assert row is not None, f"Row {test_row_key} not found after write"
        cell = row.get_cells(family=TEST_COLUMN_FAMILY, qualifier=TEST_COLUMN)[0]
        assert cell.value.decode("utf-8") == test_value

        # Run the read operation synchronously using the engine and assert
        sync_row = engine._run_as_sync(perform_read())
        assert sync_row is not None
        sync_cell = sync_row.get_cells(
            family=TEST_COLUMN_FAMILY, qualifier=TEST_COLUMN
        )[0]
        assert sync_cell.value.decode("utf-8") == test_value

        # Prepare delete mutation
        delete_mutation = DeleteAllFromRow()
        delete_entry = RowMutationEntry(test_row_key, [delete_mutation])

        # Performs a delete operation
        async def perform_delete() -> None:
            await table.bulk_mutate_rows([delete_entry])

        # Perform delete operation asynchronously using the engine and assert
        await engine._run_as_async(perform_delete())

        deleted_row = await engine._run_as_async(perform_read())
        assert deleted_row is None

    @pytest.mark.asyncio
    async def test_engine_close_behavior_async(self, project_id: str) -> None:
        """Test close behavior on an engine from async_initialize()."""
        credentials, _ = google.auth.default()
        local_engine = await BigtableEngine.async_initialize(
            project_id=project_id, credentials=credentials
        )
        await local_engine.close()
        assert local_engine._client is None
        with pytest.raises(RuntimeError, match="Client not initialized"):
            local_engine.async_client
