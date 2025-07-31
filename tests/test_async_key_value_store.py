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
import asyncio
import functools
import os
import uuid
from typing import Any, AsyncGenerator
from unittest import mock

import google.auth
import pytest
import pytest_asyncio
from google.api_core import exceptions
from google.cloud import bigtable
from google.cloud.bigtable.data import (
    BigtableDataClientAsync,
    FailedMutationEntryError,
    MutationsExceptionGroup,
    TableAsync,
)

from langchain_google_bigtable.async_key_value_store import AsyncBigtableByteStore

TEST_COLUMN_FAMILY = "cf1"
TEST_COLUMN = "test_col".encode("utf-8")
TEST_ROW_PREFIX = "pytest-bytestore-test-"
CUSTOM_COLUMN_FAMILY = "cf2"
CUSTOM_COLUMN = "custom_col".encode("utf-8")
LARGE_VALUE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_KEYS = 1000


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


@pytest_asyncio.fixture(scope="session")
async def dynamic_table_id(
    project_id: str, instance_id: str
) -> AsyncGenerator[str, None]:
    # Uses the admin client for table creation and deletion
    admin_client = bigtable.Client(project=project_id, admin=True)
    instance = admin_client.instance(instance_id)

    table_id = f"test-suite-{uuid.uuid4().hex[:8]}"
    table = instance.table(table_id)

    # Tests will only focus on recent cells
    column_families = {
        TEST_COLUMN_FAMILY: bigtable.column_family.MaxVersionsGCRule(1),
        CUSTOM_COLUMN_FAMILY: bigtable.column_family.MaxVersionsGCRule(1),
    }

    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(
            None, functools.partial(table.create, column_families=column_families)
        )
        yield table_id
    except exceptions.Conflict:  # Already exists
        yield table_id
    except Exception as e:
        pytest.fail(f"Failed to create table {table_id}: {e}")
    finally:
        try:
            await loop.run_in_executor(None, table.delete)
            admin_client.close()
        except Exception as e:
            raise e


class TestAsyncBigtableByteStore:
    """
    Integration tests for AsyncBigtableByteStore.
    """

    @pytest_asyncio.fixture()
    async def table(
        self, project_id: str, instance_id: str, dynamic_table_id: str
    ) -> AsyncGenerator[TableAsync, None]:
        """Fixture to get a TableAsync instance using BigtableDataClientAsync."""
        credentials, _ = google.auth.default()
        async_data_client = BigtableDataClientAsync(
            project=project_id, credentials=credentials
        )
        table = async_data_client.get_table(instance_id, dynamic_table_id)
        yield table
        await async_data_client.close()

    @pytest_asyncio.fixture()
    async def store(
        self, table: TableAsync
    ) -> AsyncGenerator[AsyncBigtableByteStore, None]:
        """Fixture to create an AsyncBigtableByteStore for the tests."""
        store = AsyncBigtableByteStore(
            table, column_family=TEST_COLUMN_FAMILY, column_qualifier=TEST_COLUMN
        )
        yield store

    @pytest.mark.asyncio
    async def test_amset_amget_amdelete(self, store: AsyncBigtableByteStore) -> None:
        """Test amset, amget, and amdelete methods of AsyncBigtableByteStore."""
        test_key_1 = TEST_ROW_PREFIX + uuid.uuid4().hex
        test_key_2 = TEST_ROW_PREFIX + uuid.uuid4().hex
        test_value_1 = f"value-{uuid.uuid4().hex}".encode("utf-8")
        test_value_2 = f"value-{uuid.uuid4().hex}".encode("utf-8")
        key_value_pairs = [(test_key_1, test_value_1), (test_key_2, test_value_2)]

        # Test amset
        await store.amset(key_value_pairs)

        # Test amget
        retrieved_values = await store.amget([test_key_1, test_key_2])
        assert retrieved_values == [test_value_1, test_value_2]

        # Test amdelete
        await store.amdelete([test_key_1, test_key_2])
        retrieved_values_after_delete = await store.amget([test_key_1, test_key_2])
        assert retrieved_values_after_delete == [None, None]

        # Test sync operations raising NotImplementedError
        with pytest.raises(NotImplementedError):
            store.mset(key_value_pairs)
        with pytest.raises(NotImplementedError):
            store.mget([test_key_1, test_key_2])
        with pytest.raises(NotImplementedError):
            store.mdelete([test_key_1, test_key_2])

    @pytest.mark.asyncio
    async def test_ayield_keys(self, store: AsyncBigtableByteStore) -> None:
        """Test ayield_keys method of AsyncBigtableByteStore."""
        test_key_1 = TEST_ROW_PREFIX + "key1-" + uuid.uuid4().hex
        test_key_2 = TEST_ROW_PREFIX + "key2-" + uuid.uuid4().hex
        test_key_3 = "another-prefix-" + uuid.uuid4().hex  # Key with a different prefix
        test_value_1 = f"value-{uuid.uuid4().hex}".encode("utf-8")
        test_value_2 = f"value-{uuid.uuid4().hex}".encode("utf-8")
        test_value_3 = f"value-{uuid.uuid4().hex}".encode("utf-8")
        key_value_pairs = [
            (test_key_1, test_value_1),
            (test_key_2, test_value_2),
            (test_key_3, test_value_3),
        ]

        # Set keys
        await store.amset(key_value_pairs)

        # Test ayield_keys with prefix
        retrieved_keys = [
            key async for key in store.ayield_keys(prefix=TEST_ROW_PREFIX)
        ]
        assert set(retrieved_keys) == {test_key_1, test_key_2}

        # Test ayield_keys with no prefix
        retrieved_all_keys = [key async for key in store.ayield_keys()]
        assert set(retrieved_all_keys) == {test_key_1, test_key_2, test_key_3}

        # Clean up
        await store.amdelete([test_key_1, test_key_2, test_key_3])

    def test_yield_keys_sync_raises_error(self, store: AsyncBigtableByteStore) -> None:
        """Test yield_keys sync method raising NotImplementedError."""
        with pytest.raises(NotImplementedError):
            [key for key in store.yield_keys()]

    @pytest.mark.asyncio
    async def test_amset_amget_empty_input(self, store: AsyncBigtableByteStore) -> None:
        """Test amset and amget with empty input"""
        await store.amset([])
        retrieved_values = await store.amget([])
        assert retrieved_values == []

    @pytest.mark.asyncio
    async def test_amset_large_value(self, store: AsyncBigtableByteStore) -> None:
        """Test amset with a large value."""
        test_key = TEST_ROW_PREFIX + uuid.uuid4().hex
        large_value = os.urandom(LARGE_VALUE_SIZE)  # 10MB
        await store.amset([(test_key, large_value)])
        retrieved_value = await store.amget([test_key])
        assert retrieved_value == [large_value]

    @pytest.mark.asyncio
    async def test_amset_large_number_of_keys(
        self, store: AsyncBigtableByteStore
    ) -> None:
        """Test amset with a large number of keys."""
        key_value_pairs = [
            (TEST_ROW_PREFIX + str(i), os.urandom(10)) for i in range(MAX_KEYS)
        ]
        await store.amset(key_value_pairs)
        retrieved_values = await store.amget([key for key, _ in key_value_pairs])
        assert len(retrieved_values) == MAX_KEYS
        for value in retrieved_values:
            assert value is not None

    @pytest.mark.asyncio
    async def test_duplicate_key_amset(self, store: AsyncBigtableByteStore) -> None:
        """Test amset with duplicate keys."""
        test_key = TEST_ROW_PREFIX + uuid.uuid4().hex
        test_value_1 = f"value-{uuid.uuid4().hex}".encode("utf-8")
        test_value_2 = f"value-{uuid.uuid4().hex}".encode("utf-8")

        await store.amset([(test_key, test_value_1)])
        await store.amset([(test_key, test_value_2)])
        retrieved_value = await store.amget([test_key])
        assert retrieved_value == [test_value_2]

    @pytest.mark.asyncio
    async def test_ayield_keys_prefix_no_matches(
        self, store: AsyncBigtableByteStore
    ) -> None:
        """Test ayield_keys with a prefix that does not match any key."""
        retrieved_keys = [
            key async for key in store.ayield_keys(prefix="no-match-prefix")
        ]
        assert retrieved_keys == []

    @pytest.mark.asyncio
    async def test_custom_column_family_qualifier(self, table: TableAsync) -> None:
        """Test with a custom column family and qualifier."""
        custom_store = AsyncBigtableByteStore(
            table, column_family=CUSTOM_COLUMN_FAMILY, column_qualifier=CUSTOM_COLUMN
        )
        test_key = TEST_ROW_PREFIX + uuid.uuid4().hex
        test_value = f"value-{uuid.uuid4().hex}".encode("utf-8")
        await custom_store.amset([(test_key, test_value)])
        retrieved_value = await custom_store.amget([test_key])
        assert retrieved_value == [test_value]

    @pytest.mark.asyncio
    async def test_invalid_key_type(self, store: AsyncBigtableByteStore) -> None:
        """Test that amset raises TypeError when a key is not a string."""
        with pytest.raises(TypeError, match="Keys must be of type 'str'."):
            await store.amset([(123, b"value")])  # type: ignore

    @pytest.mark.asyncio
    async def test_invalid_value_type(self, store: AsyncBigtableByteStore) -> None:
        """Test that amset raises TypeError when a value is not bytes."""
        with pytest.raises(TypeError, match="Values must be of type 'bytes'."):
            await store.amset([("key", "value")])  # type: ignore

    @pytest.mark.asyncio
    async def test_bigtable_mutation_error(self, store: AsyncBigtableByteStore) -> None:
        """Test that amset raises MutationsExceptionGroup when a mutation fails"""

        # Mock the bulk_mutate_rows method to raise a MutationsExceptionGroup
        async def mock_bulk_mutate_rows(mutations):
            # Create a mock failed_mutation_entry
            mock_failed_mutation_entry = mock.Mock()
            # Create a dummy cause
            dummy_cause = Exception("Dummy cause")
            failed_mutation = FailedMutationEntryError(
                failed_mutation_entry=mock_failed_mutation_entry,
                cause=dummy_cause,
                failed_idx=0,
            )
            raise MutationsExceptionGroup(
                [failed_mutation],
                len(mutations),
                "Test MutationsExceptionGroup message",
            )

        store.table.bulk_mutate_rows = mock_bulk_mutate_rows
        test_key = TEST_ROW_PREFIX + uuid.uuid4().hex
        test_value = f"value-{uuid.uuid4().hex}".encode("utf-8")

        with pytest.raises(MutationsExceptionGroup) as exc_info:
            await store.amset([(test_key, test_value)])
        assert "Test MutationsExceptionGroup message" in str(exc_info.value)
