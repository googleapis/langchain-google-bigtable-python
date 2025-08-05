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
from typing import AsyncIterator, Iterator, List, Tuple

import pytest
import pytest_asyncio
from google.api_core import exceptions
from google.cloud import bigtable

from langchain_google_bigtable.engine import BigtableEngine
from langchain_google_bigtable.key_value_store import (
    BigtableByteStore,
    init_key_value_store_table,
)

TEST_VALUE_COLUMN_FAMILY = "val"
# Test if additional column families can be created
TEST_ADDITIONAL_COLUMN_FAMILY = "cf2"


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
def managed_table(
    project_id: str, instance_id: str
) -> Iterator[tuple[str, str, List[str]]]:
    """
    A fixture that creates a unique Bigtable table.
    """
    column_families = [TEST_VALUE_COLUMN_FAMILY, TEST_ADDITIONAL_COLUMN_FAMILY]
    client = bigtable.Client(project=project_id, admin=True)
    instance = client.instance(instance_id)
    table_id = f"test-table-{uuid.uuid4().hex[:8]}"

    # Create the table
    init_key_value_store_table(
        instance_id=instance_id,
        table_id=table_id,
        project_id=project_id,
        client=client,
        column_families=column_families,
    )

    yield instance_id, table_id, column_families

    # Teardown
    try:
        instance.table(table_id).delete()
        client.close()
    except exceptions.NotFound:
        pass


class TestTableInitialization:
    """Tests for the table creation utility."""

    def test_init_table_already_exists(
        self, managed_table: tuple[str, str, List[str]], project_id: str
    ) -> None:
        """
        Verifies that `init_key_value_store_table` raises a ValueError
        if the table already exists, preventing accidental overwrites.
        """
        instance_id, table_id, _ = managed_table
        with pytest.raises(ValueError):
            init_key_value_store_table(
                instance_id=instance_id, table_id=table_id, project_id=project_id
            )


class TestBigtableByteStoreSync:
    """Comprehensive tests for the synchronous methods of BigtableByteStore."""

    @pytest_asyncio.fixture(scope="class")
    async def sync_store(
        self, managed_table: tuple[str, str, List[str]], project_id: str
    ) -> AsyncIterator[BigtableByteStore]:
        """Provides a sync store for the class, cleaning up the engine at the end."""
        instance_id, table_id, _ = managed_table
        store = BigtableByteStore.create_sync(
            instance_id=instance_id,
            table_id=table_id,
            project_id=project_id,
            column_family=TEST_VALUE_COLUMN_FAMILY,
        )
        yield store
        await store.get_engine().close()

    def test_sync_full_lifecycle(self, sync_store: BigtableByteStore) -> None:
        """
        Tests the complete mset -> mget -> overwrite -> mdelete -> mget -> yield_keys cycle.
        """
        # Initial Set
        sync_store.mset([("key1", b"value1"), ("key2", b"value2")])
        results = sync_store.mget(["key1", "key2", "nonexistent"])
        assert results == [b"value1", b"value2", None]

        # Overwrite
        sync_store.mset([("key1", b"value1_overwritten")])
        result = sync_store.mget(["key1"])
        assert result == [b"value1_overwritten"]

        # Delete
        sync_store.mdelete(["key2", "nonexistent_to_delete"])
        results = sync_store.mget(["key1", "key2"])
        assert results == [b"value1_overwritten", None]

        # Yield Keys
        sync_store.mdelete(list(sync_store.yield_keys()))  # Clean Slate
        sync_store.mset([("key1", b"value1"), ("ignored_key", b"value2")])
        matched_keys = sorted(list(sync_store.yield_keys(prefix="key")))
        assert matched_keys == ["key1"]

    @pytest.mark.asyncio
    async def test_async_full_lifecycle(self, sync_store: BigtableByteStore) -> None:
        """
        Tests the complete amset -> amget -> overwrite -> amdelete -> amget -> ayield_keys cycle.
        """
        # Initial Set
        await sync_store.amset([("akey1", b"avalue1"), ("akey2", b"avalue2")])
        results = await sync_store.amget(["akey1", "akey2", "anonexistent"])
        assert results == [b"avalue1", b"avalue2", None]

        # Overwrite
        await sync_store.amset([("akey1", b"avalue1_overwritten")])
        result = await sync_store.amget(["akey1"])
        assert result == [b"avalue1_overwritten"]

        # Delete
        await sync_store.amdelete(["akey2", "anonexistent_to_delete"])
        results = await sync_store.amget(["akey1", "akey2"])
        assert results == [b"avalue1_overwritten", None]

        # Yield Keys
        await sync_store.amdelete(
            [k async for k in sync_store.ayield_keys()]
        )  # Clean Slate
        await sync_store.amset([("key1", b"value1"), ("ignored_key", b"value2")])
        matched_keys = sorted([k async for k in sync_store.ayield_keys(prefix="key")])
        assert matched_keys == ["key1"]

    def test_sync_empty_and_noop_operations(
        self, sync_store: BigtableByteStore
    ) -> None:
        """Ensures that operations with empty inputs behave correctly."""
        # Set a baseline value
        sync_store.mset([("key_empty_test", b"value")])

        # mset with empty list should be a no-op
        sync_store.mset([])
        assert sync_store.mget(["key_empty_test"]) == [b"value"]

        # mget with empty list should return an empty list
        assert sync_store.mget([]) == []

        # mdelete with empty list should be a no-op
        sync_store.mdelete([])
        assert sync_store.mget(["key_empty_test"]) == [b"value"]

    def test_sync_yield_keys(self, sync_store: BigtableByteStore) -> None:
        """Tests yielding keys with and without a prefix."""
        # Clears table for this test.
        sync_store.mdelete(list(sync_store.yield_keys()))  # Clean slate

        keys_to_set = [
            ("user/1", b"u1"),
            ("user/2", b"u2"),
            ("org/1", b"o1"),
            ("session/abc", b"s1"),
        ]
        sync_store.mset(keys_to_set)

        # Yield all keys
        all_keys = sorted(list(sync_store.yield_keys()))
        assert all_keys == ["org/1", "session/abc", "user/1", "user/2"]

        # Yield with a prefix
        user_keys = sorted(list(sync_store.yield_keys(prefix="user/")))
        assert user_keys == ["user/1", "user/2"]

        # Yield with a non-matching prefix
        assert list(sync_store.yield_keys(prefix="nonexistent/")) == []

        # Yield on an empty table (after deleting)
        sync_store.mdelete(["user/1", "user/2", "org/1", "session/abc"])
        assert list(sync_store.yield_keys()) == []


@pytest.mark.asyncio
class TestBigtableByteStoreAsync:
    """Comprehensive tests for the asynchronous methods of BigtableByteStore."""

    @pytest_asyncio.fixture(scope="class")
    async def async_store(
        self, managed_table: tuple[str, str, List[str]], project_id: str
    ) -> AsyncIterator[BigtableByteStore]:
        """Provides an async store for the class, cleaning up the engine at the end."""
        instance_id, table_id, _ = managed_table
        store = await BigtableByteStore.create(
            instance_id=instance_id,
            table_id=table_id,
            project_id=project_id,
            column_family=TEST_VALUE_COLUMN_FAMILY,
        )
        yield store
        await store.get_engine().close()

    async def test_async_full_lifecycle(self, async_store: BigtableByteStore) -> None:
        """
        Tests the complete amset -> amget -> overwrite -> amdelete -> amget -> ayield_keys cycle.
        """
        # Initial Set
        await async_store.amset([("akey1", b"avalue1"), ("akey2", b"avalue2")])
        results = await async_store.amget(["akey1", "akey2", "anonexistent"])
        assert results == [b"avalue1", b"avalue2", None]

        # Overwrite
        await async_store.amset([("akey1", b"avalue1_overwritten")])
        result = await async_store.amget(["akey1"])
        assert result == [b"avalue1_overwritten"]

        # Delete
        await async_store.amdelete(["akey2", "anonexistent_to_delete"])
        results = await async_store.amget(["akey1", "akey2"])
        assert results == [b"avalue1_overwritten", None]

        # Yield Keys
        await async_store.amdelete(
            [k async for k in async_store.ayield_keys()]
        )  # Clean Slate
        await async_store.amset([("key1", b"value1"), ("ignored_key", b"value2")])
        matched_keys = sorted([k async for k in async_store.ayield_keys(prefix="key")])
        assert matched_keys == ["key1"]

    async def test_sync_full_lifecycle(self, async_store: BigtableByteStore) -> None:
        """
        Tests the complete mset -> mget -> overwrite -> mdelete -> mget -> yield_keys cycle.
        """
        # Initial Set
        async_store.mset([("key1", b"value1"), ("key2", b"value2")])
        results = async_store.mget(["key1", "key2", "nonexistent"])
        assert results == [b"value1", b"value2", None]

        # Overwrite
        async_store.mset([("key1", b"value1_overwritten")])
        result = async_store.mget(["key1"])
        assert result == [b"value1_overwritten"]

        # Delete
        async_store.mdelete(["key2", "nonexistent_to_delete"])
        results = async_store.mget(["key1", "key2"])
        assert results == [b"value1_overwritten", None]

        # Yield Keys
        async_store.mdelete(list(async_store.yield_keys()))  # Clean Slate
        async_store.mset([("key1", b"value1"), ("ignored_key", b"value2")])
        matched_keys = sorted(list(async_store.yield_keys(prefix="key")))
        assert matched_keys == ["key1"]

    async def test_async_empty_and_noop_operations(
        self, async_store: BigtableByteStore
    ) -> None:
        """Ensures that async operations with empty inputs behave correctly."""
        await async_store.amset([("akey_empty_test", b"value")])
        await async_store.amset([])
        assert await async_store.amget(["akey_empty_test"]) == [b"value"]
        assert await async_store.amget([]) == []
        await async_store.amdelete([])
        assert await async_store.amget(["akey_empty_test"]) == [b"value"]

    async def test_async_ayield_keys(self, async_store: BigtableByteStore) -> None:
        """Tests ayield_keys with and without a prefix."""
        await async_store.amdelete(
            [k async for k in async_store.ayield_keys()]
        )  # Clean slate

        keys_to_set = [
            ("async/user/1", b"u1"),
            ("async/user/2", b"u2"),
            ("async/org/1", b"o1"),
        ]
        await async_store.amset(keys_to_set)

        all_keys = sorted([key async for key in async_store.ayield_keys()])
        assert all_keys == ["async/org/1", "async/user/1", "async/user/2"]

        user_keys = sorted(
            [key async for key in async_store.ayield_keys(prefix="async/user/")]
        )
        assert user_keys == ["async/user/1", "async/user/2"]


class TestAdvancedScenarios:
    """Tests for advanced configurations, error handling, and resource management."""

    @pytest.mark.asyncio
    async def test_custom_column_family_and_qualifier(
        self, managed_table: tuple[str, str, List[str]]
    ) -> None:
        """
        Verifies the store works correctly with a non-default column family
        and a custom column qualifier (both str and bytes).
        """
        instance_id, table_id, _ = managed_table

        # Test with string qualifier
        store_str_cq = BigtableByteStore.create_sync(
            instance_id=instance_id,
            table_id=table_id,
            column_family=TEST_ADDITIONAL_COLUMN_FAMILY,
            column_qualifier="custom_qual",
        )
        store_str_cq.mset([("key1", b"val1")])
        assert store_str_cq.mget(["key1"]) == [b"val1"]
        await store_str_cq.get_engine().close()

        # Test with bytes qualifier
        store_bytes_cq = BigtableByteStore.create_sync(
            instance_id=instance_id,
            table_id=table_id,
            column_family=TEST_ADDITIONAL_COLUMN_FAMILY,
            column_qualifier=b"custom_qual_bytes",
        )
        store_bytes_cq.mset([("key2", b"val2")])
        assert store_bytes_cq.mget(["key2"]) == [b"val2"]
        await store_bytes_cq.get_engine().close()

    @pytest.mark.asyncio
    async def test_reusing_engine_across_stores(
        self, managed_table: tuple[str, str, List[str]]
    ) -> None:
        """
        Ensures that two stores can share a single BigtableEngine, which is an
        important resource management pattern.
        """
        instance_id, table_id, _ = managed_table

        # Create a single, shared engine
        engine = await BigtableEngine.async_initialize()

        # Create two stores using the same engine
        store1 = await BigtableByteStore.create(
            instance_id=instance_id,
            table_id=table_id,
            engine=engine,
            column_family=TEST_VALUE_COLUMN_FAMILY,
        )
        store2 = await BigtableByteStore.create(
            instance_id=instance_id,
            table_id=table_id,
            engine=engine,
            column_family=TEST_ADDITIONAL_COLUMN_FAMILY,
        )

        # Write with one store, read with the other
        await store1.amset([("key_1", b"value_1")])
        await store2.amset([("key_2", b"value_2")])
        result_1 = await store1.amget(["key_1"])
        result_2 = await store2.amget(["key_2"])

        assert result_1 == [b"value_1"]
        assert result_2 == [b"value_2"]

        # Clean up the shared engine
        await engine.close()

    @pytest.mark.asyncio
    async def test_operating_on_non_existent_table_raises_error(
        self, instance_id: str
    ) -> None:
        """
        Verifies that attempting to use a store pointed at a table that
        does not exist raises an error, as expected.
        """
        fake_table_id = "this-table-does-not-exist"
        store = BigtableByteStore.create_sync(
            instance_id=instance_id, table_id=fake_table_id
        )

        with pytest.raises(exceptions.NotFound):
            store.mget(["any_key"])

        await store.get_engine().close()
