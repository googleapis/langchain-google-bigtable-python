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

import asyncio
from concurrent.futures import Future
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import google.auth
from google.cloud import bigtable
from langchain_core.stores import BaseStore

from .async_key_value_store import AsyncBigtableByteStore
from .engine import BigtableEngine

if TYPE_CHECKING:
    import google.auth.credentials  # type: ignore

DEFAULT_COLUMN_FAMILY = "kv"
DEFAULT_COLUMN_QUALIFIER = "val".encode("utf-8")


def init_key_value_store_table(
    instance_id: str,
    table_id: str,
    project_id: Optional[str] = None,
    client: Optional[bigtable.Client] = None,
    column_families: List[str] = [DEFAULT_COLUMN_FAMILY],
) -> None:
    """
    Create a table for saving of LangChain Key-value pairs.

    Args:
        instance_id (str): The Instance ID for the table to be created for.
        table_id (str): The Table ID for the table.
        project_id (Optional[str]: The Cloud Project ID. It will be pulled from
              the environment if not passed.
        client (Optional[bigtable.Client]: The admin client to use for the table creation.
        column_families (Optional[List[str]]): The column families for the new table.

    Returns:
        None

    Raises:
        It raises ValueError if a table with the given table_id already exists.
    """

    if client is None:
        client = bigtable.Client(project=project_id, admin=True)

    table_client = client.instance(instance_id).table(table_id)

    if table_client.exists():
        raise ValueError(f"Table {table_id} already exists.")

    families: Dict[str, bigtable.column_family.MaxVersionsGCRule] = dict()
    for cf in column_families:
        families[cf] = bigtable.column_family.MaxVersionsGCRule(1)
    table_client.create(column_families=families)


class BigtableByteStore(BaseStore[str, bytes]):
    """
    LangChain Key-value store implementation for Google Cloud Bigtable, supporting
    both sync and async methods using BigtableEngine.
    """

    def __init__(
        self,
        engine: BigtableEngine,
        instance_id: str,
        table_id: str,
        column_family: str = DEFAULT_COLUMN_FAMILY,
        column_qualifier: bytes = DEFAULT_COLUMN_QUALIFIER,
        app_profile_id: Optional[str] = None,
    ):
        self._engine = engine
        self._instance_id = instance_id
        self._table_id = table_id
        self.app_profile_id = app_profile_id
        self._column_family = column_family
        self._column_qualifier = column_qualifier
        self._async_store: Optional[AsyncBigtableByteStore] = None

    @classmethod
    def create_sync(
        cls,
        instance_id: str,
        table_id: str,
        *,
        engine: Optional[BigtableEngine] = None,
        project_id: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        column_family: str = DEFAULT_COLUMN_FAMILY,
        column_qualifier: Union[str, bytes] = DEFAULT_COLUMN_QUALIFIER,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        client_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BigtableByteStore:
        """
        Creates a sync-initialized instance of the BigtableByteStore.

        This is the standard entry point for synchronous applications. It will create
        a new BigtableEngine if one is not provided.

        Args:
            instance_id (str): The ID of the Bigtable instance to connect to.
            table_id (str): The ID of the table to use for storing data.
            engine (BigtableEngine | None): An optional, existing BigtableEngine to share resources with.
                If you don't provide an engine, a new one will be created automatically.
                You can retrieve the created engine by calling the `get_engine()`
                method on the returned store and reuse it for other stores.
            project_id (str | None): The Google Cloud project ID. This is optional and will only be used if
                an engine is not provided.
            app_profile_id (str | None): An optional Bigtable app profile ID for routing requests.
            column_family (str | None): The column family to use for storing values. Defaults to "kv".
            column_qualifier (str | None): The column qualifier to use for storing values. Can be
                a string or bytes. Defaults to "val".
            credentials (google.auth.credentials.Credentials | None): An optional `google.auth.credentials.Credentials` object
                to use for authentication. If not provided, the default credentials
                will be used from the environment.
            client_options (dict[str, Any] | None): An optional dictionary of client options to pass to the
                `BigtableDataClientAsync`.

        Returns:
            An initialized BigtableByteStore instance.
        """
        my_engine: BigtableEngine
        if engine:
            my_engine = engine
        else:
            client_kwargs: dict[str, Any] = kwargs
            my_engine = BigtableEngine.initialize(
                project_id=project_id,
                credentials=credentials,
                client_options=client_options,
                **client_kwargs,
            )

        cq_bytes: bytes
        if isinstance(column_qualifier, str):
            cq_bytes = column_qualifier.encode("utf-8")
        else:
            cq_bytes = column_qualifier

        return cls(
            engine=my_engine,
            instance_id=instance_id,
            table_id=table_id,
            column_family=column_family,
            column_qualifier=cq_bytes,
            app_profile_id=app_profile_id,
        )

    @classmethod
    async def create(
        cls,
        instance_id: str,
        table_id: str,
        *,
        engine: Optional[BigtableEngine] = None,
        project_id: Optional[str] = None,
        app_profile_id: Optional[str] = None,
        column_family: str = DEFAULT_COLUMN_FAMILY,
        column_qualifier: Union[str, bytes] = DEFAULT_COLUMN_QUALIFIER,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        client_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BigtableByteStore:
        """
        Creates an async-initialized instance of the BigtableByteStore.

        This is the standard entry point for asynchronous applications. It will create
        a new BigtableEngine if one is not provided.

        Args:
            instance_id (str): The ID of the Bigtable instance to connect to.
            table_id (str): The ID of the table to use for storing data.
            engine (BigtableEngine | None): An optional, existing BigtableEngine to share resources with.
                If you don't provide an engine, a new one will be created automatically.
                You can retrieve the created engine by calling the `get_engine()`
                method on the returned store and reuse it for other stores.
            project_id (str | None): The Google Cloud project ID. This is optional and will only be used if
                an engine is not provided.
            app_profile_id (str | None): An optional Bigtable app profile ID for routing requests.
            column_family (str | None): The column family to use for storing values. Defaults to "kv".
            column_qualifier (str | None): The column qualifier to use for storing values. Can be
                a string or bytes. Defaults to "val".
            credentials (google.auth.credentials.Credentials | None): An optional `google.auth.credentials.Credentials` object
                to use for authentication. If not provided, the default credentials
                will be used from the environment.
            client_options (dict[str, Any] | None): An optional dictionary of client options to pass to the
                `BigtableDataClientAsync`.

        Returns:
            An initialized BigtableByteStore instance.
        """
        my_engine: BigtableEngine
        if engine:
            my_engine = engine
        else:
            client_kwargs: dict[str, Any] = kwargs
            my_engine = await BigtableEngine.async_initialize(
                project_id=project_id,
                credentials=credentials,
                client_options=client_options,
                **client_kwargs,
            )

        cq_bytes: bytes
        if isinstance(column_qualifier, str):
            cq_bytes = column_qualifier.encode("utf-8")
        else:
            cq_bytes = column_qualifier

        return cls(
            engine=my_engine,
            instance_id=instance_id,
            table_id=table_id,
            column_family=column_family,
            column_qualifier=cq_bytes,
            app_profile_id=app_profile_id,
        )

    async def _get_async_store(self, **kwargs: Any) -> AsyncBigtableByteStore:
        """
        Returns a AsyncBigtableByteStore object to be used for data operations.
        If one is not available, a new one is created.
        """
        if not self._async_store:
            async_table = await self._engine.get_async_table(
                self._instance_id,
                self._table_id,
                app_profile_id=self.app_profile_id,
                **kwargs,
            )
            self._async_store = AsyncBigtableByteStore(
                async_table, self._column_family, self._column_qualifier
            )
        return self._async_store

    def get_engine(self) -> BigtableEngine:
        """Returns the BigtableEngine being used for this object."""
        return self._engine

    async def amget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """
        Asynchronously retrieves values for a sequence of keys.

        It only reads the most recent version for each key.

        Args:
            keys: A sequence of keys to retrieve values for.

        Returns:
            A list of byte values corresponding to the input keys. If a key is not
            found, `None` is returned for that key's position in the list.
        """

        async def _internal():
            store = await self._get_async_store()
            return await store.amget(keys)

        return await self._engine._run_as_async(_internal())

    async def amset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """
        Asynchronously stores key-value pairs in the Bigtable.

        Args:
            key_value_pairs: A sequence of (key, value) tuples to store.

        Raises:
            TypeError: If any key is not a string or any value is not bytes.
        """

        async def _internal():
            store = await self._get_async_store()
            return await store.amset(key_value_pairs)

        await self._engine._run_as_async(_internal())

    async def amdelete(self, keys: Sequence[str]) -> None:
        """
        Asynchronously deletes key-value pairs from the Bigtable.

        Args:
            keys: A sequence of keys to delete.
        """

        async def _internal():
            store = await self._get_async_store()
            return await store.amdelete(keys)

        await self._engine._run_as_async(_internal())

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        """
        Asynchronously yields keys matching a given prefix.
        It only yields the row keys that match the given prefix.

        Args:
           prefix: An optional prefix to filter keys by. If `None` or an empty
             string, all keys are yielded.

        Yields:
            Keys from the table that match a given prefix.
        """
        caller_loop = asyncio.get_running_loop()
        engine_loop = self._engine._loop

        q: asyncio.Queue = asyncio.Queue()
        done = object()
        producer_future: Optional[Future] = None

        async def _producer():
            # This coroutine runs on the BigtableEngine background loop.
            try:
                store = await self._engine._run_as_async(self._get_async_store())
                async for key in store.ayield_keys(prefix=prefix):
                    # The queue was created within the caller_loop(main running event loop) context
                    # Hence, its method is called within that loop
                    caller_loop.call_soon_threadsafe(q.put_nowait, key)
            except Exception as e:
                caller_loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                caller_loop.call_soon_threadsafe(q.put_nowait, done)

        producer_future = asyncio.run_coroutine_threadsafe(_producer(), engine_loop)  # type: ignore

        while True:
            item = await q.get()
            if item is done:
                break
            elif isinstance(item, Exception):
                if producer_future and producer_future.done():
                    try:
                        producer_future.result()
                    except Exception as fut_e:
                        raise fut_e from item
                raise item
            else:
                yield item

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """
        Synchronously retrieves values for a sequence of keys.

        It only reads the most recent version for each key.

        Args:
            keys: A sequence of keys to retrieve values for.

        Returns:
            A list of byte values corresponding to the input keys. If a key is not
            found, `None` is returned for that key's position in the list.
        """

        async def _internal():
            store = await self._get_async_store()
            return await store.amget(keys)

        return self._engine._run_as_sync(_internal())

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """
        Synchronously stores key-value pairs in the Bigtable.

        Args:
            key_value_pairs: A sequence of (key, value) tuples to store.

        Raises:
            TypeError: If any key is not a string or any value is not bytes.
        """

        async def _internal():
            store = await self._get_async_store()
            return await store.amset(key_value_pairs)

        self._engine._run_as_sync(_internal())

    def mdelete(self, keys: Sequence[str]) -> None:
        """
        Synchronously deletes key-value pairs from the Bigtable.

        Args:
            keys: A sequence of keys to delete.
        """

        async def _internal():
            store = await self._get_async_store()
            return await store.amdelete(keys)

        self._engine._run_as_sync(_internal())

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """
        Synchronously yields keys matching a given prefix.
        It only yields the keys that match the given prefix.

        Args:
           prefix: An optional prefix to filter keys by. If `None` or an empty
             string, all keys are yielded.

        Yields:
            Keys from the table that match a given prefix.
        """
        done = object()

        async def _create_queue():
            queue: asyncio.Queue = asyncio.Queue()
            return queue

        queue_creation_future = asyncio.run_coroutine_threadsafe(
            _create_queue(), self._engine._loop  # type: ignore
        )
        q = queue_creation_future.result()

        async def _producer(queue):
            try:
                store = await self._get_async_store()
                async for key in store.ayield_keys(prefix=prefix):
                    await queue.put(key)
            except Exception as e:
                await queue.put(e)
            finally:
                await queue.put(done)

        producer_future = asyncio.run_coroutine_threadsafe(
            _producer(q), self._engine._loop  # type: ignore
        )

        while True:
            get_future = asyncio.run_coroutine_threadsafe(q.get(), self._engine._loop)  # type: ignore
            item = get_future.result()

            if item is done:
                break
            elif isinstance(item, Exception):
                # attempt to retrieve the full exception from the producer future
                if producer_future and producer_future.done():
                    try:
                        producer_future.result()  # might re-raise a more specific exception
                    except Exception as fut_e:
                        raise fut_e from item
                raise item  # raise the exception we got from the queue
            else:
                yield item
