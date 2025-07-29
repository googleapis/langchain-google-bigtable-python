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
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
)

import google.auth
from google.cloud.bigtable.data import BigtableDataClientAsync, TableAsync

if TYPE_CHECKING:
    import google.auth.credentials  # type: ignore


class BigtableEngine:
    """Manages the client and execution context, handling the

    async/sync conversion via a background event loop.

    This class is the core of the async/sync interoperability, providing a
    reusable component that can be shared across multiple store instances to
    conserve resources.
    """

    _default_loop: Optional[asyncio.AbstractEventLoop] = None
    _default_thread: Optional[Thread] = None
    __create_key = object()

    def __init__(
        self,
        key: object,
        client: Optional[BigtableDataClientAsync],
        loop: Optional[asyncio.AbstractEventLoop],
        thread: Optional[Thread],
    ) -> None:
        """Initializes the engine with a running event loop and a client.

        Args:
            key (object): object to prevent direct constructor usage.
            client (BigtableDataClientAsync): The async Bigtable data client.
            loop (Optional[asyncio.AbstractEventLoop]): The asyncio event loop
              running in the background thread.
            thread (Optional[Thread]): The background thread hosting the event loop.
        """
        if key != BigtableEngine.__create_key:
            raise Exception("Use factory method 'initialize'")
        self._client = client
        self._loop = loop
        self._thread = thread

    @classmethod
    def __start_background_loop(
        cls,
        project_id: Optional[str],
        credentials: Optional[google.auth.credentials.Credentials] = None,
        client_options: Optional[Any] = None,
        **kwargs: Any,
    ) -> Future:
        """Creates and starts the default background loop and thread"""
        if cls._default_loop is None or cls._default_loop.is_closed():
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()

        coro = cls._create(
            project_id=project_id,
            loop=cls._default_loop,
            thread=cls._default_thread,
            credentials=credentials,
            client_options=client_options,
            **kwargs,
        )

        return asyncio.run_coroutine_threadsafe(coro, cls._default_loop)

    @classmethod
    async def _create(
        cls,
        project_id: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        thread: Optional[Thread] = None,
        client: Optional[BigtableDataClientAsync] = None,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        client_options: Optional[Any] = None,
        **kwargs: Any,
    ) -> BigtableEngine:
        """Asynchronously instantiates the BigtableEngine Object"""
        if not client:
            client = BigtableDataClientAsync(
                project=project_id,
                credentials=credentials,
                client_options=client_options,
                **kwargs,
            )
        return cls(cls.__create_key, client, loop, thread)

    @classmethod
    async def async_initialize(
        cls,
        project_id: Optional[str] = None,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        client_options: Optional[Any] = None,
        **kwargs: Any,
    ) -> BigtableEngine:
        """Creates a BigtableEngine instance with a background event loop and a new data client asynchronously

        Args:
            project_id (Optional[str]): Google Cloud Project ID.
            credentials (Optional[google.auth.credentials.Credentials]): credentials
              to pass into the data client for this engine.
            client_options (Optional[Any]): Client options used to set user options
              for the client.

        Returns:
            A BigtableEngine Object
        """

        future = cls.__start_background_loop(
            project_id=project_id,
            credentials=credentials,
            client_options=client_options,
            **kwargs,
        )

        return await asyncio.wrap_future(future)

    @classmethod
    def initialize(
        cls,
        project_id: Optional[str] = None,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        client_options: Optional[Any] = None,
        **kwargs: Any,
    ) -> BigtableEngine:
        """Creates a BigtableEngine instance with a background event loop and a new data client synchronously.

        Args:
            project_id (Optional[str]): Google Cloud Project ID.
            credentials (Optional[google.auth.credentials.Credentials]): credentials
              to pass into the data client for this engine.
            client_options (Optional[Any]): Client options used to set user options
              for the client.

        Returns:
            A BigtableEngine Object
        """

        future = cls.__start_background_loop(
            project_id=project_id,
            credentials=credentials,
            client_options=client_options,
            **kwargs,
        )

        return future.result()

    @property
    def async_client(self) -> BigtableDataClientAsync:
        """The data client property of this class."""
        if not self._client:
            raise RuntimeError("Client not initialized.")
        return self._client

    async def get_async_table(
        self,
        instance_id: str,
        table_id: str,
        app_profile_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TableAsync:
        """Returns the table using this class's client"""
        return self.async_client.get_table(
            instance_id, table_id, app_profile_id=app_profile_id, **kwargs
        )

    def _run_as_sync(self, coro: Any) -> Any:
        """Runs a coroutine on the background loop and waits for the result.

        This is the core mechanism for providing a synchronous API.

        Args:
            coro: The coroutine to execute.

        Returns:
            The result of the coroutine.
        """
        if not self._loop:
            raise Exception(
                "Engine was not initialized with a background loop for sync methods."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _run_as_async(self, coro: Any) -> Any:
        """Runs a coroutine on the background loop without blocking the main loop.

        This is used for calling from an existing asynchronous context.

        Args:
            coro: The coroutine to execute.

        Returns:
            An awaitable future that resolves with the result of the coroutine.
        """
        if not self._loop or not self._loop.is_running():
            return await coro

        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        )

    async def close(self) -> None:
        """Closes the underlying client for this specific engine instance."""
        if self._client:
            close_coro = self._client.close()
            if self._loop and self._loop.is_running():
                # Runs the close operation on the loop associated with this engine
                future = asyncio.run_coroutine_threadsafe(close_coro, self._loop)
                try:
                    await asyncio.wrap_future(future)
                except Exception as e:
                    raise e
            else:
                # Fallback if loop is not running
                try:
                    await close_coro
                except Exception as e:
                    raise e
            self._client = None  # type: ignore

    @classmethod
    async def shutdown_default_loop(cls) -> None:
        """
        Closes the default class-level shared loop and terminates the thread associated with it.

        Note: Calling this method will prevent any new BigtableEngine instances
        from using the shared event loop. Additionally, after this method is called
        it will not be possible to run more coroutines in the previous loop.

        Raises:
            Exception: If the thread does not terminate within the timeout period.
        """
        loop = cls._default_loop
        thread = cls._default_thread

        # Clear class references
        cls._default_loop = None
        cls._default_thread = None

        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread:
            try:
                thread.join(timeout=20.0)
            finally:
                if thread.is_alive():
                    raise Exception(
                        "Warning: BigtableEngine default thread did not terminate."
                    )
                else:
                    if loop:
                        loop.close()  # Close the loop for resource cleanup.
