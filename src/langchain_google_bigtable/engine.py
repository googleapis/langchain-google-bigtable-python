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
import json
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple

from google.cloud.bigtable.data import BigtableDataClientAsync


class BigtableEngine:
    """
    Manages the client and execution context, handling the
    async/sync conversion via a background event loop.

    This class is the core of the async/sync interoperability, providing a
    reusable component that can be shared across multiple store instances to
    conserve resources.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop, thread: Thread, client: BigtableDataClientAsync):
        """
        Initializes the engine with a running event loop and a client.

        Args:
            loop: The asyncio event loop running in the background thread.
            thread: The background thread hosting the event loop.
            client: The async Bigtable data client.
        """
        self._loop = loop
        self._thread = thread
        self._client = client

    @classmethod
    def sync_initialize(cls, client: BigtableDataClientAsync) -> "BigtableEngine":
        """
        Creates a new engine instance, synchronously.

        This method is the standard way to create an engine. It spins up a new
        background thread and event loop to handle all async operations.

        Args:
            client: An optional pre-configured async Bigtable client. If not
                    provided, a default one will be created.

        Returns:
            A new, fully initialized BigtableEngine instance.
        """
        if not client:
            client = BigtableDataClientAsync()
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        return cls(loop, thread, client)

    def _run_as_sync(self, coro: Any) -> Any:
        """
        Runs a coroutine on the background loop and waits for the result.
        This is the core mechanism for providing a synchronous API.

        Args:
            coro: The coroutine to execute.

        Returns:
            The result of the coroutine.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _run_as_async(self, coro: Any) -> Any:
        """
        Runs a coroutine on the background loop without blocking the main loop.
        This is for calling from an existing async context.

        Args:
            coro: The coroutine to execute.

        Returns:
            An awaitable future that resolves with the result of the coroutine.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)
