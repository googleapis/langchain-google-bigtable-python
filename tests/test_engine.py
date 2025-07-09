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
from threading import Thread
from typing import Iterator
from unittest.mock import Mock

import pytest

from langchain_google_bigtable.engine import BigtableEngine


@pytest.fixture
def engine() -> Iterator[BigtableEngine]:
    mock_client = Mock()
    engine_instance = BigtableEngine.sync_initialize(client=mock_client)

    yield engine_instance

    if engine_instance._loop.is_running():
        engine_instance._loop.call_soon_threadsafe(engine_instance._loop.stop)
        engine_instance._thread.join(timeout=5)

    assert not engine_instance._thread.is_alive()


async def simple_test_coroutine(value: str) -> str:
    await asyncio.sleep(0)
    return f"processed: {value}"


async def coroutine_that_raises() -> None:
    await asyncio.sleep(0)
    raise ValueError("Something went wrong inside the coroutine")


def test_sync_initialize_with_client() -> None:
    mock_client = Mock()
    engine_instance = BigtableEngine.sync_initialize(client=mock_client)

    assert isinstance(engine_instance, BigtableEngine)
    assert isinstance(engine_instance._loop, asyncio.AbstractEventLoop)
    assert isinstance(engine_instance._thread, Thread)
    assert engine_instance.client == mock_client
    assert engine_instance._thread.is_alive()
    assert engine_instance._loop.is_running()

    engine_instance._loop.call_soon_threadsafe(engine_instance._loop.stop)
    engine_instance._thread.join(timeout=5)


def test_run_as_sync_success(engine: BigtableEngine) -> None:
    test_value = "sync test"
    result = engine._run_as_sync(simple_test_coroutine(test_value))
    assert result == f"processed: {test_value}"


def test_run_as_sync_propagates_exception(engine: BigtableEngine) -> None:
    with pytest.raises(ValueError, match="Something went wrong"):
        engine._run_as_sync(coroutine_that_raises())


@pytest.mark.asyncio
async def test_run_as_async_success(engine: BigtableEngine) -> None:
    test_value = "async test"
    result = await engine._run_as_async(simple_test_coroutine(test_value))
    assert result == f"processed: {test_value}"


@pytest.mark.asyncio
async def test_run_as_async_propagates_exception(engine: BigtableEngine) -> None:
    with pytest.raises(ValueError, match="Something went wrong"):
        await engine._run_as_async(coroutine_that_raises())
