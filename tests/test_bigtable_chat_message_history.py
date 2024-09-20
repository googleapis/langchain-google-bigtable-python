# Copyright 2024 Google LLC
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
import random
import re
import string
import time
import uuid
from typing import Iterator

import pytest
from google.cloud import bigtable  # type: ignore
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_google_bigtable.chat_message_history import (
    BigtableChatMessageHistory,
    init_chat_history_table,
)

TABLE_ID_PREFIX = "test-table-history-"


@pytest.fixture
def client() -> Iterator[bigtable.Client]:
    yield bigtable.Client(
        project=get_env_var("PROJECT_ID", "ID of the GCP project"), admin=True
    )


@pytest.fixture
def instance_id() -> Iterator[str]:
    yield get_env_var("INSTANCE_ID", "ID of the Cloud Bigtable instance")


@pytest.fixture
def table_id(instance_id: str, client: bigtable.Client) -> Iterator[str]:
    table_id = TABLE_ID_PREFIX + "".join(
        random.choice(string.ascii_lowercase) for _ in range(10)
    )
    # Create table and column family
    init_chat_history_table(instance_id=instance_id, table_id=table_id, client=client)

    yield table_id

    # Teardown
    client.instance(instance_id).table(table_id).delete()


def test_bigtable_full_workflow(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    session_id = uuid.uuid4().hex
    history = BigtableChatMessageHistory(
        instance_id, table_id, session_id, client=client
    )

    history.add_ai_message("Hey! I am AI!")
    history.add_user_message("Hey! I am human!")
    messages = history.messages

    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Hey! I am AI!"
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Hey! I am human!"

    history.clear()
    assert len(history.messages) == 0


def get_index_from_message(message: BaseMessage) -> int:
    match = re.search("^Hey! I am (AI|human)! Index: ([0-9]+)$", str(message.content))
    if match:
        return int(match[2])
    return 0


def test_bigtable_loads_of_messages(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    NUM_MESSAGES = 10000
    session_id = uuid.uuid4().hex
    history = BigtableChatMessageHistory(
        instance_id, table_id, session_id, client=client
    )

    ai_messages = []
    human_messages = []
    for i in range(NUM_MESSAGES):
        ai_messages.append(AIMessage(content=f"Hey! I am AI! Index: {2*i}"))
        human_messages.append(HumanMessage(content=f"Hey! I am human! Index: {2*i+1}"))
    history.add_messages(ai_messages)
    history.add_messages(human_messages)

    # wait for eventual consistency
    time.sleep(5)

    messages = history.messages

    assert len(messages) == 2 * NUM_MESSAGES

    messages.sort(key=get_index_from_message)

    for i in range(2 * NUM_MESSAGES):
        type = AIMessage if i % 2 == 0 else HumanMessage
        content = (
            f"Hey! I am AI! Index: {i}"
            if i % 2 == 0
            else f"Hey! I am human! Index: {i}"
        )
        assert isinstance(messages[i], type)
        assert messages[i].content == content

    history.clear()
    assert len(history.messages) == 0


def test_bigtable_multiple_sessions(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    session_id1 = uuid.uuid4().hex
    history1 = BigtableChatMessageHistory(
        instance_id, table_id, session_id1, client=client
    )
    session_id2 = uuid.uuid4().hex
    history2 = BigtableChatMessageHistory(
        instance_id, table_id, session_id2, client=client
    )

    history1.add_ai_message("Hey! I am AI!")
    history2.add_user_message("Hey! I am human!")
    messages1 = history1.messages
    messages2 = history2.messages

    assert len(messages1) == 1
    assert len(messages2) == 1
    assert isinstance(messages1[0], AIMessage)
    assert messages1[0].content == "Hey! I am AI!"
    assert isinstance(messages2[0], HumanMessage)
    assert messages2[0].content == "Hey! I am human!"

    history1.clear()
    assert len(history1.messages) == 0
    assert len(history2.messages) == 1

    history2.clear()
    assert len(history1.messages) == 0
    assert len(history2.messages) == 0


def test_bigtable_missing_instance(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    non_existent_instance_id = "non-existent"
    with pytest.raises(NameError) as excinfo:
        BigtableChatMessageHistory(
            non_existent_instance_id, table_id, "", client=client
        )

    assert str(excinfo.value) == f"Instance {non_existent_instance_id} does not exist"


def test_bigtable_missing_table(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    non_existent_table_id = "non_existent"
    with pytest.raises(NameError) as excinfo:
        BigtableChatMessageHistory(
            instance_id, non_existent_table_id, "", client=client
        )
    assert (
        str(excinfo.value)
        == f"Table {non_existent_table_id} does not exist on instance {instance_id}"
    )


def test_bigtable_missing_column_family(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    other_table_id = table_id + "1"
    client.instance(instance_id).table(other_table_id).create()

    with pytest.raises(NameError) as excinfo:
        BigtableChatMessageHistory(instance_id, other_table_id, "", client=client)
    assert (
        str(excinfo.value)
        == f"Column family langchain does not exist on table {other_table_id}"
    )

    client.instance(instance_id).table(other_table_id).delete()


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v
