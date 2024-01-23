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
import string
import uuid

from google.cloud import bigtable
from google.cloud.bigtable import column_family
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_bigtable import (
    BigtableChatMessageHistory,
)
import pytest


TABLE_ID_PREFIX = "test-table-"


@pytest.fixture
def client() -> bigtable.Client:
    yield bigtable.Client(
        project=get_env_var("PROJECT_ID", "ID of the GCP project"), admin=True
    )


@pytest.fixture
def instance_id() -> str:
    yield get_env_var("INSTANCE_ID", "ID of the Cloud Bigtable instance")


@pytest.fixture
def table_id(instance_id: str, client: bigtable.Client) -> str:
    table_id = TABLE_ID_PREFIX + "".join(
        random.choice(string.ascii_lowercase) for _ in range(10)
    )
    # Create table
    client.instance(instance_id).table(table_id).create(
        column_families={
            "langchain": column_family.MaxVersionsGCRule(1),
        }
    )

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

    history.init_schema()
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


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v
