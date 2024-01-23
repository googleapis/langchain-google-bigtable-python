import pytest
import uuid
import string
import random
import os

from langchain_google_bigtable import (
    BigtableChatMessageHistory,
)
from langchain_core.messages import AIMessage, HumanMessage

from google.cloud import bigtable  # noqa: F401
from google.cloud.bigtable import column_family


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
