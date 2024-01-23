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

import json
import os
import random
import string

import pytest
from google.cloud import bigtable
from google.cloud.bigtable import column_family, row_filters
from langchain_core.documents import Document

from langchain_google_bigtable import (
    BigtableLoader,
    BigtableSaver,
    Encoding,
    MetadataMapping,
)

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
            "non_default_family": column_family.MaxVersionsGCRule(1),
            "my_int_family": column_family.MaxVersionsGCRule(1),
            "my_custom_family": column_family.MaxVersionsGCRule(1),
        }
    )

    yield table_id

    # Teardown
    client.instance(instance_id).table(table_id).delete()


def test_bigtable_simple_use_case(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    saver = BigtableSaver(instance_id, table_id, client=client)
    loader = BigtableLoader(instance_id, table_id, client=client)

    written_docs = [Document(page_content="some content")]
    saver.add_documents(written_docs)
    returned_docs = loader.load()

    assert len(returned_docs) == 1
    assert returned_docs[0].page_content == "some content"
    assert returned_docs[0].metadata != {}
    assert len(returned_docs[0].metadata["rowkey"]) > 0

    saver.delete(returned_docs)
    returned_docs = loader.load()
    assert len(returned_docs) == 0


def test_bigtable_custom_content_encoding(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    content_encoding = Encoding.ASCII
    column_family = "non_default_family"
    column_name = "non_default_column"
    saver = BigtableSaver(
        instance_id,
        table_id,
        content_encoding=content_encoding,
        content_column_family=column_family,
        content_column_name=column_name,
        client=client,
    )
    loader = BigtableLoader(
        instance_id,
        table_id,
        content_encoding=content_encoding,
        content_column_family=column_family,
        content_column_name=column_name,
        client=client,
    )

    written_docs = [Document(page_content="some content")]
    saver.add_documents(written_docs)
    returned_docs = loader.load()

    assert len(returned_docs) == 1
    assert returned_docs[0].page_content == "some content"
    assert returned_docs[0].metadata != {}
    assert len(returned_docs[0].metadata["rowkey"]) > 0


def test_bigtable_filter(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    saver = BigtableSaver(instance_id, table_id, client=client)
    loader = BigtableLoader(
        instance_id, table_id, filter=row_filters.BlockAllFilter(True), client=client
    )

    written_docs = [Document(page_content="some content")]
    saver.add_documents(written_docs)
    returned_docs = loader.load()

    assert len(returned_docs) == 0


def test_bigtable_row_set(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    rowkey = "some_row_key"
    row_set = bigtable.row_set.RowSet()
    row_set.add_row_range_from_keys(start_key=rowkey, end_key=rowkey + "zzz")

    saver = BigtableSaver(instance_id, table_id, client=client)
    loader = BigtableLoader(instance_id, table_id, row_set=row_set, client=client)

    written_docs = [Document(page_content="some content", metadata={"rowkey": rowkey})]
    saver.add_documents(written_docs)
    returned_docs = loader.load()

    assert len(returned_docs) == 1
    assert returned_docs[0].page_content == "some content"
    assert returned_docs[0].metadata["rowkey"] == rowkey


def test_bigtable_metadata_mapping(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    metadata_mappings = [
        MetadataMapping(
            column_family="my_int_family",
            column_name="my_int_column",
            metadata_key="key_in_metadata_map",
            encoding=Encoding.INT_BIG_ENDIAN,
        ),
        MetadataMapping(
            column_family="my_custom_family",
            column_name="my_custom_column",
            metadata_key="custom_key",
            encoding=Encoding.CUSTOM,
            custom_decoding_func=lambda input: json.loads(input.decode()),
            custom_encoding_func=lambda input: str.encode(json.dumps(input)),
        ),
    ]
    saver = BigtableSaver(
        instance_id, table_id, client=client, metadata_mappings=metadata_mappings
    )
    loader = BigtableLoader(
        instance_id, table_id, client=client, metadata_mappings=metadata_mappings
    )

    written_docs = [
        Document(
            page_content="some content",
            metadata={"custom_key": {"a": 1, "b": 2}, "key_in_metadata_map": 5},
        )
    ]
    saver.add_documents(written_docs)
    returned_docs = loader.load()

    assert len(returned_docs) == 1
    assert returned_docs[0].page_content == "some content"
    assert (
        returned_docs[0].metadata["custom_key"]
        == written_docs[0].metadata["custom_key"]
    )
    assert (
        returned_docs[0].metadata["key_in_metadata_map"]
        == written_docs[0].metadata["key_in_metadata_map"]
    )


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v
