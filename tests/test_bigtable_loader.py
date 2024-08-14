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
import time
from typing import Iterator

import pytest
from google.api_core.exceptions import AlreadyExists
from google.cloud import bigtable  # type: ignore
from google.cloud.bigtable import column_family, row_filters  # type: ignore
from langchain_core.documents import Document

from langchain_google_bigtable.loader import (
    BigtableLoader,
    BigtableSaver,
    Encoding,
    MetadataMapping,
    init_document_table,
)

TABLE_ID_PREFIX = "test-table-loader-"


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


def test_bigtable_loads_of_messages(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    NUM_MESSAGES = 100000
    saver = BigtableSaver(instance_id, table_id, client=client)
    loader = BigtableLoader(instance_id, table_id, client=client)

    written_docs = [
        Document(page_content=f"some content {i}") for i in range(NUM_MESSAGES)
    ]
    saver.add_documents(written_docs)

    # wait for eventual consistency
    time.sleep(20)

    returned_docs = loader.load()

    assert len(returned_docs) == NUM_MESSAGES
    for i in range(NUM_MESSAGES):
        assert returned_docs[i].page_content.startswith("some content")
        assert returned_docs[i].metadata != {}
        assert len(returned_docs[i].metadata["rowkey"]) > 0

    saver.delete(returned_docs[:-1])

    # wait for eventual consistency
    time.sleep(20)

    returned_docs = loader.load()
    assert len(returned_docs) == 1

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


def test_bigtable_invalid_custom_content_encoding(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    content_encoding = Encoding.INT_BIG_ENDIAN
    error_message = f"content_encoding '{content_encoding}' not supported for content (must be {(Encoding.UTF8, Encoding.UTF16, Encoding.ASCII)})"

    with pytest.raises(ValueError) as excinfo:
        BigtableSaver(
            instance_id,
            table_id,
            content_encoding=content_encoding,
            client=client,
        )
    assert str(excinfo.value) == error_message

    with pytest.raises(ValueError) as excinfo:
        BigtableLoader(
            instance_id,
            table_id,
            content_encoding=content_encoding,
            client=client,
        )
    assert str(excinfo.value) == error_message


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
            column_name="my_big_int_column",
            metadata_key="big_endian_int",
            encoding=Encoding.INT_BIG_ENDIAN,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="my_float_column",
            metadata_key="float",
            encoding=Encoding.FLOAT,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="my_double_column",
            metadata_key="double",
            encoding=Encoding.DOUBLE,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="my_bool_column",
            metadata_key="bool",
            encoding=Encoding.BOOL,
        ),
        MetadataMapping(
            column_family="my_int_family",
            column_name="my_little_int_column",
            metadata_key="little_endian_int",
            encoding=Encoding.INT_LITTLE_ENDIAN,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="my_utf16_column",
            metadata_key="utf16",
            encoding=Encoding.UTF16,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="non existent",
            metadata_key="in dictionary",
            encoding=Encoding.UTF16,
        ),
        MetadataMapping(
            column_family="my_custom_family",
            column_name="my_custom_column",
            metadata_key="custom_key",
            encoding=Encoding.CUSTOM,
            custom_decoding_func=lambda input: json.loads(input.decode()),
            custom_encoding_func=lambda input: json.dumps(input).encode(),
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
            metadata={
                "custom_key": {"a": 1, "b": 2},
                "big_endian_int": 5,
                "float": 3.14,
                "double": 2.71,
                "little_endian_int": 9,
                "utf16": "string encoded as utf16",
                "bool": True,
                "rowkey": "SomeKey",
                "key without mapping": "value",
            },
        )
    ]
    saver.add_documents(written_docs)
    returned_docs = loader.load()

    assert len(returned_docs) == 1
    assert returned_docs[0].page_content == "some content"

    # Not meant to be in result set
    written_docs[0].metadata.pop("key without mapping")

    # Floats are encoded differently, losing precious, causing equality impossible
    written_float = written_docs[0].metadata.pop("float")
    returned_float = returned_docs[0].metadata.pop("float")
    assert abs(written_float - returned_float) <= 0.001

    # Compare the rest of the metadata dictionary
    assert returned_docs[0].metadata == written_docs[0].metadata


def test_bigtable_empty_custom_mapping(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    metadata_mappings = [
        MetadataMapping(
            column_family="my_custom_family",
            column_name="my_custom_column",
            metadata_key="custom_key",
            encoding=Encoding.CUSTOM,
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
            metadata={"custom_key": {"a": 1, "b": 2}},
        )
    ]
    saver.add_documents(written_docs)

    metadata_mappings = [
        MetadataMapping(
            column_family="my_custom_family",
            column_name="my_custom_column",
            metadata_key="custom_key",
            encoding=Encoding.CUSTOM,
        ),
    ]

    saver = BigtableSaver(
        instance_id, table_id, client=client, metadata_mappings=metadata_mappings
    )

    with pytest.raises(NotImplementedError) as excinfo:
        saver.add_documents(written_docs)
    assert (
        str(excinfo.value)
        == "decoding/encoding function not set for custom encoded metadata key"
    )

    with pytest.raises(NotImplementedError) as excinfo:
        loader.load()
    assert (
        str(excinfo.value)
        == "decoding/encoding function not set for custom encoded metadata key"
    )


def test_bigtable_missing_column_family(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    non_existent_family = "non_existent_family"
    error_prefix = f"column family '{non_existent_family}' doesn't exist in table. Existing column families are "
    # Metadata mapping content family
    metadata_mappings = [
        MetadataMapping(
            column_family=non_existent_family,
            column_name="some_column",
            metadata_key="some_key",
            encoding=Encoding.INT_BIG_ENDIAN,
        ),
    ]
    with pytest.raises(ValueError) as excinfo:
        BigtableLoader(
            instance_id, table_id, client=client, metadata_mappings=metadata_mappings
        )
    assert str(excinfo.value).startswith(error_prefix)
    with pytest.raises(ValueError) as excinfo:
        BigtableSaver(
            instance_id, table_id, client=client, metadata_mappings=metadata_mappings
        )
    assert str(excinfo.value).startswith(error_prefix)

    # Content column family
    with pytest.raises(ValueError) as excinfo:
        BigtableLoader(
            instance_id,
            table_id,
            client=client,
            content_column_family=non_existent_family,
        )
    assert str(excinfo.value).startswith(error_prefix)
    with pytest.raises(ValueError) as excinfo:
        BigtableSaver(
            instance_id,
            table_id,
            client=client,
            content_column_family=non_existent_family,
        )
    assert str(excinfo.value).startswith(error_prefix)

    # Metadata as JSON column family
    with pytest.raises(ValueError) as excinfo:
        BigtableLoader(
            instance_id,
            table_id,
            client=client,
            metadata_as_json_column_family=non_existent_family,
            metadata_as_json_column_name="not_None",
        )
    assert str(excinfo.value).startswith(error_prefix)
    with pytest.raises(ValueError) as excinfo:
        BigtableSaver(
            instance_id,
            table_id,
            client=client,
            metadata_as_json_column_family=non_existent_family,
            metadata_as_json_column_name="not_None",
        )
    assert str(excinfo.value).startswith(error_prefix)


def test_bigtable_metadata_as_json_invalid_encoding(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    content_encoding = Encoding.INT_BIG_ENDIAN
    error_message = f"metadata_as_json_encoding '{content_encoding}' not supported for content (must be {(Encoding.UTF8, Encoding.UTF16, Encoding.ASCII)})"

    with pytest.raises(ValueError) as excinfo:
        BigtableSaver(
            instance_id,
            table_id,
            metadata_as_json_encoding=content_encoding,
            client=client,
        )
    assert str(excinfo.value) == error_message

    with pytest.raises(ValueError) as excinfo:
        BigtableLoader(
            instance_id,
            table_id,
            metadata_as_json_encoding=content_encoding,
            client=client,
        )
    assert str(excinfo.value) == error_message


def test_bigtable_metadata_as_json_only_one_parameter_provided(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    error_message = "when metadata_as_json_column_family is set, metadata_as_json_column_name must also be set"

    with pytest.raises(ValueError) as excinfo:
        BigtableSaver(
            instance_id,
            table_id,
            client=client,
            metadata_as_json_column_family="not_None",
        )
    assert str(excinfo.value) == error_message

    with pytest.raises(ValueError) as excinfo:
        BigtableLoader(
            instance_id,
            table_id,
            client=client,
            metadata_as_json_column_family="not_None",
        )
    assert str(excinfo.value) == error_message

    error_message = "when metadata_as_json_column_name is set, metadata_as_json_column_family must also be set"
    with pytest.raises(ValueError) as excinfo:
        BigtableSaver(
            instance_id,
            table_id,
            client=client,
            metadata_as_json_column_name="not_None",
        )
    assert str(excinfo.value) == error_message

    with pytest.raises(ValueError) as excinfo:
        BigtableLoader(
            instance_id,
            table_id,
            client=client,
            metadata_as_json_column_name="not_None",
        )
    assert str(excinfo.value) == error_message


def test_bigtable_metadata_as_json(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    metadata_mappings = [
        MetadataMapping(
            column_family="my_int_family",
            column_name="my_big_int_column",
            metadata_key="big_endian_int",
            encoding=Encoding.INT_BIG_ENDIAN,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="my_float_column",
            metadata_key="float",
            encoding=Encoding.FLOAT,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="my_double_column",
            metadata_key="double",
            encoding=Encoding.DOUBLE,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="my_bool_column",
            metadata_key="bool",
            encoding=Encoding.BOOL,
        ),
        MetadataMapping(
            column_family="my_int_family",
            column_name="my_little_int_column",
            metadata_key="little_endian_int",
            encoding=Encoding.INT_LITTLE_ENDIAN,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="my_utf16_column",
            metadata_key="utf16",
            encoding=Encoding.UTF16,
        ),
        MetadataMapping(
            column_family="langchain",
            column_name="non existent",
            metadata_key="in dictionary",
            encoding=Encoding.UTF16,
        ),
        MetadataMapping(
            column_family="my_custom_family",
            column_name="my_custom_column",
            metadata_key="custom_key",
            encoding=Encoding.CUSTOM,
            custom_decoding_func=lambda input: json.loads(input.decode()),
            custom_encoding_func=lambda input: json.dumps(input).encode(),
        ),
    ]
    saver = BigtableSaver(
        instance_id,
        table_id,
        client=client,
        metadata_mappings=metadata_mappings,
        metadata_as_json_encoding=Encoding.ASCII,
        metadata_as_json_column_family="langchain",
        metadata_as_json_column_name="metadata_as_json",
    )
    loader = BigtableLoader(
        instance_id,
        table_id,
        client=client,
        metadata_mappings=metadata_mappings,
        metadata_as_json_encoding=Encoding.ASCII,
        metadata_as_json_column_family="langchain",
        metadata_as_json_column_name="metadata_as_json",
    )

    written_docs = [
        Document(
            page_content="some content",
            metadata={
                "custom_key": {"a": 1, "b": 2},
                "big_endian_int": 5,
                "float": 3.14,
                "double": 2.71,
                "little_endian_int": 9,
                "utf16": "string encoded as utf16",
                "bool": True,
                "rowkey": "SomeKey",
                "key without mapping": "value",
                "another key without mapping": 45,
                "yet another key without mapping": False,
            },
        )
    ]
    saver.add_documents(written_docs)
    returned_docs = loader.load()

    assert len(returned_docs) == 1
    assert returned_docs[0].page_content == "some content"

    # Floats are encoded differently, losing precious, causing equality impossible
    written_float = written_docs[0].metadata.pop("float")
    returned_float = returned_docs[0].metadata.pop("float")
    assert abs(written_float - returned_float) <= 0.001

    # Compare the rest of the metadata dictionary
    assert returned_docs[0].metadata == written_docs[0].metadata


def test_bigtable_metadata_as_json_execution_order(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    write_metadata_mappings = [
        MetadataMapping(
            column_family="langchain",
            column_name="column_name",
            metadata_key="some_key",
            encoding=Encoding.UTF16,
        ),
    ]
    read_metadata_mapping = [
        MetadataMapping(
            column_family="langchain",
            column_name="column_name",
            metadata_key="another_key",
            encoding=Encoding.UTF16,
        ),
    ]
    saver = BigtableSaver(
        instance_id,
        table_id,
        client=client,
        metadata_mappings=write_metadata_mappings,
        metadata_as_json_encoding=Encoding.ASCII,
        metadata_as_json_column_family="langchain",
        metadata_as_json_column_name="metadata_as_json",
    )
    loader = BigtableLoader(
        instance_id,
        table_id,
        client=client,
        metadata_mappings=read_metadata_mapping,
        metadata_as_json_encoding=Encoding.ASCII,
        metadata_as_json_column_family="langchain",
        metadata_as_json_column_name="metadata_as_json",
    )

    written_docs = [
        Document(
            page_content="some content",
            metadata={
                "some_key": "expected value",
                "another_key": "ignored value",
                "rowkey": "SomeKey",
            },
        )
    ]
    saver.add_documents(written_docs)
    returned_docs = loader.load()

    assert len(returned_docs) == 1
    assert returned_docs[0].page_content == "some content"

    # Compare the metadata dictionary
    assert returned_docs[0].metadata == {
        "another_key": "expected value",
        "rowkey": "SomeKey",
    }


def test_table_creation(
    instance_id: str, table_id: str, client: bigtable.Client
) -> None:
    # Cleanup default table created by test framework.
    table_client = client.instance(instance_id).table(table_id)
    table_client.delete()

    # Create table.
    init_document_table(instance_id, table_id, client)

    # Assert table exists.
    assert table_client.exists()
    assert sorted(table_client.list_column_families().keys()) == ["langchain"]
    # Expect second creation to fail.
    with pytest.raises(AlreadyExists):
        init_document_table(instance_id, table_id, client)

    # Delete table.
    table_client.delete()
    assert not table_client.exists()

    # Create with column families.
    content_column_family = "content_column_family"
    first_column_from_mapping = "first_column_from_mapping"
    second_column_from_mapping = "second_column_from_mapping"
    metadata_as_json_column_family = "metadata_as_json_column_family"
    init_document_table(
        instance_id,
        table_id,
        client,
        content_column_family,
        [
            MetadataMapping(first_column_from_mapping, "", "", Encoding.ASCII),
            MetadataMapping(second_column_from_mapping, "", "", Encoding.ASCII),
        ],
        metadata_as_json_column_family,
    )
    expected_families = [
        content_column_family,
        first_column_from_mapping,
        second_column_from_mapping,
        metadata_as_json_column_family,
    ]

    # Assert successful creation.
    assert table_client.exists()
    created_families = table_client.list_column_families().keys()
    assert len(created_families) == len(expected_families)
    assert sorted(created_families) == sorted(expected_families)


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v
