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
from typing import AsyncGenerator, AsyncIterator, Iterator, List

import google.auth
import numpy as np
import pytest
import pytest_asyncio
from google.api_core import exceptions
from google.cloud import bigtable
from google.cloud.bigtable.data import (
    BigtableDataClientAsync,
    ReadRowsQuery,
    RowRange,
)
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from typing_extensions import override

from langchain_google_bigtable.async_vector_store import (
    AsyncBigtableVectorStore,
    ColumnConfig,
    DistanceStrategy,
    Encoding,
    MetadataMapping,
    QueryParameters,
    VectorDataType,
)

TEST_ROW_PREFIX = "pytest-advanced-vstore-1-"
CONTENT_COLUMN_FAMILY = "content-cf"
EMBEDDING_COLUMN_FAMILY = "embedding-cf"
METADATA_COLUMN_FAMILY = "md"
VECTOR_SIZE = 3

# Use a deterministic fake embedding service for consistent test results.
# It hashes the input text to create a seed, ensuring that the same
# text will always produce the same embedding vector.
embedding_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.fixture(scope="session")
def project_id() -> Iterator[str]:
    yield get_env_var("PROJECT_ID", "GCP Project ID")


@pytest.fixture(scope="session")
def instance_id() -> Iterator[str]:
    yield get_env_var("INSTANCE_ID", "Bigtable Instance ID")


@pytest.fixture(scope="session")
def dynamic_table_id(project_id: str, instance_id: str) -> Iterator[str]:
    admin_client = bigtable.Client(project=project_id, admin=True)
    instance = admin_client.instance(instance_id)
    table_id = f"test-advanced-{uuid.uuid4().hex[:8]}"
    table = instance.table(table_id)
    column_families = {
        CONTENT_COLUMN_FAMILY: bigtable.column_family.MaxVersionsGCRule(1),
        EMBEDDING_COLUMN_FAMILY: bigtable.column_family.MaxVersionsGCRule(1),
        METADATA_COLUMN_FAMILY: bigtable.column_family.MaxVersionsGCRule(1),
    }
    try:
        table.create(column_families=column_families)
        yield table_id
    except exceptions.Conflict:  # Already exists
        yield table_id
    finally:
        try:
            table.delete()
        except Exception as e:
            pytest.fail(f"Failed to delete table {table_id}: {e}")
        finally:
            admin_client.close()


@pytest.mark.asyncio(loop_scope="class")
class TestAdvancedFeatures:
    """Tests for filtering, edge cases, and other advanced functionality."""

    @pytest_asyncio.fixture(scope="class", loop_scope="class")
    async def async_data_client(
        self,
        project_id: str,
    ) -> AsyncGenerator[BigtableDataClientAsync, None]:
        credentials, _ = google.auth.default()
        client = BigtableDataClientAsync(project=project_id, credentials=credentials)
        yield client
        await client.close()

    @pytest_asyncio.fixture(scope="class", loop_scope="class")
    async def store(
        self,
        async_data_client: BigtableDataClientAsync,
        instance_id: str,
        dynamic_table_id: str,
    ) -> AsyncIterator[AsyncBigtableVectorStore]:
        table = async_data_client.get_table(instance_id, dynamic_table_id)
        vector_store = AsyncBigtableVectorStore(
            client=async_data_client,
            instance_id=instance_id,
            async_table=table,
            embedding_service=embedding_service,
            content_column=ColumnConfig(
                column_family=CONTENT_COLUMN_FAMILY, column_qualifier="content"
            ),
            embedding_column=ColumnConfig(
                column_family=EMBEDDING_COLUMN_FAMILY, column_qualifier="embedding"
            ),
            metadata_mappings=[
                MetadataMapping("color", Encoding.UTF8),
                MetadataMapping("number", Encoding.INT_BIG_ENDIAN),
                MetadataMapping("is_good", Encoding.BOOL),
                MetadataMapping("rating", Encoding.FLOAT),
            ],
            collection=TEST_ROW_PREFIX,
        )
        yield vector_store
        # Cleanup logic
        row_range = RowRange(start_key=TEST_ROW_PREFIX.encode("utf-8"))
        row_iterator = await vector_store.async_table.read_rows(
            ReadRowsQuery(row_ranges=[row_range])
        )
        all_keys = [key.row_key.decode() for key in row_iterator]
        ids_to_delete = [key.split(":", 1)[1] for key in all_keys if ":" in key]
        if ids_to_delete:
            await vector_store.adelete(ids_to_delete)

    @pytest_asyncio.fixture(scope="class", loop_scope="class")
    async def store_with_json(
        self,
        async_data_client: BigtableDataClientAsync,
        instance_id: str,
        dynamic_table_id: str,
    ) -> AsyncIterator[AsyncBigtableVectorStore]:
        table = async_data_client.get_table(instance_id, dynamic_table_id)
        vector_store = AsyncBigtableVectorStore(
            client=async_data_client,
            instance_id=instance_id,
            async_table=table,
            embedding_service=embedding_service,
            content_column=ColumnConfig(
                column_family=CONTENT_COLUMN_FAMILY, column_qualifier="content"
            ),
            embedding_column=ColumnConfig(
                column_family=EMBEDDING_COLUMN_FAMILY, column_qualifier="embedding"
            ),
            metadata_as_json_column=ColumnConfig(
                column_family=METADATA_COLUMN_FAMILY, column_qualifier="metadata_json"
            ),
            metadata_mappings=[
                MetadataMapping("color", Encoding.UTF8),
                MetadataMapping("number", Encoding.INT_BIG_ENDIAN),
                MetadataMapping("is_good", Encoding.BOOL),
                MetadataMapping("rating", Encoding.FLOAT),
            ],
            collection=TEST_ROW_PREFIX,
        )
        yield vector_store
        row_range = RowRange(start_key=TEST_ROW_PREFIX.encode("utf-8"))
        row_iterator = await vector_store.async_table.read_rows(
            ReadRowsQuery(row_ranges=[row_range])
        )
        all_keys = [key.row_key.decode() for key in row_iterator]
        ids_to_delete = [key.split(":", 1)[1] for key in all_keys if ":" in key]
        if ids_to_delete:
            await vector_store.adelete(ids_to_delete)

    @pytest_asyncio.fixture(scope="class", loop_scope="class")
    async def store_all_encodings(
        self,
        async_data_client: BigtableDataClientAsync,
        instance_id: str,
        dynamic_table_id: str,
    ) -> AsyncIterator[AsyncBigtableVectorStore]:
        table = async_data_client.get_table(instance_id, dynamic_table_id)
        vector_store = AsyncBigtableVectorStore(
            client=async_data_client,
            instance_id=instance_id,
            async_table=table,
            embedding_service=embedding_service,
            content_column=ColumnConfig(
                column_family=CONTENT_COLUMN_FAMILY, column_qualifier="content"
            ),
            embedding_column=ColumnConfig(
                column_family=EMBEDDING_COLUMN_FAMILY, column_qualifier="embedding"
            ),
            metadata_mappings=[
                MetadataMapping("md_utf8", Encoding.UTF8),
                MetadataMapping("md_utf16", Encoding.UTF16),
                MetadataMapping("md_ascii", Encoding.ASCII),
                MetadataMapping("md_int_le", Encoding.INT_LITTLE_ENDIAN),
                MetadataMapping("md_int_be", Encoding.INT_BIG_ENDIAN),
                MetadataMapping("md_double", Encoding.DOUBLE),
                MetadataMapping("md_float", Encoding.FLOAT),
                MetadataMapping("md_bool", Encoding.BOOL),
            ],
            collection=TEST_ROW_PREFIX,
        )
        yield vector_store
        row_range = RowRange(start_key=TEST_ROW_PREFIX.encode("utf-8"))
        row_iterator = await vector_store.async_table.read_rows(
            ReadRowsQuery(row_ranges=[row_range])
        )
        all_keys = [key.row_key.decode() for key in row_iterator]
        ids_to_delete = [key.split(":", 1)[1] for key in all_keys if ":" in key]
        if ids_to_delete:
            await vector_store.adelete(ids_to_delete)

    @pytest.mark.parametrize(
        "operator, value, expected_pages",
        [
            (">", 20, ["item 30", "item 40"]),
            ("<", 30, ["item 10", "item 20"]),
            (">=", 30, ["item 30", "item 40"]),
            ("<=", 20, ["item 10", "item 20"]),
            ("!=", 30, ["item 10", "item 20", "item 40"]),
        ],
    )
    async def test_filtering_numerical_operators(
        self,
        store: AsyncBigtableVectorStore,
        operator: str,
        value: int,
        expected_pages: List[str],
    ) -> None:
        """Tests individual numerical comparison filters: >, <, >=, <=, !="""
        added_doc_ids = await store.aadd_texts(
            ["item 10", "item 20", "item 30", "item 40"],
            metadatas=[{"number": n} for n in [10, 20, 30, 40]],
        )
        query_params = QueryParameters(filters={"number": {operator: value}})
        results = await store.asimilarity_search(
            "any", k=4, query_parameters=query_params
        )
        assert len(results) == len(expected_pages)
        assert set(expected_pages) == {doc.page_content for doc in results}

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_string_operators(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests string-based filters 'like' (regex) and 'contains'."""
        added_doc_ids = await store.aadd_texts(
            ["A regular item", "Another item", "A special thing"],
            metadatas=[{"color": c} for c in ["regular", "another", "special"]],
        )
        query_params_like = QueryParameters(filters={"color": {"like": "reg.*"}})
        results_like = await store.asimilarity_search(
            "any", k=3, query_parameters=query_params_like
        )
        assert len(results_like) == 1
        assert results_like[0].page_content == "A regular item"

        query_params_contains = QueryParameters(filters={"color": {"contains": "othe"}})
        results_contains = await store.asimilarity_search(
            "any", k=3, query_parameters=query_params_contains
        )
        assert len(results_contains) == 1
        assert results_contains[0].page_content == "Another item"

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_qualifier_filters(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests qualifier-based filters: Qualifiers, ColumnQualifierPrefix, ColumnQualifierRegex."""
        added_doc_ids = await store.aadd_texts(
            ["doc1", "doc2", "doc3"],
            metadatas=[
                {"color": "red", "number": 1},
                {"color": "blue"},
                {"rating": 5.0},
            ],
        )
        query_params_exist = QueryParameters(
            filters={"Qualifiers": ["color", "number"]}
        )
        results_exist = await store.asimilarity_search(
            "any", k=3, query_parameters=query_params_exist
        )
        assert len(results_exist) == 1
        assert results_exist[0].page_content == "doc1"

        query_params_prefix = QueryParameters(filters={"ColumnQualifierPrefix": "rat"})
        results_prefix = await store.asimilarity_search(
            "any", k=3, query_parameters=query_params_prefix
        )
        assert len(results_prefix) == 1
        assert results_prefix[0].page_content == "doc3"

        query_params_regex = QueryParameters(filters={"ColumnQualifierRegex": "^c.*r$"})
        results_regex = await store.asimilarity_search(
            "any", k=3, query_parameters=query_params_regex
        )
        assert len(results_regex) == 2
        assert {"doc1", "doc2"} == {doc.page_content for doc in results_regex}

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_collection(self, store: AsyncBigtableVectorStore) -> None:
        """Tests filtering by a custom collection row key prefix, overriding the default collection prefix."""
        added_doc_ids = await store.aadd_texts(
            ["doc-default-A", "doc-default-B"], ids=["doc-a", "doc-b"]
        )
        await store.aadd_texts(["doc-other-C"], ids=["other/doc-c"])
        added_doc_ids.append("other/doc-c")

        results_default = await store.asimilarity_search("any doc", k=3)
        assert len(results_default) == 3
        assert {"doc-default-A", "doc-default-B", "doc-other-C"} == {
            doc.page_content for doc in results_default
        }

        row_key_prefix = "other/"
        query_params_override = QueryParameters(
            filters={"rowKeyFilter": row_key_prefix}
        )
        results_override = await store.asimilarity_search(
            "any doc", k=3, query_parameters=query_params_override
        )
        assert len(results_override) == 1
        assert results_override[0].page_content == "doc-other-C"

        query_params_fail = QueryParameters(
            filters={"rowKeyFilter": "non-existent-prefix"}
        )
        results_fail = await store.asimilarity_search(
            "any doc", k=3, query_parameters=query_params_fail
        )
        assert len(results_fail) == 0

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_exact_match(self, store: AsyncBigtableVectorStore) -> None:
        """Tests filtering with an exact match (==)."""
        added_doc_ids = await store.aadd_texts(
            ["red apple", "red car", "blue car"],
            metadatas=[
                {"color": "red", "number": 1},
                {"color": "red", "number": 2},
                {"color": "blue", "number": 3},
            ],
        )
        query_params = QueryParameters(filters={"color": {"==": "blue"}})
        results = await store.asimilarity_search_by_vector(
            [1.0, 0.0, 0.0], k=3, query_parameters=query_params
        )
        assert len(results) == 1
        assert results[0].page_content == "blue car"

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_comparison(self, store: AsyncBigtableVectorStore) -> None:
        """Tests filtering with comparison operators."""
        added_doc_ids = await store.aadd_texts(
            ["item 10", "item 20", "item 30"],
            metadatas=[{"number": 10}, {"number": 20}, {"number": 30}],
        )
        query_params = QueryParameters(filters={"number": {">=": 20}})
        results = await store.asimilarity_search_by_vector(
            [1.0, 0.0, 0.0], k=3, query_parameters=query_params
        )
        assert len(results) == 2
        assert {"item 20", "item 30"} == {doc.page_content for doc in results}

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_list_operators(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests filtering with 'in' and 'nin' operators."""
        added_doc_ids = await store.aadd_texts(
            ["red", "blue", "green", "yellow"],
            metadatas=[
                {"color": "red"},
                {"color": "blue"},
                {"color": "green"},
                {"color": "yellow"},
            ],
        )
        query_params_in = QueryParameters(filters={"color": {"in": ["blue", "yellow"]}})
        results_in = await store.asimilarity_search_by_vector(
            [1.0, 0.0, 0.0], k=4, query_parameters=query_params_in
        )
        assert len(results_in) == 2
        assert {"blue", "yellow"} == {doc.page_content for doc in results_in}

        query_params_nin = QueryParameters(
            filters={"color": {"nin": ["blue", "yellow"]}}
        )
        results_nin = await store.asimilarity_search_by_vector(
            [1.0, 0.0, 0.0], k=4, query_parameters=query_params_nin
        )
        assert len(results_nin) == 2
        assert {"red", "green"} == {doc.page_content for doc in results_nin}

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_nested_chain(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests filtering with nested AND conditions (ColumnValueChainFilter)."""
        added_doc_ids = await store.aadd_texts(
            ["good red", "bad red", "good blue"],
            metadatas=[
                {"color": "red", "is_good": True},
                {"color": "red", "is_good": False},
                {"color": "blue", "is_good": True},
            ],
        )
        query_params = QueryParameters(
            filters={
                "ColumnValueChainFilter": {
                    "color": {"==": "red"},
                    "is_good": {"==": True},
                }
            }
        )
        results = await store.asimilarity_search_by_vector(
            [1.0, 0.0, 0.0], k=3, query_parameters=query_params
        )
        assert len(results) == 1
        assert results[0].page_content == "good red"

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_by_row_key_and_multiple_values(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """
        Tests filtering by a combination of rowKeyFilter, multiple metadata value
        filters, and a qualifier filter simultaneously.
        """
        added_doc_ids = await store.aadd_texts(
            ["doc-A", "doc-B", "doc-C", "doc-D", "doc-E"],
            ids=[
                "group1/doc-a",
                "group1/doc-b",
                "group2/doc-c",
                "group1/doc-d",
                "group1/doc-e",
            ],
            metadatas=[
                {"color": "red", "number": 1, "is_good": True, "rating": 4.1},
                {"color": "blue", "number": 2, "is_good": False, "rating": 3.5},
                {"color": "blue", "number": 3, "is_good": True, "rating": 4.8},
                {"color": "blue", "is_good": True, "rating": 4.9},
                {
                    "color": "blue",
                    "number": 5,
                    "is_good": True,
                    "rating": 4.9,
                },  # The only expected match
            ],
        )

        query_params = QueryParameters(
            filters={
                "rowKeyFilter": "group1/",
                "color": {"==": "blue"},
                "is_good": {"==": True},
                "rating": {">=": 4.9},
                "Qualifiers": ["number", "color"],
            }
        )
        results = await store.asimilarity_search(
            "any", k=5, query_parameters=query_params
        )

        assert len(results) == 1
        assert results[0].page_content == "doc-E"

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_nested_union(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests filtering with nested OR conditions (ColumnValueUnionFilter)."""
        added_doc_ids = await store.aadd_texts(
            ["item 1", "item 2", "item 3"],
            metadatas=[{"number": 1}, {"number": 2}, {"number": 3}],
        )
        query_params = QueryParameters(
            filters={"ColumnValueUnionFilter": {"number": {"<": 2, ">": 2}}}
        )
        results = await store.asimilarity_search_by_vector(
            [1.0, 0.0, 0.0], k=3, query_parameters=query_params
        )
        assert len(results) == 2
        assert {"item 1", "item 3"} == {doc.page_content for doc in results}

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_filtering_complex_nested(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests a complex combination of nested filters (OR of (AND and condition))."""
        added_doc_ids = await store.aadd_texts(
            ["A", "B", "C", "D", "E"],
            metadatas=[
                {"color": "red", "is_good": True, "number": 1},  # Match 1 (by chain)
                {"color": "red", "is_good": False, "number": 2},
                {"color": "blue", "is_good": True, "number": 3},
                {
                    "color": "green",
                    "is_good": True,
                    "number": 10,
                },  # Match 2 (by number)
                {
                    "color": "green",
                    "is_good": False,
                    "number": 20,
                },  # Match 3 (by number)
            ],
        )
        query_params = QueryParameters(
            filters={
                "ColumnValueUnionFilter": {
                    "ColumnValueChainFilter": {
                        "color": {"==": "red"},
                        "is_good": {"==": True},
                    },
                    "number": {">=": 10},
                }
            }
        )
        results = await store.asimilarity_search_by_vector(
            [1.0, 0.0, 0.0], k=5, query_parameters=query_params
        )
        assert len(results) == 3
        assert {"A", "D", "E"} == {doc.page_content for doc in results}

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_store_with_json_metadata_lifecycle(
        self, store_with_json: AsyncBigtableVectorStore
    ) -> None:
        """
        Tests the full lifecycle (add, get, search, delete) for a store
        configured to use a single JSON column for metadata.
        """
        docs_to_add = [
            Document(
                page_content="The sky is a brilliant blue today.",
                metadata={
                    "color": "blue",
                    "rating": 4.5,
                },
                id="json_doc1",
            ),
            Document(
                page_content="An orange cat is napping in the sun.",
                metadata={"color": "orange", "rating": 5.0},
                id="json_doc2",
            ),
        ]
        added_ids = await store_with_json.aadd_documents(docs_to_add)
        assert set(added_ids) == {"json_doc1", "json_doc2"}

        retrieved_docs = await store_with_json.aget_by_ids(["json_doc2"])
        assert len(retrieved_docs) == 1
        retrieved_doc = retrieved_docs[0]
        assert retrieved_doc.page_content == "An orange cat is napping in the sun."
        expected_metadata = {
            "color": "orange",
            "rating": 5.0,
        }
        assert retrieved_doc.metadata == expected_metadata

        query = "The sky is a brilliant blue today."

        sim_results = await store_with_json.asimilarity_search(query, k=1)
        assert len(sim_results) == 1
        assert sim_results[0].id == "json_doc1"
        assert sim_results[0].metadata["color"] == "blue"

        sim_results_score = await store_with_json.asimilarity_search_with_score(
            query, k=1
        )
        assert len(sim_results_score) == 1
        doc, score = sim_results_score[0]
        assert doc.id == "json_doc1"
        assert isinstance(score, float)
        assert doc.metadata["rating"] == 4.5

        mmr_results = await store_with_json.amax_marginal_relevance_search(query, k=1)
        assert len(mmr_results) == 1
        assert mmr_results[0].id == "json_doc1"
        assert mmr_results[0].metadata["color"] == "blue"

        await store_with_json.adelete(["json_doc1"])

        retrieved_after_delete = await store_with_json.aget_by_ids(["json_doc1"])
        assert len(retrieved_after_delete) == 0

        remaining_docs = await store_with_json.aget_by_ids(["json_doc2"])
        assert len(remaining_docs) == 1
        assert remaining_docs[0].id == "json_doc2"

        # Clean up
        await store_with_json.adelete(
            [remaining_doc.id for remaining_doc in remaining_docs]
        )

    async def test_all_metadata_encodings_lifecycle(
        self, store_all_encodings: AsyncBigtableVectorStore
    ) -> None:
        """
        Tests the full add-get-search-delete lifecycle for every supported metadata encoding.
        """
        original_metadata = {
            "md_utf8": "UTF-8 text: 你好",
            "md_utf16": "UTF-16 text: 你好",
            "md_ascii": "Simple ASCII",
            "md_int_le": -1_000_000_000,
            "md_int_be": 9_000_000_000,
            "md_double": 3.141592653589793,
            "md_float": -2.718,
            "md_bool": True,
        }
        doc = Document(
            page_content="multi-encoded metadata",
            metadata=original_metadata,
            id="all-encodings-doc",
        )
        await store_all_encodings.aadd_documents([doc])
        retrieved_docs = await store_all_encodings.aget_by_ids(["all-encodings-doc"])
        assert len(retrieved_docs) == 1
        retrieved_metadata = retrieved_docs[0].metadata

        for key, value in original_metadata.items():
            if isinstance(value, float):
                assert retrieved_metadata[key] == pytest.approx(value)
            else:
                assert retrieved_metadata[key] == value

        for key, value in original_metadata.items():
            query_params = QueryParameters(filters={key: {"==": value}})
            search_results = await store_all_encodings.asimilarity_search(
                "any", k=1, query_parameters=query_params
            )
            assert len(search_results) == 1, f"Filter failed for key: {key}"
            assert search_results[0].id.endswith("all-encodings-doc")  # type: ignore

        await store_all_encodings.adelete(["all-encodings-doc"])
        assert len(await store_all_encodings.aget_by_ids(["all-encodings-doc"])) == 0

    async def test_search_with_double64_vector_type(
        self,
        async_data_client: BigtableDataClientAsync,
        instance_id: str,
        dynamic_table_id: str,
    ) -> None:
        """Tests a DOUBLE64 vector search."""
        table = async_data_client.get_table(instance_id, dynamic_table_id)
        store_double_encoding = AsyncBigtableVectorStore(
            client=async_data_client,
            instance_id=instance_id,
            async_table=table,
            embedding_service=embedding_service,
            content_column=ColumnConfig(
                column_family="content-cf", column_qualifier="content"
            ),
            embedding_column=ColumnConfig(
                column_family="embedding-cf",
                column_qualifier="embedding",
                encoding=Encoding.DOUBLE,
            ),
            collection="pytest-async-vstore-",
        )
        texts_to_add = ["alpha document", "beta document"]
        doc_ids = ["doc_alpha", "doc_beta"]
        try:
            await store_double_encoding.aadd_texts(texts_to_add, ids=doc_ids)
            query = "beta document"
            query_params = QueryParameters(vector_data_type=VectorDataType.DOUBLE64)
            results = await store_double_encoding.asimilarity_search(
                query, k=1, query_parameters=query_params
            )
            assert len(results) == 1
            assert results[0].page_content == "beta document"
        finally:
            await store_double_encoding.adelete(doc_ids)

    async def test_search_with_different_distance_strategies(
        self,
        async_data_client: BigtableDataClientAsync,
        instance_id: str,
        dynamic_table_id: str,
    ) -> None:
        """Tests that EUCLIDEAN and COSINE strategies return different results using a custom embedding class."""

        class CustomDistanceEmbedding(DeterministicFakeEmbedding):
            @override
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                v_cosine = np.array([4.0, 3.0])  # Aligned with query (Cosine winner)
                v_euclidean = np.array(
                    [0.7, 0.8]
                )  # Physically closer to query (Euclidean winner)
                output_vectors = []
                for text in texts:
                    if text == "cosine_doc":
                        output_vectors.append(v_cosine.tolist())
                    elif text == "euclidean_doc":
                        output_vectors.append(v_euclidean.tolist())
                return output_vectors

            @override
            def embed_query(self, text: str) -> list[float]:
                q_vec = np.array([0.8, 0.6])
                return q_vec.tolist()

        local_embedding_service = CustomDistanceEmbedding(size=2)
        table = async_data_client.get_table(instance_id, dynamic_table_id)
        local_store = AsyncBigtableVectorStore(
            client=async_data_client,
            instance_id=instance_id,
            async_table=table,
            embedding_service=local_embedding_service,
            content_column=ColumnConfig(
                column_family="content-cf", column_qualifier="content"
            ),
            embedding_column=ColumnConfig(
                column_family="embedding-cf", column_qualifier="embedding"
            ),
            collection="pytest-async-vstore-",
        )
        doc_ids = ["cosine_doc", "euclidean_doc"]
        try:
            await local_store.aadd_texts(doc_ids, ids=doc_ids)
            query_params_cosine = QueryParameters(
                distance_strategy=DistanceStrategy.COSINE
            )
            results_cosine = await local_store.asimilarity_search(
                "query", k=1, query_parameters=query_params_cosine
            )
            assert results_cosine[0].page_content == "cosine_doc"

            query_params_euclidean = QueryParameters(
                distance_strategy=DistanceStrategy.EUCLIDEAN
            )
            results_euclidean = await local_store.asimilarity_search(
                "query", k=1, query_parameters=query_params_euclidean
            )
            assert results_euclidean[0].page_content == "euclidean_doc"
        finally:
            await local_store.adelete(doc_ids)

    async def test_edge_cases_and_empty_inputs(
        self,
        async_data_client: BigtableDataClientAsync,
        instance_id: str,
        dynamic_table_id: str,
    ) -> None:
        """Tests edge cases and empty inputs on empty stores"""

        table = async_data_client.get_table(instance_id, dynamic_table_id)
        local_store = AsyncBigtableVectorStore(
            client=async_data_client,
            instance_id=instance_id,
            async_table=table,
            embedding_service=embedding_service,
            content_column=ColumnConfig(
                column_family="content-cf", column_qualifier="content"
            ),
            embedding_column=ColumnConfig(
                column_family="embedding-cf", column_qualifier="embedding"
            ),
            collection="empty-store-1",
        )

        assert await local_store.aadd_texts([]) == []
        assert await local_store.aget_by_ids([]) == []
        assert await local_store.adelete([]) is None
        assert await local_store.aget_by_ids(["non-existent-id"]) == []
        assert await local_store.adelete(["non-existent-id"]) is True
        assert await local_store.asimilarity_search("no matching docs", k=1) == []

    async def test_filtering_no_results(self, store: AsyncBigtableVectorStore) -> None:
        """Tests that a filter returning no results behaves correctly."""
        added_doc_ids = await store.aadd_texts(["item 1"], metadatas=[{"color": "red"}])
        query_params = QueryParameters(filters={"color": {"==": "blue"}})
        results = await store.asimilarity_search(
            "item 1", k=1, query_parameters=query_params
        )

        assert len(results) == 0

        # Clean Up
        await store.adelete(added_doc_ids)

    async def test_invalid_metadata_type_on_add(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests that adding data with a type mismatch for a mapped metadata field raises an error."""
        with pytest.raises(ValueError, match="Failed to encode value"):
            await store.aadd_texts(["bad data"], metadatas=[{"number": "not-a-number"}])

    async def test_sync_methods_raise_not_implemented(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Ensures synchronous methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            store.add_texts(["test"])
        with pytest.raises(NotImplementedError):
            store.search("query", "similarity")
        with pytest.raises(NotImplementedError):
            store.delete(["some-id"])
        with pytest.raises(NotImplementedError):
            store.similarity_search("query")
        with pytest.raises(NotImplementedError):
            store.similarity_search_by_vector([1.0])
        with pytest.raises(NotImplementedError):
            store.similarity_search_with_score("query")
        with pytest.raises(NotImplementedError):
            store.max_marginal_relevance_search("query")
        with pytest.raises(NotImplementedError):
            store.from_texts(["text"])
        with pytest.raises(NotImplementedError):
            store.from_documents(Document(page_content="text"))
        with pytest.raises(NotImplementedError):
            store.max_marginal_relevance_search_by_vector([1.0])
        with pytest.raises(NotImplementedError):
            store.similarity_search_with_relevance_scores("query")
