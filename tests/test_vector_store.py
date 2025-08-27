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

import os
import uuid
from typing import AsyncIterator, Iterator, List

import pytest
import pytest_asyncio
from google.api_core import exceptions
from google.cloud import bigtable
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_google_bigtable.engine import BigtableEngine
from langchain_google_bigtable.vector_store import (
    BigtableVectorStore,
    ColumnConfig,
    DistanceStrategy,
    Encoding,
    QueryParameters,
    VectorMetadataMapping,
    init_vector_store_table,
)

TEST_ROW_PREFIX = "pytest-vstore-"
CONTENT_COLUMN_FAMILY = "content_cf"
EMBEDDING_COLUMN_FAMILY = "embedding_cf"
METADATA_COLUMN_FAMILY = "md"
VECTOR_SIZE = 3

content_column = ColumnConfig(
    column_family=CONTENT_COLUMN_FAMILY, column_qualifier="content"
)
embedding_column = ColumnConfig(
    column_family=EMBEDDING_COLUMN_FAMILY, column_qualifier="embedding"
)


def get_env_var(key: str, desc: str) -> str:
    """Gets an environment variable or raises an error."""
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.fixture(scope="session")
def project_id() -> Iterator[str]:
    """Returns the GCP Project ID from environment variables."""
    yield get_env_var("PROJECT_ID", "GCP Project ID")


@pytest.fixture(scope="session")
def instance_id() -> Iterator[str]:
    """Returns the Bigtable Instance ID from environment variables."""
    yield get_env_var("INSTANCE_ID", "Bigtable Instance ID")


@pytest.fixture(scope="session")
def managed_table(project_id: str, instance_id: str) -> Iterator[str]:
    """Creates a unique Bigtable table for the test session and tears it down."""
    table_id = f"test-vector-store-{uuid.uuid4().hex[:8]}"
    client = bigtable.Client(project=project_id, admin=True)
    try:
        init_vector_store_table(
            instance_id=instance_id,
            table_id=table_id,
            project_id=project_id,
            client=client,
            content_column_family=CONTENT_COLUMN_FAMILY,
            embedding_column_family=EMBEDDING_COLUMN_FAMILY,
        )
        yield table_id
    finally:
        try:
            instance = client.instance(instance_id)
            instance.table(table_id).delete()
        except exceptions.NotFound:
            pass
        finally:
            client.close()


class TestTableInitialization:
    """Tests for the table creation utility."""

    def test_init_table_already_exists(
        self, managed_table: str, project_id: str, instance_id: str
    ) -> None:
        """
        Verifies that init_vector_store_table raises a ValueError if the table already exists.
        """
        with pytest.raises(ValueError):
            init_vector_store_table(
                instance_id=instance_id, table_id=managed_table, project_id=project_id
            )


@pytest.mark.asyncio
class TestBigtableVectorStoreSync:
    """Tests for a synchronously initialized BigtableVectorStore."""

    @pytest_asyncio.fixture(scope="class")
    async def sync_store(
        self, project_id: str, instance_id: str, managed_table: str
    ) -> AsyncIterator[BigtableVectorStore]:
        """Provides a sync store for the class, cleaning up the engine at the end."""
        store = BigtableVectorStore.create_sync(
            project_id=project_id,
            instance_id=instance_id,
            table_id=managed_table,
            embedding_service=DeterministicFakeEmbedding(size=VECTOR_SIZE),
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_mappings=[
                VectorMetadataMapping("color", Encoding.UTF8),
                VectorMetadataMapping("number", Encoding.INT_BIG_ENDIAN),
            ],
            collection=f"{TEST_ROW_PREFIX}sync-",
        )
        yield store
        await store.close()

    async def test_full_lifecycle(self, sync_store: BigtableVectorStore) -> None:
        """Tests the complete add -> get -> delete cycle for sync and async methods."""
        docs = [Document(page_content="sync test doc", metadata={}, id="s_doc1")]
        doc_ids = ["s_doc1"]

        # Sync methods
        sync_store.add_documents(docs)
        assert len(sync_store.get_by_ids(doc_ids)) == 1
        sync_store.delete(doc_ids)
        assert not sync_store.get_by_ids(doc_ids)

        # Async methods
        await sync_store.aadd_documents(docs)
        assert len(await sync_store.aget_by_ids(doc_ids)) == 1
        await sync_store.adelete(doc_ids)
        assert not await sync_store.aget_by_ids(doc_ids)

    async def test_search_methods(self, sync_store: BigtableVectorStore) -> None:
        """Tests all search variants on a sync-created store."""
        texts = ["apple", "banana"]
        added_ids = sync_store.add_texts(
            texts, metadatas=[{"color": "red"}, {"color": "yellow"}]
        )

        # Basic search
        assert len(sync_store.similarity_search("apple", k=1)) == 1
        assert len(await sync_store.asimilarity_search("apple", k=1)) == 1

        # Search with score
        results_with_score = sync_store.similarity_search_with_score("banana", k=1)
        assert len(results_with_score) == 1
        assert results_with_score[0][0].page_content == "banana"

        # Relevance score
        results_relevance = sync_store.similarity_search_with_relevance_scores(
            "apple", k=1
        )
        assert len(results_relevance) == 1
        assert 0.0 <= results_relevance[0][1] <= 1.0

        # MMR
        assert len(sync_store.max_marginal_relevance_search("banana", k=1)) == 1
        assert len(await sync_store.amax_marginal_relevance_search("banana", k=1)) == 1

        sync_store.delete(added_ids)


@pytest.mark.asyncio
class TestBigtableVectorStoreAsync:
    """Tests for an asynchronously initialized BigtableVectorStore."""

    @pytest_asyncio.fixture(scope="class", loop_scope="class")
    async def async_store(
        self, project_id: str, instance_id: str, managed_table: str
    ) -> AsyncIterator[BigtableVectorStore]:
        """Provides an async store for the class, cleaning it up at the end."""
        store = await BigtableVectorStore.create(
            project_id=project_id,
            instance_id=instance_id,
            table_id=managed_table,
            embedding_service=DeterministicFakeEmbedding(size=VECTOR_SIZE),
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_mappings=[
                VectorMetadataMapping("color", Encoding.UTF8),
                VectorMetadataMapping("number", Encoding.INT_BIG_ENDIAN),
            ],
            collection=f"{TEST_ROW_PREFIX}async-",
        )
        yield store
        await store.close()

    async def test_full_lifecycle(self, async_store: BigtableVectorStore) -> None:
        """Tests the complete add -> get -> delete cycle for sync and async methods."""
        docs = [Document(page_content="async test doc", metadata={}, id="a_doc1")]
        doc_ids = ["a_doc1"]

        # Async methods
        await async_store.aadd_documents(docs)
        assert len(await async_store.aget_by_ids(doc_ids)) == 1
        await async_store.adelete(doc_ids)
        assert not await async_store.aget_by_ids(doc_ids)

        # Sync methods
        async_store.add_documents(docs)
        assert len(async_store.get_by_ids(doc_ids)) == 1
        async_store.delete(doc_ids)
        assert not async_store.get_by_ids(doc_ids)

    async def test_search_methods(self, async_store: BigtableVectorStore) -> None:
        """Tests all search variants on an async-created store."""
        texts = ["carrot", "dragonfruit"]
        added_ids = await async_store.aadd_texts(
            texts, metadatas=[{"color": "orange"}, {"color": "pink"}]
        )

        # Basic search
        assert len(await async_store.asimilarity_search("carrot", k=1)) == 1
        assert len(async_store.similarity_search("carrot", k=1)) == 1

        # Search with score
        results_with_score = await async_store.asimilarity_search_with_score(
            "dragonfruit", k=1
        )
        assert len(results_with_score) == 1
        assert results_with_score[0][0].page_content == "dragonfruit"

        # MMR
        assert (
            len(await async_store.amax_marginal_relevance_search("dragonfruit", k=1))
            == 1
        )
        assert len(async_store.max_marginal_relevance_search("dragonfruit", k=1)) == 1

        await async_store.adelete(added_ids)


@pytest.mark.asyncio
class TestAdvancedScenarios:
    """Tests for advanced configurations, factory methods, and error handling."""

    @pytest_asyncio.fixture(scope="class", loop_scope="class")
    async def engine(self, project_id: str) -> AsyncIterator[BigtableEngine]:
        """Provides a shared engine for advanced tests."""
        engine = await BigtableEngine.async_initialize(project_id=project_id)
        yield engine
        await engine.close()

    async def test_from_methods(
        self, engine: BigtableEngine, instance_id: str, managed_table: str
    ) -> None:
        """Tests the from_texts and from_documents class methods."""
        docs = [Document(page_content="doc content", id="d1")]
        texts = ["text content"]
        embedding = DeterministicFakeEmbedding(size=VECTOR_SIZE)

        # Sync from_texts
        text_store = BigtableVectorStore.from_texts(
            texts,
            embedding,
            instance_id=instance_id,
            table_id=managed_table,
            engine=engine,
            ids=["t1"],
            content_column=content_column,
            embedding_column=embedding_column,
            collection="test_coll_from_texts",
        )
        assert len(text_store.get_by_ids(["t1"])) == 1
        text_store.delete(["t1"])

        # Async from_documents
        doc_store = await BigtableVectorStore.afrom_documents(
            docs,
            embedding,
            instance_id=instance_id,
            table_id=managed_table,
            engine=engine,
            ids=["d1"],
            content_column=content_column,
            embedding_column=embedding_column,
            collection="test_coll_from_docs",
        )
        assert len(await doc_store.aget_by_ids(["d1"])) == 1
        await doc_store.adelete(["d1"])

    async def test_filtering(
        self, engine: BigtableEngine, instance_id: str, managed_table: str
    ) -> None:
        """Tests that metadata filters are applied correctly."""
        embedding_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)
        store = BigtableVectorStore(
            instance_id=instance_id,
            table_id=managed_table,
            embedding_service=embedding_service,
            content_column=content_column,
            embedding_column=embedding_column,
            engine=engine,
            metadata_mappings=[VectorMetadataMapping("color", Encoding.UTF8)],
            collection=f"{TEST_ROW_PREFIX}filter-",
        )
        docs = [
            Document(page_content="red car", metadata={"color": "red"}),
            Document(page_content="blue car", metadata={"color": "blue"}),
        ]
        added_ids = store.add_documents(docs)

        query_params = QueryParameters(
            filters={"ColumnValueFilter": {"color": {"==": "blue"}}}
        )
        results = store.similarity_search("any", k=2, query_parameters=query_params)
        assert len(results) == 1
        assert results[0].page_content == "blue car"

        store.delete(added_ids)

    async def test_as_retriever(
        self, engine: BigtableEngine, instance_id: str, managed_table: str
    ) -> None:
        """Tests that the store can be converted to a retriever."""
        embedding_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)
        store = BigtableVectorStore(
            instance_id=instance_id,
            table_id=managed_table,
            embedding_service=embedding_service,
            engine=engine,
            content_column=content_column,
            embedding_column=embedding_column,
            collection="test_coll_2",
            metadata_mappings=[
                VectorMetadataMapping(metadata_key="color", encoding=Encoding.UTF8)
            ],
        )
        retriever = store.as_retriever(
            search_kwargs={
                "k": 1,
                "filter": {"ColumnValueFilter": {"color": {"==": "yellow"}}},
            }
        )
        assert isinstance(retriever, VectorStoreRetriever)
        store.add_texts(
            ["retriever test"], ids=["retriever1"], metadatas=[{"color": "yellow"}]
        )
        retriever_results = retriever.invoke("retriever test")
        assert len(retriever_results) == 1
        store.delete(["retriever1"])

    async def test_edge_cases_and_errors(
        self, engine: BigtableEngine, instance_id: str, managed_table: str
    ) -> None:
        """Tests behavior with empty inputs, non-existent keys, and errors."""
        embedding_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)
        store = BigtableVectorStore(
            instance_id=instance_id,
            table_id=managed_table,
            embedding_service=embedding_service,
            collection="test_coll_3",
            content_column=content_column,
            embedding_column=embedding_column,
            engine=engine,
            metadata_mappings=[
                VectorMetadataMapping("number", Encoding.INT_BIG_ENDIAN)
            ],
        )

        # Empty operations
        assert store.add_texts([]) == []
        assert store.get_by_ids([]) == []
        assert await store.aget_by_ids([]) == []
        assert store.delete([]) is True
        assert store.similarity_search("empty", k=0) == []

        # Non-existent keys
        assert store.get_by_ids(["non-existent-id"]) == []
        assert store.delete(["non-existent-id"]) is True  # No error on missing key

        # Search in empty table
        assert store.similarity_search("empty", k=1) == []

        # Invalid metadata type
        with pytest.raises(ValueError, match="Failed to encode value"):
            store.add_texts(["bad data"], metadatas=[{"number": "not-a-number"}])

        # Empty from_texts/from_documents
        empty_store = BigtableVectorStore.from_texts(
            [],
            embedding_service,
            instance_id=instance_id,
            table_id=managed_table,
            engine=engine,
            content_column=content_column,
            embedding_column=embedding_column,
            collection="test_coll_empty_from_texts",
        )

    async def test_large_batch_operations(
        self, engine: BigtableEngine, instance_id: str, managed_table: str
    ) -> None:
        """Tests adding, getting, and deleting a large batch of documents."""
        embedding_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)
        store = BigtableVectorStore(
            instance_id=instance_id,
            table_id=managed_table,
            embedding_service=embedding_service,
            content_column=content_column,
            embedding_column=embedding_column,
            engine=engine,
            collection=f"{TEST_ROW_PREFIX}large-batch-",
        )

        num_docs = 200
        docs = [
            Document(page_content=f"doc num {i}", id=f"large_{i}", metadata={})
            for i in range(num_docs)
        ]
        ids = [f"large_{i}" for i in range(num_docs)]

        added_ids = store.add_documents(docs)
        assert len(added_ids) == num_docs

        retrieved_docs = store.get_by_ids(ids)
        assert len(retrieved_docs) == num_docs

        store.delete(ids)
        retrieved_after_delete = store.get_by_ids(ids)
        assert len(retrieved_after_delete) == 0

    async def test_collection_isolation(
        self, engine: BigtableEngine, instance_id: str, managed_table: str
    ) -> None:
        """Tests that different collections in the same table are isolated."""
        embedding_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)
        store1 = BigtableVectorStore(
            instance_id=instance_id,
            table_id=managed_table,
            embedding_service=embedding_service,
            content_column=content_column,
            embedding_column=embedding_column,
            engine=engine,
            collection="collection1",
        )
        store2 = BigtableVectorStore(
            instance_id=instance_id,
            table_id=managed_table,
            embedding_service=embedding_service,
            engine=engine,
            content_column=content_column,
            embedding_column=embedding_column,
            collection="collection2",
        )

        store1.add_texts(["doc from collection 1"], ids=["c1_doc1"])
        store2.add_texts(["doc from collection 2"], ids=["c2_doc1"])

        # Search store 1, should only find its own doc
        results1 = store1.similarity_search("doc", k=5)
        assert len(results1) == 1
        assert results1[0].page_content == "doc from collection 1"

        # Search store 2, should only find its own doc
        results2 = store2.similarity_search("doc", k=5)
        assert len(results2) == 1
        assert results2[0].page_content == "doc from collection 2"

        store1.delete(["c1_doc1"])
        store2.delete(["c2_doc1"])
