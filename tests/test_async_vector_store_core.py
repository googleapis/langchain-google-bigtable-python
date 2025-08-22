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
from typing import AsyncGenerator, AsyncIterator, Iterator

import google.auth
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

from langchain_google_bigtable.async_vector_store import (
    AsyncBigtableVectorStore,
    ColumnConfig,
    Encoding,
    MetadataMapping,
    QueryParameters,
)

TEST_ROW_PREFIX = "pytest-core-vstore-"
CONTENT_COLUMN_FAMILY = "content-cf"
EMBEDDING_COLUMN_FAMILY = "embedding-cf"
METADATA_COLUMN_FAMILY = "md"
VECTOR_SIZE = 3

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
    table_id = f"test-core-{uuid.uuid4().hex[:8]}"
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
class TestCoreFunctionality:
    """Tests for core vector store functionality."""

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
        embedding_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)
        vector_store = AsyncBigtableVectorStore(
            client=async_data_client,
            instance_id=instance_id,
            async_table=table,
            embedding_service=embedding_service,
            content_column=ColumnConfig(CONTENT_COLUMN_FAMILY, "content"),
            embedding_column=ColumnConfig(EMBEDDING_COLUMN_FAMILY, "embedding"),
            metadata_mappings=[
                MetadataMapping("color", Encoding.UTF8),
                MetadataMapping("number", Encoding.INT_BIG_ENDIAN),
                MetadataMapping("is_good", Encoding.BOOL),
                MetadataMapping("rating", Encoding.FLOAT),
            ],
            collection=TEST_ROW_PREFIX,
        )
        yield vector_store

    async def test_aadd_adelete_aget(self, store: AsyncBigtableVectorStore) -> None:
        """Tests adding, getting, and deleting documents."""
        docs = [
            Document(
                page_content="the sky is blue",
                metadata={"color": "blue", "number": 10},
                id="doc1",
            ),
            Document(
                page_content="the sun is yellow",
                metadata={"color": "yellow", "number": 20},
                id="doc2",
            ),
        ]
        added_doc_ids = ["doc1", "doc2"]
        added_row_keys = await store.aadd_documents(docs)
        assert added_row_keys == added_doc_ids

        retrieved = await store.aget_by_ids(added_doc_ids)
        assert len(retrieved) == 2
        assert {doc.id for doc in retrieved} == set(added_doc_ids)

        await store.adelete(["doc1"])
        retrieved_after_delete = await store.aget_by_ids(added_doc_ids)
        assert len(retrieved_after_delete) == 1
        assert retrieved_after_delete[0].id == "doc2"

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_similarity_search_methods(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests standard similarity search methods."""
        texts = ["apple", "banana", "cake", "kite"]
        added_doc_ids = await store.aadd_texts(texts)
        query = "apple"
        embedding = await store.embedding_service.aembed_query(query)

        results_sim = await store.asimilarity_search(query, k=2)
        assert len(results_sim) == 2
        assert "apple" in [doc.page_content for doc in results_sim]

        results_sim_vec = await store.asimilarity_search_by_vector(embedding, k=2)
        assert len(results_sim_vec) == 2
        assert {doc.page_content for doc in results_sim} == {
            doc.page_content for doc in results_sim_vec
        }

        results_score = await store.asimilarity_search_with_score(query, k=1)
        assert len(results_score) == 1
        doc, score = results_score[0]
        assert doc.page_content == "apple"
        assert isinstance(score, float)

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_asimilarity_search_with_relevance_scores(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests that relevance scores are returned correctly for different distance strategies."""
        texts_to_add = ["a document about cats", "a document about dogs"]
        added_doc_ids = await store.aadd_texts(texts_to_add)

        query = "a document about cats"
        results_cosine = await store.asimilarity_search_with_relevance_scores(
            query, k=1
        )

        assert len(results_cosine) == 1
        doc, score = results_cosine[0]
        assert isinstance(doc, Document)
        assert doc.page_content == "a document about cats"
        assert isinstance(score, float)

        query_params_euclidean = QueryParameters(distance_strategy="EUCLIDEAN")
        results_euclidean = await store.asimilarity_search_with_relevance_scores(
            query, k=1, query_parameters=query_params_euclidean
        )

        assert len(results_euclidean) == 1
        doc_euc, score_euc = results_euclidean[0]
        assert isinstance(doc_euc, Document)
        assert doc_euc.page_content == "a document about cats"
        assert isinstance(score_euc, float)

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_amax_marginal_relevance_search(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests amax_marginal_relevance_search returns the correct top result."""
        texts = ["foo", "bar", "baz", "boo"]
        added_doc_ids = await store.aadd_texts(texts)
        results = await store.amax_marginal_relevance_search("bar")
        assert results[0].page_content == "bar"

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_amax_marginal_relevance_search_by_vector(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests amax_marginal_relevance_search_by_vector returns the correct top result."""
        texts = ["foo", "bar", "baz", "boo"]
        added_doc_ids = await store.aadd_texts(texts)
        embedding = await store.embedding_service.aembed_query("bar")
        results = await store.amax_marginal_relevance_search_by_vector(embedding)
        assert results[0].page_content == "bar"

        # Clean up
        await store.adelete(added_doc_ids)

    async def test_afrom_documents(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the afrom_texts class method for initialization and adding data."""
        ids = ["doc01", "doc02"]
        texts = ["text1", "text2"]
        metadatas = [{"color": "green"}, {"color": "orange"}]
        documents = [
            Document(page_content=text, metadata=metadata, id=id)
            for id, text, metadata in zip(ids, texts, metadatas)
        ]
        new_store = await AsyncBigtableVectorStore.afrom_documents(
            documents=documents,
            embedding=store.embedding_service,
            client=store.client,
            instance_id=store.instance_id,
            async_table=store.async_table,
            content_column=store.content_column,
            embedding_column=store.embedding_column,
            metadata_mappings=store.metadata_mappings,
            collection=store.collection,
        )
        results = await new_store.asimilarity_search("text1", k=2)
        assert len(results) == 2
        assert {doc.page_content for doc in results} == set(texts)

        # Clean up
        await new_store.adelete(ids)

    async def test_afrom_texts(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the afrom_texts class method for initialization and adding data."""
        ids = ["doc01", "doc02"]
        texts = ["text1", "text2"]
        metadatas = [{"color": "green"}, {"color": "orange"}]
        new_store = await AsyncBigtableVectorStore.afrom_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embedding=store.embedding_service,
            client=store.client,
            instance_id=store.instance_id,
            async_table=store.async_table,
            content_column=store.content_column,
            embedding_column=store.embedding_column,
            metadata_mappings=store.metadata_mappings,
            collection=store.collection,
        )
        results = await new_store.asimilarity_search("text1", k=2)
        assert len(results) == 2
        assert {doc.page_content for doc in results} == set(texts)

        # Clean up
        await new_store.adelete(ids)
