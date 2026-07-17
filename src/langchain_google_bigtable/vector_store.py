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

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import google.auth.credentials
from google.api_core.exceptions import (
    GoogleAPIError,
    PermissionDenied,
)
from google.cloud import bigtable
from google.cloud.bigtable.column_family import MaxVersionsGCRule
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import (
    VectorStore,
    VectorStoreRetriever,
)

from .async_vector_store import (
    METADATA_COLUMN_FAMILY,
    AsyncBigtableVectorStore,
    ColumnConfig,
    DistanceStrategy,
    Encoding,
    QueryParameters,
    VectorDataType,
    VectorMetadataMapping,
)
from .engine import BigtableEngine

CONTENT_COLUMN_FAMILY = "langchain"
EMBEDDING_COLUMN_FAMILY = "langchain"


def init_vector_store_table(
    instance_id: str,
    table_id: str,
    project_id: Optional[str] = None,
    client: Optional[bigtable.Client] = None,
    content_column_family: str = CONTENT_COLUMN_FAMILY,
    embedding_column_family: str = EMBEDDING_COLUMN_FAMILY,
) -> None:
    """Creates a Bigtable table with the necessary column families for the vector store.

    It always creates the "md" column family for metadata and allows you to specify
    the name of the column family used for content and embeddings.

    Args:
        instance_id (str): The ID of the Bigtable instance.
        table_id (str): The ID of the table to create.
        project_id (Optional[str]): Your Google Cloud project ID.
        client (Optional[bigtable.Client]): An optional pre-configured Bigtable admin client.
        content_column_family (str): The name of the column family to store document content.
                                     Defaults to "langchain".
        embedding_column_family (str): The name of the column family to store embeddings. Defaults to "langchain".

    Raises:
        ValueError: If the table already exists.
        PermissionDenied: If the client lacks permission to create the table.
        GoogleAPIError: If any other Google Cloud API error occurs.
    """
    if client is None:
        client = bigtable.Client(project=project_id, admin=True)

    instance = client.instance(instance_id)
    table = instance.table(table_id)

    if table.exists():
        raise ValueError(f"Table {table_id} already exists")

    families_to_create = {
        content_column_family,
        embedding_column_family,
        METADATA_COLUMN_FAMILY,
    }

    gc_rule = MaxVersionsGCRule(1)
    column_family_rules = {cf: gc_rule for cf in families_to_create}

    try:
        table.create(column_families=column_family_rules)
    except PermissionDenied as e:
        raise PermissionDenied(f"Permission denied while creating table: {e}") from e
    except GoogleAPIError as e:
        raise GoogleAPIError(f"A Google Cloud error occurred: {e}") from e


class BigtableVectorStore(VectorStore):
    """
    A vector store implementation using Google Cloud Bigtable.

    This class provides the main user-facing interface, conforming to the
    `langchain_core.vectorstores.VectorStore` standard, and handles both
    synchronous and asynchronous operations by wrapping an async core.
    """

    def __init__(
        self,
        instance_id: str,
        table_id: str,
        embedding_service: Embeddings,
        collection: str,
        content_column: ColumnConfig = ColumnConfig(
            column_family=CONTENT_COLUMN_FAMILY, column_qualifier="content"
        ),
        embedding_column: ColumnConfig = ColumnConfig(
            column_family=EMBEDDING_COLUMN_FAMILY, column_qualifier="embedding"
        ),
        project_id: Optional[str] = None,
        metadata_mappings: Optional[List[VectorMetadataMapping]] = None,
        metadata_as_json_column: Optional[ColumnConfig] = None,
        engine: Optional[BigtableEngine] = None,
        app_profile_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initializes the BigtableVectorStore.

        Args:
            instance_id (str): Your Bigtable instance ID.
            table_id (str): The ID of the table to use for the vector store.
            embedding_service (Embeddings): The embedding service to use.
            collection (str): A name for the collection of vectors for this store. Internally, this is used as the row key prefix.
            content_column (ColumnConfig): Configuration for the document content column.
            embedding_column (ColumnConfig): Configuration for the vector embedding column.
            project_id (Optional[str]): Your Google Cloud project ID.
            metadata_mappings (Optional[List[VectorMetadataMapping]]): Mappings for storing metadata in separate columns.
            metadata_as_json_column (Optional[ColumnConfig]): Configuration for storing all metadata in a
                                                               single JSON column.
            engine (Optional[BigtableEngine]): The BigtableEngine to use for connecting to Bigtable.
            app_profile_id (Optional[str]): The Bigtable app profile ID to use for requests.
            **kwargs (Any): Additional arguments for engine creation if an engine is not provided.
        """
        self.instance_id = instance_id
        self.table_id = table_id
        self.embedding_service = embedding_service
        self.project_id = project_id
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_mappings = metadata_mappings
        self.collection = collection
        self.metadata_as_json_column = metadata_as_json_column
        self.app_profile_id = app_profile_id
        if engine:
            self.engine = engine
        else:
            self.engine = BigtableEngine.initialize(
                project_id=project_id,
                instance_id=instance_id,
                **kwargs,
            )
        self.async_store: Optional[AsyncBigtableVectorStore] = None

    @property
    def embeddings(self) -> Embeddings:
        """
        Access the query embedding object.

        Returns:
            (Embeddings): The embedding service object.
        """
        return self.embedding_service

    async def _get_async_store(self) -> AsyncBigtableVectorStore:
        """
        Lazily initializes and returns the underlying async store.

        Returns:
            (AsyncBigtableVectorStore): The initialized asynchronous vector store.
        """
        if not self.async_store:
            if not self.engine:
                raise ValueError(
                    "BigtableEngine not initialized. Call 'create' or 'create_sync'."
                )

            async_table = await self.engine.get_async_table(
                self.instance_id, self.table_id, app_profile_id=self.app_profile_id
            )
            self.async_store = AsyncBigtableVectorStore(
                client=self.engine.async_client,
                instance_id=self.instance_id,
                async_table=async_table,
                embedding_service=self.embedding_service,
                content_column=self.content_column,
                embedding_column=self.embedding_column,
                collection=self.collection,
                metadata_as_json_column=self.metadata_as_json_column,
                metadata_mappings=self.metadata_mappings,
            )
        return self.async_store

    @classmethod
    def create_sync(
        cls,
        instance_id: str,
        table_id: str,
        embedding_service: Embeddings,
        collection: str,
        engine: Optional[BigtableEngine] = None,
        content_column: ColumnConfig = ColumnConfig(
            column_family=CONTENT_COLUMN_FAMILY, column_qualifier="content"
        ),
        embedding_column: ColumnConfig = ColumnConfig(
            column_family=EMBEDDING_COLUMN_FAMILY, column_qualifier="embedding"
        ),
        project_id: Optional[str] = None,
        metadata_mappings: Optional[List[VectorMetadataMapping]] = None,
        metadata_as_json_column: Optional[ColumnConfig] = None,
        app_profile_id: Optional[str] = None,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        client_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BigtableVectorStore:
        """
        Synchronously initializes the engine and creates an instance of the vector store.

        Args:
            instance_id (str): Your Bigtable instance ID.
            table_id (str): The ID of the table to use for the vector store.
            embedding_service (Embeddings): The embedding service to use.
            collection (str): A name for the collection of vectors for this store. Internally, this is used as the row key prefix.
            engine (Optional[BigtableEngine]): An optional, existing BigtableEngine.
            content_column (ColumnConfig): Configuration for the document content column.
            embedding_column (ColumnConfig): Configuration for the vector embedding column.
            project_id (Optional[str]): Your Google Cloud project ID.
            metadata_mappings (Optional[List[VectorMetadataMapping]]): Mappings for metadata columns.
            metadata_as_json_column (Optional[ColumnConfig]): Configuration for a single JSON metadata column.
            app_profile_id (Optional[str]): The Bigtable app profile ID to use.
            credentials (Optional[google.auth.credentials.Credentials]): Custom credentials to use.
            client_options (Optional[Dict[str, Any]]): Client options for the Bigtable client.
            **kwargs (Any): Additional keyword arguments for engine initialization.

        Returns:
            (BigtableVectorStore): An instance of the vector store.
        """
        my_engine: BigtableEngine
        if engine:
            my_engine = engine
        else:
            my_engine = BigtableEngine.initialize(
                project_id=project_id,
                instance_id=instance_id,
                credentials=credentials,
                client_options=client_options,
                **kwargs,
            )
        return cls(
            instance_id=instance_id,
            table_id=table_id,
            embedding_service=embedding_service,
            project_id=project_id,
            collection=collection,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_mappings=metadata_mappings,
            metadata_as_json_column=metadata_as_json_column,
            engine=my_engine,
            app_profile_id=app_profile_id,
        )

    @classmethod
    async def create(
        cls,
        instance_id: str,
        table_id: str,
        embedding_service: Embeddings,
        collection: str,
        engine: Optional[BigtableEngine] = None,
        content_column: ColumnConfig = ColumnConfig(
            column_family=CONTENT_COLUMN_FAMILY, column_qualifier="content"
        ),
        embedding_column: ColumnConfig = ColumnConfig(
            column_family=EMBEDDING_COLUMN_FAMILY, column_qualifier="embedding"
        ),
        project_id: Optional[str] = None,
        metadata_mappings: Optional[List[VectorMetadataMapping]] = None,
        metadata_as_json_column: Optional[ColumnConfig] = None,
        app_profile_id: Optional[str] = None,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        client_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BigtableVectorStore:
        """
        Asynchronously initializes the engine and creates an instance of the vector store.

        Args:
            instance_id (str): Your Bigtable instance ID.
            table_id (str): The ID of the table to use for the vector store.
            embedding_service (Embeddings): The embedding service to use.
            collection (str): A name for the collection of vectors for this store. Internally, this is used as the row key prefix.
            engine (Optional[BigtableEngine]): An optional, existing BigtableEngine.
            content_column (ColumnConfig): Configuration for the document content column.
            embedding_column (ColumnConfig): Configuration for the vector embedding column.
            project_id (Optional[str]): Your Google Cloud project ID.
            query_parameters (Optional[QueryParameters]): Default QueryParameters for searches.
            metadata_mappings (Optional[List[VectorMetadataMapping]]): Mappings for metadata columns.
            metadata_as_json_column (Optional[ColumnConfig]): Configuration for a single JSON metadata column.
            app_profile_id (Optional[str]): The Bigtable app profile ID to use.
            credentials (Optional[google.auth.credentials.Credentials]): Custom credentials to use.
            client_options (Optional[Dict[str, Any]]): Client options for the Bigtable client.
            **kwargs (Any): Additional keyword arguments for engine initialization.

        Returns:
            (BigtableVectorStore): An instance of the vector store.
        """
        my_engine: BigtableEngine
        if engine:
            my_engine = engine
        else:
            my_engine = await BigtableEngine.async_initialize(
                project_id=project_id,
                instance_id=instance_id,
                credentials=credentials,
                client_options=client_options,
                **kwargs,
            )
        return cls(
            instance_id=instance_id,
            table_id=table_id,
            embedding_service=embedding_service,
            project_id=project_id,
            collection=collection,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_mappings=metadata_mappings,
            metadata_as_json_column=metadata_as_json_column,
            engine=my_engine,
            app_profile_id=app_profile_id,
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> BigtableVectorStore:
        """
        Return VectorStore initialized from documents and embeddings.
        This is a synchronous method that creates the store and adds documents.

        Args:
            documents (List[Document]): List of documents to add.
            embedding (Embeddings): The embedding service to use.
            ids (Optional[list]): list of IDs for the texts.
            **kwargs (Any): Keyword arguments to pass to the `create_sync` method.

        Returns:
            (BigtableVectorStore): An instance of the vector store.
        """
        store = cls.create_sync(embedding_service=embedding, **kwargs)
        store.add_documents(documents, ids)
        return store

    @classmethod
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> BigtableVectorStore:
        """
        Return VectorStore initialized from documents and embeddings.
        This is an asynchronous method that creates the store and adds documents.

        Args:
            documents (List[Document]): List of documents to add.
            embedding (Embeddings): The embedding service to use.
             ids (Optional[list]): list of IDs for the documents.
            **kwargs (Any): Keyword arguments to pass to the `create` method.

        Returns:
            (BigtableVectorStore): An instance of the vector store.
        """
        store = await cls.create(embedding_service=embedding, **kwargs)
        await store.aadd_documents(documents, ids)
        return store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> BigtableVectorStore:
        """
        Return VectorStore initialized from texts and embeddings.

        Args:
            texts (List[str]): List of text strings to add.
            embedding (Embeddings): The embedding service to use.
            metadatas (Optional[List[dict]]): Optional list of metadata for each text.
            ids (Optional[list]): list of IDs for the texts.
            **kwargs (Any): Keyword arguments to pass to the `create_sync` factory method.

        Returns:
            (BigtableVectorStore): An instance of the vector store.
        """
        store = cls.create_sync(embedding_service=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> BigtableVectorStore:
        """
        Return VectorStore initialized from texts and embeddings.

        Args:
            texts (List[str]): List of text strings to add.
            embedding (Embeddings): The embedding service to use.
            metadatas (Optional[List[dict]]): Optional list of metadata for each text.
            ids (Optional[list]): list of IDs for the texts.
            **kwargs (Any): Keyword arguments to pass to the `acreate` method.

        Returns:
            (BigtableVectorStore): An instance of the vector store.
        """
        store = await cls.create(embedding_service=embedding, **kwargs)
        await store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): An iterable of texts to add.
            metadatas (Optional[List[Dict]]): Optional list of metadatas.
            ids (Optional[list]): list of IDs for the texts.
            **kwargs (Any): Additional arguments.

        Returns:
            (List[str]): A list of the row keys of the added texts.
        """

        async def _internal() -> List[str]:
            store = await self._get_async_store()
            return await store.aadd_texts(
                texts=list(texts), metadatas=metadatas, ids=ids, **kwargs
            )

        return self.engine._run_as_sync(_internal())

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): An iterable of texts to add.
            metadatas (Optional[List[dict]]): Optional list of metadatas.
            ids (Optional[list]): list of IDs for the texts.
            **kwargs (Any): Additional arguments.

        Returns:
            (List[str]): A list of the row keys of the added texts.
        """

        async def _internal() -> List[str]:
            store = await self._get_async_store()
            return await store.aadd_texts(
                texts=list(texts), metadatas=metadatas, ids=ids, **kwargs
            )

        return await self.engine._run_as_async(_internal())

    def add_documents(
        self, documents: List[Document], ids: Optional[list] = None, **kwargs: Any
    ) -> List[str]:
        """
        Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]): A list of documents to add.
            ids (Optional[list]): list of IDs for the documents.
            **kwargs (Any): Additional arguments.

        Returns:
            (List[str]): A list of the row keys of the added documents.
        """

        async def _internal() -> List[str]:
            store = await self._get_async_store()
            return await store.aadd_documents(documents, ids=ids, **kwargs)

        return self.engine._run_as_sync(_internal())

    async def aadd_documents(
        self, documents: List[Document], ids: Optional[list] = None, **kwargs: Any
    ) -> List[str]:
        """
        Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]): A list of documents to add.
            ids (Optional[list]): list of IDs for the documents.
            **kwargs (Any): Additional arguments.

        Returns:
            (List[str]): A list of the row keys of the added documents.
        """

        async def _internal() -> List[str]:
            store = await self._get_async_store()
            return await store.aadd_documents(documents, ids=ids, **kwargs)

        return await self.engine._run_as_async(_internal())

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete by vector ID.

        Args:
            ids (Optional[List[str]]): A list of document IDs to delete.
            **kwargs (Any): Additional arguments.

        Returns:
            (Optional[bool]): True if the deletion was successful.
        """
        if ids is None:
            return False

        async def _internal() -> None:
            store = await self._get_async_store()
            await store.adelete(ids, **kwargs)

        self.engine._run_as_sync(_internal())
        return True

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """
        Delete by vector ID.

        Args:
            ids (Optional[List[str]]): A list of document IDs to delete.
            **kwargs (Any): Additional arguments.

        Returns:
            (Optional[bool]): True if the deletion was successful.
        """
        if ids is None:
            return False

        async def _internal() -> None:
            store = await self._get_async_store()
            await store.adelete(ids, **kwargs)

        await self.engine._run_as_async(_internal())
        return True

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs most similar to query.

        Args:
            query (str): The text to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Document]): A list of documents most similar to the query.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.asimilarity_search(
                query, k=k, query_parameters=query_parameters, **kwargs
            )

        return self.engine._run_as_sync(_internal())

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs most similar to query.

        Args:
            query (str): The text to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Document]): A list of documents most similar to the query.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.asimilarity_search(
                query, k=k, query_parameters=query_parameters, **kwargs
            )

        return await self.engine._run_as_async(_internal())

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs most similar to embedding vector.

        Args:
            embedding (List[float]): The embedding vector to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Document]): A list of documents most similar to the embedding.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.asimilarity_search_by_vector(
                embedding, k=k, query_parameters=query_parameters, **kwargs
            )

        return self.engine._run_as_sync(_internal())

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs most similar to embedding vector.

        Args:
            embedding (List[float]): The embedding vector to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Document]): A list of documents most similar to the embedding.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.asimilarity_search_by_vector(
                embedding, k=k, query_parameters=query_parameters, **kwargs
            )

        return await self.engine._run_as_async(_internal())

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Run similarity search with distance.

        Args:
            query (str): The text to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Tuple[Document, float]]): A list of tuples containing the document and its distance score.
        """

        async def _internal() -> List[Tuple[Document, float]]:
            store = await self._get_async_store()
            return await store.asimilarity_search_with_score(
                query, k=k, query_parameters=query_parameters, **kwargs
            )

        return self.engine._run_as_sync(_internal())

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Run similarity search with distance.

        Args:
            query (str): The text to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Tuple[Document, float]]): A list of tuples containing the document and its distance score.
        """

        async def _internal() -> List[Tuple[Document, float]]:
            store = await self._get_async_store()
            return await store.asimilarity_search_with_score(
                query, k=k, query_parameters=query_parameters, **kwargs
            )

        return await self.engine._run_as_async(_internal())

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Run similarity search with distance by vector.

        Args:
            embedding (List[float]): The embedding vector to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Tuple[Document, float]]): A list of tuples containing the document and its distance score.
        """

        async def _internal() -> List[Tuple[Document, float]]:
            store = await self._get_async_store()
            return await store.asimilarity_search_with_score_by_vector(
                embedding, k=k, query_parameters=query_parameters, **kwargs
            )

        return self.engine._run_as_sync(_internal())

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Run similarity search with distance by vector.

        Args:
            embedding (List[float]): The embedding vector to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Tuple[Document, float]]): A list of tuples containing the document and its distance score.
        """

        async def _internal() -> List[Tuple[Document, float]]:
            store = await self._get_async_store()
            return await store.asimilarity_search_with_score_by_vector(
                embedding, k=k, query_parameters=query_parameters, **kwargs
            )

        return await self.engine._run_as_async(_internal())

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return docs and relevance scores in the range [0, 1].

        Args:
            query (str): The text to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
        """

        async def _internal() -> List[Tuple[Document, float]]:
            store = await self._get_async_store()
            return await store.asimilarity_search_with_relevance_scores(
                query, k=k, query_parameters=query_parameters, **kwargs
            )

        return self.engine._run_as_sync(_internal())

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return docs and relevance scores in the range [0, 1].

        Args:
            query (str): The text to search for.
            k (int): The number of results to return.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
        """

        async def _internal() -> List[Tuple[Document, float]]:
            store = await self._get_async_store()
            return await store.asimilarity_search_with_relevance_scores(
                query, k=k, query_parameters=query_parameters, **kwargs
            )

        return await self.engine._run_as_async(_internal())

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs selected using the maximal marginal relevance.

        Args:
            query (str): The text to search for.
            k (int): The number of results to return.
            fetch_k (int): The number of documents to fetch for MMR.
            lambda_mult (float): The lambda multiplier for MMR.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Document]): A list of documents selected by MMR.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.amax_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                query_parameters=query_parameters,
                **kwargs,
            )

        return self.engine._run_as_sync(_internal())

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs selected using the maximal marginal relevance.

        Args:
            query (str): The text to search for.
            k (int): The number of results to return.
            fetch_k (int): The number of documents to fetch for MMR.
            lambda_mult (float): The lambda multiplier for MMR.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Document]): A list of documents selected by MMR.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.amax_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                query_parameters=query_parameters,
                **kwargs,
            )

        return await self.engine._run_as_async(_internal())

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs selected using the maximal marginal relevance.

        Args:
            embedding (List[float]): The embedding vector to search for.
            k (int): The number of results to return.
            fetch_k (int): The number of documents to fetch for MMR.
            lambda_mult (float): The lambda multiplier for MMR.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Document]): A list of documents selected by MMR.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.amax_marginal_relevance_search_by_vector(
                embedding=embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                query_parameters=query_parameters,
                **kwargs,
            )

        return self.engine._run_as_sync(_internal())

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Return docs selected using the maximal marginal relevance.

        Args:
            embedding (List[float]): The embedding vector to search for.
            k (int): The number of results to return.
            fetch_k (int): The number of documents to fetch for MMR.
            lambda_mult (float): The lambda multiplier for MMR.
            query_parameters (Optional[QueryParameters]): Custom query parameters for this search.
            **kwargs (Any): Additional keyword arguments (e.g., filter).

        Returns:
            (List[Document]): A list of documents selected by MMR.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.amax_marginal_relevance_search_by_vector(
                embedding=embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                query_parameters=query_parameters,
                **kwargs,
            )

        return await self.engine._run_as_async(_internal())

    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """
        Return documents by their IDs.

        Args:
            ids (List[str]): A list of document IDs to retrieve.

        Returns:
            (List[Document]): A list of the retrieved documents.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.aget_by_ids(ids)

        return self.engine._run_as_sync(_internal())

    async def aget_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """
        Return documents by their IDs.

        Args:
            ids (List[str]): A list of document IDs to retrieve.

        Returns:
            (List[Document]): A list of the retrieved documents.
        """

        async def _internal() -> List[Document]:
            store = await self._get_async_store()
            return await store.aget_by_ids(ids)

        return await self.engine._run_as_async(_internal())

    async def get_engine(self) -> Optional[BigtableEngine]:
        """
        Get the BigtableEngine instance.

        Returns:
            (Optional[BigtableEngine]): The engine instance if it exists.
        """
        if hasattr(self, "engine"):
            return self.engine
        return None

    async def close(self) -> None:
        """
        Close the engine connection.
        """
        if hasattr(self, "engine") and self.engine:
            await self.engine.close()

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """
        Return VectorStoreRetriever initialized from this VectorStore.

        Args:
            **kwargs (Any): Keyword arguments to pass to the retriever.

        Returns:
            (VectorStoreRetriever): The initialized retriever.
        """
        return super().as_retriever(**kwargs)
