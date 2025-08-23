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

import copy
import json
import struct
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)
from uuid import uuid4

import numpy as np
from google.api_core.exceptions import (
    GoogleAPIError,
    InvalidArgument,
    PermissionDenied,
    ResourceExhausted,
)
from google.cloud import bigtable
from google.cloud.bigtable.data import (
    BigtableDataClientAsync,
    DeleteAllFromRow,
    ReadRowsQuery,
    RowMutationEntry,
    SetCell,
    TableAsync,
)
from google.cloud.bigtable.data.execute_query import SqlType as Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import (
    VectorStore,
    utils,
)

from langchain_google_bigtable.loader import Encoding

METADATA_COLUMN_FAMILY = "md"


class DistanceStrategy(Enum):
    """Specifies the distance metric for vector similarity search."""

    COSINE = "COSINE"
    EUCLIDEAN = "EUCLIDEAN"


class VectorDataType(Enum):
    """Specifies the data type of the vector's elements."""

    FLOAT32 = "float32"
    DOUBLE64 = "double64"


@dataclass
class ColumnConfig:
    """
    Represents the configuration for a single column in Bigtable.

    Attributes:
        column_family (str): The name of the column family.
        column_qualifier (str): The name of the column qualifier.
        encoding (Encoding): The data encoding to use for the column's value.
    """

    column_family: str
    column_qualifier: str
    encoding: Encoding = Encoding.UTF8


class MetadataMapping(ColumnConfig):
    """
    A specific ColumnConfig for mapping a metadata key to its own Bigtable column.
    It is used for filtering tasks and must be called in order to use this metadata
    key for filtering tasks.

    This class automatically sets the column family to the default metadata family
    and defaults the column name to the metadata key if not specified.

    Attributes:
        metadata_key (str): The key in the metadata dictionary that this column maps to.
    """

    metadata_key: str

    def __init__(
        self,
        metadata_key: str,
        encoding: Encoding,
        column_qualifier: Optional[str] = None,
    ):
        """Initializes the MetadataMapping.

        Args:
            metadata_key (str): The metadata dictionary key to map.
            encoding (Encoding): The data encoding to use for the column's value.
            column_qualifier (Optional[str]): Optional name to use for the column qualifier. If not
                                              provided, the `metadata_key` is used as the column name.
        """
        final_column_name = column_qualifier or metadata_key
        super().__init__(
            column_family=METADATA_COLUMN_FAMILY,
            column_qualifier=final_column_name,
            encoding=encoding,
        )
        self.metadata_key = metadata_key


@dataclass
class QueryParameters:
    """
    Holds the parameters for a vector search query.

    Attributes:
        algorithm (str): The search algorithm to use (e.g., "kNN").
        distance_strategy (DistanceStrategy): The distance metric to use for comparison.
        vector_data_type (VectorDataType): The data type of the stored vectors.
        filters (Optional[Dict[str, Any]]): Optional dictionary of metadata filters to apply to the search.
    """

    algorithm: str = "kNN"
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
    vector_data_type: VectorDataType = VectorDataType.FLOAT32
    filters: Optional[Dict[str, Any]] = None


class AsyncBigtableVectorStore(VectorStore):
    def __init__(
        self,
        client: BigtableDataClientAsync,
        instance_id: str,
        async_table: TableAsync,
        embedding_service: Embeddings,
        content_column: ColumnConfig,
        embedding_column: ColumnConfig,
        collection: Optional[str] = None,
        metadata_as_json_column: Optional[ColumnConfig] = None,
        metadata_mappings: Optional[List[MetadataMapping]] = None,
    ):
        """Initializes the AsyncBigtableVectorStore.

        Args:
            client (BigtableDataClientAsync): The Bigtable async client instance.
            instance_id (str): The Bigtable instance ID.
            async_table (TableAsync): The Bigtable TableAsync instance.
            embedding_service (Embeddings): The embedding service to use.
            content_column (ColumnConfig): ColumnConfig for document content.
            embedding_column (ColumnConfig): ColumnConfig for vector embeddings.
            collection (Optional[str]): The name of the collection (optional). It is used as a prefix for row keys.
            metadata_as_json_column (Optional[ColumnConfig]): ColumnConfig for metadata as JSON column.
            metadata_mappings (Optional[List[MetadataMapping]]): List of MetadataMapping for additional metadata columns.
        """
        self.client = client
        self.instance_id = instance_id
        self.async_table = async_table
        self.embedding_service = embedding_service
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_as_json_column = metadata_as_json_column
        self.metadata_mappings = metadata_mappings or []
        self.collection = collection
        self._metadata_lookup = {
            m.column_qualifier: (
                m.column_family,
                m.column_qualifier,
                m.metadata_key,
                m.encoding,
            )
            for m in self.metadata_mappings
        }

    @classmethod
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> AsyncBigtableVectorStore:
        """Asynchronously creates a BigtableVectorStore from a list of documents.

        Args:
            documents (List[Document]): List of documents to add.
            embedding (Embeddings): Embedding service to use.
            **kwargs (Any): Additional arguments for AsyncBigtableVectorStore constructor.
                            Requires: client, instance_id, async_table.

        Returns:
            AsyncBigtableVectorStore: An instance of the vector store.
        """
        store = cls(embedding_service=embedding, **kwargs)
        await store.aadd_documents(documents)
        return store

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncBigtableVectorStore:
        """Asynchronously creates a BigtableVectorStore from a list of texts.

        Args:
            texts (List[str]): List of texts to add.
            embedding (Embeddings): Embedding service to use.
            metadatas (Optional[List[dict]]): Optional list of metadatas.
            ids (Optional[List[str]]): Optional list of ids for each text.
            **kwargs (Any): Additional arguments for AsyncBigtableVectorStore constructor.

        Returns:
            AsyncBigtableVectorStore: An instance of the vector store.
        """
        store = cls(embedding_service=embedding, **kwargs)
        await store.aadd_texts(texts, metadatas, ids)
        return store

    def _get_query_params(
        self,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> QueryParameters:
        """Merges instance, method, and kwarg filters into a single QueryParameters object.

        Args:
            query_parameters (Optional[QueryParameters]): Parameters passed directly to the method.
            **kwargs (Any): Additional keyword arguments, may contain a 'filter'.

        Returns:
            QueryParameters: The final resolved query parameters.
        """
        final_params = query_parameters or QueryParameters()
        if "filter" in kwargs:
            final_params = copy.deepcopy(final_params)
            final_params.filters = kwargs.pop("filter")
        return final_params

    def _encode_value(self, value: Any, encoding: Encoding) -> bytes:
        """Encodes a value into bytes based on the specified encoding.

        Args:
            value (Any): The value to encode.
            encoding (Encoding): The encoding to use.

        Returns:
            bytes: The encoded value.
        """
        if isinstance(value, bytes):
            return value
        try:
            if encoding == Encoding.UTF8:
                return str(value).encode("utf-8")
            if encoding == Encoding.UTF16:
                return str(value).encode("utf-16")
            if encoding == Encoding.ASCII:
                return str(value).encode("ascii")
            if encoding == Encoding.BOOL:
                return struct.pack("?", bool(value))
            if encoding == Encoding.INT_LITTLE_ENDIAN:
                return struct.pack("<q", int(value))
            if encoding == Encoding.FLOAT:
                return struct.pack(">f", float(value))
            if encoding == Encoding.DOUBLE:
                return struct.pack(">d", float(value))
            if encoding == Encoding.INT_BIG_ENDIAN:
                return struct.pack(">q", int(value))
        except (ValueError, struct.error) as e:
            raise ValueError(
                f"Failed to encode value '{value}' with encoding {encoding}: {e}"
            )
        return str(value).encode("utf-8")

    def _decode_value(self, value: bytes, encoding: Encoding) -> Any:
        """Decodes bytes into a value based on the specified encoding.

        Args:
            value (bytes): The bytes to decode.
            encoding (Encoding): The encoding used for the value.

        Returns:
            Any: The decoded value.
        """
        if not isinstance(value, bytes) or encoding == Encoding.CUSTOM:
            return value
        try:
            if encoding == Encoding.UTF8:
                return value.decode("utf-8")
            if encoding == Encoding.UTF16:
                return value.decode("utf-16")
            if encoding == Encoding.ASCII:
                return value.decode("ascii")
            if encoding == Encoding.BOOL:
                return struct.unpack("?", value)[0]
            if encoding == Encoding.INT_LITTLE_ENDIAN:
                return struct.unpack("<q", value)[0]
            if encoding == Encoding.FLOAT:
                return struct.unpack(">f", value)[0]
            if encoding == Encoding.DOUBLE:
                return struct.unpack(">d", value)[0]
            if encoding == Encoding.INT_BIG_ENDIAN:
                return struct.unpack(">q", value)[0]
        except (UnicodeDecodeError, struct.error) as e:
            raise ValueError(f"Failed to decode value with encoding {encoding}: {e}")
        return value.decode("utf-8")

    def _embeddings_to_bytes(
        self, data: List[float], encoding: Optional[Encoding] = Encoding.FLOAT
    ) -> bytes:
        """Converts a list of numbers (embeddings) to bytes.

        Args:
            data (List[float]): The embeddings data to convert.
            encoding (Optional[Encoding]): The encoding to use (FLOAT or DOUBLE).

        Returns:
            bytes: The byte representation of the data.
        """
        if not isinstance(data, list):
            raise ValueError("Data must be a list of numbers.")
        struct_format = ">d" if encoding == Encoding.DOUBLE else ">f"
        return b"".join(struct.pack(struct_format, value) for value in data)

    def _bytes_to_embeddings(
        self, byte_data: bytes, encoding: Optional[Encoding] = Encoding.FLOAT
    ) -> List[float]:
        """Converts a bytes object back into a list of embedding values.

        Args:
            byte_data (bytes): The bytes object to convert.
            encoding (Optional[Encoding]): The encoding that was used to pack the numbers.

        Returns:
            List[float]: A list of floats or doubles representing the embeddings.
        """
        if not isinstance(byte_data, bytes):
            raise TypeError("Input data must be a bytes object.")
        struct_format = ">d" if encoding == Encoding.DOUBLE else ">f"
        value_size = struct.calcsize(struct_format)
        if len(byte_data) % value_size != 0:
            raise ValueError(
                f"Byte data length is not a multiple of value size {value_size}."
            )
        return [
            struct.unpack(struct_format, chunk)[0]
            for chunk in (
                byte_data[i : i + value_size]
                for i in range(0, len(byte_data), value_size)
            )
        ]

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Asynchronously adds texts to the Bigtable, generating embeddings.

        Args:
            texts (List[str]): List of text strings to add.
            metadatas (Optional[List[Dict[str, Any]]]): Optional list of metadata dictionaries for each text.
            ids (Optional[List[str]]): Optional list of ids for each text.

        Returns:
            List[str]: List of added document ids.
        """
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid4()) for _ in texts]
        doc_embeddings = await self.embedding_service.aembed_documents(list(texts))

        mutations = []
        added_doc_ids = []
        row_keys = []
        for text, embeddings, metadata, doc_id in zip(
            texts, doc_embeddings, metadatas, ids
        ):
            # Check if the embeddings is a NumPy array
            if isinstance(embeddings, np.ndarray):
                # If so, change to list
                embeddings = embeddings.tolist()

            # Convert any np.float type objects to float to preparing for encoding
            if embeddings and isinstance(embeddings[0], np.number):
                embeddings = [float(x) for x in embeddings]

            row_key = f"{self.collection}:{doc_id}" if self.collection else doc_id
            row_keys.append(row_key)
            added_doc_ids.append(doc_id)
            mutation_entries = [
                SetCell(
                    self.content_column.column_family,
                    self.content_column.column_qualifier,
                    self._encode_value(text, self.content_column.encoding),
                ),
                SetCell(
                    self.embedding_column.column_family,
                    self.embedding_column.column_qualifier,
                    self._embeddings_to_bytes(
                        embeddings, self.embedding_column.encoding
                    ),
                ),
            ]
            if self.metadata_as_json_column and metadata:
                mutation_entries.append(
                    SetCell(
                        self.metadata_as_json_column.column_family,
                        self.metadata_as_json_column.column_qualifier,
                        self._encode_value(
                            json.dumps(metadata), self.metadata_as_json_column.encoding
                        ),
                    )
                )
            if self.metadata_mappings and metadata:
                for mapping in self.metadata_mappings:
                    if mapping.metadata_key in metadata:
                        mutation_entries.append(
                            SetCell(
                                mapping.column_family,
                                mapping.column_qualifier.encode("utf-8"),
                                self._encode_value(
                                    metadata[mapping.metadata_key], mapping.encoding
                                ),
                            )
                        )
            mutations.append(
                RowMutationEntry(row_key=row_key, mutations=mutation_entries)  # type: ignore
            )

        if mutations:
            await self.async_table.bulk_mutate_rows(mutations)
        return added_doc_ids

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Asynchronously adds documents to the Bigtable.

        Args:
            documents (List[Document]): List of Document objects to add.

        Returns:
            List[str]: List of added document ids.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if not ids:
            ids = [doc.id if hasattr(doc, "id") else str(uuid4()) for doc in documents]
        return await self.aadd_texts(texts, metadatas, ids, **kwargs)

    async def adelete(
        self, ids: Optional[list] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Asynchronously deletes documents from Bigtable.

        Args:
            ids (Optional[list]): List of document IDs to delete.

        Returns:
            Optional[bool]: True if the deletion was attempted, None if no IDs were provided.
        """
        if not ids:
            return None

        mutations = [
            RowMutationEntry(
                f"{self.collection}:{key}" if self.collection else key,
                [DeleteAllFromRow()],
            )
            for key in ids
        ]
        if mutations:
            await self.async_table.bulk_mutate_rows(mutations)
        return True

    async def aget_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Asynchronously retrieves documents by their IDs.

        Args:
            ids (Sequence[str]): List of document IDs to retrieve.

        Returns:
            List[Document]: List of Document objects retrieved.

        Raises:
            GoogleAPIError: If there's an error during the read operation.
        """
        if not ids:
            return []
        row_keys = [
            f"{self.collection}:{key}" if self.collection else key for key in ids
        ]
        row_filter = bigtable.data.row_filters.CellsColumnLimitFilter(1)

        query = bigtable.data.ReadRowsQuery(row_keys=row_keys, row_filter=row_filter)  # type: ignore
        try:
            rows = await self.async_table.read_rows(query)
        except GoogleAPIError as e:
            raise GoogleAPIError(f"Error while getting documents by ids: {e}") from e
        documents = []
        for row in rows:
            content = None

            content = self._decode_value(
                row.get_cells(
                    family=self.content_column.column_family,
                    qualifier=self.content_column.column_qualifier,
                )[0].value,
                self.content_column.encoding,
            )
            metadata = {}
            if self.metadata_as_json_column and row.__contains__(
                (
                    self.metadata_as_json_column.column_family,
                    self.metadata_as_json_column.column_qualifier,
                )
            ):
                metadata = json.loads(
                    self._decode_value(
                        row.get_cells(
                            family=self.metadata_as_json_column.column_family,
                            qualifier=self.metadata_as_json_column.column_qualifier,
                        )[0].value,
                        self.metadata_as_json_column.encoding,
                    )
                )

            elif self.metadata_mappings:
                for mapping in self.metadata_mappings:
                    if row.__contains__(
                        (
                            mapping.column_family,
                            mapping.column_qualifier.encode("utf-8"),
                        )
                    ):
                        metadata[mapping.metadata_key] = self._decode_value(
                            row.get_cells(
                                family=mapping.column_family,
                                qualifier=mapping.column_qualifier.encode("utf-8"),
                            )[0].value,
                            mapping.encoding,
                        )

            if content:
                row_id = (
                    row.row_key.decode("utf-8")
                    if isinstance(row.row_key, bytes)
                    else row.row_key
                )
                if self.collection:
                    row_id = row_id.replace(
                        f"{self.collection}:", ""
                    )  # remove collection prefix when retrieving keys
                documents.append(
                    Document(page_content=content, metadata=metadata, id=row_id)
                )
        return documents

    def _build_where_clause(
        self, filters: Optional[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Builds the WHERE clause for a BTQL query from a dictionary of filters.

        Args:
            filters (Optional[Dict[str, Any]]): A dictionary of filters to apply to the query.

        Returns:
            Tuple[str, Dict[str, Any], Dict[str, Any]]: A tuple containing the WHERE clause
            as a string, a dictionary of parameters for the query, and a dictionary of
            parameter types.
        """

        btql_where_clause = ""
        params = {}
        params_type = {}
        param_count = 0
        new_line = "\n"
        tab_space = "\t"
        line_padding = " " * 1
        row_prefix_clause = None
        if self.collection:
            # The Vector Store instance is associated with a specific collection.
            # Default to filtering rows using the collection name as a prefix.
            param_key = f"rowPrefix_{param_count}"
            collection_prefix = f"{self.collection}:"
            params[param_key] = collection_prefix.encode("utf-8")
            params_type[param_key] = Type.Bytes()
            row_prefix_clause = (
                f"{new_line}{line_padding}{tab_space}{tab_space}"
                f"(STARTS_WITH(_key, @{param_key})) {new_line}"
            )
            param_count += 1
            if filters and "collectionFilter" in filters:
                # User provided a 'collectionPrefix' filter, overriding the default.
                # Update the parameter with the user-specified prefix.
                params[param_key] = filters["collectionFilter"].encode("utf-8")
        if not self.collection and filters and "collectionFilter" in filters:
            # User provided a 'collectionPrefix' filter, but the Vector Store
            # is NOT tied to a specific collection. Apply the filter to all rows.
            param_key = f"rowPrefix_{param_count}"
            params[param_key] = filters["collectionFilter"].encode("utf-8")
            params_type[param_key] = Type.Bytes()
            row_prefix_clause = (
                f"{new_line}{line_padding}{tab_space}{tab_space}"
                f"(STARTS_WITH(_key, @{param_key})) {new_line}"
            )
            param_count += 1

        metadata_clause = ""
        if filters and "metadataFilter" in filters:
            (
                metadata_clause,
                metadata_params,
                metadata_params_type,
                param_count,
            ) = self._process_metadata_filter(
                filters["metadataFilter"], param_count, line_padding + tab_space
            )
            params.update(metadata_params)
            params_type.update(metadata_params_type)

        if row_prefix_clause and metadata_clause:
            joiner = f"{new_line}{line_padding}{tab_space}{tab_space} AND {new_line}"
            btql_where_clause = row_prefix_clause + joiner + metadata_clause
        else:
            btql_where_clause = row_prefix_clause or metadata_clause

        return btql_where_clause, params, params_type

    def _process_metadata_filter(
        self, filter_dict: Dict[str, Any], param_count: int, line_padding: str
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], int]:
        """Processes metadata-specific filters for the WHERE clause.

        Args:
            filter_dict (Dict[str, Any]): The dictionary containing metadata filters.
            param_count (int): The current count of parameters to ensure unique names.
            line_padding (str): A string for formatting the query with appropriate indentation.

        Returns:
            Tuple[str, Dict[str, Any], Dict[str, Any], int]: A tuple containing the generated
            clause string, a dictionary of parameters, a dictionary of parameter types, and
            the updated parameter count.
        """
        conditions = []
        local_params: Dict[Any, Any] = {}
        local_params_type: Dict[Any, Any] = {}
        new_line = "\n"
        tab_space = "\t"

        remaining_filters = filter_dict.copy()

        if "Qualifiers" in remaining_filters:
            param_key = f"qualifiers_{param_count}"
            local_params[param_key] = [
                q.encode("utf-8") for q in remaining_filters.pop("Qualifiers")
            ]
            local_params_type[param_key] = Type.Array(Type.Bytes())
            new_clause = (
                f"{new_line}{line_padding}{tab_space}"
                f"(ARRAY_INCLUDES_ALL(MAP_KEYS({METADATA_COLUMN_FAMILY}), @{param_key}))"
                f"{new_line}"
            )
            conditions.append(new_clause)
            param_count += 1

        if "ColumnQualifierPrefix" in remaining_filters:
            param_key = f"qualifiersPrefix_{param_count}"
            local_params[param_key] = remaining_filters.pop(
                "ColumnQualifierPrefix"
            ).encode("utf-8")
            local_params_type[param_key] = Type.Bytes()
            new_clause = (
                f"{new_line}{line_padding}{tab_space}"
                f"(ARRAY_LENGTH(ARRAY_FILTER(MAP_KEYS({METADATA_COLUMN_FAMILY}), e -> STARTS_WITH(e, @{param_key}))) > 0)"
                f"{new_line}"
            )
            conditions.append(new_clause)
            param_count += 1

        if "ColumnQualifierRegex" in remaining_filters:
            param_key = f"qualifiersRegex_{param_count}"
            local_params[param_key] = remaining_filters.pop("ColumnQualifierRegex")
            local_params_type[param_key] = Type.String()
            new_clause = (
                f"{new_line}{line_padding}{tab_space}"
                f"(ARRAY_LENGTH(ARRAY_FILTER(MAP_KEYS({METADATA_COLUMN_FAMILY}), e -> REGEXP_CONTAINS(SAFE_CONVERT_BYTES_TO_STRING(e), @{param_key}))) > 0)"
                f"{new_line}"
            )
            conditions.append(new_clause)
            param_count += 1

        if remaining_filters:
            line_padding += tab_space

            (
                value_clause,
                value_params,
                value_params_type,
                param_count,
            ) = self._process_value_filters(
                remaining_filters, "AND", param_count, line_padding
            )
            if value_clause:
                value_clause = f"{new_line}{line_padding}{value_clause}"
                conditions.append(value_clause)
                local_params.update(value_params)
                local_params_type.update(value_params_type)

        joiner = f"{new_line}{line_padding} AND {new_line}"
        joined_conditions = joiner.join(conditions)

        return joined_conditions, local_params, local_params_type, param_count

    def _process_value_filters(
        self,
        filter_dict: Dict[str, Any],
        logical_operator: str,
        param_count: int,
        line_padding: str,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], int]:
        """Processes value-based filters for the WHERE clause.

        Args:
            filter_dict (Dict[str, Any]): The dictionary of filters to process.
            logical_operator (str): The logical operator ("AND" or "OR") to join conditions.
            param_count (int): The current count of parameters for unique naming.
            line_padding (str): A string for formatting the query with indentation.

        Returns:
            Tuple[str, Dict[str, Any], Dict[str, Any], int]: A tuple containing the generated
            clause string, a dictionary of parameters, a dictionary of parameter types, and
            the updated parameter count.
        """
        conditions = []
        local_params = {}
        local_params_type = {}
        new_line = "\n"
        tab_space = "\t"

        for key, value in filter_dict.items():
            if key in [
                "rowPrefix",
                "Qualifiers",
                "ColumnQualifierPrefix",
                "ColumnQualifierRegex",
            ]:
                continue
            if key not in self._metadata_lookup and key not in [
                "QualifierUnionFilter",
                "QualifierChainFilter",
            ]:
                raise ValueError(
                    f"Unsupported filter or Metadata Column: {key}. Initialize the class with this metadatamapping to filter using this metadata."
                )

            if key == "QualifierChainFilter":
                sub_clause, sub_params, sub_params_type, param_count = (
                    self._process_value_filters(
                        value, "AND", param_count, line_padding + tab_space
                    )
                )
                conditions.append(sub_clause)
                local_params.update(sub_params)
                local_params_type.update(sub_params_type)
            elif key == "QualifierUnionFilter":
                sub_clause, sub_params, sub_params_type, param_count = (
                    self._process_value_filters(
                        value, "OR", param_count, line_padding + tab_space
                    )
                )
                conditions.append(sub_clause)
                local_params.update(sub_params)
                local_params_type.update(sub_params_type)
            elif isinstance(value, dict):
                op_map = {
                    ">": (">", "gt"),
                    "<": ("<", "lt"),
                    ">=": (">=", "gte"),
                    "<=": ("<=", "lte"),
                    "==": ("=", "eq"),
                    "!=": ("!=", "ne"),
                    "in": ("IN UNNEST", "in"),
                    "nin": ("NOT IN UNNEST", "nin"),
                    "contains": ("contains", "contains"),
                    "like": ("like", "like"),
                }
                for op, op_val in value.items():
                    if op not in op_map:
                        raise ValueError(f"Unsupported filter operator: {op}")

                    op_str, op_prefix = op_map[op]
                    param_key = f"{op_prefix}_{param_count}"

                    inner_expr = ""
                    if op in {">", "<", ">=", "<=", "==", "!="}:
                        local_params[param_key] = self._encode_value(
                            op_val, self._metadata_lookup[key][3]
                        )
                        local_params_type[param_key] = Type.Bytes()
                        inner_expr = (
                            f"{METADATA_COLUMN_FAMILY}['{key}'] {op_str} @{param_key}"
                        )
                    elif op in {"in", "nin"}:
                        local_params[param_key] = [
                            self._encode_value(v, self._metadata_lookup[key][3])
                            for v in op_val
                        ]
                        local_params_type[param_key] = Type.Array(Type.Bytes())
                        inner_expr = (
                            f"{METADATA_COLUMN_FAMILY}['{key}'] {op_str}(@{param_key})"
                        )
                    elif op == "contains":
                        local_params[param_key] = self._encode_value(
                            op_val, self._metadata_lookup[key][3]
                        )
                        local_params_type[param_key] = Type.Bytes()
                        inner_expr = f"STRPOS({METADATA_COLUMN_FAMILY}['{key}'], @{param_key}) > 0"
                    elif op == "like":
                        local_params[param_key] = str(op_val)
                        local_params_type[param_key] = Type.String()
                        inner_expr = f"REGEXP_CONTAINS(SAFE_CONVERT_BYTES_TO_STRING({METADATA_COLUMN_FAMILY}['{key}']), @{param_key})"

                    clause = (
                        f"{new_line}{line_padding}{tab_space}({new_line}{line_padding}{tab_space}{tab_space}"
                        f"{inner_expr}"
                        f"{new_line}{tab_space}{line_padding}){new_line}"
                    )
                    conditions.append(clause)
                    param_count += 1
            else:
                raise ValueError(f"Unsupported filter type: {value}")

        if not conditions:
            return "", {}, {}, param_count

        joiner = f"{new_line}{line_padding}{tab_space}{logical_operator}{line_padding}{new_line}"
        joined_conditions = joiner.join(conditions)
        full_clause = f"{new_line}{line_padding}({new_line}{joined_conditions}{new_line}{line_padding}){new_line}"

        return full_clause, local_params, local_params_type, param_count

    def _prepare_btql_query(
        self,
        query_vector: List[float],
        k: Optional[int] = 4,
        query_parameters: Optional[QueryParameters] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Prepares the full BTQL query string, parameters, and types for a vector search.

        Args:
            query_vector (List[float]): The vector to search with.
            k (Optional[int]): The number of nearest neighbors to retrieve.
            query_parameters (Optional[QueryParameters]): The parameters for the query, including distance
                                                strategy and filters.

        Returns:
            Tuple[str, Dict[str, Any], Dict[str, Any]]: A tuple containing the BTQL query string,
            a dictionary of query parameters, and a dictionary of parameter types.
        """
        if not query_parameters:
            query_parameters = QueryParameters()
        distance_metric = (
            "COSINE_DISTANCE"
            if query_parameters.distance_strategy == DistanceStrategy.COSINE
            else "EUCLIDEAN_DISTANCE"
        )
        vector_data_type = (
            "TO_VECTOR32"
            if query_parameters.vector_data_type == VectorDataType.FLOAT32
            else "TO_VECTOR64"
        )

        where_clause, params, params_type = self._build_where_clause(
            query_parameters.filters
        )

        btql_query = f"""
                SELECT
                    _key,
                    `{self.content_column.column_family}`['{self.content_column.column_qualifier}'] AS content,
                    `{self.embedding_column.column_family}`['{self.embedding_column.column_qualifier}'] AS embedding,"""

        # Select each metadata column defined in metadata_mappings to retrieve the metadata,
        # Metadata not defined in metadata_mappings won't be retrieved
        if self.metadata_mappings:
            for mapping in self.metadata_mappings:
                btql_query += f"""
                    {mapping.column_family}['{mapping.column_qualifier}'] AS {mapping.metadata_key},"""

        btql_query += f"""
                    {distance_metric}(
                        {vector_data_type}(`{self.embedding_column.column_family}`['{self.embedding_column.column_qualifier}']), 
                        {query_vector}) 
                    AS distance
                FROM 
                    `{self.async_table.table_id}`
                {"WHERE " + where_clause if where_clause else ""}
                ORDER 
                    BY distance ASC
                LIMIT 
                    {k};
        """
        return btql_query, params, params_type

    async def query_vector_store(
        self,
        query_vector: List[float],
        k: Optional[int] = 4,
        query_parameters: Optional[QueryParameters] = None,
    ) -> Any:
        """Executes a vector search query against the Bigtable instance.

        Args:
            query_vector (List[float]): The vector to search for.
            k (Optional[int]): The number of results to return.
            query_parameters (Optional[QueryParameters]): Query parameters including filters.

        Returns:
            A list of the query result rows.
        """
        # Check if the embeddings is a NumPy array
        if isinstance(query_vector, np.ndarray):
            # If so, change to list
            query_vector = query_vector.tolist()

        # Convert any np.float type objects to float
        if query_vector and isinstance(query_vector[0], np.number):
            query_vector = [float(x) for x in query_vector]

        query_parameters = query_parameters or QueryParameters()
        btql_query, params, params_type = self._prepare_btql_query(
            query_vector, k, query_parameters
        )
        try:
            results = []
            async for row in await self.client.execute_query(
                btql_query,
                instance_id=self.instance_id,
                parameters=params,
                parameter_types=params_type,
            ):
                results.append(row)
            return results
        except (ResourceExhausted, GoogleAPIError, InvalidArgument) as e:
            raise e

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query (str): The text to find similar documents for.
            k (int): The number of documents to return. Defaults to 4.
            query_parameters (Optional[QueryParameters]): Optional query parameters.
            **kwargs (Any): Additional keyword arguments, can include 'filter'.

        Returns:
            List[Document]: A list of documents most similar to the query.
        """
        embeddings = await self.embedding_service.aembed_query(query)
        return await self.asimilarity_search_by_vector(
            embeddings, k=k, query_parameters=query_parameters, **kwargs
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding (List[float]): The embedding vector to search with.
            k (int): The number of documents to return. Defaults to 4.
            query_parameters (Optional[QueryParameters]): Optional query parameters.
            **kwargs (Any): Additional keyword arguments, can include 'filter'.

        Returns:
            List[Document]: A list of documents most similar to the embedding.
        """
        docs_with_scores = await self.asimilarity_search_with_score_by_vector(
            embedding, k=k, query_parameters=query_parameters, **kwargs
        )
        return [doc for doc, _ in docs_with_scores]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance.

        Args:
            query (str): The text to find similar documents for.
            k (int): The number of documents to return. Defaults to 4.
            query_parameters (Optional[QueryParameters]): Optional query parameters.
            **kwargs (Any): Additional keyword arguments, can include 'filter'.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing a
            document and its distance score.
        """
        embeddings = await self.embedding_service.aembed_query(query)

        return await self.asimilarity_search_with_score_by_vector(
            embeddings, k=k, query_parameters=query_parameters, **kwargs
        )

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance by vector.

        Args:
            embedding (List[float]): The embedding vector to search with.
            k (int): The number of documents to return. Defaults to 4.
            query_parameters (Optional[QueryParameters]): Optional query parameters.
            **kwargs (Any): Additional keyword arguments, can include 'filter'.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing a
            document and its distance score.
        """
        final_query_params = self._get_query_params(query_parameters, **kwargs)
        results = await self.query_vector_store(
            embedding, k, query_parameters=final_query_params
        )
        docs_with_scores = []
        for res in results:
            res_selected_cols = []

            for col in res.fields:
                res_selected_cols.append(col[0])
            metadata = {}
            for mapping in self.metadata_mappings:
                if mapping.metadata_key in res_selected_cols:
                    metadata[mapping.metadata_key] = self._decode_value(
                        res[mapping.metadata_key], mapping.encoding
                    )

            doc_id = res["_key"]
            doc_id = doc_id.decode("utf-8") if isinstance(doc_id, bytes) else doc_id
            if self.collection:
                doc_id = doc_id.replace(f"{self.collection}:", "")

            doc = Document(
                id=doc_id,
                page_content=self._decode_value(
                    res["content"], self.content_column.encoding
                ),
                metadata=metadata,
            )
            docs_with_scores.append((doc, res["distance"]))
        return docs_with_scores

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        Args:
            query (str): The text to find similar documents for.
            k (int): The number of documents to return. Defaults to 4.
            query_parameters (Optional[QueryParameters]): Optional query parameters.
            **kwargs (Any): Additional keyword arguments, can include 'filter'.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing a
            document and its relevance score.
        """
        docs_with_dist = await self.asimilarity_search_with_score(
            query, k=k, query_parameters=query_parameters, **kwargs
        )
        final_query_params = self._get_query_params(query_parameters, **kwargs)

        if final_query_params.distance_strategy == DistanceStrategy.COSINE:
            relevance_score_fn = self._cosine_relevance_score_fn
        else:  # If distance to use not declared as COSINE, uses EUCLIDEAN by default
            relevance_score_fn = self._euclidean_relevance_score_fn
        return [(doc, relevance_score_fn(dist)) for doc, dist in docs_with_dist]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Args:
            query (str): The text to find relevant and diverse documents for.
            k (int): Number of documents to return. Defaults to 4.
            fetch_k (int): Number of documents to fetch to pass to MMR algorithm.
                           Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                                 of diversity returned. Defaults to 0.5.
            query_parameters (Optional[QueryParameters]): Optional query parameters.
            **kwargs (Any): Additional keyword arguments, can include 'filter'.

        Returns:
            List[Document]: A list of documents selected by MMR.
        """
        embeddings = await self.embedding_service.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embeddings,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            query_parameters=query_parameters,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Args:
            embedding (list[float]): The embedding vector for the query.
            k (int): Number of documents to return. Defaults to 4.
            fetch_k (int): Number of documents to fetch for MMR. Defaults to 20.
            lambda_mult (float): Diversity factor. Defaults to 0.5.
            query_parameters (Optional[QueryParameters]): Optional query parameters.
            **kwargs (Any): Additional keyword arguments, can include 'filter'.

        Returns:
            List[Document]: A list of documents selected by MMR.
        """
        query_vector = embedding
        documents = [
            doc
            for doc, score in await self.amax_marginal_relevance_search_with_score_by_vector(
                query_vector,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                query_parameters=query_parameters,
                **kwargs,
            )
        ]
        return documents

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_parameters: Optional[QueryParameters] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and scores selected using the maximal marginal relevance.

        Args:
            embedding (list[float]): The embedding vector for the query.
            k (Optional[int]): Number of documents to return. Defaults to 4.
            fetch_k (Optional[int]): Number of documents to fetch for MMR. Defaults to 20.
            lambda_mult (float): Diversity factor. Defaults to 0.5.
            query_parameters (Optional[QueryParameters]): Optional query parameters.
            **kwargs (Any): Additional keyword arguments, can include 'filter'.

        Returns:
            list[tuple[Document, float]]: A list of tuples containing a document
            and its distance score.
        """
        query_parameters = self._get_query_params(query_parameters, **kwargs)
        fetch_k_results = await self.query_vector_store(
            embedding,
            k=fetch_k,
            query_parameters=query_parameters,
        )
        embedding_list = []

        for res in fetch_k_results:
            # Convert bytes to embeddings list
            embeddings = self._bytes_to_embeddings(
                res["embedding"], self.embedding_column.encoding
            )
            embedding_list.append(embeddings)

        mmr_selected_indices = utils.maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        results = []
        for i, result in enumerate(fetch_k_results):
            # Take only the rows selected by MMR
            if i in mmr_selected_indices:
                distance = result["distance"]

                metadata = {}
                res_selected_cols = []

                for col in result.fields:
                    res_selected_cols.append(col[0])
                for mapping in self.metadata_mappings:
                    if mapping.metadata_key in res_selected_cols:
                        metadata[mapping.metadata_key] = self._decode_value(
                            result[mapping.metadata_key], mapping.encoding
                        )

                doc_id = result["_key"]
                doc_id = doc_id.decode("utf-8") if isinstance(doc_id, bytes) else doc_id
                if self.collection:
                    doc_id = doc_id.replace(f"{self.collection}:", "")
                document = Document(
                    id=doc_id,
                    page_content=self._decode_value(
                        result["content"], self.content_column.encoding
                    ),
                    metadata=metadata,
                )
                results.append((document, distance))
        return results

    def add_texts(self, *args, **kwargs):
        """Synchronous 'add_texts' is not supported. Use 'aadd_texts' instead."""
        raise NotImplementedError(
            "This is the async vector store, use aadd_texts instead."
        )

    def add_documents(self, *args, **kwargs):
        """Synchronous 'add_documents' is not supported. Use 'aadd_documents' instead."""
        raise NotImplementedError(
            "This is the async vector store, use aadd_documents instead."
        )

    def delete(self, *args, **kwargs):
        """Synchronous 'delete' is not supported. Use 'adelete' instead."""
        raise NotImplementedError(
            "This is the async vector store, use adelete instead."
        )

    def search(self, *args, **kwargs):
        """Synchronous 'search' is not supported. Use 'asearch' instead."""
        raise NotImplementedError(
            "This is the async vector store, use asearch instead."
        )

    def similarity_search(self, *args, **kwargs):
        """Synchronous 'similarity_search' is not supported. Use 'asimilarity_search' instead."""
        raise NotImplementedError(
            "This is the async vector store, use asimilarity_search instead."
        )

    def similarity_search_by_vector(self, *args, **kwargs):
        """Synchronous 'similarity_search_by_vector' is not supported. Use 'asimilarity_search_by_vector' instead."""
        raise NotImplementedError(
            "This is the async vector store, use asimilarity_search_by_vector instead."
        )

    def similarity_search_with_score(self, *args, **kwargs):
        """Synchronous 'similarity_search_with_score' is not supported. Use 'asimilarity_search_with_score' instead."""
        raise NotImplementedError(
            "This is the async vector store, use asimilarity_search_with_score instead."
        )

    def max_marginal_relevance_search(self, *args, **kwargs):
        """Synchronous 'max_marginal_relevance_search' is not supported. Use 'amax_marginal_relevance_search' instead."""
        raise NotImplementedError(
            "This is the async vector store, use amax_marginal_relevance_search instead."
        )

    @classmethod
    def from_documents(cls, *args, **kwargs):
        """Synchronous 'from_documents' is not supported. Use 'afrom_documents' instead."""
        raise NotImplementedError(
            "This is the async vector store, use afrom_documents instead."
        )

    @classmethod
    def from_texts(cls, *args, **kwargs):
        """Synchronous 'from_texts' is not supported. Use 'afrom_texts' instead."""
        raise NotImplementedError(
            "This is the async vector store, use afrom_texts instead."
        )

    def max_marginal_relevance_search_by_vector(self, *args, **kwargs):
        """Synchronous 'max_marginal_relevance_search_by_vector' is not supported. Use 'amax_marginal_relevance_search_by_vector' instead."""
        raise NotImplementedError(
            "This is the async vector store, use amax_marginal_relevance_search_by_vector instead."
        )

    def similarity_search_with_relevance_scores(self, *args, **kwargs):
        """Synchronous 'similarity_search_with_relevance_scores' is not supported. Use 'asimilarity_search_with_relevance_scores' instead."""
        raise NotImplementedError(
            "This is the async vector store, use asimilarity_search_with_relevance_scores instead."
        )
