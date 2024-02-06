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
from __future__ import annotations

import struct
import uuid
from enum import Enum
from typing import Any, Callable, Iterator, List, Optional

from google.cloud import bigtable
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

COLUMN_FAMILY = "langchain"
CONTENT_COLUMN_NAME = "content"
ID_METADATA_KEY = "rowkey"


class Encoding(Enum):
    UTF8 = "utf-8"
    UTF16 = "utf-16"
    ASCII = "ascii"
    INT_LITTLE_ENDIAN = "int_little_endian"
    INT_BIG_ENDIAN = "int_big_endian"
    DOUBLE = "double"
    FLOAT = "float"
    BOOL = "bool"
    CUSTOM = "custom"


def _not_implemented(_: Any) -> Any:
    raise NotImplementedError(
        "decoding/encoding function not set for custom encoded metadata key"
    )


class MetadataMapping:
    def __init__(
        self,
        column_family: str,
        column_name: str,
        metadata_key: str,
        encoding: Encoding,
        custom_encoding_func: Callable[[Any], bytes] = _not_implemented,
        custom_decoding_func: Callable[[bytes], Any] = _not_implemented,
    ):
        self.column_family = column_family
        self.column_name = column_name
        self.metadata_key = metadata_key
        self.encoding = encoding
        self.custom_encoding_func = custom_encoding_func
        self.custom_decoding_func = custom_decoding_func


SUPPORTED_DOC_ENCODING = (Encoding.UTF8, Encoding.UTF16, Encoding.ASCII)

default_client: Optional[bigtable.Client] = None


class BigtableLoader(BaseLoader):
    """Load from the Google Cloud Platform `Bigtable`."""

    def __init__(
        self,
        instance_id: str,
        table_id: str,
        row_set: Optional[bigtable.row_set.RowSet] = None,
        filter: Optional[bigtable.row_filters.RowFilter] = None,
        client: Optional[bigtable.Client] = None,
        content_encoding: Encoding = Encoding.UTF8,
        content_column_family: str = COLUMN_FAMILY,
        content_column_name: str = CONTENT_COLUMN_NAME,
        metadata_mappings: List[MetadataMapping] = [],
    ):
        """Initialize Bigtable document loader.

        Args:
            instance_id: The Bigtable instance to load data from.
            table_id: The Bigtable table to load data from.
            row_set: Optional. The row set to read data from.
            filter: Optional. The filter to apply to the query to load data from bigtable.
            client : Optional. The pre-created client to query bigtable.
            content_encoding: Optional. The encoding in which to load the page content. Defaults to UTF-8. Allowed values are  UTF8, UTF16 and ASCII.
            content_column_family: Optional. The column family in which the content is stored. Defaults to "langchain".
            content_column_name: Optional. The column in which the content is stored. Defaults to "content".
            metadata_mappings: Optional. The array of mappings that maps from Bigtable columns to keys on the metadata dictionary, including the encoding to use when mapping from Bigtable bytes to a python type.
        """
        self.row_set = row_set
        self.filter = filter
        self.client = (
            (client or self.__get_default_client())
            .instance(instance_id)
            .table(table_id)
        )
        if content_encoding not in SUPPORTED_DOC_ENCODING:
            raise ValueError(f"{content_encoding} not in {SUPPORTED_DOC_ENCODING}")
        families = self.client.list_column_families()
        for mapping in metadata_mappings:
            if mapping.column_family not in families:
                raise ValueError(
                    f"column family '{mapping.column_family}' doesn't exist in table. Existing column families are {families.keys()}"
                )
        self.content_encoding = content_encoding
        self.content_column_family = content_column_family
        self.content_column_name = content_column_name
        self.metadata_mappings = metadata_mappings

    def __get_default_client(self) -> bigtable.Client:
        global default_client
        if default_client is None:
            default_client = bigtable.Client(admin=True)
        return default_client

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """A lazy loader for Documents."""
        rows = self.client.read_rows(row_set=self.row_set, filter_=self.filter)
        for row in rows:
            metadata = {ID_METADATA_KEY: row.row_key.decode()}
            for mapping in self.metadata_mappings:
                if (
                    mapping.column_family in row.cells
                    and mapping.column_name.encode() in row.cells[mapping.column_family]
                ):
                    metadata[mapping.metadata_key] = self._decode(
                        row.cells[mapping.column_family][mapping.column_name.encode()][
                            0
                        ].value,
                        mapping,
                    )

            content = ""
            if (
                self.content_column_name.encode()
                in row.cells[self.content_column_family]
            ):
                content = row.cells[self.content_column_family][
                    self.content_column_name.encode()
                ][0].value.decode(self.content_encoding.value)

            yield Document(
                page_content=content,
                metadata=metadata,
            )

    def _decode(self, value: bytes, mapping: MetadataMapping) -> Any:
        match mapping.encoding:
            case Encoding.UTF8:
                return value.decode(mapping.encoding.value)
            case Encoding.UTF16:
                return value.decode(mapping.encoding.value)
            case Encoding.ASCII:
                return value.decode(mapping.encoding.value)
            case Encoding.INT_LITTLE_ENDIAN:
                return int.from_bytes(value, "little")
            case Encoding.INT_BIG_ENDIAN:
                return int.from_bytes(value, "big")
            case Encoding.DOUBLE:
                return struct.unpack("d", value)[0]
            case Encoding.FLOAT:
                return struct.unpack("f", value)[0]
            case Encoding.BOOL:
                return bool.from_bytes(value)
            case Encoding.CUSTOM:
                return mapping.custom_decoding_func(value)
            case _:
                raise ValueError(f"Invalid encoding {mapping.encoding}")


class BigtableSaver:
    """Load from the Google Cloud Platform `Bigtable`."""

    def __init__(
        self,
        instance_id: str,
        table_id: str,
        client: Optional[bigtable.Client] = None,
        content_encoding: Encoding = Encoding.UTF8,
        content_column_family: str = COLUMN_FAMILY,
        content_column_name: str = CONTENT_COLUMN_NAME,
        metadata_mappings: List[MetadataMapping] = [],
    ):
        """Initialize Bigtable document saver.

        Args:
            instance: The Bigtable instance to load data from.
            table: The Bigtable table to load data from.
            client : Optional. The pre-created client to query bigtable.
            content_encoding: Optional. The encoding in which to load the page content. Defaults to UTF-8. Allowed values are  UTF8, UTF16 and ASCII.
            content_column_family: Optional. The column family in which the content is stored. Defaults to "langchain".
            content_column_name: Optional. The column in which the content is stored. Defaults to "content".
            metadata_mappings: Optional. The array of mappings that maps from Bigtable columns to keys on the metadata dictionary, including the encoding to use when mapping from Bigtable bytes to a python type.
        """
        self.client = (
            (client or bigtable.Client(admin=True))
            .instance(instance_id)
            .table(table_id)
        )
        if content_encoding not in SUPPORTED_DOC_ENCODING:
            raise ValueError(f"{content_encoding} not in {SUPPORTED_DOC_ENCODING}")
        families = self.client.list_column_families()
        for mapping in metadata_mappings:
            if mapping.column_family not in families:
                raise ValueError(
                    f"column family '{mapping.column_family}' doesn't exist in table. Existing column families are {families.keys()}"
                )
        self.content_encoding = content_encoding
        self.content_column_family = content_column_family
        self.content_column_name = content_column_name
        self.metadata_mappings = metadata_mappings

    def add_documents(self, docs: List[Document]):
        batcher = self.client.mutations_batcher()
        for doc in docs:
            row_key = doc.metadata.get(ID_METADATA_KEY) or uuid.uuid4().hex
            row = self.client.direct_row(row_key)
            row.set_cell(
                self.content_column_family,
                self.content_column_name,
                doc.page_content.encode(self.content_encoding.value),
            )
            for mapping in self.metadata_mappings:
                if mapping.metadata_key in doc.metadata:
                    value = self._encode(doc.metadata[mapping.metadata_key], mapping)
                    row.set_cell(mapping.column_family, mapping.column_name, value)
            batcher.mutate(row)
        batcher.flush()

    def delete(self, docs: List[Document]):
        batcher = self.client.mutations_batcher()
        for doc in docs:
            row = self.client.direct_row(doc.metadata.get(ID_METADATA_KEY))
            row.delete()
            batcher.mutate(row)
        batcher.flush()

    def _encode(self, value: Any, mapping: MetadataMapping) -> bytes:
        match mapping.encoding:
            case Encoding.UTF8:
                return value.encode(mapping.encoding.value)
            case Encoding.UTF16:
                return value.encode(mapping.encoding.value)
            case Encoding.ASCII:
                return value.encode(mapping.encoding.value)
            case Encoding.INT_LITTLE_ENDIAN:
                return int.to_bytes(value, byteorder="little")
            case Encoding.INT_BIG_ENDIAN:
                return int.to_bytes(value, byteorder="big")
            case Encoding.DOUBLE:
                return struct.pack("d", value)
            case Encoding.FLOAT:
                return struct.pack("f", value)
            case Encoding.BOOL:
                return bool.to_bytes(value)
            case Encoding.CUSTOM:
                return mapping.custom_encoding_func(value)
            case _:
                raise ValueError(f"Invalid encoding {mapping.encoding}")
