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
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from google.cloud import bigtable
from google.cloud.bigtable.data import (
    DeleteAllFromRow,
    ReadRowsQuery,
    RowMutationEntry,
    RowRange,
    TableAsync,
)
from langchain_core.stores import BaseStore


class AsyncBigtableByteStore(BaseStore[str, bytes]):
    """
    Async-only LangChain ByteStore implementation for Bigtable.

    This class provides an asynchronous methods for storing and retrieving
    byte values in a Bigtable table. It uses row keys for keys and stores
    values in a specified column family and qualifier.

    Attributes:
        table: The `TableAsync` instance used for Bigtable operations.
        value_column_family: The column family where values are stored.
        value_column_qualifier: The column qualifier where values are stored.
    """

    DEFAULT_COLUMN_FAMILY = "kv"
    DEFAULT_COLUMN_QUALIFIER = "val".encode("utf-8")

    def __init__(
        self,
        async_table: TableAsync,
        column_family: str = DEFAULT_COLUMN_FAMILY,
        column_qualifier: bytes = DEFAULT_COLUMN_QUALIFIER,
    ):
        """
        Initializes a new AsyncBigtableByteStore.

        Args:
            async_table: The `TableAsync` instance to use for Bigtable operations.
            column_family: The column family to store values in.
            column_qualifier: The column qualifier to store values in.
        """
        self._table = async_table
        self._column_family = column_family
        self._column_qualifier = column_qualifier

    @property
    def table(self):
        """Returns the underlying Bigtable table instance."""
        return self._table

    @property
    def value_column_family(self):
        """Returns the column family used to store values."""
        return self._column_family

    @property
    def value_column_qualifier(self):
        """Returns the column qualifier used to store values."""
        return self._column_qualifier

    async def amget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """
        Asynchronously retrieves values for a sequence of keys.

        It only reads the most recent version for each key.

        Args:
            keys: A sequence of keys to retrieve values for.

        Returns:
            A list of byte values corresponding to the input keys. If a key is not
            found, `None` is returned for that key's position in the list.
        """
        row_keys = [key.encode() for key in keys]
        results = {}
        row_filter = bigtable.data.row_filters.CellsColumnLimitFilter(1)

        query = bigtable.data.ReadRowsQuery(
            row_keys=cast(List[Union[str, bytes]], row_keys), row_filter=row_filter
        )

        # It only reads the most recent version for each row
        rows_read = await self.table.read_rows(query)
        for row in rows_read:
            cell = row.get_cells(
                family=self.value_column_family, qualifier=self.value_column_qualifier
            )[0]
            if cell:
                results[row.row_key] = cell.value

        res = []
        for key in keys:
            if key.encode() in results:
                res.append(results[key.encode()])
            else:
                res.append(None)

        return res

    async def amset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """
        Asynchronously stores key-value pairs in the Bigtable.

        Args:
            key_value_pairs: A sequence of (key, value) tuples to store.

        Raises:
            TypeError: If any key is not a string or any value is not bytes.
        """
        mutations = []
        for key, value in key_value_pairs:
            if not isinstance(key, str):
                raise TypeError("Keys must be of type 'str'.")
            if not isinstance(value, bytes):
                raise TypeError("Values must be of type 'bytes'.")

            mutation = bigtable.data.SetCell(
                family=self.value_column_family,
                qualifier=self.value_column_qualifier,
                new_value=value,
            )

            row_mutation = bigtable.data.RowMutationEntry(
                row_key=key, mutations=[mutation]
            )
            mutations.append(row_mutation)

        await self.table.bulk_mutate_rows(mutations)

    async def amdelete(self, keys: Sequence[str]) -> None:
        """
        Asynchronously deletes key-value pairs from the Bigtable.

        Args:
            keys: A sequence of keys to delete.
        """
        mutations = []
        for key in keys:
            mutation = DeleteAllFromRow()
            row_mutation = RowMutationEntry(key, [mutation])
            mutations.append(row_mutation)

        if mutations:
            await self.table.bulk_mutate_rows(mutations)

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        """
        Asynchronously yields keys matching a given prefix.

        It only yields the row keys that match the given prefix.

        Args:
           prefix: An optional prefix to filter keys by. If `None` or an empty
             string, all keys are yielded.

        Yields:
            Keys from the table that match a given prefix.
        """
        row_filter = bigtable.data.row_filters.StripValueTransformerFilter(True)
        if not prefix or prefix == "":
            # Return all keys
            query = ReadRowsQuery(row_filter=row_filter)
            async for row in await self.table.read_rows_stream(query):
                yield row.row_key.decode("utf-8")

        else:
            # Return keys matching the prefix
            end_key = prefix[:-1] + chr(ord(prefix[-1]) + 1)
            prefix_range = RowRange(start_key=prefix, end_key=end_key)
            query = ReadRowsQuery(row_ranges=[prefix_range], row_filter=row_filter)

            async for row in await self._table.read_rows_stream(query):
                yield row.row_key.decode("utf-8")

    # Sync methods are not implemented in the Async-only version
    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        raise NotImplementedError("Use amget for async operations.")

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        raise NotImplementedError("Use amset for async operations.")

    def mdelete(self, keys: Sequence[str]) -> None:
        raise NotImplementedError("Use amdelete for async operations.")

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        raise NotImplementedError("Use ayield_keys for async operations.")
