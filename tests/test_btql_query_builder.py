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

import struct
from unittest.mock import Mock

import pytest

from langchain_google_bigtable.async_vector_store import (
    AsyncBigtableVectorStore,
    ColumnConfig,
    Encoding,
    MetadataMapping,
    QueryParameters,
)

MOCK_CLIENT = Mock()
MOCK_INSTANCE_ID = "test-instance"
MOCK_TABLE = Mock()
MOCK_TABLE.table_id = "test-table"
MOCK_EMBEDDING_SERVICE = Mock()


class TestPrepareBtqlQuery:
    """
    Unit tests for the BTQL query builder, using string matching
    removing newlines, and tabs.
    """

    @pytest.fixture
    def store(self) -> AsyncBigtableVectorStore:
        return AsyncBigtableVectorStore(
            client=MOCK_CLIENT,
            instance_id=MOCK_INSTANCE_ID,
            async_table=MOCK_TABLE,
            embedding_service=MOCK_EMBEDDING_SERVICE,
            content_column=ColumnConfig("cf", "content"),
            embedding_column=ColumnConfig("cf", "embedding"),
            metadata_mappings=[
                MetadataMapping("color", Encoding.UTF8),
                MetadataMapping("is_good", Encoding.BOOL),
                MetadataMapping("number", Encoding.INT_BIG_ENDIAN),
                MetadataMapping("rating", Encoding.FLOAT),
            ],
        )

    def test_query_with_collection_only(self, store: AsyncBigtableVectorStore) -> None:
        """Tests a query with only a collection prefix filter."""
        store.collection = "my-collection"
        params = QueryParameters(filters={})

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE  (STARTS_WITH(_key, @rowPrefix_0))"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["rowPrefix_0"] == b"my-collection:"

    def test_filter_equal(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the '==' (equal) operator and verifies UTF8 encoding."""
        params = QueryParameters(filters={"metadataFilter": {"color": {"==": "red"}}})

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( md['color'] = @eq_0 ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["eq_0"] == b"red"

    def test_filter_not_equal(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the '!=' (not equal) operator and INT_BIG_ENDIAN encoding."""
        params = QueryParameters(filters={"metadataFilter": {"number": {"!=": 50}}})

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( md['number'] != @ne_0 ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["ne_0"] == struct.pack(">q", 50)

    def test_filter_greater_than(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the '>' (greater than) operator and INT_BIG_ENDIAN encoding."""
        params = QueryParameters(filters={"metadataFilter": {"number": {">": 100}}})

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( md['number'] > @gt_0 ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["gt_0"] == struct.pack(">q", 100)

    def test_filter_greater_than_or_equal(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the '>=' (greater than or equal) operator and FLOAT encoding."""
        params = QueryParameters(filters={"metadataFilter": {"rating": {">=": 3.5}}})

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( md['rating'] >= @gte_0 ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["gte_0"] == struct.pack(">f", 3.5)

    def test_filter_less_than(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the '<' (less than) operator and INT_BIG_ENDIAN encoding."""
        params = QueryParameters(filters={"metadataFilter": {"number": {"<": 10}}})

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( md['number'] < @lt_0 ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["lt_0"] == struct.pack(">q", 10)

    def test_filter_less_than_or_equal(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the '<=' (less than or equal) operator and FLOAT encoding."""
        params = QueryParameters(filters={"metadataFilter": {"rating": {"<=": 4.9}}})

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( md['rating'] <= @lte_0 ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["lte_0"] == struct.pack(">f", 4.9)

    def test_filter_in(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the 'in' operator and verifies correct encoding for multiple values."""
        params = QueryParameters(
            filters={"metadataFilter": {"color": {"in": ["red", "blue"]}}}
        )

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( md['color'] IN UNNEST(@in_0) ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["in_0"] == [b"red", b"blue"]

    def test_filter_not_in(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the 'nin' (not in) operator and verifies correct encoding for multiple int values."""
        params = QueryParameters(
            filters={"metadataFilter": {"number": {"nin": [1, 2, 3]}}}
        )

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( md['number'] NOT IN UNNEST(@nin_0) ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["nin_0"] == [struct.pack(">q", i) for i in [1, 2, 3]]

    def test_filter_contains(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the 'contains' string operator."""
        params = QueryParameters(
            filters={"metadataFilter": {"color": {"contains": "gree"}}}
        )

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( STRPOS(md['color'], @contains_0) > 0 ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["contains_0"] == b"gree"

    def test_filter_like(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the 'like' (regex) operator."""
        params = QueryParameters(
            filters={"metadataFilter": {"color": {"like": "bl.e"}}}
        )

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( REGEXP_CONTAINS(SAFE_CONVERT_BYTES_TO_STRING(md['color']), @like_0) ) ) "

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["like_0"] == "bl.e"

    def test_filter_qualifiers_exist(self, store: AsyncBigtableVectorStore) -> None:
        """Tests the 'Qualifiers' filter to check for column existence."""
        params = QueryParameters(
            filters={"metadataFilter": {"Qualifiers": ["color", "rating"]}}
        )

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = (
            "WHERE  (ARRAY_INCLUDES_ALL(MAP_KEYS(md), @qualifiers_0))"
        )

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["qualifiers_0"] == [b"color", b"rating"]

    def test_query_with_collection_and_metadata_filter(
        self, store: AsyncBigtableVectorStore
    ) -> None:
        """Tests combining a collection prefix with a metadata filter and verifies parameter values."""
        store.collection = "my-docs"
        params = QueryParameters(filters={"metadataFilter": {"number": {">=": 42}}})

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE  (STARTS_WITH(_key, @rowPrefix_0))   AND   ( ( md['number'] >= @gte_1 ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["rowPrefix_0"] == b"my-docs:"
        assert query_params["gte_1"] == struct.pack(">q", 42)

    def test_filter_chain(self, store: AsyncBigtableVectorStore) -> None:
        """Tests a simple 'QualifierChainFilter' (AND) and verifies parameter values."""
        params = QueryParameters(
            filters={
                "metadataFilter": {
                    "QualifierChainFilter": {
                        "color": {"==": "blue"},
                        "is_good": {"==": True},
                    }
                }
            }
        )
        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = (
            "WHERE   ( ( ( md['color'] = @eq_0 ) AND  ( md['is_good'] = @eq_1 ) ) )"
        )

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["eq_0"] == b"blue"
        assert query_params["eq_1"] == struct.pack("?", True)

    def test_filter_union(self, store: AsyncBigtableVectorStore) -> None:
        """Tests a simple 'QualifierUnionFilter' (OR) and verifies parameter values."""
        params = QueryParameters(
            filters={
                "metadataFilter": {
                    "QualifierUnionFilter": {"number": {"<": 10}, "rating": {">": 4.9}}
                }
            }
        )
        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces, new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = (
            "WHERE   ( ( ( md['number'] < @lt_0 ) OR  ( md['rating'] > @gt_1 ) ) )"
        )

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["lt_0"] == struct.pack(">q", 10)
        assert query_params["gt_1"] == struct.pack(">f", 4.9)

    def test_filter_chain_in_union(self, store: AsyncBigtableVectorStore) -> None:
        """Tests a complex nested filter: (A AND B) OR C."""
        params = QueryParameters(
            filters={
                "metadataFilter": {
                    "QualifierUnionFilter": {
                        "QualifierChainFilter": {
                            "color": {"==": "red"},
                            "is_good": {"==": True},
                        },
                        "number": {">": 100},
                    }
                }
            }
        )
        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces and new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( ( ( md['color'] = @eq_0 ) AND  ( md['is_good'] = @eq_1 ) ) OR  ( md['number'] > @gt_2 ) ) ) "

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["eq_0"] == b"red"
        assert query_params["eq_1"] == struct.pack("?", True)
        assert query_params["gt_2"] == struct.pack(">q", 100)

    def test_filter_union_in_chain(self, store: AsyncBigtableVectorStore) -> None:
        """Tests a complex nested filter: A AND (B OR C)."""
        params = QueryParameters(
            filters={
                "metadataFilter": {
                    "QualifierChainFilter": {
                        "color": {"==": "green"},
                        "QualifierUnionFilter": {
                            "number": {"<": 5},
                            "is_good": {"==": False},
                        },
                    }
                }
            }
        )

        # Prepare Query
        btql, query_params, _ = store._prepare_btql_query([0.1], 5, params)

        # Remove tab spaces and new lines
        btql_removed = btql.replace("\n", "").replace("\t", "")

        expected_where_clause = "WHERE   ( ( ( md['color'] = @eq_0 ) AND  ( ( md['number'] < @lt_1 ) OR  ( md['is_good'] = @eq_2 ) ) ) )"

        assert expected_where_clause in btql_removed

        # Assert the placeholder values are correctly encoded
        assert query_params["eq_0"] == b"green"
        assert query_params["lt_1"] == struct.pack(">q", 5)
        assert query_params["eq_2"] == struct.pack("?", False)
