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

"""Bigtable-based chat message history"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import List, Optional

from google.cloud import bigtable  # type: ignore
from google.cloud.bigtable.row_filters import RowKeyRegexFilter  # type: ignore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict

from .common import use_client_or_default

COLUMN_FAMILY = "langchain"
COLUMN_NAME = "history"


def create_chat_history_table(
    instance_id: str,
    table_id: str,
    client: Optional[bigtable.Client] = None,
) -> None:
    table_client = (
        use_client_or_default(client, "chat_history")
        .instance(instance_id)
        .table(table_id)
    )
    if not table_client.exists():
        table_client.create()

    families = table_client.list_column_families()
    if COLUMN_FAMILY not in families:
        table_client.column_family(
            COLUMN_FAMILY, gc_rule=bigtable.column_family.MaxVersionsGCRule(1)
        ).create()


class BigtableChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Bigtable.

    Args:
        instance_id: The Bigtable instance to use for chat message history.
        table_id: The Bigtable table to use for chat message history.
        session_id: The session ID.
        client : Optional. The pre-created client to query bigtable.
    """

    def __init__(
        self,
        instance_id: str,
        table_id: str,
        session_id: str,
        client: Optional[bigtable.Client] = None,
    ) -> None:
        instance = use_client_or_default(client, "chat_history").instance(instance_id)
        if not instance.exists():
            raise NameError(f"Instance {instance_id} does not exist")

        self.table_client = instance.table(table_id)
        if not self.table_client.exists():
            raise NameError(
                f"Table {table_id} does not exist on instance {instance_id}"
            )
        if COLUMN_FAMILY not in self.table_client.list_column_families():
            raise NameError(
                f"Column family {COLUMN_FAMILY} does not exist on table {table_id}"
            )

        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all session messages from DB"""
        rows = self.table_client.read_rows(
            filter_=RowKeyRegexFilter(
                str.encode("^" + re.escape(self.session_id) + "#.*")
            )
        )
        items = [
            json.loads(row.cells[COLUMN_FAMILY][COLUMN_NAME.encode()][0].value.decode())
            for row in rows
        ]
        messages = messages_from_dict(
            [{"type": item["type"], "data": item} for item in items]
        )
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Write a message to the table"""

        row_key = str.encode(
            self.session_id
            + "#"
            + str(time.time_ns()).rjust(25, "0")
            + "#"
            + uuid.uuid4().hex
        )
        row = self.table_client.direct_row(row_key)
        value = str.encode(message.json())
        row.set_cell(COLUMN_FAMILY, COLUMN_NAME, value)
        row.commit()

    def clear(self) -> None:
        """Clear session memory from DB"""
        row_key_prefix = self.session_id
        self.table_client.drop_by_prefix(row_key_prefix)
