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
from typing import List, Optional
import uuid

from google.cloud import bigtable
from google.cloud.bigtable.row_filters import RowKeyRegexFilter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

COLUMN_FAMILY = "langchain"
COLUMN_NAME = "history"


class BigtableChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Bigtable.

    Args:
        instance_id: The Bigtable instance to use for chat message history.
        table_id: The Bigtable table to use for chat message history.
        session_id: Optional. The existing session ID.
    """

    def __init__(
        self,
        instance_id: str,
        table_id: str,
        session_id: Optional[str] = None,
        client: Optional[bigtable.Client] = None,
    ) -> None:
        self.client = (
            (client or bigtable.Client(admin=True))
            .instance(instance_id)
            .table(table_id)
        )

        self.session_id = session_id or uuid.uuid4().hex

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all session messages from DB"""
        rows = self.client.read_rows(
            filter_=RowKeyRegexFilter(
                str.encode("^" + re.escape(self.session_id) + "#.*")
            )
        )
        items = [
            json.loads(row.cells[COLUMN_FAMILY][COLUMN_NAME.encode()][0].value.decode())
            for row in rows
        ]
        messages = messages_from_dict(items)
        return messages

    def init_schema(self):
        families = self.client.list_column_families()
        if COLUMN_FAMILY not in families:
            self.client.create(
                column_families={
                    COLUMN_FAMILY: bigtable.column_family.MaxVersionsGCRule(1)
                }
            )

    def add_message(self, message: BaseMessage) -> None:
        """Write a message to the table"""

        row_key = str.encode(
            self.session_id
            + "#"
            + str(time.time()).rjust(25, "0")
            + "#"
            + uuid.uuid4().hex
        )
        row = self.client.direct_row(row_key)
        value = str.encode(json.dumps(message_to_dict(message)))
        row.set_cell(COLUMN_FAMILY, COLUMN_NAME, value)
        row.commit()

    def clear(self) -> None:
        """Clear session memory from DB"""
        row_key_prefix = self.session_id
        self.client.drop_by_prefix(row_key_prefix, timeout=200)
