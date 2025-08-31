from google.cloud.bigtable import Client
from google.cloud.bigtable.instance import Instance
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Type

# The number of rows to scan to find possible column qualifiers.
DEFAULT_COL_QUALIFIER_SCAN_LIMIT = 1  



class BigtableGetInstancesAndTableSchemaArgs(BaseModel):
    """Input for GetTableTool."""
    pass


class BigtableGetInstancesAndTableSchemaTool(BaseTool):
    """
    A tool to interact with Google Bigtable and retrieve metadata and data of a project.
    """

    name: str = "GetBigtableInstancesAndTableSchema"
    description: str = (
        "Gets the schema of all Bigtable resources in a project: all instances, their tables, "
        "and for each table the column families and their column qualifiers. "
        "Bigtable uses a two-tier column model: Column Families -> Column Qualifiers "
        "Since Bigtable has no fixed schema, the tool scans a few rows to infer possible column qualifiers. "
    )
    args_schema: Type[BaseModel] = BigtableGetInstancesAndTableSchemaArgs
    _client: Client

    def __init__(
        self,
        client: Client,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._client = client

    def get_instances(self):
        """
        Retrieve all instances in the project (synchronous).
        """
        instances: List[Instance]
        instances, _ = self._client.list_instances()
        return [instance.instance_id for instance in instances]

    def get_tables(self, instance_id: str):
        """
        Retrieve all tables in a specific instance (synchronous).
        """
        instance = self._client.instance(instance_id)
        tables = instance.list_tables()
        return [table.table_id for table in tables]

    def get_column_families(self, instance_id: str, table_id: str):
        """
        Retrieve all column families in a specific table (synchronous).
        """
        instance = self._client.instance(instance_id)
        table = instance.table(table_id)
        column_families = table.list_column_families()
        return list(column_families.keys())

    def get_possible_columns(
        self,
        instance_id: str,
        table_id: str,
        row_limit: Optional[int] = DEFAULT_COL_QUALIFIER_SCAN_LIMIT,
    ):
        """
        Retrieve possible columns by scanning the first (possibly few) rows of a table.
        """
        instance = self._client.instance(instance_id)
        table = instance.table(table_id)
        rows = table.read_rows(limit=row_limit)
        rows.consume_all()

        possible_columns = set()
        for row in rows.rows.values():
            for family_name, columns in row.cells.items():
                for column_name in columns.keys():
                    # Format as `family['qualifier']` to match Bigtable SQL syntax
                    possible_columns.add(f"{family_name}['{column_name.decode('utf-8')}']")

        return list(possible_columns)

    def get_metadata(self):
        """
        Retrieve all instances, tables, and column families (synchronous).
        """
        metadata = {}
        instances = self.get_instances()

        for instance_id in instances:
            metadata[instance_id] = {}
            tables = self.get_tables(instance_id)
            for table_id in tables:
                # TODO: write a comment on why this is still necessary.
                column_families = self.get_column_families(instance_id, table_id)
                possible_columns = self.get_possible_columns(instance_id, table_id)
                metadata[instance_id][table_id] = {
                    "column_families": column_families,
                    "possible_columns": possible_columns,
                }
        return metadata

    def _run(self, *args, **kwargs):
        """
        Implementation of the abstract method `_run` (synchronous).
        """
        return self.get_metadata()

