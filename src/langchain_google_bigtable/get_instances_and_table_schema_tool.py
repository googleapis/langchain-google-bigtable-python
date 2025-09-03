from google.cloud.bigtable import Client
from google.cloud.bigtable.instance import Instance
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from typing import List
from google.cloud.bigtable.row_data import PartialRowsData

# The number of rows to scan to find possible column qualifiers.
DEFAULT_COL_QUALIFIER_SCAN_LIMIT = 1  

@dataclass
class FamilyQualifiers:
    column_family_name: str
    column_qualifiers: List[str] = field(default_factory=list)

def extract_family_qualifiers(rows: PartialRowsData) -> List[FamilyQualifiers]:
    """
    Extracts a list of FamilyQualifiers, each containing a family name and a sorted list of qualifiers.
    Raises ValueError if no valid data found.
    """
    if not hasattr(rows, "rows") or not rows.rows:
        raise ValueError("No rows data found for extracting family qualifiers.")
    family_map = {}
    for row in rows.rows.values():
        for family_name, qualifier_cells in row.cells.items():
            if family_name not in family_map:
                family_map[family_name] = set()
            for qualifier in qualifier_cells.keys():
                family_map[family_name].add(qualifier.decode('utf-8'))
    if not family_map:
        raise ValueError("No column families or qualifiers found in the scanned rows.")
    return [FamilyQualifiers(family, sorted(list(qualifiers))) for family, qualifiers in family_map.items()]


def format_family_qualifiers(family_qualifiers: List[FamilyQualifiers]) -> List[str]:
    """
    Formats the list of FamilyQualifiers into a list like: ["cf1['col1']", "cf1['col2']", "cf2['col1']"]
    """
    qualified_columns = []
    for fq in family_qualifiers:
        for qualifier in fq.column_qualifiers:
            qualified_columns.append(f"{fq.column_family_name}['{qualifier}']")
    return qualified_columns

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
    ) -> List[str]:
        """
        Retrieve possible columns by scanning the first (possibly few) rows of a table.
        """
        try:
            instance = self._client.instance(instance_id)
            table = instance.table(table_id)
            rows = table.read_rows(limit=row_limit)
            rows.consume_all()
            family_qualifiers = extract_family_qualifiers(rows)
            return format_family_qualifiers(family_qualifiers)
        except Exception as e:
            return f"Error extracting possible columns: {str(e)}"


    def get_metadata(self) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        """
        Retrieve all instances, tables, and column families (synchronous).
        """
        metadata = {}
        instances = self.get_instances()

        for instance_id in instances:
            metadata[instance_id] = {}
            tables = self.get_tables(instance_id)
            for table_id in tables:
                """
                We explicitly retrieve column families here because scanning rows may miss
                families that have no data in the sampled rows. Only the metadata API can
                guarantee a complete list of all defined column families, even for empty tables.
                """
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
