import os
import pytest
import pytest_asyncio
from google.cloud import bigtable
from langchain_google_bigtable.engine import BigtableEngine

"""
Fixtures for Google Bigtable tests.
These fixtures provide the necessary setup for testing Bigtable tools.
"""


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.fixture(scope="session")
def project_id() -> str:
    return get_env_var("PROJECT_ID", "GCP Project ID")


@pytest.fixture(scope="session")
def instance_id() -> str:
    return get_env_var("INSTANCE_ID", "Bigtable Instance ID")


@pytest.fixture(scope="session")
def admin_client(project_id: str):
    """
    Fixture to create a Bigtable client.
    """
    client = bigtable.Client(project=project_id, admin=True)
    yield client


@pytest_asyncio.fixture(scope="session")
def bigtable_engine(project_id):
    """
    Fixture to create a BigtableEngine instance for testing.
    """
    return BigtableEngine.initialize(project_id=project_id)
