import os
import pytest
import pytest_asyncio
from google.cloud import bigtable
from google.cloud.bigtable.data import BigtableDataClient, BigtableDataClientAsync

""""
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

@pytest.fixture(scope="session")
def sync_data_client(project_id: str):
    """
    Fixture to create a Bigtable client.
    """
    try:
        client = BigtableDataClient(project=project_id, admin=True)
        yield client
    finally:
        client.close()

@pytest_asyncio.fixture(scope="function")
async def async_data_client(project_id: str):
    """
    Fixture to create an async Bigtable client.
    """
    try:
        client = BigtableDataClientAsync(project=project_id, admin=True)
        yield client
    finally:
        await client.close()