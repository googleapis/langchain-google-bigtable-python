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
import os
from typing import Generator

import pytest
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
def admin_client(project_id: str) -> Generator[bigtable.Client, None, None]:
    """
    Fixture to create a Bigtable client.
    """
    client = bigtable.Client(project=project_id, admin=True)
    yield client


@pytest.fixture(scope="session")
def bigtable_engine(project_id: str) -> BigtableEngine:
    """
    Fixture to create a BigtableEngine instance for testing.
    """
    return BigtableEngine.initialize(project_id=project_id)
