"""Tests run against a throwaway SQLite database and artifact directory."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="sentinelgraph-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_tmp, 'test.db')}"
os.environ["ARTIFACT_DIR"] = os.path.join(_tmp, "artifacts")
os.environ.setdefault("GNN_BACKEND", "sgc")  # fast CPU fallback; GraphSAGE is exercised by the real seed

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def seeded():
    from app.cli import seed

    seed(n_accounts=500)
    return True


@pytest.fixture(scope="session")
def client(seeded):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
