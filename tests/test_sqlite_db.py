from __future__ import annotations

import pytest

from rag.db import SqliteAdapter, get_db_adapter


def test_sqlite_adapter_round_trip_and_source_management(tmp_path) -> None:
    db_path = tmp_path / "rag.sqlite3"
    adapter = SqliteAdapter(f"sqlite://{db_path}", "embeddings")
    adapter.setup(embedding_dim=2)
    adapter.add(
        [
            {"text": "alpha", "source": "a.txt", "vector": [1.0, 0.0]},
            {"text": "beta", "source": "b.txt", "vector": [0.0, 1.0]},
        ]
    )

    assert adapter.exists()
    assert adapter.query(query_vector=[0.9, 0.1], k=1) == ["alpha"]
    assert adapter.has_source("a.txt")
    assert adapter.list_sources() == ["a.txt", "b.txt"]
    assert adapter.info()["rows"] == 2
    assert adapter.info()["embedding_dim"] == 2

    adapter.delete_source("b.txt")

    assert not adapter.has_source("b.txt")
    assert adapter.list_sources() == ["a.txt"]


def test_sqlite_adapter_rejects_dimension_mismatch(tmp_path) -> None:
    db_path = tmp_path / "rag.sqlite3"
    adapter = SqliteAdapter(f"sqlite://{db_path}", "embeddings")
    adapter.setup(embedding_dim=2)
    adapter.add([{"text": "alpha", "source": "a.txt", "vector": [1.0, 0.0]}])

    with pytest.raises(RuntimeError, match="query dim 3 doesn't match index dim 2"):
        adapter.query(query_vector=[1.0, 0.0, 0.0], k=1)


def test_sqlite_adapter_uses_content_hash_for_idempotent_upserts(tmp_path) -> None:
    db_path = tmp_path / "rag.sqlite3"
    adapter = SqliteAdapter(f"sqlite://{db_path}", "embeddings")
    adapter.setup(embedding_dim=2)

    adapter.add([{"text": "same", "source": "a.txt", "vector": [1.0, 0.0]}])
    adapter.add([{"text": "same", "source": "a.txt", "vector": [0.0, 1.0]}])

    assert adapter.info()["rows"] == 1
    assert adapter.query(query_vector=[0.0, 1.0], k=1) == ["same"]


def test_sqlite_adapter_routing(tmp_path) -> None:
    adapter = get_db_adapter(f"sqlite://{tmp_path / 'rag.sqlite3'}", "embeddings")
    assert isinstance(adapter, SqliteAdapter)
