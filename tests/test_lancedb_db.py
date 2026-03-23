from __future__ import annotations

from rag.db import LanceDbAdapter, QueryChunk, get_db_adapter


def test_lancedb_adapter_round_trip_and_source_management(tmp_path) -> None:
    adapter = LanceDbAdapter(str(tmp_path / "db"), "embeddings")
    adapter.setup(embedding_dim=2)
    adapter.add(
        [
            {"text": "alpha", "source": "a.txt", "vector": [1.0, 0.0]},
            {"text": "beta", "source": "o'hara.txt", "vector": [0.0, 1.0]},
        ]
    )

    assert adapter.exists()
    assert adapter.query(query_vector=[0.9, 0.1], k=1) == ["alpha"]
    assert adapter.query_chunks(query_vector=[0.9, 0.1], k=2) == [
        QueryChunk(text="alpha", source="a.txt"),
        QueryChunk(text="beta", source="o'hara.txt"),
    ]
    assert adapter.has_source("a.txt")
    assert adapter.has_source("o'hara.txt")
    assert adapter.list_sources() == ["a.txt", "o'hara.txt"]
    assert adapter.info()["rows"] == 2

    adapter.delete_source("o'hara.txt")

    assert not adapter.has_source("o'hara.txt")
    assert adapter.list_sources() == ["a.txt"]


def test_get_db_adapter_defaults_to_lancedb() -> None:
    adapter = get_db_adapter("./db", "embeddings")
    assert isinstance(adapter, LanceDbAdapter)
