from __future__ import annotations

import pytest

from rag.db import _bedrock_kb_parse_uri, get_db_adapter, BedrockKbAdapter


# ── URI parsing ──────────────────────────────────────────────────────────────

def test_bedrock_kb_parse_uri_kb_only() -> None:
    kb_id, ds_id = _bedrock_kb_parse_uri("bedrock-kb://KBID123")
    assert kb_id == "KBID123"
    assert ds_id is None


def test_bedrock_kb_parse_uri_with_data_source() -> None:
    kb_id, ds_id = _bedrock_kb_parse_uri("bedrock-kb://KBID123/DSID456")
    assert kb_id == "KBID123"
    assert ds_id == "DSID456"


def test_bedrock_kb_parse_uri_rejects_bad_scheme() -> None:
    with pytest.raises(ValueError, match="Invalid Bedrock KB URI"):
        _bedrock_kb_parse_uri("s3://bucket/path")


def test_bedrock_kb_parse_uri_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="Knowledge Base ID"):
        _bedrock_kb_parse_uri("bedrock-kb://")


# ── Adapter behaviour ────────────────────────────────────────────────────────

def _make_adapter(monkeypatch, kb_id: str = "TESTKB", ds_id: str | None = None) -> BedrockKbAdapter:
    """Return a BedrockKbAdapter with BedrockKbService fully stubbed out."""

    class StubService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_knowledge_base(self) -> dict:
            return {"knowledgeBaseId": kb_id}

        def upload_files(self, paths, ds_id=None) -> str:
            return "job-123"

        def wait_for_ingestion(self, job_id, ds_id=None) -> None:
            pass

        def retrieval_query(self, text, top_k=5) -> list[str]:
            return ["chunk1", "chunk2"]

        def s3_object_exists(self, key, ds_id=None) -> bool:
            return False

        def delete_s3_object(self, key, ds_id=None) -> None:
            pass

        def list_s3_objects(self, ds_id=None) -> list[str]:
            return []

    monkeypatch.setattr("rag.bedrock_kb.BedrockKbService", StubService)

    uri = f"bedrock-kb://{kb_id}" + (f"/{ds_id}" if ds_id else "")
    return BedrockKbAdapter(uri, "embeddings")


def test_bedrock_kb_adapter_add_expects_file_paths(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    with pytest.raises(NotImplementedError, match="expects rows with 'path'"):
        adapter.add([{"text": "hello", "vector": [0.1, 0.2]}])


def test_bedrock_kb_adapter_query_raises(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    with pytest.raises(NotImplementedError, match="retrieves by text"):
        adapter.query(query_vector=[0.1, 0.2], k=5)


def test_bedrock_kb_adapter_has_query_by_text(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert hasattr(adapter, "query_by_text")
    result = adapter.query_by_text("What is AFM?", 3)
    assert result == ["chunk1", "chunk2"]


def test_bedrock_kb_adapter_has_wait_for_ingestion(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch)
    assert hasattr(adapter, "wait_for_ingestion")


def test_bedrock_kb_adapter_routing(monkeypatch) -> None:
    """get_db_adapter('bedrock-kb://X', 't') returns BedrockKbAdapter."""

    class StubService:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("rag.bedrock_kb.BedrockKbService", StubService)
    adapter = get_db_adapter("bedrock-kb://MYKB", "embeddings")
    assert isinstance(adapter, BedrockKbAdapter)
