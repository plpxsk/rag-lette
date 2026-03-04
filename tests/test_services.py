from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunk import Chunk
from rag.config import RagConfig
from rag.services import IngestionService, QueryService


class StubVectorAdapter:
    def __init__(self) -> None:
        self.added_rows: list[dict] = []

    def preflight(self) -> None:
        return

    def exists(self) -> bool:
        return True

    def has_source(self, source: str) -> bool:
        return False

    def delete_source(self, source: str) -> None:
        return

    def setup(self, *, embedding_dim: int) -> None:
        return

    def add(self, rows: list[dict]) -> None:
        self.added_rows.extend(rows)

    def query(self, *, query_vector: list[float], k: int) -> list[str]:
        assert query_vector == [0.42, 0.24]
        assert k == 3
        return ["vector-result"]


class StubVertexLikeAdapter:
    def exists(self) -> bool:
        return True

    def query_by_text(self, question: str, k: int) -> list[str]:
        assert question == "What is AFM?"
        assert k == 3
        return ["text-result"]


class StubMissingDbAdapter:
    def exists(self) -> bool:
        return False


class StubEmbedder:
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts == ["What is AFM?"]:
            return [[0.42, 0.24]]
        return [[0.1, 0.2] for _ in texts]


def test_query_service_uses_query_by_text_when_available(monkeypatch) -> None:
    cfg = RagConfig(top_k=3)

    def fake_get_db_adapter(uri: str, table: str) -> StubVertexLikeAdapter:
        return StubVertexLikeAdapter()

    monkeypatch.setattr("rag.services.get_db_adapter", fake_get_db_adapter)
    result = QueryService().retrieve(cfg, "What is AFM?")
    assert result == ["text-result"]


def test_query_service_uses_vector_search_when_needed(monkeypatch) -> None:
    cfg = RagConfig(top_k=3)

    def fake_get_db_adapter(uri: str, table: str) -> StubVectorAdapter:
        return StubVectorAdapter()

    monkeypatch.setattr("rag.services.get_db_adapter", fake_get_db_adapter)
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    result = QueryService().retrieve(cfg, "What is AFM?")
    assert result == ["vector-result"]


def test_query_service_raises_when_db_missing(monkeypatch) -> None:
    cfg = RagConfig()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: StubMissingDbAdapter())
    with pytest.raises(RuntimeError, match="No database found"):
        QueryService().retrieve(cfg, "What is AFM?")


def test_ingestion_service_single_file_emits_progress_and_writes(monkeypatch, tmp_path: Path) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello")
    adapter = StubVectorAdapter()

    def fake_get_db_adapter(uri: str, table: str) -> StubVectorAdapter:
        return adapter

    monkeypatch.setattr("rag.services.get_db_adapter", fake_get_db_adapter)
    monkeypatch.setattr(
        "rag.services.chunk_file",
        lambda *args, **kwargs: [Chunk(text="chunk-a", source="doc.txt"), Chunk(text="chunk-b", source="doc.txt")],
    )
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    events: list[tuple[str, str]] = []

    result = IngestionService().ingest(
        cfg,
        source_file,
        skip_extensions=set(),
        overwrite=False,
        progress=lambda event, stage: events.append((event, stage)),
    )

    assert result.mode == "vector"
    assert result.rows_written == 2
    assert len(adapter.added_rows) == 2
    assert events == [
        ("start", "chunking"),
        ("end", "chunking"),
        ("start", "embedding"),
        ("end", "embedding"),
        ("start", "writing"),
        ("end", "writing"),
    ]


def test_ingestion_service_skips_existing_file_without_overwrite(monkeypatch, tmp_path: Path) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello")

    class ExistingAdapter(StubVectorAdapter):
        def has_source(self, source: str) -> bool:
            return True

    adapter = ExistingAdapter()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)

    result = IngestionService().ingest(
        cfg,
        source_file,
        skip_extensions=set(),
        overwrite=False,
    )

    assert result.rows_written == 0
    assert result.skipped_files == ["doc.txt"]
    assert adapter.added_rows == []


def test_ingestion_service_raises_when_chunking_returns_no_chunks(monkeypatch, tmp_path: Path) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello")

    adapter = StubVectorAdapter()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr("rag.services.chunk_file", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match="No chunks extracted"):
        IngestionService().ingest(cfg, source_file, skip_extensions=set(), overwrite=False)


def test_ingestion_service_collects_directory_failures(monkeypatch, tmp_path: Path) -> None:
    cfg = RagConfig()
    file_ok = tmp_path / "ok.txt"
    file_bad = tmp_path / "bad.txt"
    file_ok.write_text("ok")
    file_bad.write_text("bad")

    adapter = StubVectorAdapter()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr("rag.services.list_chunkable_files", lambda *args, **kwargs: [file_ok, file_bad])

    def fake_chunk(path: Path, **kwargs) -> list[Chunk]:
        if path.name == "bad.txt":
            raise ValueError("broken file")
        return [Chunk(text="good", source=path.name)]

    monkeypatch.setattr("rag.services.chunk_file", fake_chunk)
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    result = IngestionService().ingest(cfg, tmp_path, skip_extensions=set(), overwrite=False)
    assert result.rows_written == 1
    assert len(result.failures) == 1
    assert result.failures[0][0].name == "bad.txt"


def test_ingestion_service_bedrock_kb_emits_upload_and_indexing_progress(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig(db="bedrock-kb://MYKB")
    source_file = tmp_path / "doc.pdf"
    source_file.write_text("pdf")

    class StubBedrockKbAdapter:
        def __init__(self) -> None:
            self.uploaded_rows: list[dict] = []
            self.indexing_called = False

        def preflight(self) -> None:
            return

        def exists(self) -> bool:
            return False

        def has_source(self, source: str) -> bool:
            return False

        def delete_source(self, source: str) -> None:
            return

        def setup(self, *, embedding_dim: int) -> None:
            return

        def add(self, rows: list[dict]) -> None:
            self.uploaded_rows.extend(rows)

        def wait_for_ingestion(self) -> None:
            self.indexing_called = True

    adapter = StubBedrockKbAdapter()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)

    events: list[tuple[str, str]] = []
    result = IngestionService().ingest(
        cfg,
        source_file,
        skip_extensions=set(),
        overwrite=False,
        progress=lambda event, stage: events.append((event, stage)),
    )

    assert result.mode == "bedrock-kb"
    assert result.rows_written == 1
    assert adapter.indexing_called
    assert events == [
        ("start", "uploading"),
        ("end", "uploading"),
        ("start", "indexing"),
        ("end", "indexing"),
    ]


def test_ingestion_service_bedrock_kb_skips_indexing_without_wait(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig(db="bedrock-kb://MYKB")
    source_file = tmp_path / "doc.pdf"
    source_file.write_text("pdf")

    class StubBedrockKbAdapterNoWait:
        def preflight(self) -> None:
            return

        def exists(self) -> bool:
            return False

        def has_source(self, source: str) -> bool:
            return False

        def delete_source(self, source: str) -> None:
            return

        def setup(self, *, embedding_dim: int) -> None:
            return

        def add(self, rows: list[dict]) -> None:
            pass

    adapter = StubBedrockKbAdapterNoWait()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)

    events: list[tuple[str, str]] = []
    result = IngestionService().ingest(
        cfg,
        source_file,
        skip_extensions=set(),
        overwrite=False,
        progress=lambda event, stage: events.append((event, stage)),
    )

    assert result.rows_written == 1
    assert events == [("start", "uploading"), ("end", "uploading")]


def test_ingestion_service_vertex_upload_emits_progress(monkeypatch, tmp_path: Path) -> None:
    cfg = RagConfig(db="vertex://project/corpus")
    source_file = tmp_path / "doc.pdf"
    source_file.write_text("pdf")

    class StubVertexAdapter:
        def __init__(self) -> None:
            self.uploaded_rows: list[dict] = []

        def preflight(self) -> None:
            return

        def exists(self) -> bool:
            return False

        def has_source(self, source: str) -> bool:
            return False

        def delete_source(self, source: str) -> None:
            return

        def setup(self, *, embedding_dim: int) -> None:
            return

        def add(self, rows: list[dict]) -> None:
            self.uploaded_rows.extend(rows)

    adapter = StubVertexAdapter()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)

    events: list[tuple[str, str]] = []
    result = IngestionService().ingest(
        cfg,
        source_file,
        skip_extensions=set(),
        overwrite=False,
        progress=lambda event, stage: events.append((event, stage)),
    )

    assert result.mode == "vertex"
    assert result.rows_written == 1
    assert len(adapter.uploaded_rows) == 1
    assert events == [("start", "uploading"), ("end", "uploading")]
