from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunk import Chunk
from rag.config import RagConfig
from rag.db import QueryChunk
from rag.services import IngestionService, QueryService


class StubVectorAdapter:
    def __init__(self, *, exists: bool = True, existing_sources: set[str] | None = None) -> None:
        self._exists = exists
        self.existing_sources = set(existing_sources or set())
        self.added_rows: list[dict] = []
        self.deleted_sources: list[str] = []
        self.embedding_dims: list[int] = []
        self.recorded_embedding_configs: list[tuple[str, str, int]] = []
        self.validated_embedding_configs: list[tuple[str, str]] = []
        self.preflight_called = False

    def preflight(self) -> None:
        self.preflight_called = True

    def exists(self) -> bool:
        return self._exists

    def has_source(self, source: str) -> bool:
        return source in self.existing_sources

    def delete_source(self, source: str) -> None:
        self.deleted_sources.append(source)
        self.existing_sources.discard(source)

    def setup(self, *, embedding_dim: int) -> None:
        self.embedding_dims.append(embedding_dim)

    def record_embedding_config(
        self,
        *,
        embed_provider: str,
        embed_model: str,
        embedding_dim: int,
    ) -> None:
        self.recorded_embedding_configs.append((embed_provider, embed_model, embedding_dim))

    def validate_embedding_config(self, *, embed_provider: str, embed_model: str) -> None:
        self.validated_embedding_configs.append((embed_provider, embed_model))

    def add(self, rows: list[dict]) -> None:
        self.added_rows.extend(rows)

    def query(self, *, query_vector: list[float], k: int) -> list[str]:
        assert query_vector == [0.42, 0.24]
        assert k == 3
        return ["vector-result"]

    def query_chunks(self, *, query_vector: list[float], k: int) -> list[QueryChunk]:
        assert query_vector == [0.42, 0.24]
        assert k == 3
        return [QueryChunk(text="vector-result", source="doc.txt")]


class StubVertexLikeAdapter:
    def exists(self) -> bool:
        return True

    def query_by_text(self, question: str, k: int) -> list[str]:
        assert question == "What is AFM?"
        assert k == 3
        return ["text-result"]

    def query_chunks_by_text(self, question: str, k: int) -> list[QueryChunk]:
        assert question == "What is AFM?"
        assert k == 3
        return [QueryChunk(text="text-result")]


class StubMissingDbAdapter:
    def exists(self) -> bool:
        return False


class StubEmbedder:
    dim = 2

    def __init__(self, *, question_vector: list[float] | None = None) -> None:
        self.question_vector = question_vector or [0.42, 0.24]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts == ["What is AFM?"]:
            return [self.question_vector]
        return [[0.1, 0.2] for _ in texts]


def test_query_service_uses_query_by_text_when_available(monkeypatch) -> None:
    cfg = RagConfig(top_k=3)

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: StubVertexLikeAdapter())

    result = QueryService().retrieve(cfg, "What is AFM?")
    assert result.texts == ["text-result"]
    assert result.source_counts == []


def test_query_service_uses_vector_search_when_needed(monkeypatch) -> None:
    cfg = RagConfig(top_k=3)

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: StubVectorAdapter())
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    result = QueryService().retrieve(cfg, "What is AFM?")
    assert result.texts == ["vector-result"]
    assert result.source_counts == [("doc.txt", 1)]
    assert result.chunks[0].source == "doc.txt"


def test_query_service_validates_vector_backend_embedding_config(monkeypatch) -> None:
    cfg = RagConfig(top_k=3, embed_provider="openai", embed_model="text-embedding-3-small")
    adapter = StubVectorAdapter()

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    QueryService().retrieve(cfg, "What is AFM?")

    assert adapter.validated_embedding_configs == [("openai", "text-embedding-3-small")]


def test_query_service_raises_when_db_missing(monkeypatch) -> None:
    cfg = RagConfig()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: StubMissingDbAdapter())

    with pytest.raises(RuntimeError, match="No database found"):
        QueryService().retrieve(cfg, "What is AFM?")


def test_ingestion_service_single_file_emits_progress_and_writes(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello", encoding="utf-8")
    adapter = StubVectorAdapter()

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr(
        "rag.services.chunk_file",
        lambda *args, **kwargs: [
            Chunk(text="chunk-a", source="doc.txt"),
            Chunk(text="chunk-b", source="doc.txt"),
        ],
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

    assert adapter.preflight_called
    assert result.mode == "vector"
    assert result.rows_written == 2
    assert len(adapter.added_rows) == 2
    assert adapter.embedding_dims == [2]
    assert adapter.recorded_embedding_configs == [("mistral", "mistral-embed", 2)]
    assert {row["source"] for row in adapter.added_rows} == {"doc.txt"}
    assert events == [
        ("start", "chunking"),
        ("end", "chunking"),
        ("start", "embedding"),
        ("end", "embedding"),
        ("start", "writing"),
        ("end", "writing"),
    ]


def test_ingestion_service_skips_existing_file_without_overwrite(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello", encoding="utf-8")
    adapter = StubVectorAdapter(existing_sources={"doc.txt"})

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
    assert adapter.deleted_sources == []


def test_ingestion_service_single_file_overwrite_deletes_only_after_successful_chunk(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello", encoding="utf-8")
    adapter = StubVectorAdapter(existing_sources={"doc.txt"})

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr(
        "rag.services.chunk_file",
        lambda *args, **kwargs: [Chunk(text="chunk-a", source="doc.txt")],
    )
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    IngestionService().ingest(cfg, source_file, skip_extensions=set(), overwrite=True)

    assert adapter.deleted_sources == ["doc.txt"]
    assert len(adapter.added_rows) == 1


def test_ingestion_service_single_file_overwrite_preserves_existing_rows_on_chunk_failure(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello", encoding="utf-8")
    adapter = StubVectorAdapter(existing_sources={"doc.txt"})

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr("rag.services.chunk_file", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("broken file")))

    with pytest.raises(RuntimeError, match="Failed to chunk doc.txt: broken file"):
        IngestionService().ingest(cfg, source_file, skip_extensions=set(), overwrite=True)

    assert adapter.deleted_sources == []
    assert adapter.added_rows == []


def test_ingestion_service_raises_when_chunking_returns_no_chunks(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello", encoding="utf-8")

    adapter = StubVectorAdapter()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr("rag.services.chunk_file", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match="No chunks extracted"):
        IngestionService().ingest(cfg, source_file, skip_extensions=set(), overwrite=False)


def test_ingestion_service_collects_directory_failures(monkeypatch, tmp_path: Path) -> None:
    cfg = RagConfig()
    file_ok = tmp_path / "ok.txt"
    file_bad = tmp_path / "bad.txt"
    file_ok.write_text("ok", encoding="utf-8")
    file_bad.write_text("bad", encoding="utf-8")

    adapter = StubVectorAdapter()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr(
        "rag.services.discover_chunkable_files",
        lambda *args, **kwargs: ([file_ok, file_bad], []),
    )

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
    assert adapter.deleted_sources == []


def test_ingestion_service_directory_overwrite_deletes_only_successfully_chunked_sources(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig()
    file_ok = tmp_path / "ok.txt"
    file_bad = tmp_path / "bad.txt"
    file_ok.write_text("ok", encoding="utf-8")
    file_bad.write_text("bad", encoding="utf-8")
    adapter = StubVectorAdapter(existing_sources={"ok.txt", "bad.txt"})

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr(
        "rag.services.discover_chunkable_files",
        lambda *args, **kwargs: ([file_ok, file_bad], []),
    )

    def fake_chunk(path: Path, **kwargs) -> list[Chunk]:
        if path.name == "bad.txt":
            raise ValueError("broken file")
        return [Chunk(text="good", source=path.name)]

    monkeypatch.setattr("rag.services.chunk_file", fake_chunk)
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    result = IngestionService().ingest(cfg, tmp_path, skip_extensions=set(), overwrite=True)

    assert result.rows_written == 1
    assert adapter.deleted_sources == ["ok.txt"]
    assert len(result.failures) == 1
    assert result.failures[0][0].name == "bad.txt"


def test_ingestion_service_raises_on_embedding_count_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig()
    source_file = tmp_path / "doc.txt"
    source_file.write_text("hello", encoding="utf-8")
    adapter = StubVectorAdapter()

    class ShortEmbedder(StubEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]]

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr(
        "rag.services.chunk_file",
        lambda *args, **kwargs: [
            Chunk(text="chunk-a", source="doc.txt"),
            Chunk(text="chunk-b", source="doc.txt"),
        ],
    )
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: ShortEmbedder())

    with pytest.raises(RuntimeError, match="different number of vectors than chunks"):
        IngestionService().ingest(cfg, source_file, skip_extensions=set(), overwrite=False)

    assert adapter.added_rows == []


@pytest.mark.parametrize("db_uri", ["./db", "weaviate://localhost:8080"])
def test_ingestion_service_ingests_unstructured_directory_for_vector_backends(
    monkeypatch, tmp_path: Path, db_uri: str
) -> None:
    cfg = RagConfig(db=db_uri, chunk_method="unstructured")
    adapter = StubVectorAdapter()
    for name in ["a.pdf", "b.docx", "c.pptx", "d.xlsx", "e.md", "f.txt", ".hidden.docx"]:
        (tmp_path / name).write_text(name, encoding="utf-8")

    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr(
        "rag.services.chunk_file",
        lambda path, **kwargs: [Chunk(text=f"chunk:{path.name}", source=path.name)],
    )
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    result = IngestionService().ingest(cfg, tmp_path, skip_extensions=set(), overwrite=False)

    assert result.rows_written == 6
    assert result.failures == []
    assert {row["source"] for row in adapter.added_rows} == {
        "a.pdf",
        "b.docx",
        "c.pptx",
        "d.xlsx",
        "e.md",
        "f.txt",
    }
    assert all(row["source"] != ".hidden.docx" for row in adapter.added_rows)


def test_ingestion_service_reports_ignored_hidden_and_unsupported_files(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig()
    (tmp_path / "ok.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "slides.pptx").write_text("slides", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("hidden", encoding="utf-8")

    adapter = StubVectorAdapter()
    monkeypatch.setattr("rag.services.get_db_adapter", lambda uri, table: adapter)
    monkeypatch.setattr(
        "rag.services.chunk_file",
        lambda path, **kwargs: [Chunk(text=f"chunk:{path.name}", source=path.name)],
    )
    monkeypatch.setattr("rag.services.get_embed_adapter", lambda provider, model: StubEmbedder())

    result = IngestionService().ingest(cfg, tmp_path, skip_extensions=set(), overwrite=False)

    assert result.rows_written == 1
    assert [(ignored.path.name, ignored.reason) for ignored in result.ignored_files] == [
        (".hidden.txt", "hidden file"),
        ("slides.pptx", "unsupported file type (.pptx)"),
    ]


def test_ingestion_service_bedrock_kb_emits_upload_and_indexing_progress(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = RagConfig(db="bedrock-kb://MYKB")
    source_file = tmp_path / "doc.pdf"
    source_file.write_text("pdf", encoding="utf-8")

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
    assert adapter.uploaded_rows == [
        {"path": str(source_file.resolve()), "source": "doc.pdf"}
    ]
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
    source_file.write_text("pdf", encoding="utf-8")

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
    source_file.write_text("pdf", encoding="utf-8")

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
    assert adapter.uploaded_rows == [
        {"path": str(source_file.resolve()), "source": "doc.pdf"}
    ]
    assert events == [("start", "uploading"), ("end", "uploading")]
